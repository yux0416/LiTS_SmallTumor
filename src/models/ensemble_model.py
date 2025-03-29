# src/models/ensemble_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import argparse
import time

# 添加项目根目录到路径
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

# 导入工具函数
from src.utils.metrics import dice_coefficient
from src.models.attention_unet import get_attention_unet_model
from src.models.unet import UNet
from dataloader import LiTSDataset, get_transforms


class EnsembleModel(nn.Module):
    """
    集成模型类
    集成多个训练好的模型，根据加权平均进行预测
    """

    def __init__(self, models, weights=None):
        """
        初始化集成模型

        参数:
            models: 模型列表
            weights: 每个模型的权重列表，如果为None则使用相等权重
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)  # 使用ModuleList管理模型
        self.num_models = len(models)

        # 如果没有提供权重，则使用等权重
        if weights is None:
            self.weights = torch.ones(self.num_models) / self.num_models
        else:
            # 确保权重和为1
            weights = torch.tensor(weights)
            self.weights = weights / weights.sum()

        # 将权重注册为缓冲区，这样它会被正确地移动到设备上
        self.register_buffer('model_weights', self.weights.view(-1, 1, 1, 1))

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入图像

        返回:
            加权平均的预测结果
        """
        # 将所有模型设置为评估模式
        for model in self.models:
            model.eval()

        # 获取每个模型的预测
        with torch.no_grad():
            # 初始化加权结果为零张量
            device = x.device
            first_output = self.models[0](x)
            if isinstance(first_output, tuple):
                first_output = first_output[0]

            # 初始化与第一个模型输出相同形状的零张量
            weighted_sum = torch.zeros_like(first_output)

            # 手动计算加权和
            for i, model in enumerate(self.models):
                # 获取模型输出
                model_output = model(x)

                # 如果模型返回多个输出，只使用主输出
                if isinstance(model_output, tuple):
                    model_output = model_output[0]

                # 对输出应用softmax获取概率
                model_probs = F.softmax(model_output, dim=1)

                # 直接使用标量权重
                weight = self.weights[i].item()
                weighted_sum += model_probs * weight

            # 转换回logits形式
            # 添加小值防止log(0)
            ensemble_logits = torch.log(weighted_sum + 1e-7)

        return ensemble_logits


def load_model(model_path, model_type, attention_type=None, n_channels=3, n_classes=3, init_features=32):
    """
    加载预训练模型

    参数:
        model_path: 模型权重路径
        model_type: 模型类型
        attention_type: 注意力类型
        n_channels: 输入通道数
        n_classes: 输出类别数
        init_features: 初始特征数

    返回:
        加载好权重的模型
    """
    # 初始化模型
    if model_type == 'standard':
        model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            init_features=init_features
        )
    else:
        model = get_attention_unet_model(
            model_type=model_type,
            n_channels=n_channels,
            n_classes=n_classes,
            init_features=init_features,
            attention_type=attention_type
        )

    # 加载权重
    model.load_state_dict(torch.load(model_path))

    return model


def create_ensemble(models_info, device):
    """
    创建集成模型

    参数:
        models_info: 包含模型信息的列表，每个元素是一个字典
        device: 计算设备

    返回:
        集成模型
    """
    models = []
    weights = []

    for info in models_info:
        model = load_model(
            model_path=info['path'],
            model_type=info['type'],
            attention_type=info.get('attention_type'),
            n_channels=info.get('n_channels', 3),
            n_classes=info.get('n_classes', 3),
            init_features=info.get('init_features', 32)
        ).to(device)

        models.append(model)
        weights.append(info['weight'])

    # 创建集成模型
    ensemble = EnsembleModel(models, weights).to(device)

    return ensemble


def evaluate_ensemble(ensemble, val_loader, device, n_classes=3, tumor_class=2):
    """
    评估集成模型性能

    参数:
        ensemble: 集成模型
        val_loader: 验证数据加载器
        device: 计算设备
        n_classes: 类别数
        tumor_class: 肿瘤类别索引

    返回:
        评估指标
    """
    ensemble.eval()

    # 初始化指标
    total_dice = 0
    class_dices = {str(i): 0 for i in range(n_classes)}

    # 肿瘤检测指标
    tumor_metrics = {'tp': 0, 'fp': 0, 'fn': 0}

    # 小型肿瘤指标
    small_tumor_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    medium_tumor_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    large_tumor_metrics = {'tp': 0, 'fp': 0, 'fn': 0}

    batch_count = 0

    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # 获取数据
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # 处理weight_map (如果存在)
            weight_map = batch.get('weight_map')
            if weight_map is not None:
                weight_map = weight_map.to(device)

            # 前向传播
            outputs = ensemble(images)

            # 计算每个类别的Dice系数
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            for c in range(n_classes):
                dice = dice_coefficient((preds == c).float(), (masks == c).float())
                class_dices[str(c)] += dice.item()
                if c == tumor_class:  # 肿瘤类别
                    total_dice += dice.item()

            # 计算肿瘤TP, FP, FN
            pred_tumor = (preds == tumor_class).cpu().numpy()
            true_tumor = (masks == tumor_class).cpu().numpy()

            tumor_metrics['tp'] += np.sum(pred_tumor & true_tumor)
            tumor_metrics['fp'] += np.sum(pred_tumor & ~true_tumor)
            tumor_metrics['fn'] += np.sum(~pred_tumor & true_tumor)

            # 按肿瘤大小分类
            if weight_map is not None:
                weight_map = weight_map.cpu().numpy()

                # 小型肿瘤: 权重 > 0.8
                small_mask = (weight_map > 0.8) & true_tumor
                pred_small = pred_tumor & small_mask

                small_tumor_metrics['tp'] += np.sum(pred_small)
                small_tumor_metrics['fp'] += np.sum(pred_tumor & ~true_tumor & (weight_map > 0.8))
                small_tumor_metrics['fn'] += np.sum(~pred_tumor & small_mask)

                # 中型肿瘤: 0.4 < 权重 <= 0.8
                medium_mask = ((weight_map > 0.4) & (weight_map <= 0.8)) & true_tumor
                pred_medium = pred_tumor & medium_mask

                medium_tumor_metrics['tp'] += np.sum(pred_medium)
                medium_tumor_metrics['fp'] += np.sum(
                    pred_tumor & ~true_tumor & ((weight_map > 0.4) & (weight_map <= 0.8)))
                medium_tumor_metrics['fn'] += np.sum(~pred_tumor & medium_mask)

                # 大型肿瘤: 权重 <= 0.4
                large_mask = (weight_map <= 0.4) & true_tumor
                pred_large = pred_tumor & large_mask

                large_tumor_metrics['tp'] += np.sum(pred_large)
                large_tumor_metrics['fp'] += np.sum(pred_tumor & ~true_tumor & (weight_map <= 0.4))
                large_tumor_metrics['fn'] += np.sum(~pred_tumor & large_mask)

            batch_count += 1

    # 计算平均值
    avg_dice = total_dice / batch_count
    avg_class_dices = {k: v / batch_count for k, v in class_dices.items()}

    # 计算肿瘤指标
    tp, fp, fn = tumor_metrics['tp'], tumor_metrics['fp'], tumor_metrics['fn']
    tumor_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    tumor_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    tumor_f1 = 2 * tumor_precision * tumor_recall / (tumor_precision + tumor_recall) if (
                                                                                                    tumor_precision + tumor_recall) > 0 else 0

    # 计算小型肿瘤指标
    size_metrics = {}

    # 小型肿瘤
    tp, fp, fn = small_tumor_metrics['tp'], small_tumor_metrics['fp'], small_tumor_metrics['fn']
    small_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    small_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    small_f1 = 2 * small_precision * small_recall / (small_precision + small_recall) if (
                                                                                                    small_precision + small_recall) > 0 else 0

    size_metrics['small'] = {
        'precision': small_precision,
        'recall': small_recall,
        'f1': small_f1
    }

    # 中型肿瘤
    tp, fp, fn = medium_tumor_metrics['tp'], medium_tumor_metrics['fp'], medium_tumor_metrics['fn']
    medium_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    medium_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    medium_f1 = 2 * medium_precision * medium_recall / (medium_precision + medium_recall) if (
                                                                                                         medium_precision + medium_recall) > 0 else 0

    size_metrics['medium'] = {
        'precision': medium_precision,
        'recall': medium_recall,
        'f1': medium_f1
    }

    # 大型肿瘤
    tp, fp, fn = large_tumor_metrics['tp'], large_tumor_metrics['fp'], large_tumor_metrics['fn']
    large_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    large_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    large_f1 = 2 * large_precision * large_recall / (large_precision + large_recall) if (
                                                                                                    large_precision + large_recall) > 0 else 0

    size_metrics['large'] = {
        'precision': large_precision,
        'recall': large_recall,
        'f1': large_f1
    }

    # 整合所有指标
    metrics = {
        'dice': avg_dice,
        'class_dice': avg_class_dices,
        'tumor_metrics': {
            'precision': tumor_precision,
            'recall': tumor_recall,
            'f1': tumor_f1
        },
        'size_metrics': size_metrics
    }

    return metrics


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='创建和评估集成模型')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed',
                        help='预处理数据目录')
    parser.add_argument('--output_dir', type=str, default='results/models/ensemble',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载器工作进程数')
    parser.add_argument('--img_size', type=str, default='256,256',
                        help='图像大小，格式为"高,宽"')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置集成模型
    models_info = [
        {
            'path': 'results/models/attention/attention_local_contrast_20250324_224246/best_model.pth',
            'type': 'attention',
            'attention_type': 'local_contrast',
            'weight': 0.40  # 最高权重给予表现最好的模型
        },
        {
            'path': 'results/models/attention/attention_scale_aware_small_20250325_232710/best_model.pth',
            'type': 'attention',
            'attention_type': 'scale_aware_small',
            'weight': 0.25
        },
        {
            'path': 'results/models/attention/attention_enhanced_tumor_size_20250326_111517/best_model.pth',
            'type': 'attention',
            'attention_type': 'enhanced_tumor_size',
            'weight': 0.20
        },
        {
            'path': 'results/models/attention/attention_tuned_local_contrast_20250327_065046/best_model.pth',
            'type': 'attention',
            'attention_type': 'tuned_local_contrast',
            'weight': 0.15
        }
    ]

    # 创建集成模型
    print("创建集成模型...")
    ensemble = create_ensemble(models_info, device)
    print("集成模型创建完成")

    # 准备验证数据加载器
    img_size = tuple(map(int, args.img_size.split(',')))
    val_transform = get_transforms('val', img_size=img_size)

    val_dataset = LiTSDataset(
        data_dir=args.data_dir,
        slice_list_file=Path(args.data_dir) / "splits" / "val_slices.txt",
        transform=val_transform,
        phase="val",
        small_tumor_focus=True,
        return_weight_map=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"验证集大小: {len(val_dataset)} 切片")

    # 评估集成模型
    print("开始评估集成模型...")
    start_time = time.time()
    metrics = evaluate_ensemble(ensemble, val_loader, device)
    eval_time = time.time() - start_time
    print(f"评估完成，耗时: {eval_time:.2f}秒")

    # 保存评估结果
    import json
    metrics_file = output_dir / 'ensemble_metrics.json'
    with open(metrics_file, 'w') as f:
        # 将numpy数组转换为列表
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                metrics_serializable[k] = {}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, np.ndarray):
                        metrics_serializable[k][sub_k] = sub_v.tolist()
                    else:
                        metrics_serializable[k][sub_k] = sub_v
            elif isinstance(v, np.ndarray):
                metrics_serializable[k] = v.tolist()
            else:
                metrics_serializable[k] = v

        json.dump(metrics_serializable, f, indent=2)

    print(f"评估指标已保存到: {metrics_file}")

    # 打印关键指标
    print("\n集成模型性能:")
    print(f"整体Dice: {metrics['dice']:.4f}")

    print("\n各类别Dice:")
    for cls, dice in metrics['class_dice'].items():
        class_name = "背景" if cls == '0' else "肝脏" if cls == '1' else "肿瘤"
        print(f"  {class_name}: {dice:.4f}")

    print("\n小型肿瘤性能:")
    small_metrics = metrics['size_metrics']['small']
    print(f"  F1分数: {small_metrics['f1']:.4f}")
    print(f"  召回率: {small_metrics['recall']:.4f}")
    print(f"  精确率: {small_metrics['precision']:.4f}")

    # 保存模型架构和权重信息
    config = {
        'models': models_info,
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }

    with open(output_dir / 'ensemble_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"集成模型配置已保存到: {output_dir / 'ensemble_config.json'}")

    return ensemble, metrics


# 预测单个样本函数，用于可视化
def predict_sample(ensemble, image, device):
    """
    使用集成模型预测单个样本

    参数:
        ensemble: 集成模型
        image: 输入图像
        device: 计算设备

    返回:
        预测结果
    """
    ensemble.eval()
    with torch.no_grad():
        # 添加批次维度
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # 将图像移到设备上
        image = image.to(device)

        # 预测
        output = ensemble(image)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)

        # 移回CPU并转换为numpy数组
        pred = pred.cpu().numpy()

        # 如果是单个样本，去掉批次维度
        if pred.shape[0] == 1:
            pred = pred[0]

    return pred


if __name__ == "__main__":
    ensemble, metrics = main()