# temp_evaluate.py
import os
import json
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# 添加项目根目录到路径
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

# 导入相关模块
from src.models.unet import UNet
from src.models.attention_unet import get_attention_unet_model
from src.utils.losses import CombinedLoss
from dataloader import LiTSDataset, get_transforms


# 评估指标函数
def dice_coefficient(pred, target, smooth=1e-6):
    """计算Dice系数"""
    # 确保pred是二值化的
    if pred.dtype != torch.bool:
        pred = (pred > 0.5).float()

    # 获取维度信息
    batch_size = pred.size(0)

    # 展平tensors
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    # 计算交集和并集
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)

    # 计算Dice
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice.mean()


def class_wise_dice(pred, target, n_classes):
    """计算每个类别的Dice系数"""
    # 确保pred是分类结果
    if pred.dim() == 4:  # [B, C, H, W]
        pred = pred.argmax(dim=1)  # [B, H, W]

    # 计算每个类别的Dice
    dice_scores = []

    for i in range(n_classes):
        pred_i = (pred == i).float()
        target_i = (target == i).float()

        # 计算交集和并集
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()

        # 计算Dice
        dice_i = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice_i.item())

    return dice_scores


def compute_metrics(pred, target, class_idx=None):
    """计算分类指标"""
    if class_idx is not None:
        # 二分类情况 - 只关注特定类别
        pred_class = (pred == class_idx)
        target_class = (target == class_idx)
    else:
        # 使用所有类别
        pred_class = pred
        target_class = target

    # 转换为numpy数组
    if isinstance(pred_class, torch.Tensor):
        pred_class = pred_class.cpu().numpy()
    if isinstance(target_class, torch.Tensor):
        target_class = target_class.cpu().numpy()

    # 计算TP, FP, FN
    tp = np.sum(pred_class & target_class)
    fp = np.sum(pred_class & ~target_class)
    fn = np.sum(~pred_class & target_class)

    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_custom(model, dataloader, criterion, device, n_classes=3, tumor_class=2):
    """评估模型性能"""
    model.eval()

    # 初始化指标
    total_loss = 0
    total_dice = 0
    class_dices = [0] * n_classes
    tumor_metrics = {'tp': 0, 'fp': 0, 'fn': 0}

    # 小型肿瘤指标
    small_tumor_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    medium_tumor_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    large_tumor_metrics = {'tp': 0, 'fp': 0, 'fn': 0}

    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 获取数据
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            weight_map = batch.get('weight_map')
            if weight_map is not None:
                weight_map = weight_map.to(device)
                loss = criterion(outputs, masks, weight_map)
            else:
                loss = criterion(outputs, masks)

            total_loss += loss.item()

            # 获取预测
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            # 计算整体Dice系数
            batch_dice = 0
            for c in range(n_classes):
                dice = dice_coefficient((preds == c).float(), (masks == c).float())
                class_dices[c] += dice.item()
                if c == tumor_class:  # 肿瘤类别
                    batch_dice = dice.item()  # 使用肿瘤的Dice

            total_dice += batch_dice

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

            num_batches += 1

    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_class_dices = [d / num_batches for d in class_dices]

    # 计算肿瘤指标
    tumor_precision = tumor_metrics['tp'] / (tumor_metrics['tp'] + tumor_metrics['fp']) if (tumor_metrics['tp'] +
                                                                                            tumor_metrics[
                                                                                                'fp']) > 0 else 0
    tumor_recall = tumor_metrics['tp'] / (tumor_metrics['tp'] + tumor_metrics['fn']) if (tumor_metrics['tp'] +
                                                                                         tumor_metrics['fn']) > 0 else 0
    tumor_f1 = 2 * tumor_precision * tumor_recall / (tumor_precision + tumor_recall) if (
                                                                                                    tumor_precision + tumor_recall) > 0 else 0

    # 计算小型肿瘤指标
    size_metrics = {}

    # 小型肿瘤
    small_precision = small_tumor_metrics['tp'] / (small_tumor_metrics['tp'] + small_tumor_metrics['fp']) if (
                                                                                                                         small_tumor_metrics[
                                                                                                                             'tp'] +
                                                                                                                         small_tumor_metrics[
                                                                                                                             'fp']) > 0 else 0
    small_recall = small_tumor_metrics['tp'] / (small_tumor_metrics['tp'] + small_tumor_metrics['fn']) if (
                                                                                                                      small_tumor_metrics[
                                                                                                                          'tp'] +
                                                                                                                      small_tumor_metrics[
                                                                                                                          'fn']) > 0 else 0
    small_f1 = 2 * small_precision * small_recall / (small_precision + small_recall) if (
                                                                                                    small_precision + small_recall) > 0 else 0

    size_metrics['small'] = {
        'precision': small_precision,
        'recall': small_recall,
        'f1': small_f1
    }

    # 中型肿瘤
    medium_precision = medium_tumor_metrics['tp'] / (medium_tumor_metrics['tp'] + medium_tumor_metrics['fp']) if (
                                                                                                                             medium_tumor_metrics[
                                                                                                                                 'tp'] +
                                                                                                                             medium_tumor_metrics[
                                                                                                                                 'fp']) > 0 else 0
    medium_recall = medium_tumor_metrics['tp'] / (medium_tumor_metrics['tp'] + medium_tumor_metrics['fn']) if (
                                                                                                                          medium_tumor_metrics[
                                                                                                                              'tp'] +
                                                                                                                          medium_tumor_metrics[
                                                                                                                              'fn']) > 0 else 0
    medium_f1 = 2 * medium_precision * medium_recall / (medium_precision + medium_recall) if (
                                                                                                         medium_precision + medium_recall) > 0 else 0

    size_metrics['medium'] = {
        'precision': medium_precision,
        'recall': medium_recall,
        'f1': medium_f1
    }

    # 大型肿瘤
    large_precision = large_tumor_metrics['tp'] / (large_tumor_metrics['tp'] + large_tumor_metrics['fp']) if (
                                                                                                                         large_tumor_metrics[
                                                                                                                             'tp'] +
                                                                                                                         large_tumor_metrics[
                                                                                                                             'fp']) > 0 else 0
    large_recall = large_tumor_metrics['tp'] / (large_tumor_metrics['tp'] + large_tumor_metrics['fn']) if (
                                                                                                                      large_tumor_metrics[
                                                                                                                          'tp'] +
                                                                                                                      large_tumor_metrics[
                                                                                                                          'fn']) > 0 else 0
    large_f1 = 2 * large_precision * large_recall / (large_precision + large_recall) if (
                                                                                                    large_precision + large_recall) > 0 else 0

    size_metrics['large'] = {
        'precision': large_precision,
        'recall': large_recall,
        'f1': large_f1
    }

    # 整合所有指标
    metrics = {
        'loss': avg_loss,
        'dice': avg_dice,
        'class_dice': {str(i): dice for i, dice in enumerate(avg_class_dices)},
        'tumor_metrics': {
            'precision': tumor_precision,
            'recall': tumor_recall,
            'f1': tumor_f1
        },
        'size_metrics': size_metrics
    }

    return metrics


def evaluate_trained_model(model_dir, data_dir, device, img_size=(256, 256), batch_size=4, num_workers=2):
    """评估训练好的模型"""
    model_dir = Path(model_dir)

    # 检查模型文件和配置是否存在
    model_path = model_dir / 'best_model.pth'
    config_path = model_dir / 'config.json'

    if not model_path.exists() or not config_path.exists():
        print(f"错误：在 {model_dir} 中未找到模型或配置文件")
        return

    # 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 创建模型
    model_type = config.get('model_type', 'standard')
    n_channels = config.get('n_channels', 3)
    n_classes = config.get('n_classes', 3)
    init_features = config.get('init_features', 64)

    print(f"正在评估模型: {model_dir.name}, 类型: {model_type}")

    # 初始化模型
    try:
        if model_type == 'standard':
            model = UNet(
                n_channels=n_channels,
                n_classes=n_classes,
                init_features=init_features
            ).to(device)
        else:
            model = get_attention_unet_model(
                model_type=model_type,
                n_channels=n_channels,
                n_classes=n_classes,
                init_features=init_features,
                attention_type=config.get('attention_type', 'cbam')
            ).to(device)

        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 准备验证数据加载器
    val_transform = get_transforms('val', img_size=img_size)

    val_dataset = LiTSDataset(
        data_dir=data_dir,
        slice_list_file=Path(data_dir) / "splits" / "val_slices.txt",
        transform=val_transform,
        phase="val",
        small_tumor_focus=True,
        return_weight_map=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"验证集大小: {len(val_dataset)} 切片")

    # 定义损失函数
    criterion = CombinedLoss(
        dice_weight=config.get('dice_weight', 1.0),
        focal_weight=config.get('focal_weight', 1.0),
        boundary_weight=config.get('boundary_weight', 0.5),
        size_weight=config.get('size_weight', 1.0)
    )

    # 详细评估
    print("开始详细评估...")
    metrics = evaluate_custom(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        n_classes=n_classes,
        tumor_class=2
    )

    # 保存评估指标
    metrics_file = model_dir / 'evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        # 确保所有值都是标准Python类型
        metrics_serializable = json.dumps(metrics, ensure_ascii=False)
        f.write(metrics_serializable)

    print(f"评估完成，指标已保存到 {metrics_file}")

    # 打印关键指标
    print("\n关键性能指标:")
    print(f"整体 Dice: {metrics['dice']:.4f}")

    print("\n各类别 Dice:")
    for cls, dice in metrics['class_dice'].items():
        class_name = "背景" if cls == '0' else "肝脏" if cls == '1' else "肿瘤"
        print(f"  {class_name}: {dice:.4f}")

    if 'size_metrics' in metrics and 'small' in metrics['size_metrics']:
        small_tumor = metrics['size_metrics']['small']
        print("\n小型肿瘤性能:")
        print(f"  F1分数: {small_tumor['f1']:.4f}")
        print(f"  召回率: {small_tumor['recall']:.4f}")
        print(f"  精确率: {small_tumor['precision']:.4f}")

    return metrics


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if args.all_models:
        # 评估所有模型
        models_dir = Path(args.models_dir)
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                try:
                    evaluate_trained_model(
                        model_dir,
                        args.data_dir,
                        device,
                        img_size=tuple(map(int, args.img_size.split(','))),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers
                    )
                    print("\n" + "=" * 50 + "\n")
                except Exception as e:
                    print(f"评估模型 {model_dir.name} 时出错: {e}")
    else:
        # 评估单个模型
        evaluate_trained_model(
            args.models_dir,
            args.data_dir,
            device,
            img_size=tuple(map(int, args.img_size.split(','))),
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='模型目录或包含多个模型的目录')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed',
                        help='预处理数据目录')
    parser.add_argument('--img_size', type=str, default='256,256',
                        help='图像大小，格式为"高,宽"')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载器工作进程数')
    parser.add_argument('--all_models', action='store_true',
                        help='评估目录下的所有模型')

    args = parser.parse_args()
    main(args)