# src/utils/metrics.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def dice_coefficient(pred, target, smooth=1e-6, reduction='mean'):
    """
    计算Dice系数

    参数:
        pred: 预测掩码，形状为[B, C, H, W]或[B, H, W]，已经过sigmoid或softmax
        target: 目标掩码，形状为[B, H, W]或[B, C, H, W]
        smooth: 平滑项，避免除零错误
        reduction: 'mean', 'sum'或'none'

    返回:
        Dice系数
    """
    # 获取维度信息
    batch_size = pred.size(0)

    # 确保pred是二值化的 - 使用0.5阈值
    if pred.dtype != torch.bool:
        pred = (pred > 0.5).float()

    # 处理不同形状的情况
    if pred.dim() == 3 and target.dim() == 3:  # [B, H, W] 和 [B, H, W]
        # 扁平化预测和目标
        pred_flat = pred.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)

        # 计算Dice
        dice = (2. * intersection + smooth) / (union + smooth)

    elif pred.dim() == 4 and target.dim() == 3:  # [B, C, H, W] 和 [B, H, W]
        # 如果预测是多通道的，而目标是单通道的类别索引
        target = target.long()
        n_classes = pred.size(1)

        # 将目标转换为one-hot表示
        target_one_hot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()

        # 扁平化预测和目标
        pred_flat = pred.reshape(batch_size, n_classes, -1)
        target_flat = target_one_hot.reshape(batch_size, n_classes, -1)

        # 计算每个类别的Dice系数
        intersection = (pred_flat * target_flat).sum(2)
        union = pred_flat.sum(2) + target_flat.sum(2)

        # 计算Dice
        dice = (2. * intersection + smooth) / (union + smooth)

        # 平均每个类别的Dice
        dice = dice.mean(1)

    elif pred.dim() == 4 and target.dim() == 4:  # [B, C, H, W] 和 [B, C, H, W]
        # 扁平化预测和目标
        pred_flat = pred.reshape(batch_size, pred.size(1), -1)
        target_flat = target.reshape(batch_size, target.size(1), -1)

        # 计算每个类别的Dice系数
        intersection = (pred_flat * target_flat).sum(2)
        union = pred_flat.sum(2) + target_flat.sum(2)

        # 计算Dice
        dice = (2. * intersection + smooth) / (union + smooth)

        # 平均每个类别的Dice
        dice = dice.mean(1)

    else:
        raise ValueError(f"不支持的形状组合: pred {pred.shape}, target {target.shape}")

    # 应用reduction
    if reduction == 'mean':
        return dice.mean()
    elif reduction == 'sum':
        return dice.sum()
    else:  # 'none'
        return dice


def class_wise_dice(pred, target, n_classes, smooth=1e-6):
    """
    计算每个类别的Dice系数

    参数:
        pred: 预测掩码，形状为[B, C, H, W]或[B, H, W]
        target: 目标掩码，形状为[B, H, W]
        n_classes: 类别数量
        smooth: 平滑项

    返回:
        每个类别的Dice系数列表
    """
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
        dice_i = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_i.item())

    return dice_scores


def recall_precision_f1(pred, target, n_classes):
    """
    计算召回率、精确率和F1分数

    参数:
        pred: 预测掩码，形状为[B, C, H, W]或[B, H, W]
        target: 目标掩码，形状为[B, H, W]
        n_classes: 类别数量

    返回:
        每个类别的召回率、精确率和F1分数
    """
    # 确保pred是分类结果
    if pred.dim() == 4:  # [B, C, H, W]
        pred = pred.argmax(dim=1)  # [B, H, W]

    # 展平预测和目标
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()

    # 计算混淆矩阵
    cm = confusion_matrix(target_flat, pred_flat, labels=range(n_classes))

    # 初始化结果列表
    recall = np.zeros(n_classes)
    precision = np.zeros(n_classes)
    f1 = np.zeros(n_classes)

    # 计算每个类别的指标
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        # 计算指标，处理除零情况
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    return {
        'recall': recall,
        'precision': precision,
        'f1': f1
    }


def tumor_detection_metrics(pred, target, tumor_class=2, size_map=None, size_thresholds=[0, 10, 20]):
    """
    计算肿瘤检测指标，按大小分类

    参数:
        pred: 预测掩码，形状为[B, C, H, W]或[B, H, W]
        target: 目标掩码，形状为[B, H, W]
        tumor_class: 肿瘤类别索引
        size_map: 肿瘤大小图，形状为[B, H, W]
        size_thresholds: 肿瘤大小阈值列表，如[0, 10, 20]对应小型(<10mm)、中型(10-20mm)、大型(>20mm)

    返回:
        不同大小肿瘤的检测指标
    """
    # 确保pred是分类结果
    if pred.dim() == 4:  # [B, C, H, W]
        pred = pred.argmax(dim=1)  # [B, H, W]

    # 如果没有提供size_map，仅计算整体指标
    if size_map is None:
        pred_tumor = (pred == tumor_class).float()
        target_tumor = (target == tumor_class).float()

        # 计算交集和并集
        intersection = (pred_tumor * target_tumor).sum().item()
        pred_area = pred_tumor.sum().item()
        target_area = target_tumor.sum().item()

        # 计算指标
        recall = intersection / target_area if target_area > 0 else 0
        precision = intersection / pred_area if pred_area > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        return {
            'overall': {
                'recall': recall,
                'precision': precision,
                'f1': f1
            }
        }

    # 按大小分类计算指标
    size_categories = ["small", "medium", "large"]
    results = {}

    for i in range(len(size_thresholds) - 1):
        lower = size_thresholds[i]
        upper = size_thresholds[i + 1]
        category = size_categories[i]

        # 创建当前大小类别的掩码
        if i < len(size_thresholds) - 2:
            # 中间类别：lower <= size < upper
            size_mask = ((size_map >= lower) & (size_map < upper)).float()
        else:
            # 最后一个类别：size >= lower
            size_mask = (size_map >= lower).float()

        # 获取当前大小类别的目标和预测
        target_tumor = ((target == tumor_class) & (size_mask > 0)).float()
        pred_tumor = ((pred == tumor_class) & (size_mask > 0)).float()

        # 计算指标
        intersection = (pred_tumor * target_tumor).sum().item()
        pred_area = pred_tumor.sum().item()
        target_area = target_tumor.sum().item()

        recall = intersection / target_area if target_area > 0 else 0
        precision = intersection / pred_area if pred_area > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        results[category] = {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }

    return results


class MetricsTracker:
    """
    度量指标跟踪器，用于训练和验证
    """

    def __init__(self):
        self.metrics = {
            'loss': [],
            'dice': [],
            'class_dice': {0: [], 1: [], 2: []},  # 假设3个类别
            'tumor_metrics': {'recall': [], 'precision': [], 'f1': []},
            'size_metrics': {'small': [], 'medium': [], 'large': []}
        }

    def update(self, loss, pred, target, size_map=None):
        """更新指标值"""
        # 更新损失
        self.metrics['loss'].append(loss)

        # 计算并更新Dice系数
        dice = dice_coefficient(pred, target).item()
        self.metrics['dice'].append(dice)

        # 计算并更新各类别Dice系数
        class_dice = class_wise_dice(pred, target, n_classes=3)
        for i, d in enumerate(class_dice):
            self.metrics['class_dice'][i].append(d)

        # 计算肿瘤检测指标
        if size_map is not None:
            # 按大小分层的指标
            size_metrics = tumor_detection_metrics(
                pred, target, tumor_class=2,
                size_map=size_map, size_thresholds=[0, 10, 20]
            )

            for size, metrics in size_metrics.items():
                if size not in self.metrics['size_metrics']:
                    self.metrics['size_metrics'][size] = {'recall': [], 'precision': [], 'f1': []}

                for k, v in metrics.items():
                    self.metrics['size_metrics'][size][k].append(v)

        # 肿瘤整体指标
        tumor_metrics = recall_precision_f1(pred, target, n_classes=3)
        for k, v in tumor_metrics.items():
            self.metrics['tumor_metrics'][k].append(v[2])  # 取索引2对应肿瘤

    def get_average(self):
        """计算指标平均值"""
        avg_metrics = {}

        # 计算平均损失和Dice
        avg_metrics['loss'] = np.mean(self.metrics['loss'])
        avg_metrics['dice'] = np.mean(self.metrics['dice'])

        # 计算各类别Dice平均值
        avg_metrics['class_dice'] = {}
        for cls, dice_list in self.metrics['class_dice'].items():
            avg_metrics['class_dice'][cls] = np.mean(dice_list)

        # 计算肿瘤指标平均值
        avg_metrics['tumor_metrics'] = {}
        for k, v in self.metrics['tumor_metrics'].items():
            avg_metrics['tumor_metrics'][k] = np.mean(v)

        # 计算不同大小肿瘤指标平均值
        avg_metrics['size_metrics'] = {}
        for size, metrics in self.metrics['size_metrics'].items():
            avg_metrics['size_metrics'][size] = {}
            for k, v in metrics.items():
                if v:  # 检查列表非空
                    avg_metrics['size_metrics'][size][k] = np.mean(v)
                else:
                    avg_metrics['size_metrics'][size][k] = 0.0

                return avg_metrics

        def reset(self):
            """重置所有指标"""
            for key in self.metrics:
                if isinstance(self.metrics[key], list):
                    self.metrics[key] = []
                elif isinstance(self.metrics[key], dict):
                    for sub_key in self.metrics[key]:
                        if isinstance(self.metrics[key][sub_key], list):
                            self.metrics[key][sub_key] = []
                        elif isinstance(self.metrics[key][sub_key], dict):
                            for size_key in self.metrics[key][sub_key]:
                                self.metrics[key][sub_key][size_key] = []

    def evaluate_model(model, dataloader, criterion, device, n_classes=3, tumor_class=2):
        """
        评估模型性能

        参数:
            model: 待评估的模型
            dataloader: 数据加载器
            criterion: 损失函数
            device: 计算设备
            n_classes: 类别数量
            tumor_class: 肿瘤类别索引

        返回:
            评估指标结果
        """
        # 初始化指标跟踪器
        metrics = MetricsTracker()

        # 设置模型为评估模式
        model.eval()

        with torch.no_grad():
            for batch in dataloader:
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

                # 获取预测结果
                if n_classes > 1:
                    # 多类别情况
                    probs = torch.softmax(outputs, dim=1)
                    preds = probs.argmax(dim=1)
                else:
                    # 二分类情况
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()

                # 更新指标
                metrics.update(loss.item(), preds, masks, weight_map)

        # 返回平均指标
        return metrics.get_average()

    if __name__ == "__main__":
        # 快速测试评估指标
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 生成测试数据
        batch_size, channels, height, width = 2, 3, 64, 64
        pred = torch.randn(batch_size, channels, height, width).to(device)
        pred_class = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        target = torch.randint(0, channels, (batch_size, height, width)).to(device)
        size_map = torch.rand(batch_size, height, width).to(device) * 30  # 0-30mm范围

        # 测试各指标
        dice = dice_coefficient(torch.softmax(pred, dim=1), target)
        class_dice = class_wise_dice(pred_class, target, n_classes=channels)
        metrics = recall_precision_f1(pred_class, target, n_classes=channels)
        tumor_metrics = tumor_detection_metrics(
            pred_class, target, tumor_class=2, size_map=size_map, size_thresholds=[0, 10, 20]
        )

        print(f"Overall Dice: {dice.item():.4f}")
        print("Class-wise Dice:")
        for i, d in enumerate(class_dice):
            print(f"  Class {i}: {d:.4f}")

        print("\nRecall, Precision, F1:")
        for i in range(channels):
            print(
                f"  Class {i}: R={metrics['recall'][i]:.4f}, P={metrics['precision'][i]:.4f}, F1={metrics['f1'][i]:.4f}")

        print("\nTumor Detection by Size:")
        for size, metrics in tumor_metrics.items():
            print(f"  {size}: R={metrics['recall']:.4f}, P={metrics['precision']:.4f}, F1={metrics['f1']:.4f}")


def evaluate_model(model, dataloader, criterion, device, n_classes=3, tumor_class=2):
    """
    评估模型性能

    参数:
        model: 待评估的模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        n_classes: 类别数量
        tumor_class: 肿瘤类别索引

    返回:
        评估指标结果
    """
    # 初始化指标跟踪器
    metrics = MetricsTracker()

    # 设置模型为评估模式
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
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

            # 获取预测结果
            if n_classes > 1:
                # 多类别情况
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
            else:
                # 二分类情况
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

            # 更新指标
            metrics.update(loss.item(), preds, masks, weight_map)

    # 返回平均指标
    return metrics.get_average()