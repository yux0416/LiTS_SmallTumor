# src/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0, reduction='mean'):
    """
    Dice损失函数

    参数:
        pred: 预测logits，形状为[B, C, H, W]
        target: 目标掩码，形状为[B, H, W]或[B, C, H, W]
        smooth: 平滑项，避免除零错误
        reduction: 'mean', 'sum'或'none'

    返回:
        损失值
    """
    # 获取维度信息
    batch_size = pred.size(0)

    # 确保pred是经过sigmoid的
    if not torch.max(pred) <= 1.0 and not torch.min(pred) >= 0.0:
        pred = torch.sigmoid(pred)

    # 扁平化预测
    pred_flat = pred.reshape(batch_size, -1)

    # 处理目标格式
    if target.dim() == 3:  # [B, H, W]
        # 将目标转为long类型，确保可以用于one_hot
        target = target.long()

        # 将目标转为one-hot表示
        if pred.size(1) > 1:  # 多类分割
            target = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

    # 扁平化目标
    target_flat = target.reshape(batch_size, -1)

    # 计算Dice系数
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)

    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice

    # 应用reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal损失函数 - 处理类别不平衡

    参数:
        pred: 预测logits，形状为[B, C, H, W]
        target: 目标掩码，形状为[B, H, W]或[B, C, H, W]
        alpha: 平衡正负样本的系数
        gamma: 聚焦参数，减少易分类样本的权重
        reduction: 'mean', 'sum'或'none'

    返回:
        损失值
    """
    # 确保pred经过sigmoid处理
    if not torch.max(pred) <= 1.0 and not torch.min(pred) >= 0.0:
        pred = torch.sigmoid(pred)

    # 处理目标格式
    if target.dim() == 3:  # [B, H, W]
        # 确保目标是long类型
        target = target.long()

        if pred.size(1) > 1:  # 多类分割
            target = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
    # 计算二元交叉熵
    bce = F.binary_cross_entropy(pred, target, reduction='none')

    # 计算Focal项
    pt = target * pred + (1 - target) * (1 - pred)
    focal_weight = (1 - pt) ** gamma

    # 应用alpha平衡因子
    if alpha is not None:
        alpha_weight = target * alpha + (1 - target) * (1 - alpha)
        focal_weight = focal_weight * alpha_weight

    loss = focal_weight * bce

    # 应用reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def boundary_loss(pred, target, sigma=1.0, reduction='mean'):
    """
    边界损失函数 - 增强对目标边界的敏感度

    参数:
        pred: 预测logits，形状为[B, C, H, W]
        target: 目标掩码，形状为[B, H, W]或[B, C, H, W]
        sigma: 高斯滤波器标准差
        reduction: 'mean', 'sum'或'none'

    返回:
        损失值
    """
    # 确保pred经过sigmoid处理
    if not torch.max(pred) <= 1.0 and not torch.min(pred) >= 0.0:
        pred = torch.sigmoid(pred)

    # 计算目标边界
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

    # 处理目标格式
    if target.dim() == 3:  # [B, H, W]
        # 确保目标是long类型
        target = target.long()

        if pred.size(1) > 1:  # 多类分割
            target = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

    # 计算目标和预测的边界
    target_boundaries = []
    pred_boundaries = []

    for i in range(target.size(1)):
        # 目标边界
        target_channel = target[:, i:i + 1]
        target_boundary = F.conv2d(target_channel, laplacian_kernel, padding=1).abs()
        target_boundary = torch.exp(-target_boundary ** 2 / (2 * sigma ** 2))
        target_boundaries.append(target_boundary)

        # 预测边界
        pred_channel = pred[:, i:i + 1]
        pred_boundary = F.conv2d(pred_channel, laplacian_kernel, padding=1).abs()
        pred_boundary = torch.exp(-pred_boundary ** 2 / (2 * sigma ** 2))
        pred_boundaries.append(pred_boundary)

    target_boundary = torch.cat(target_boundaries, dim=1)
    pred_boundary = torch.cat(pred_boundaries, dim=1)

    # 计算边界差异
    loss = F.mse_loss(pred_boundary, target_boundary, reduction=reduction)

    return loss


def size_weighted_loss(pred, target, weight_map, loss_fn, reduction='mean'):
    """
    基于大小的加权损失函数

    参数:
        pred: 预测logits，形状为[B, C, H, W]
        target: 目标掩码，形状为[B, H, W]或[B, C, H, W]
        weight_map: 权重图，形状为[B, H, W]或[B, 1, H, W]
        loss_fn: 基础损失函数
        reduction: 'mean', 'sum'或'none'

    返回:
        加权后的损失
    """
    # 计算基础损失 - 使用'none'以应用权重
    if 'reduction' in loss_fn.__code__.co_varnames:
        base_loss = loss_fn(pred, target, reduction='none')
    else:
        # 如果损失函数不接受reduction参数
        base_loss = loss_fn(pred, target)

    # 如果base_loss只有批次维度[B]，但weight_map有空间维度[B, 1, H, W]
    if base_loss.dim() == 1 and weight_map.dim() == 4:
        # 我们需要对每个样本的空间区域计算加权损失
        batch_size = base_loss.size(0)
        weighted_losses = []

        for i in range(batch_size):
            # 对每个样本计算加权损失
            sample_loss = base_loss[i]
            sample_weight = weight_map[i]

            # 计算样本权重的平均值作为缩放因子
            weight_scale = sample_weight.mean()

            # 应用权重缩放
            weighted_sample_loss = sample_loss * weight_scale
            weighted_losses.append(weighted_sample_loss)

        # 重新组合为批次
        weighted_loss = torch.stack(weighted_losses)

        # 应用reduction
        if reduction == 'mean':
            return weighted_loss.mean()
        elif reduction == 'sum':
            return weighted_loss.sum()
        else:  # 'none'
            return weighted_loss

    elif base_loss.dim() == weight_map.dim():
        # 如果维度相同，尝试直接相乘
        weighted_loss = base_loss * weight_map
    else:
        print(f"警告: base_loss ({base_loss.shape}) 和 weight_map ({weight_map.shape}) 维度不匹配")

        # 返回与输入批次大小匹配的损失
        if base_loss.dim() == 0:  # 标量
            return base_loss.expand(pred.size(0))  # 扩展为批次大小
        else:
            return base_loss  # 保持原样

    # 应用reduction
    if reduction == 'mean':
        return weighted_loss.mean()
    elif reduction == 'sum':
        return weighted_loss.sum()
    else:  # 'none'
        return weighted_loss


class CombinedLoss(nn.Module):
    """
    组合多种损失函数
    """

    def __init__(self, dice_weight=1.0, focal_weight=1.0, boundary_weight=0.5,
                 size_weight=1.0, alpha=0.25, gamma=2.0, sigma=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.size_weight = size_weight
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, pred, target, weight_map=None):
        loss = torch.zeros(pred.size(0), device=pred.device)  # 初始化为批次大小的零张量

        # Dice损失
        if self.dice_weight > 0:
            dice = dice_loss(pred, target)
            if dice.dim() == 0:  # 如果是标量
                dice = dice.expand(pred.size(0))  # 扩展为批次大小
            loss = loss + self.dice_weight * dice

        # Focal损失
        if self.focal_weight > 0:
            focal = focal_loss(pred, target, alpha=self.alpha, gamma=self.gamma)
            if focal.dim() == 0:  # 如果是标量
                focal = focal.expand(pred.size(0))  # 扩展为批次大小
            loss = loss + self.focal_weight * focal

        # 边界损失
        if self.boundary_weight > 0:
            boundary = boundary_loss(pred, target, sigma=self.sigma)
            if boundary.dim() == 0:  # 如果是标量
                boundary = boundary.expand(pred.size(0))  # 扩展为批次大小
            loss = loss + self.boundary_weight * boundary

        # 小型肿瘤加权损失
        if weight_map is not None and self.size_weight > 0:
            size_weighted = size_weighted_loss(
                pred, target, weight_map,
                lambda p, t: dice_loss(p, t, reduction='none')
            )
            if size_weighted.dim() == 0:  # 如果是标量
                size_weighted = size_weighted.expand(pred.size(0))  # 扩展为批次大小
            loss = loss + self.size_weight * size_weighted

        # 返回平均损失
        return loss.mean()


if __name__ == "__main__":
    # 快速测试损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成测试数据
    batch_size, channels, height, width = 2, 3, 64, 64
    pred = torch.randn(batch_size, channels, height, width).to(device)
    target = torch.randint(0, channels, (batch_size, height, width)).to(device)
    weight_map = torch.rand(batch_size, height, width).to(device)

    # 测试各损失函数
    dice = dice_loss(torch.sigmoid(pred), target)
    focal = focal_loss(torch.sigmoid(pred), target)
    bound = boundary_loss(torch.sigmoid(pred), target)
    weighted = size_weighted_loss(
        torch.sigmoid(pred), target, weight_map,
        lambda p, t: dice_loss(p, t, reduction='none')
    )

    # 测试组合损失
    combined_loss = CombinedLoss()
    loss = combined_loss(pred, target, weight_map)

    # 修改输出方式，处理非标量张量
    print(f"Dice Loss: {dice.mean().item():.4f}")
    print(f"Focal Loss: {focal.mean().item():.4f}")
    print(f"Boundary Loss: {bound.mean().item():.4f}")

    # 对于非标量的 weighted，使用 mean() 先计算平均值
    if weighted.dim() > 0 and weighted.numel() > 1:
        print(f"Size-Weighted Loss (mean): {weighted.mean().item():.4f}")
        print(f"Size-Weighted Loss (shape): {weighted.shape}")
    else:
        print(f"Size-Weighted Loss: {weighted.item():.4f}")

    print(f"Combined Loss: {loss.item():.4f}")