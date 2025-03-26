# src/models/attention_modules/spatial_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    空间注意力模块

    通过卷积层生成空间注意力图，突出关键区域的特征
    特别有利于检测图像中的小目标（如小型肿瘤）
    """

    def __init__(self, kernel_size=7):
        """
        初始化空间注意力模块

        参数:
            kernel_size: 卷积核大小，较大的核可以捕获更广的上下文
        """
        super(SpatialAttention, self).__init__()

        # 确保kernel_size是奇数，便于padding
        assert kernel_size % 2 == 1, "卷积核大小必须为奇数"
        padding = kernel_size // 2

        # 沿通道维度连接平均池化和最大池化的结果以捕获不同的特征
        # 然后通过卷积生成注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            加权后的特征，形状相同
        """
        # 沿通道维度计算平均值和最大值
        # [B, C, H, W] -> [B, 1, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # 沿通道维度连接池化结果
        # [B, 1, H, W] + [B, 1, H, W] -> [B, 2, H, W]
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # 生成空间注意力图
        # [B, 2, H, W] -> [B, 1, H, W]
        attention = self.conv(pooled)
        attention = self.sigmoid(attention)

        # 通过注意力权重增强原始特征
        # [B, C, H, W] * [B, 1, H, W] -> [B, C, H, W]
        return x * attention


class SpatialGate(nn.Module):
    """
    空间门控模块 - 增强版空间注意力

    使用多尺度感受野来更好地捕获不同大小目标的空间信息
    特别适用于检测大小差异很大的目标（从小型到大型肿瘤）
    """

    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化空间门控模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 降维比例，用于减少计算成本
        """
        super(SpatialGate, self).__init__()

        # 降维以减少计算量
        reduced_channels = max(1, in_channels // reduction_ratio)

        # 多尺度卷积，分别用3x3、5x5和7x7卷积核捕获不同尺度的空间信息
        self.conv_1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels, reduced_channels, kernel_size=5, padding=2)
        self.conv_5 = nn.Conv2d(in_channels, reduced_channels, kernel_size=7, padding=3)

        # 融合不同尺度的特征生成最终的空间注意力图
        self.fusion = nn.Conv2d(reduced_channels * 3, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            加权后的特征，形状相同
        """
        # 通过不同尺度的卷积提取多尺度特征
        feat_1 = F.relu(self.conv_1(x))
        feat_3 = F.relu(self.conv_3(x))
        feat_5 = F.relu(self.conv_5(x))

        # 连接多尺度特征
        # [B, C', H, W] * 3 -> [B, 3*C', H, W]
        feat_concat = torch.cat([feat_1, feat_3, feat_5], dim=1)

        # 生成空间注意力图
        # [B, 3*C', H, W] -> [B, 1, H, W]
        spatial_attention = self.fusion(feat_concat)
        spatial_attention = self.bn(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)

        # 通过空间注意力权重增强原始特征
        # [B, C, H, W] * [B, 1, H, W] -> [B, C, H, W]
        return x * spatial_attention


class SmallObjectSpatialAttention(nn.Module):
    """
    小目标空间注意力模块

    专门为增强小型病变的检测而设计，如小型肿瘤
    使用梯度反向传播和形态学特征来关注小目标区域
    """

    def __init__(self, in_channels, size_threshold=0.05):
        """
        初始化小目标空间注意力模块

        参数:
            in_channels: 输入通道数
            size_threshold: 小目标的相对大小阈值（相对于图像大小）
        """
        super(SmallObjectSpatialAttention, self).__init__()

        self.size_threshold = size_threshold

        # 特征提取和注意力生成
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)

        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            加权后的特征，形状相同
        """
        # 提取高级特征
        feat = F.relu(self.bn1(self.conv1(x)))
        feat = F.relu(self.bn2(self.conv2(feat)))

        # 生成空间注意力图
        attention = self.conv3(feat)

        # 应用形态学增强 - 使用高斯差分突出小目标
        attention_pooled = F.avg_pool2d(attention, kernel_size=3, stride=1, padding=1)
        attention_diff = attention - attention_pooled
        attention_diff = F.relu(attention_diff)  # 只保留正差值

        # 归一化并应用sigmoid
        attention = self.sigmoid(attention + attention_diff * 2.0)  # 增强小目标响应

        # 通过注意力权重增强原始特征
        return x * attention


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机输入
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试基本空间注意力
    sa = SpatialAttention().to(device)
    output_sa = sa(x)
    print(f"空间注意力输出形状: {output_sa.shape}")

    # 测试空间门控
    sg = SpatialGate(channels).to(device)
    output_sg = sg(x)
    print(f"空间门控输出形状: {output_sg.shape}")

    # 测试小目标空间注意力
    sosa = SmallObjectSpatialAttention(channels).to(device)
    output_sosa = sosa(x)
    print(f"小目标空间注意力输出形状: {output_sosa.shape}")