# src/models/attention_modules/enhanced_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入其他需要的注意力模块
from .spatial_attention import SmallObjectSpatialAttention
from .channel_attention import ChannelAttention
from .hybrid_attention import CBAM


class MultiScaleEnhancedAttention(nn.Module):
    """
    多尺度特征增强注意力模块
    专为小型肿瘤设计，通过集成不同尺度特征增强小型肿瘤的表示
    """

    def __init__(self, in_channels, reduction_ratio=8):
        super(MultiScaleEnhancedAttention, self).__init__()

        # 多种尺寸的卷积核提取不同尺度特征
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=5, padding=2)

        # 小目标特征增强
        self.small_enhancer = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        )

        # 通道注意力计算权重
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 多尺度特征提取
        feat_1x1 = self.conv1x1(x)
        feat_3x3 = self.conv3x3(x)
        feat_5x5 = self.conv5x5(x)

        # 针对小目标增强
        small_feat = self.small_enhancer(feat_1x1)

        # 合并多尺度特征
        multi_scale = torch.cat([small_feat, feat_3x3, feat_5x5, feat_1x1], dim=1)

        # 应用通道注意力
        attn_weights = self.channel_attention(multi_scale)
        enhanced = multi_scale * attn_weights

        # 融合特征
        output = self.fusion(enhanced)

        return output + x  # 残差连接


class LocalContrastEnhancedAttention(nn.Module):
    """
    局部对比度增强注意力
    增强小型肿瘤与周围肝脏组织的对比度
    """

    def __init__(self, in_channels, kernel_size=7):
        super(LocalContrastEnhancedAttention, self).__init__()

        self.local_context = nn.Conv2d(in_channels, in_channels,
                                       kernel_size=kernel_size,
                                       padding=kernel_size // 2,
                                       groups=in_channels)

        self.contrast_calc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 边缘检测
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 获取局部上下文
        local_feat = self.local_context(x)

        # 计算局部与原始特征的对比
        contrast_input = torch.cat([x, local_feat], dim=1)
        contrast_map = self.contrast_calc(contrast_input)

        # 检测边缘
        edge_map = self.edge_detector(x)

        # 综合对比度和边缘信息
        attention = contrast_map * (1.0 + edge_map)

        # 应用注意力
        return x * attention


class ScaleAwareSmallObjectModule(nn.Module):
    """
    尺度感知型小目标增强模块
    通过自适应组合不同尺度的注意力机制，特别关注小型肿瘤
    """

    def __init__(self, in_channels, reduction_ratio=8):
        super(ScaleAwareSmallObjectModule, self).__init__()

        # 小目标专用空间注意力
        self.small_spatial_attn = SmallObjectSpatialAttention(in_channels)

        # 通道注意力
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)

        # 尺度检测器 - 检测当前特征是否包含小尺度目标
        self.scale_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, kernel_size=1),  # 2个通道：小目标权重和大目标权重
            nn.Softmax(dim=1)
        )

        # 特征增强
        self.enhancement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 检测尺度偏好
        scale_weights = self.scale_detector(x)
        small_weight = scale_weights[:, 0:1, :, :]
        large_weight = scale_weights[:, 1:2, :, :]

        # 应用不同注意力
        small_enhanced = self.small_spatial_attn(x)
        channel_enhanced = self.channel_attn(x)

        # 根据尺度权重组合
        combined = small_enhanced * small_weight + channel_enhanced * large_weight

        # 最终增强
        enhancement = self.enhancement(combined)

        return x * enhancement


class EnhancedTumorSizeAttention(nn.Module):
    """
    增强型肿瘤大小感知注意力
    优化对小型肿瘤的检测能力
    """

    def __init__(self, in_channels, reduction_ratio=8):
        super(EnhancedTumorSizeAttention, self).__init__()

        # 小型肿瘤感知分支 - 使用小卷积核和密集连接
        self.small_tumor_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        )

        # 中型肿瘤感知分支
        self.medium_tumor_branch = CBAM(in_channels // 2, reduction_ratio)

        # 大型肿瘤感知分支
        self.large_tumor_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        )

        # 尺度选择器 - 动态选择或组合不同分支
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 3, kernel_size=1),  # 3个权重
            nn.Softmax(dim=1)
        )

        # 输出层
        self.output_layer = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

    def forward(self, x):
        # 获取大小权重
        weights = self.scale_selector(x)
        small_weight = weights[:, 0:1, :, :]
        medium_weight = weights[:, 1:2, :, :]
        large_weight = weights[:, 2:3, :, :]

        # 应用不同分支
        small_out = self.small_tumor_branch(x)
        medium_out = self.medium_tumor_branch(x[:, :x.size(1) // 2, :, :])  # 修改以使用正确的通道数
        large_out = self.large_tumor_branch(x)

        # 获取权重后的前向传播
        # 注意力路径选择
        small_path = small_out * small_weight
        medium_path = medium_out * medium_weight
        large_path = large_out * large_weight

        # 权重融合，加倍强调小型肿瘤
        combined = (small_path * 1.5 + medium_path + large_path) / (
                    small_weight * 1.5 + medium_weight + large_weight + 1e-8)

        # 最终输出
        output = self.output_layer(combined)

        return x * torch.sigmoid(output)  # 应用作为注意力


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机输入
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试多尺度增强注意力
    ms_attn = MultiScaleEnhancedAttention(channels).to(device)
    output_ms = ms_attn(x)
    print(f"多尺度增强注意力输出形状: {output_ms.shape}")

    # 测试局部对比度增强注意力
    lc_attn = LocalContrastEnhancedAttention(channels).to(device)
    output_lc = lc_attn(x)
    print(f"局部对比度增强注意力输出形状: {output_lc.shape}")

    # 测试尺度感知小目标模块
    sa_attn = ScaleAwareSmallObjectModule(channels).to(device)
    output_sa = sa_attn(x)
    print(f"尺度感知小目标模块输出形状: {output_sa.shape}")

    # 测试增强型肿瘤大小注意力
    et_attn = EnhancedTumorSizeAttention(channels).to(device)
    output_et = et_attn(x)
    print(f"增强型肿瘤大小注意力输出形状: {output_et.shape}")