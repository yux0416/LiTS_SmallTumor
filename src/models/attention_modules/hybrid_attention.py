# src/models/attention_modules/hybrid_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入已定义的注意力模块
from .spatial_attention import SpatialAttention, SmallObjectSpatialAttention
from .channel_attention import ChannelAttention, CBAM_Channel


class CBAM(nn.Module):
    """
    卷积块注意力模块 (CBAM)

    结合了通道注意力和空间注意力
    先应用通道注意力再应用空间注意力
    """

    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        """
        初始化CBAM模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 通道注意力中的降维比例
            spatial_kernel_size: 空间注意力的卷积核大小
        """
        super(CBAM, self).__init__()

        # 通道注意力模块
        self.channel_attention = CBAM_Channel(in_channels, reduction_ratio)

        # 空间注意力模块
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            注意力增强的特征，形状相同
        """
        # 先应用通道注意力
        x = self.channel_attention(x)

        # 再应用空间注意力
        x = self.spatial_attention(x)

        return x


class SmallObjectAttention(nn.Module):
    """
    小目标注意力模块

    专门为增强小型肿瘤等小目标的检测设计
    结合通道和空间注意力，偏重于空间注意力
    """

    def __init__(self, in_channels, reduction_ratio=8, size_threshold=0.05):
        """
        初始化小目标注意力模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 通道注意力中的降维比例
            size_threshold: 小目标的相对大小阈值
        """
        super(SmallObjectAttention, self).__init__()

        # 通道注意力 - 使用更小的降维比例，保留更多特征
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)

        # 小目标空间注意力 - 专为小型目标设计
        self.spatial_attention = SmallObjectSpatialAttention(in_channels, size_threshold)

        # 融合层 - 学习如何结合两种注意力机制
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            注意力增强的特征，形状相同
        """
        # 应用通道注意力
        channel_out = self.channel_attention(x)

        # 应用空间注意力
        spatial_out = self.spatial_attention(x)

        # 融合两种注意力的结果
        # 这里我们直接相加，也可以用融合层学习更复杂的组合方式
        out = channel_out + spatial_out
        fusion_weight = self.fusion(out)

        # 输出融合后的结果
        return x * fusion_weight


class ScaleAwareAttention(nn.Module):
    """
    尺度感知注意力模块

    使用多尺度特征和金字塔池化增强对不同大小目标的感知
    特别适合同时检测小型和大型肿瘤
    """

    def __init__(self, in_channels, reduction_ratio=16, pool_sizes=[1, 2, 4, 8]):
        """
        初始化尺度感知注意力模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 通道降维比例
            pool_sizes: 金字塔池化的尺寸列表
        """
        super(ScaleAwareAttention, self).__init__()

        # 通道降维
        reduced_channels = max(1, in_channels // reduction_ratio)

        # 金字塔池化层
        self.pool_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size=(size, size)) for size in pool_sizes
        ])

        # 特征转换层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
            for _ in range(len(pool_sizes))
        ])

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_channels * len(pool_sizes), reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            注意力增强的特征，形状相同
        """
        batch_size, channels, height, width = x.size()

        # 多尺度特征提取
        pyramid_features = []
        for pool_layer, conv_layer in zip(self.pool_layers, self.conv_layers):
            # 池化、卷积处理
            pooled = pool_layer(x)
            conv_out = conv_layer(pooled)

            # 上采样回原始大小
            upsampled = F.interpolate(conv_out, size=(height, width), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)

        # 连接多尺度特征
        concat_features = torch.cat(pyramid_features, dim=1)

        # 融合多尺度特征，生成注意力图
        attention = self.fusion(concat_features)

        # 应用注意力增强输入特征
        return x * attention


class MultiModalityAttention(nn.Module):
    """
    多模态注意力模块

    用于融合不同输入模态或特征
    在医学图像中可用于结合不同窗口下的CT特征
    """

    def __init__(self, in_channels, num_modalities=2, reduction_ratio=8):
        """
        初始化多模态注意力模块

        参数:
            in_channels: 输入通道数
            num_modalities: 模态数量
            reduction_ratio: 特征降维比例
        """
        super(MultiModalityAttention, self).__init__()

        self.num_modalities = num_modalities
        reduced_channels = max(1, in_channels // reduction_ratio)

        # 为每个模态创建特征转换层
        self.modal_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_modalities)
        ])

        # 交叉注意力
        self.cross_attention = nn.Sequential(
            nn.Conv2d(reduced_channels * num_modalities, in_channels * num_modalities, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * num_modalities),
            nn.Sigmoid()
        )

        # 输出转换
        self.output_transform = nn.Conv2d(in_channels * num_modalities, in_channels, kernel_size=1, bias=False)

    def forward(self, x_list):
        """
        前向传播

        参数:
            x_list: 输入特征列表，每个元素形状为 [B, C, H, W]

        返回:
            融合后的特征，形状为 [B, C, H, W]
        """
        assert len(x_list) == self.num_modalities, f"预期{self.num_modalities}个模态，但只提供了{len(x_list)}个"

        # 变换每个模态的特征
        transformed_features = [
            transform(x) for transform, x in zip(self.modal_transforms, x_list)
        ]

        # 连接变换后的特征
        concat_features = torch.cat(transformed_features, dim=1)

        # 生成交叉注意力权重
        cross_attn = self.cross_attention(concat_features)

        # 应用注意力到原始特征
        original_concat = torch.cat(x_list, dim=1)
        attended_features = original_concat * cross_attn

        # 融合得到最终输出
        output = self.output_transform(attended_features)

        return output


class DualAttention(nn.Module):
    """
    双重注意力模块

    同时学习空间和通道维度上的注意力权重
    通过并行分支处理，然后融合两种注意力的结果
    """

    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化双重注意力模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 通道降维比例
        """
        super(DualAttention, self).__init__()

        # 通道注意力分支
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)

        # 空间注意力分支
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            注意力增强的特征，形状相同
        """
        # 通道注意力分支
        channel_out = self.channel_attention(x)

        # 空间注意力分支
        spatial_out = self.spatial_attention(x)

        # 连接两种注意力的输出
        concat_features = torch.cat([channel_out, spatial_out], dim=1)

        # 融合特征
        fusion_weights = self.fusion(concat_features)

        # 应用融合权重
        enhanced_features = x * fusion_weights

        return enhanced_features


class TumorSizeSpecificAttention(nn.Module):
    """
    肿瘤大小特定注意力模块

    专门为肝脏肿瘤检测设计，针对不同大小的肿瘤使用不同的注意力策略
    小型肿瘤(<10mm)、中型肿瘤(10-20mm)和大型肿瘤(>20mm)
    """

    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化肿瘤大小特定注意力模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 通道降维比例
        """
        super(TumorSizeSpecificAttention, self).__init__()

        # 小型肿瘤注意力 - 重点关注局部细节和对比度
        self.small_tumor_attention = SmallObjectAttention(in_channels, reduction_ratio // 2)

        # 中型肿瘤注意力 - 平衡局部和上下文信息
        self.medium_tumor_attention = CBAM(in_channels, reduction_ratio)

        # 大型肿瘤注意力 - 关注更广泛的上下文
        self.large_tumor_attention = ScaleAwareAttention(in_channels, reduction_ratio)

        # 注意力选择网络 - 学习根据输入特征选择适当的注意力机制
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.attention_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 3, kernel_size=1, bias=False),  # 3个注意力权重
            nn.Softmax(dim=1)  # 归一化权重
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            注意力增强的特征，形状相同
        """
        # 应用三种不同的注意力
        small_out = self.small_tumor_attention(x)
        medium_out = self.medium_tumor_attention(x)
        large_out = self.large_tumor_attention(x)

        # 计算注意力选择权重
        weights = self.attention_selector(x)  # [B, 3, 1, 1]

        # 应用权重融合不同大小肿瘤的注意力结果
        small_weight = weights[:, 0:1, :, :]
        medium_weight = weights[:, 1:2, :, :]
        large_weight = weights[:, 2:3, :, :]

        # 加权融合
        output = small_out * small_weight + medium_out * medium_weight + large_out * large_weight

        return output


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机输入
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试CBAM
    cbam = CBAM(channels).to(device)
    output_cbam = cbam(x)
    print(f"CBAM输出形状: {output_cbam.shape}")

    # 测试小目标注意力
    soa = SmallObjectAttention(channels).to(device)
    output_soa = soa(x)
    print(f"小目标注意力输出形状: {output_soa.shape}")

    # 测试尺度感知注意力
    saa = ScaleAwareAttention(channels).to(device)
    output_saa = saa(x)
    print(f"尺度感知注意力输出形状: {output_saa.shape}")

    # 测试多模态注意力
    x_list = [torch.randn(batch_size, channels, height, width).to(device) for _ in range(2)]
    mma = MultiModalityAttention(channels, num_modalities=2).to(device)
    output_mma = mma(x_list)
    print(f"多模态注意力输出形状: {output_mma.shape}")

    # 测试双重注意力
    da = DualAttention(channels).to(device)
    output_da = da(x)
    print(f"双重注意力输出形状: {output_da.shape}")

    # 测试肿瘤大小特定注意力
    tssa = TumorSizeSpecificAttention(channels).to(device)
    output_tssa = tssa(x)
    print(f"肿瘤大小特定注意力输出形状: {output_tssa.shape}")