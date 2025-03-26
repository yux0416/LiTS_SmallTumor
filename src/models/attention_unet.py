# src/models/attention_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# 导入基础U-Net组件
from .unet import DoubleConv, Down, Up, OutConv

# 导入注意力模块
from .attention_modules.channel_attention import (
    ChannelAttention,
    CBAM_Channel,
    ECABlock,
    GCTModule
)
from .attention_modules.spatial_attention import (
    SpatialAttention,
    SpatialGate,
    SmallObjectSpatialAttention
)
from .attention_modules.hybrid_attention import (
    CBAM,
    SmallObjectAttention,
    ScaleAwareAttention,
    MultiModalityAttention,
    DualAttention,
    TumorSizeSpecificAttention
)
from .attention_modules.dynamic_attention import (
    DynamicAttention,
    MultiScaleFeatureIntegration,
    SmallObjectEnhancedAttention
)

from .attention_modules.enhanced_attention import (
    MultiScaleEnhancedAttention,
    LocalContrastEnhancedAttention,
    ScaleAwareSmallObjectModule,
    EnhancedTumorSizeAttention
)

from .attention_modules.enhanced_attention_tuned import (
    TunedLocalContrastAttention,
    HybridLocalContrastSmallObjectAttention,
    IntegratedSmallTumorAttention
)


class AttentionUNet(nn.Module):
    """
    注意力增强的U-Net

    基于标准U-Net架构，在跳跃连接处增加了注意力模块
    可以使用不同类型的注意力机制
    """

    def __init__(
            self,
            n_channels=3,
            n_classes=3,
            init_features=64,
            attention_type='cbam',
            bilinear=False
    ):
        """
        初始化注意力增强的U-Net

        参数:
            n_channels: 输入通道数
            n_classes: 输出类别数
            init_features: 初始特征数量
            attention_type: 使用的注意力类型 ('channel', 'spatial', 'cbam', 'small', 'scale')
            bilinear: 是否使用双线性上采样
        """
        super(AttentionUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention_type = attention_type

        # 特征数量
        features = init_features

        # 编码器
        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)

        # 是否使用双线性上采样
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)

        # 解码器
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)

        # 输出层
        self.outc = OutConv(features, n_classes)

        # 添加注意力模块
        self.attentions = nn.ModuleDict()

        # 为每个解码器阶段创建注意力模块
        attention_dims = [
            features * 8,
            features * 4,
            features * 2,
            features
        ]

        for i, dim in enumerate(attention_dims):
            self.attentions[f'attn{i + 1}'] = self._create_attention_module(dim)

    def _create_attention_module(self, dim):
        """创建指定类型的注意力模块"""
        if self.attention_type == 'channel':
            return ChannelAttention(dim)
        elif self.attention_type == 'spatial':
            return SpatialAttention()
        elif self.attention_type == 'cbam':
            return CBAM(dim)
        elif self.attention_type == 'small':
            return SmallObjectAttention(dim)
        elif self.attention_type == 'scale':
            return ScaleAwareAttention(dim)
        elif self.attention_type == 'dual':
            return DualAttention(dim)
        elif self.attention_type == 'tumor_size':
            return TumorSizeSpecificAttention(dim)
        elif self.attention_type == 'gct':
            return GCTModule(dim)
        elif self.attention_type == 'eca':
            return ECABlock(dim)
        # 添加对新注意力模块的支持
        elif self.attention_type == 'multi_scale_enhanced':
            return MultiScaleEnhancedAttention(dim)
        elif self.attention_type == 'local_contrast':
            return LocalContrastEnhancedAttention(dim)
        elif self.attention_type == 'scale_aware_small':
            return ScaleAwareSmallObjectModule(dim)
        elif self.attention_type == 'enhanced_tumor_size':
            return EnhancedTumorSizeAttention(dim)
        # 添加到_create_attention_module方法中的条件选择
        elif self.attention_type == 'tuned_local_contrast':
            return TunedLocalContrastAttention(dim)
        elif self.attention_type == 'hybrid_local_small':
            return HybridLocalContrastSmallObjectAttention(dim)
        elif self.attention_type == 'integrated_small_tumor':
            return IntegratedSmallTumorAttention(dim)
        else:
            raise ValueError(f"不支持的注意力类型: {self.attention_type}")

    def forward(self, x):
        """前向传播"""
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器路径 + 注意力增强的跳跃连接
        x = self.up1(x5, self.attentions['attn1'](x4))
        x = self.up2(x, self.attentions['attn2'](x3))
        x = self.up3(x, self.attentions['attn3'](x2))
        x = self.up4(x, self.attentions['attn4'](x1))

        # 输出层
        logits = self.outc(x)

        return logits


class DeepAttentionUNet(nn.Module):
    """
    深度注意力U-Net

    在U-Net架构的每个层次都添加注意力模块
    包括编码器和解码器内部以及跳跃连接
    """

    def __init__(
            self,
            n_channels=3,
            n_classes=3,
            init_features=64,
            attention_type='cbam',
            bilinear=False
    ):
        """
        初始化深度注意力U-Net

        参数:
            n_channels: 输入通道数
            n_classes: 输出类别数
            init_features: 初始特征数量
            attention_type: 使用的注意力类型 ('channel', 'spatial', 'cbam', 'small', 'scale')
            bilinear: 是否使用双线性上采样
        """
        super(DeepAttentionUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention_type = attention_type

        # 特征数量
        features = init_features

        # 编码器 (带有注意力)
        self.inc = self._double_conv_with_attention(n_channels, features)
        self.down1 = self._down_with_attention(features, features * 2)
        self.down2 = self._down_with_attention(features * 2, features * 4)
        self.down3 = self._down_with_attention(features * 4, features * 8)

        # 是否使用双线性上采样
        factor = 2 if bilinear else 1
        self.down4 = self._down_with_attention(features * 8, features * 16 // factor)

        # 解码器 (带有注意力)
        self.up1 = self._up_with_attention(features * 16, features * 8 // factor, bilinear)
        self.up2 = self._up_with_attention(features * 8, features * 4 // factor, bilinear)
        self.up3 = self._up_with_attention(features * 4, features * 2 // factor, bilinear)
        self.up4 = self._up_with_attention(features * 2, features, bilinear)

        # 输出层
        self.outc = OutConv(features, n_classes)

        # 跳跃连接处的注意力模块
        self.skip_attentions = nn.ModuleDict()

        # 为每个解码器阶段创建注意力模块
        attention_dims = [
            features * 8,
            features * 4,
            features * 2,
            features
        ]

        for i, dim in enumerate(attention_dims):
            self.skip_attentions[f'skip_attn{i + 1}'] = self._create_attention_module(dim)

    def _create_attention_module(self, dim):
        """创建指定类型的注意力模块"""
        if self.attention_type == 'channel':
            return ChannelAttention(dim)
        elif self.attention_type == 'spatial':
            return SpatialAttention()
        elif self.attention_type == 'cbam':
            return CBAM(dim)
        elif self.attention_type == 'small':
            return SmallObjectAttention(dim)
        elif self.attention_type == 'scale':
            return ScaleAwareAttention(dim)
        elif self.attention_type == 'dual':
            return DualAttention(dim)
        elif self.attention_type == 'tumor_size':
            return TumorSizeSpecificAttention(dim)
        elif self.attention_type == 'gct':
            return GCTModule(dim)
        elif self.attention_type == 'eca':
            return ECABlock(dim)
        # 添加对新注意力模块的支持
        elif self.attention_type == 'multi_scale_enhanced':
            return MultiScaleEnhancedAttention(dim)
        elif self.attention_type == 'local_contrast':
            return LocalContrastEnhancedAttention(dim)
        elif self.attention_type == 'scale_aware_small':
            return ScaleAwareSmallObjectModule(dim)
        elif self.attention_type == 'enhanced_tumor_size':
            return EnhancedTumorSizeAttention(dim)
        # 添加到_create_attention_module方法中的条件选择
        elif self.attention_type == 'tuned_local_contrast':
            return TunedLocalContrastAttention(dim)
        elif self.attention_type == 'hybrid_local_small':
            return HybridLocalContrastSmallObjectAttention(dim)
        elif self.attention_type == 'integrated_small_tumor':
            return IntegratedSmallTumorAttention(dim)
        else:
            raise ValueError(f"不支持的注意力类型: {self.attention_type}")

    def _double_conv_with_attention(self, in_channels, out_channels):
        """带注意力的双卷积块"""
        double_conv = DoubleConv(in_channels, out_channels)
        attention = self._create_attention_module(out_channels)

        return nn.Sequential(OrderedDict([
            ('double_conv', double_conv),
            ('attention', attention)
        ]))

    def _down_with_attention(self, in_channels, out_channels):
        """带注意力的下采样块"""
        down = Down(in_channels, out_channels)
        attention = self._create_attention_module(out_channels)

        return nn.Sequential(OrderedDict([
            ('down', down),
            ('attention', attention)
        ]))

    def _up_with_attention(self, in_channels, out_channels, bilinear):
        """带注意力的上采样块"""

        # 注意：Up模块已经包含了DoubleConv，我们只在它之后添加注意力
        class UpWithAttention(nn.Module):
            def __init__(self, up, attention):
                super(UpWithAttention, self).__init__()
                self.up = up
                self.attention = attention

            def forward(self, x1, x2):
                x = self.up(x1, x2)
                return self.attention(x)

        up = Up(in_channels, out_channels, bilinear)
        attention = self._create_attention_module(out_channels)

        return UpWithAttention(up, attention)

    def forward(self, x):
        """前向传播"""
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器路径 + 注意力增强的跳跃连接
        x = self.up1(x5, self.skip_attentions['skip_attn1'](x4))
        x = self.up2(x, self.skip_attentions['skip_attn2'](x3))
        x = self.up3(x, self.skip_attentions['skip_attn3'](x2))
        x = self.up4(x, self.skip_attentions['skip_attn4'](x1))

        # 输出层
        logits = self.outc(x)

        return logits


class HierarchicalAttentionUNet(nn.Module):
    """
    层次化注意力U-Net

    在不同层级使用不同类型的注意力机制
    深层使用通道注意力，浅层使用空间注意力，跳跃连接使用混合注意力
    """

    def __init__(
            self,
            n_channels=3,
            n_classes=3,
            init_features=64,
            use_small_object_attention=True,
            bilinear=False
    ):
        """
        初始化层次化注意力U-Net

        参数:
            n_channels: 输入通道数
            n_classes: 输出类别数
            init_features: 初始特征数量
            use_small_object_attention: 是否使用小目标注意力
            bilinear: 是否使用双线性上采样
        """
        super(HierarchicalAttentionUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_small_object_attention = use_small_object_attention

        # 特征数量
        features = init_features

        # 编码器
        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)

        # 是否使用双线性上采样
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)

        # 解码器
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)

        # 输出层
        self.outc = OutConv(features, n_classes)

        # 编码器注意力 - 使用通道注意力 (关注特征通道重要性)
        self.enc_attentions = nn.ModuleDict({
            'enc1': ChannelAttention(features),
            'enc2': ChannelAttention(features * 2),
            'enc3': ChannelAttention(features * 4),
            'enc4': ChannelAttention(features * 8),
            'enc5': ChannelAttention(features * 16 // factor)
        })

        # 跳跃连接注意力 - 使用混合注意力 (CBAM或小目标注意力)
        self.skip_attentions = nn.ModuleDict()

        skip_attention_dims = [
            features * 8,
            features * 4,
            features * 2,
            features
        ]

        for i, dim in enumerate(skip_attention_dims):
            if use_small_object_attention and i >= 2:  # 在浅层使用小目标注意力
                self.skip_attentions[f'skip{i + 1}'] = SmallObjectAttention(dim)
            else:  # 在深层使用CBAM
                self.skip_attentions[f'skip{i + 1}'] = CBAM(dim)

        # 解码器注意力 - 使用空间注意力 (关注空间位置重要性)
        self.dec_attentions = nn.ModuleDict({
            'dec1': SpatialAttention(),
            'dec2': SpatialAttention(),
            'dec3': SpatialAttention(),
            'dec4': SpatialAttention()
        })

    def forward(self, x):
        """前向传播"""
        # 编码器路径 (带通道注意力)
        x1 = self.enc_attentions['enc1'](self.inc(x))
        x2 = self.enc_attentions['enc2'](self.down1(x1))
        x3 = self.enc_attentions['enc3'](self.down2(x2))
        x4 = self.enc_attentions['enc4'](self.down3(x3))
        x5 = self.enc_attentions['enc5'](self.down4(x4))

        # 解码器路径 (带混合注意力的跳跃连接和空间注意力的特征处理)
        x = self.up1(x5, self.skip_attentions['skip1'](x4))
        x = self.dec_attentions['dec1'](x)

        x = self.up2(x, self.skip_attentions['skip2'](x3))
        x = self.dec_attentions['dec2'](x)

        x = self.up3(x, self.skip_attentions['skip3'](x2))
        x = self.dec_attentions['dec3'](x)

        x = self.up4(x, self.skip_attentions['skip4'](x1))
        x = self.dec_attentions['dec4'](x)

        # 输出层
        logits = self.outc(x)

        return logits


class SmallTumorFocusUNet(nn.Module):
    """
    小肿瘤聚焦U-Net

    专门为增强小型肝脏肿瘤检测能力设计
    在浅层特征图使用小目标注意力，在深层使用尺度感知注意力
    """

    def __init__(
            self,
            n_channels=3,
            n_classes=3,
            init_features=64,
            bilinear=False
    ):
        """
        初始化小肿瘤聚焦U-Net

        参数:
            n_channels: 输入通道数
            n_classes: 输出类别数
            init_features: 初始特征数量
            bilinear: 是否使用双线性上采样
        """
        super(SmallTumorFocusUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 特征数量
        features = init_features

        # 编码器
        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)

        # 是否使用双线性上采样
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)

        # 解码器
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)

        # 输出层
        self.outc = OutConv(features, n_classes)

        # 瓶颈层注意力 - 使用尺度感知注意力，关注多尺度特征
        self.bottleneck_attention = ScaleAwareAttention(features * 16 // factor)

        # 跳跃连接注意力 - 深层用CBAM，浅层用小目标注意力
        self.skip_attentions = nn.ModuleDict({
            'skip1': CBAM(features * 8),  # 深层
            'skip2': CBAM(features * 4),  # 深层
            'skip3': SmallObjectAttention(features * 2),  # 浅层，关注小目标
            'skip4': SmallObjectAttention(features)  # 浅层，关注小目标
        })

        # 特征融合注意力 - 用于增强最终特征
        self.final_attention = SmallObjectAttention(features)

    def forward(self, x):
        """前向传播"""
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 瓶颈层注意力
        x5 = self.bottleneck_attention(x5)

        # 解码器路径 (带注意力增强的跳跃连接)
        x = self.up1(x5, self.skip_attentions['skip1'](x4))
        x = self.up2(x, self.skip_attentions['skip2'](x3))
        x = self.up3(x, self.skip_attentions['skip3'](x2))
        x = self.up4(x, self.skip_attentions['skip4'](x1))

        # 最终特征增强
        x = self.final_attention(x)

        # 输出层
        logits = self.outc(x)

        return logits


class MultiScaleTumorUNet(nn.Module):
    """
    多尺度肿瘤U-Net

    使用多尺度特征融合和尺度特定注意力机制
    适合同时检测不同大小的肿瘤
    """

    def __init__(
            self,
            n_channels=3,
            n_classes=3,
            init_features=64,
            bilinear=False
    ):
        """
        初始化多尺度肿瘤U-Net

        参数:
            n_channels: 输入通道数
            n_classes: 输出类别数
            init_features: 初始特征数量
            bilinear: 是否使用双线性上采样
        """
        super(MultiScaleTumorUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 特征数量
        features = init_features

        # 编码器
        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)

        # 是否使用双线性上采样
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)

        # 解码器
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)

        # 输出层
        self.outc = OutConv(features, n_classes)

        # 多尺度特征融合
        self.multi_scale_fusion = nn.ModuleDict()

        # 侧边连接 - 将深层特征直接连接到浅层
        self.lateral_connections = nn.ModuleDict()

        # 特征尺寸
        dims = [
            features * 16 // factor,  # 瓶颈层
            features * 8,  # 第4层
            features * 4,  # 第3层
            features * 2,  # 第2层
            features  # 第1层
        ]

        # 为每一层创建侧边连接和多尺度融合
        for i in range(1, len(dims)):
            # 侧边连接: 深层 -> 浅层
            self.lateral_connections[f'lateral{i}'] = nn.Conv2d(
                dims[i - 1], dims[i], kernel_size=1, bias=False
            )

            # 多尺度融合
            self.multi_scale_fusion[f'fusion{i}'] = nn.Sequential(
                nn.Conv2d(dims[i] * 2, dims[i], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(dims[i]),
                nn.ReLU(inplace=True)
            )

        # 肿瘤大小特定注意力模块
        self.tumor_size_attention = TumorSizeSpecificAttention(features)

    def forward(self, x):
        """前向传播"""
        # 编码器路径
        x1 = self.inc(x)  # 层1特征
        x2 = self.down1(x1)  # 层2特征
        x3 = self.down2(x2)  # 层3特征
        x4 = self.down3(x3)  # 层4特征
        x5 = self.down4(x4)  # 瓶颈层特征

        # 多尺度特征融合
        # 侧边连接: 将深层特征上采样并与浅层特征融合
        lateral1 = F.interpolate(
            self.lateral_connections['lateral1'](x5),
            size=x4.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        x4_enhanced = self.multi_scale_fusion['fusion1'](torch.cat([x4, lateral1], dim=1))

        lateral2 = F.interpolate(
            self.lateral_connections['lateral2'](x4_enhanced),
            size=x3.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        x3_enhanced = self.multi_scale_fusion['fusion2'](torch.cat([x3, lateral2], dim=1))

        lateral3 = F.interpolate(
            self.lateral_connections['lateral3'](x3_enhanced),
            size=x2.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        x2_enhanced = self.multi_scale_fusion['fusion3'](torch.cat([x2, lateral3], dim=1))

        lateral4 = F.interpolate(
            self.lateral_connections['lateral4'](x2_enhanced),
            size=x1.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        x1_enhanced = self.multi_scale_fusion['fusion4'](torch.cat([x1, lateral4], dim=1))

        # 解码器路径
        x = self.up1(x5, x4_enhanced)
        x = self.up2(x, x3_enhanced)
        x = self.up3(x, x2_enhanced)
        x = self.up4(x, x1_enhanced)

        # 应用肿瘤大小特定注意力
        x = self.tumor_size_attention(x)

        # 输出层
        logits = self.outc(x)

        return logits


def get_attention_unet_model(model_type='standard', **kwargs):
    """
    获取指定类型的注意力U-Net模型

    参数:
        model_type: 模型类型
            - 'standard': 标准U-Net (没有注意力)
            - 'attention': 注意力U-Net (跳跃连接处有注意力)
            - 'deep_attention': 深度注意力U-Net (每层都有注意力)
            - 'hierarchical': 层次化注意力U-Net (不同层用不同类型注意力)
            - 'small_tumor': 小肿瘤聚焦U-Net (专注于小肿瘤检测)
            - 'multi_scale': 多尺度肿瘤U-Net (适合检测不同大小肿瘤)
        **kwargs: 传递给模型构造函数的额外参数

    返回:
        选定的模型实例
    """
    from .unet import UNet

    # 复制kwargs以避免修改原始字典
    model_kwargs = kwargs.copy()

    if model_type == 'standard':
        # 标准UNet不需要attention_type参数
        if 'attention_type' in model_kwargs:
            model_kwargs.pop('attention_type')
        return UNet(**model_kwargs)
    elif model_type == 'attention':
        return AttentionUNet(**model_kwargs)
    elif model_type == 'deep_attention':
        return DeepAttentionUNet(**model_kwargs)
    elif model_type == 'hierarchical':
        # 层次化注意力UNet不需要attention_type参数
        if 'attention_type' in model_kwargs:
            model_kwargs.pop('attention_type')
        return HierarchicalAttentionUNet(**model_kwargs)
    elif model_type == 'small_tumor':
        # 小肿瘤聚焦UNet不需要attention_type参数
        if 'attention_type' in model_kwargs:
            model_kwargs.pop('attention_type')
        return SmallTumorFocusUNet(**model_kwargs)
    elif model_type == 'multi_scale':
        # 多尺度肿瘤UNet不需要attention_type参数
        if 'attention_type' in model_kwargs:
            model_kwargs.pop('attention_type')
        return MultiScaleTumorUNet(**model_kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机输入
    batch_size, channels, height, width = 2, 3, 256, 256
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试不同类型的模型
    model_types = [
        'standard',
        'attention',
        'deep_attention',
        'hierarchical',
        'small_tumor',
        'multi_scale'
    ]

    for model_type in model_types:
        print(f"\n测试 {model_type} 模型:")
        model = get_attention_unet_model(
            model_type=model_type,
            n_channels=channels,
            n_classes=3,
            init_features=64
        ).to(device)

        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {total_params:,}")

        # 测试前向传播
        output = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")