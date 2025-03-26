# src/models/enhanced_attention_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parents[2]))

# 导入基础U-Net组件
from src.models.unet import DoubleConv, Down, Up, OutConv

# 导入注意力模块
from src.models.attention_modules.channel_attention import ChannelAttention, CBAM_Channel, ECABlock, GCTModule
from src.models.attention_modules.spatial_attention import SpatialAttention, SmallObjectSpatialAttention
from src.models.attention_modules.hybrid_attention import CBAM, SmallObjectAttention, ScaleAwareAttention


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
        medium_out = self.medium_tumor_branch(x[:, :in_channels // 2, :, :])
        large_out = self.large_tumor_branch(x)

        # 获取权重后的前向传播
        # 注意力路径选择
        small_path = small_out * small_weight
        medium_path = medium_out * medium_weight
        large_path = large_out * large_weight

        # 权重融合，加倍强调小型肿瘤
        combined = (small_path * 1.5 + medium_path + large_path) / (small_weight * 1.5 + medium_weight + large_weight)

        # 最终输出
        output = self.output_layer(combined)

        return x * torch.sigmoid(output)  # 应用作为注意力

class DynamicAttention(nn.Module):
    """
    动态注意力模块

    智能地平衡空间和通道注意力，根据输入特征自适应调整注意力分配策略
    特别适合检测大小差异很大的目标，如从小型到大型肿瘤
    """

    def __init__(self, in_channels, reduction_ratio=16, dilation_sizes=[1, 2, 4]):
        """
        初始化动态注意力模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 降维比例，用于减少计算成本
            dilation_sizes: 空间上下文感受野的膨胀率列表
        """
        super(DynamicAttention, self).__init__()

        # 确保降维后的通道数至少为1
        reduced_channels = max(1, in_channels // reduction_ratio)

        # 通道注意力门控
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 多尺度空间上下文模块 - 使用不同膨胀率的卷积获取多尺度信息
        self.spatial_contexts = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // len(dilation_sizes),
                      kernel_size=3, padding=d, dilation=d, groups=in_channels // len(dilation_sizes))
            for d in dilation_sizes
        ])

        # 空间注意力融合
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 动态平衡器 - 学习通道和空间注意力之间的平衡
        self.dynamic_balancer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 小目标增强器 - 特别关注小型目标区域
        self.small_object_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, kernel_size=1),
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

        # 1. 计算通道注意力
        channel_attn = self.channel_gate(x)

        # 2. 计算多尺度空间上下文
        spatial_contexts = []
        for context_layer in self.spatial_contexts:
            spatial_contexts.append(context_layer(x))

        # 连接多尺度上下文特征
        multi_scale_context = torch.cat(spatial_contexts, dim=1)

        # 3. 生成空间注意力图
        spatial_attn = self.spatial_fusion(multi_scale_context)

        # 4. 动态平衡通道和空间注意力
        balance_weights = self.dynamic_balancer(x)
        channel_weight = balance_weights[:, 0:1, :, :]
        spatial_weight = balance_weights[:, 1:2, :, :]

        # 5. 检测并增强小目标区域
        # 小区域通常有更高的局部变化，我们使用局部特征变化来发现它们
        local_var = F.avg_pool2d(
            (x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)) ** 2,
            kernel_size=3, stride=1, padding=1
        )
        small_object_attn = self.small_object_enhancer(local_var)

        # 6. 组合各种注意力机制
        # 基本注意力 = 通道权重*通道注意力 + 空间权重*空间注意力
        basic_attn = channel_attn * channel_weight + spatial_attn * spatial_weight

        # 将小目标注意力与基本注意力结合
        # 增强系数: 小目标区域获得额外增强
        enhanced_attn = basic_attn * (1.0 + small_object_attn)

        # 应用最终注意力
        return x * enhanced_attn


class MultiScaleFeatureIntegration(nn.Module):
    """
    多尺度特征集成模块

    整合不同尺度的特征，提升对不同大小目标的感知能力
    特别适合同时检测小型和大型肿瘤
    """

    def __init__(self, in_channels, out_channels=None):
        """
        初始化多尺度特征集成模块

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数，默认与输入相同
        """
        super(MultiScaleFeatureIntegration, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        branch_channels = in_channels // 4

        # 分支1: 1x1卷积处理点级特征
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 分支2: 3x3卷积处理局部特征
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 分支3: 3x3空洞卷积，扩大感受野
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 分支4: 池化后1x1卷积，获取全局信息
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 特征重校准 - 根据通道重要性调整权重
        self.channel_calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            多尺度融合后的特征，形状与输入相同
        """
        # 获取输入尺寸
        height, width = x.size(2), x.size(3)

        # 分支1: 点级特征
        branch1_out = self.branch1(x)

        # 分支2: 局部特征
        branch2_out = self.branch2(x)

        # 分支3: 扩展感受野
        branch3_out = self.branch3(x)

        # 分支4: 全局特征，需要上采样回原始尺寸
        branch4_out = self.branch4(x)
        branch4_out = F.interpolate(branch4_out, size=(height, width),
                                    mode='bilinear', align_corners=False)

        # 连接所有分支的输出
        concat_features = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

        # 融合多尺度特征
        fused_features = self.fusion(concat_features)

        # 通道重校准 - 调整各通道的重要性
        channel_weights = self.channel_calibration(fused_features)

        # 应用通道权重
        calibrated_features = fused_features * channel_weights

        return calibrated_features


class SmallObjectEnhancedAttention(nn.Module):
    """
    小目标增强注意力模块

    结合动态注意力和多尺度特征集成，专门针对小型肿瘤检测优化
    动态调整注意力分配，特别强化对小目标区域的关注
    """

    def __init__(self, in_channels, reduction_ratio=8):
        """
        初始化小目标增强注意力模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 降维比例
        """
        super(SmallObjectEnhancedAttention, self).__init__()

        # 动态注意力模块
        self.dynamic_attention = DynamicAttention(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
            dilation_sizes=[1, 2, 3]
        )

        # 多尺度特征集成模块
        self.feature_integration = MultiScaleFeatureIntegration(
            in_channels=in_channels,
            out_channels=in_channels
        )

        # 局部-全局上下文对比模块 - 检测小目标的关键
        # 小目标往往在局部和全局上下文之间有较大差异
        self.local_global_contrast = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 边缘感知模块 - 小肿瘤通常有相对明显的边界
        self.edge_awareness = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 注意力融合
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            增强后的特征，形状与输入相同
        """
        # 1. 应用动态注意力
        dynamic_attn_out = self.dynamic_attention(x)

        # 2. 多尺度特征集成
        integrated_features = self.feature_integration(x)

        # 3. 局部-全局上下文对比
        # 提取局部特征（小卷积核）和全局特征（全局池化）
        local_features = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        global_features = F.adaptive_avg_pool2d(x, 1)
        global_features = F.interpolate(global_features, size=x.shape[2:],
                                        mode='bilinear', align_corners=False)

        # 连接局部和全局特征
        context_contrast = torch.cat([local_features, global_features], dim=1)
        contrast_map = self.local_global_contrast(context_contrast)

        # 4. 边缘感知 - 通过计算特征梯度来检测边缘
        edge_map = self.edge_awareness(x)

        # 5. 增强特征
        # 结合动态注意力、多尺度特征，以及局部-全局对比
        enhanced_features = dynamic_attn_out + integrated_features

        # 根据对比图和边缘图进一步增强特征
        enhanced_features = enhanced_features * (1.0 + contrast_map)
        enhanced_features = enhanced_features * (1.0 + edge_map)

        # 最终融合
        output = self.fusion(enhanced_features)

        return output


class EnhancedAttentionUNet(nn.Module):
    """
    增强型注意力U-Net

    整合了最新的动态注意力机制，特别针对小型肿瘤检测进行优化
    对网络不同层级应用不同类型的注意力，实现层次化的特征增强
    """

    def __init__(
            self,
            n_channels=3,
            n_classes=3,
            init_features=64,
            small_tumor_focus=True,
            use_dynamic_attention=True,
            bilinear=False
    ):
        """
        初始化增强型注意力U-Net

        参数:
            n_channels: 输入通道数
            n_classes: 输出类别数
            init_features: 初始特征图数量
            small_tumor_focus: 是否特别关注小型肿瘤
            use_dynamic_attention: 是否使用动态注意力机制
            bilinear: 是否使用双线性上采样
        """
        super(EnhancedAttentionUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.small_tumor_focus = small_tumor_focus
        self.use_dynamic_attention = use_dynamic_attention

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

        # 瓶颈层注意力 - 使用多尺度特征集成
        self.bottleneck_attention = MultiScaleFeatureIntegration(
            features * 16 // factor
        )

        # 编码器注意力 - 不同层级使用不同类型的注意力
        self.encoder_attentions = nn.ModuleDict({
            'enc1': ChannelAttention(features),
            'enc2': ChannelAttention(features * 2),
            'enc3': CBAM(features * 4),
            'enc4': CBAM(features * 8)
        })

        # 跳跃连接注意力
        attention_dims = [
            features * 8,  # 第4层
            features * 4,  # 第3层
            features * 2,  # 第2层
            features  # 第1层
        ]

        self.skip_attentions = nn.ModuleDict()

        for i, dim in enumerate(attention_dims):
            # 深层使用CBAM，浅层使用小目标注意力
            if i >= 2 and self.small_tumor_focus:
                if self.use_dynamic_attention:
                    # 使用新的动态注意力和小目标增强
                    self.skip_attentions[f'skip{i + 1}'] = SmallObjectEnhancedAttention(dim)
                else:
                    # 使用原来的小目标注意力
                    self.skip_attentions[f'skip{i + 1}'] = SmallObjectAttention(dim)
            else:
                if self.use_dynamic_attention:
                    # 使用新的动态注意力
                    self.skip_attentions[f'skip{i + 1}'] = DynamicAttention(dim)
                else:
                    # 使用原来的CBAM注意力
                    self.skip_attentions[f'skip{i + 1}'] = CBAM(dim)

        # 解码器注意力 - 使用空间注意力
        self.decoder_attentions = nn.ModuleDict({
            'dec1': SpatialAttention(),
            'dec2': SpatialAttention(),
            'dec3': SmallObjectSpatialAttention(features * 2) if self.small_tumor_focus else SpatialAttention(),
            'dec4': SmallObjectSpatialAttention(features) if self.small_tumor_focus else SpatialAttention()
        })

        # 最终特征增强 - 小目标增强注意力
        if self.small_tumor_focus and self.use_dynamic_attention:
            self.final_attention = SmallObjectEnhancedAttention(features)
        elif self.small_tumor_focus:
            self.final_attention = SmallObjectAttention(features)
        else:
            self.final_attention = CBAM(features)

    def forward(self, x):
        """前向传播"""
        # 编码器路径（带有注意力增强）
        x1 = self.encoder_attentions['enc1'](self.inc(x))
        x2 = self.encoder_attentions['enc2'](self.down1(x1))
        x3 = self.encoder_attentions['enc3'](self.down2(x2))
        x4 = self.encoder_attentions['enc4'](self.down3(x3))

        # 瓶颈层注意力增强
        x5 = self.down4(x4)
        x5 = self.bottleneck_attention(x5)

        # 解码器路径 + 注意力增强的跳跃连接
        x = self.up1(x5, self.skip_attentions['skip1'](x4))
        x = self.decoder_attentions['dec1'](x)

        x = self.up2(x, self.skip_attentions['skip2'](x3))
        x = self.decoder_attentions['dec2'](x)

        x = self.up3(x, self.skip_attentions['skip3'](x2))
        x = self.decoder_attentions['dec3'](x)

        x = self.up4(x, self.skip_attentions['skip4'](x1))
        x = self.decoder_attentions['dec4'](x)

        # 最终特征增强
        x = self.final_attention(x)

        # 输出层
        logits = self.outc(x)

        return logits


class BalancedRecallUNet(nn.Module):
    """
    平衡召回率的增强U-Net

    针对提高召回率优化的模型架构，在各层级添加额外的特征路径
    同时集成动态注意力机制，特别关注小型肿瘤
    """

    def __init__(
            self,
            n_channels=3,
            n_classes=3,
            init_features=64,
            use_deep_supervision=True,
            bilinear=False
    ):
        """
        初始化平衡召回率的增强U-Net

        参数:
            n_channels: 输入通道数
            n_classes: 输出类别数
            init_features: 初始特征图数量
            use_deep_supervision: 是否使用深度监督（多层输出）
            bilinear: 是否使用双线性上采样
        """
        super(BalancedRecallUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_deep_supervision = use_deep_supervision

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

        # 解码器 - 使用残差连接结构
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)

        # 主输出层
        self.outc = OutConv(features, n_classes)

        # 深度监督用的附加输出层（不同分辨率）
        if self.use_deep_supervision:
            self.deep_outc1 = OutConv(features * 8 // factor, n_classes)
            self.deep_outc2 = OutConv(features * 4 // factor, n_classes)
            self.deep_outc3 = OutConv(features * 2 // factor, n_classes)

        # 瓶颈层注意力 - 使用多尺度特征集成
        self.bottleneck_attention = MultiScaleFeatureIntegration(
            features * 16 // factor
        )

        # 融合跳跃连接的特征和上采样特征
        self.fusion1 = nn.Conv2d(features * 16 // factor, features * 8 // factor, kernel_size=1)
        self.fusion2 = nn.Conv2d(features * 8 // factor, features * 4 // factor, kernel_size=1)
        self.fusion3 = nn.Conv2d(features * 4 // factor, features * 2 // factor, kernel_size=1)
        self.fusion4 = nn.Conv2d(features * 2 // factor, features, kernel_size=1)

        # 添加跳跃连接注意力
        self.skip_attention1 = SmallObjectEnhancedAttention(features * 8)
        self.skip_attention2 = SmallObjectEnhancedAttention(features * 4)
        self.skip_attention3 = SmallObjectEnhancedAttention(features * 2)
        self.skip_attention4 = SmallObjectEnhancedAttention(features)

        # 添加最终特征增强
        self.final_attention = SmallObjectEnhancedAttention(features)

    def forward(self, x):
        """前向传播"""
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck_attention(self.down4(x4))

        # 解码器路径 - 使用增强的跳跃连接
        # 第一层上采样 - 增强x4特征
        d1 = self.up1(x5, self.skip_attention1(x4))

        # 深度监督1
        if self.use_deep_supervision:
            out1 = self.deep_outc1(d1)

        # 第二层上采样 - 增强x3特征
        d2 = self.up2(d1, self.skip_attention2(x3))

        # 深度监督2
        if self.use_deep_supervision:
            out2 = self.deep_outc2(d2)

        # 第三层上采样 - 增强x2特征
        d3 = self.up3(d2, self.skip_attention3(x2))

        # 深度监督3
        if self.use_deep_supervision:
            out3 = self.deep_outc3(d3)

        # 第四层上采样 - 增强x1特征
        d4 = self.up4(d3, self.skip_attention4(x1))

        # 最终特征增强
        enhanced = self.final_attention(d4)

        # 主输出
        main_out = self.outc(enhanced)

        # 如果使用深度监督，返回所有输出；否则只返回主输出
        if self.use_deep_supervision:
            # 确保所有输出的大小一致 - 上采样到原始尺寸
            out1 = F.interpolate(out1, size=main_out.shape[2:], mode='bilinear', align_corners=False)
            out2 = F.interpolate(out2, size=main_out.shape[2:], mode='bilinear', align_corners=False)
            out3 = F.interpolate(out3, size=main_out.shape[2:], mode='bilinear', align_corners=False)

            return main_out, out1, out2, out3
        else:
            return main_out


class RecallFocusedLoss(nn.Module):
    """
    针对召回率优化的损失函数

    特别关注减少假阴性（提高召回率），同时保持精确率
    对小型肿瘤区域给予更高权重
    """

    def __init__(self, recall_weight=2.0, size_weight=1.5, smooth=1e-5):
        """
        初始化召回率优化损失

        参数:
            recall_weight: 召回率权重，控制假阴性惩罚力度
            size_weight: 小目标权重增强系数
            smooth: 平滑因子，避免除零错误
        """
        super(RecallFocusedLoss, self).__init__()
        self.recall_weight = recall_weight
        self.size_weight = size_weight
        self.smooth = smooth

    def forward(self, pred, target, weight_map=None):
        """
        计算损失

        参数:
            pred: 预测结果，形状为[B, C, H, W]
            target: 目标掩码，形状为[B, H, W]
            weight_map: 像素权重图，用于加权小型肿瘤区域的损失

        返回:
            加权损失值
        """
        # 获取维度信息
        batch_size, num_classes = pred.size(0), pred.size(1)

        # 计算softmax概率
        pred_softmax = F.softmax(pred, dim=1)

        # 将目标转换为one-hot编码
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # 初始化损失
        total_loss = 0.0

        # 假设肿瘤类别是2
        tumor_class = 2

        # 获取肿瘤预测和真实标签
        pred_tumor = pred_softmax[:, tumor_class]
        target_tumor = target_one_hot[:, tumor_class]

        # 计算Dice系数 (对所有类别)
        dice_loss = 0.0
        for c in range(num_classes):
            pred_c = pred_softmax[:, c]
            target_c = target_one_hot[:, c]

            # 如果是肿瘤类别且提供了权重图，则应用权重图
            if c == tumor_class and weight_map is not None:
                # 对小肿瘤区域给予更高权重
                weighted_pred = pred_c * weight_map
                weighted_target = target_c * weight_map

                intersection = torch.sum(weighted_pred * weighted_target, dim=(1, 2))
                cardinality = torch.sum(weighted_pred + weighted_target, dim=(1, 2))
            else:
                intersection = torch.sum(pred_c * target_c, dim=(1, 2))
                cardinality = torch.sum(pred_c + target_c, dim=(1, 2))

            dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
            dice_loss += (1.0 - dice.mean())

        dice_loss = dice_loss / num_classes

        # 计算肿瘤召回率损失 (假阴性惩罚)
        # 真阳性: 正确预测的肿瘤区域
        true_positives = torch.sum(pred_tumor * target_tumor, dim=(1, 2))

        # 假阴性: 漏检的肿瘤区域
        false_negatives = torch.sum((1.0 - pred_tumor) * target_tumor, dim=(1, 2))

        # 如果提供了权重图，则应用于假阴性区域
        if weight_map is not None:
            # 假阴性区域的权重
            false_negative_weights = torch.sum(weight_map * (1.0 - pred_tumor) * target_tumor, dim=(1, 2))
            false_negative_weights = false_negative_weights / (torch.sum(target_tumor, dim=(1, 2)) + self.smooth)

            # 增强权重
            false_negative_weights = false_negative_weights * self.size_weight

            # 将权重应用于假阴性
            weighted_false_negatives = false_negatives * (1.0 + false_negative_weights)

            # 召回率损失 (越低召回率越高)
            recall_loss = weighted_false_negatives / (true_positives + weighted_false_negatives + self.smooth)
        else:
            recall_loss = false_negatives / (true_positives + false_negatives + self.smooth)

        recall_loss = recall_loss.mean()

        # 组合损失 - Dice损失加上加权的召回率损失
        total_loss = dice_loss + self.recall_weight * recall_loss

        return total_loss


class DeepSupervisionLoss(nn.Module):
    """
    深度监督损失函数

    结合主输出和多个辅助输出的损失
    帮助网络在不同深度层都学习有效的特征表示
    """

    def __init__(self, main_loss, aux_weights=[0.4, 0.3, 0.2]):
        """
        初始化深度监督损失

        参数:
            main_loss: 主损失函数
            aux_weights: 辅助输出的权重列表
        """
        super(DeepSupervisionLoss, self).__init__()
        self.main_loss = main_loss
        self.aux_weights = aux_weights

    def forward(self, outputs, target, weight_map=None):
        """
        计算深度监督损失

        参数:
            outputs: 模型输出的元组 (主输出, 辅助输出1, 辅助输出2, ...)
            target: 目标掩码
            weight_map: 像素权重图

        返回:
            加权组合损失
        """
        if not isinstance(outputs, tuple):
            # 如果只有一个输出，直接使用主损失函数
            return self.main_loss(outputs, target, weight_map)

        # 分离主输出和辅助输出
        main_output = outputs[0]
        aux_outputs = outputs[1:]

        # 确保权重数量与辅助输出数量一致
        assert len(aux_outputs) == len(self.aux_weights), \
            f"辅助输出数量 ({len(aux_outputs)}) 必须与权重数量 ({len(self.aux_weights)}) 一致"

        # 计算主损失
        main_loss = self.main_loss(main_output, target, weight_map)

        # 计算辅助损失
        total_aux_loss = 0.0
        for aux_output, weight in zip(aux_outputs, self.aux_weights):
            aux_loss = self.main_loss(aux_output, target, weight_map)
            total_aux_loss += weight * aux_loss

        # 组合损失 (主损失 + 加权辅助损失)
        total_loss = main_loss + total_aux_loss

        return total_loss


def get_enhanced_model(model_type='enhanced_attention', **kwargs):
    """
    获取增强注意力模型实例

    参数:
        model_type: 模型类型
            - 'enhanced_attention': 增强型注意力U-Net
            - 'balanced_recall': 平衡召回率的增强U-Net
        **kwargs: 传递给模型的额外参数

    返回:
        模型实例
    """
    # 复制kwargs以避免修改原始字典
    model_kwargs = kwargs.copy()

    if model_type == 'enhanced_attention':
        return EnhancedAttentionUNet(**model_kwargs)
    elif model_type == 'balanced_recall':
        return BalancedRecallUNet(**model_kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机输入
    batch_size, channels, height, width = 2, 3, 256, 256
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试增强型注意力U-Net
    print("\n测试增强型注意力U-Net:")
    model1 = EnhancedAttentionUNet(
        n_channels=channels,
        n_classes=3,
        init_features=64,
        small_tumor_focus=True,
        use_dynamic_attention=True
    ).to(device)

    # 计算模型参数数量
    total_params1 = sum(p.numel() for p in model1.parameters())
    print(f"模型参数数量: {total_params1:,}")

    # 测试前向传播
    output1 = model1(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output1.shape}")

    # 测试平衡召回率的增强U-Net
    print("\n测试平衡召回率的增强U-Net:")
    model2 = BalancedRecallUNet(
        n_channels=channels,
        n_classes=3,
        init_features=64,
        use_deep_supervision=True
    ).to(device)

    # 计算模型参数数量
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"模型参数数量: {total_params2:,}")

    # 测试前向传播
    outputs2 = model2(x)
    if isinstance(outputs2, tuple):
        print(f"输入形状: {x.shape}")
        print(f"主输出形状: {outputs2[0].shape}")
        print(f"辅助输出1形状: {outputs2[1].shape}")
        print(f"辅助输出2形状: {outputs2[2].shape}")
        print(f"辅助输出3形状: {outputs2[3].shape}")
    else:
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {outputs2.shape}")

    # 测试召回率优化损失
    print("\n测试召回率优化损失:")
    # 创建模拟数据
    pred = torch.randn(2, 3, 64, 64).to(device)
    target = torch.randint(0, 3, (2, 64, 64)).to(device)
    weight_map = torch.rand(2, 64, 64).to(device)

    # 创建损失函数
    recall_loss = RecallFocusedLoss(recall_weight=2.0, size_weight=1.5)
    loss_value = recall_loss(pred, target, weight_map)
    print(f"召回率优化损失值: {loss_value.item()}")

    # 测试深度监督损失
    print("\n测试深度监督损失:")
    deep_supervision_loss = DeepSupervisionLoss(recall_loss, aux_weights=[0.4, 0.3, 0.2])

    # 创建多输出模拟数据
    outputs = (pred, pred.clone(), pred.clone(), pred.clone())
    ds_loss_value = deep_supervision_loss(outputs, target, weight_map)
    print(f"深度监督损失值: {ds_loss_value.item()}")