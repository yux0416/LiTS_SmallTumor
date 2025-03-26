# src/models/attention_modules/dynamic_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机输入
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试动态注意力模块
    da = DynamicAttention(channels).to(device)
    output_da = da(x)
    print(f"动态注意力输出形状: {output_da.shape}")

    # 测试多尺度特征集成模块
    msfi = MultiScaleFeatureIntegration(channels).to(device)
    output_msfi = msfi(x)
    print(f"多尺度特征集成输出形状: {output_msfi.shape}")

    # 测试小目标增强注意力模块
    soea = SmallObjectEnhancedAttention(channels).to(device)
    output_soea = soea(x)
    print(f"小目标增强注意力输出形状: {output_soea.shape}")