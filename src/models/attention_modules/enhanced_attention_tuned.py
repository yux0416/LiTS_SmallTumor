# src/models/attention_modules/enhanced_attention_tuned.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TunedLocalContrastAttention(nn.Module):
    """
    调优版局部对比度增强注意力
    微调参数以进一步提高小型肿瘤检测性能
    """

    def __init__(self, in_channels, kernel_size=5, contrast_weight=1.5, edge_weight=1.2):
        """
        初始化调优的局部对比度增强注意力模块

        参数:
            in_channels: 输入通道数
            kernel_size: 局部上下文卷积核大小 (减小为5，更适合小型结构)
            contrast_weight: 对比度增强权重 (提高到1.5增强对比效果)
            edge_weight: 边缘增强权重 (提高到1.2增强边缘特征)
        """
        super(TunedLocalContrastAttention, self).__init__()

        # 使用更小的组卷积来更精确地捕获局部上下文
        self.local_context = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels // 4  # 减少分组数，增强特征交互
        )

        # 强化对比度计算 - 使用更深的网络
        self.contrast_calc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 改进的边缘检测器 - 多尺度边缘检测
        self.edge_detector = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, groups=in_channels // 4),
                nn.Conv2d(in_channels // 4, 1, kernel_size=1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=5, padding=2, groups=in_channels // 4),
                nn.Conv2d(in_channels // 4, 1, kernel_size=1),
                nn.Sigmoid()
            )
        ])

        # 添加自注意力机制，捕获更广泛的空间依赖关系
        self.self_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 保存增强权重
        self.contrast_weight = contrast_weight
        self.edge_weight = edge_weight

    def forward(self, x):
        # 获取局部上下文
        local_feat = self.local_context(x)

        # 计算局部与原始特征的对比
        contrast_input = torch.cat([x, local_feat], dim=1)
        contrast_map = self.contrast_calc(contrast_input)

        # 多尺度边缘检测
        edge_maps = [detector(x) for detector in self.edge_detector]
        edge_map = torch.mean(torch.cat(edge_maps, dim=1), dim=1, keepdim=True)

        # 计算自注意力图
        self_attn = self.self_attention(x)

        # 综合对比度、边缘信息和自注意力
        # 使用调优后的权重来增强对比度和边缘特征
        attention = contrast_map * (1.0 + self.contrast_weight * edge_map) * (1.0 + self_attn)

        # 应用注意力
        return x * attention


class HybridLocalContrastSmallObjectAttention(nn.Module):
    """
    混合局部对比度与小目标增强注意力
    结合局部对比度和专门为小型目标设计的增强机制
    """

    def __init__(self, in_channels, reduction_ratio=4):
        super(HybridLocalContrastSmallObjectAttention, self).__init__()

        # 局部对比度注意力分支
        self.local_contrast = TunedLocalContrastAttention(
            in_channels,
            kernel_size=5,
            contrast_weight=1.5,
            edge_weight=1.2
        )

        # 小型目标感知分支 - 专注于捕获微小特征
        self.small_object_branch = nn.Sequential(
            # 首先降维
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            # 连续多个小卷积核，增强对小目标特征的捕获
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            # 恢复通道数
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 这个模块识别具有高局部变化的区域，通常是小目标的特征
        self.local_variation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 门控机制，动态调整两个分支的权重
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 输出转换
        self.output_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # 局部对比度特征
        contrast_feat = self.local_contrast(x)

        # 小目标特征
        small_object_feat = x * self.small_object_branch(x)

        # 计算局部变化图 - 帮助识别小目标
        # 小目标区域通常有较高的局部变化
        local_var = F.avg_pool2d(
            (x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)) ** 2,
            kernel_size=3, stride=1, padding=1
        )
        variation_weight = self.local_variation(local_var)

        # 计算门控权重，决定两个分支的重要性
        gates = self.gate(x)
        contrast_gate = gates[:, 0:1, :, :]
        small_object_gate = gates[:, 1:2, :, :]

        # 融合两个分支的输出
        # 小目标分支的权重会根据局部变化进行增强
        output = (contrast_feat * contrast_gate +
                  small_object_feat * small_object_gate * (1.0 + variation_weight))

        # 最终转换
        return self.output_transform(output) + x  # 残差连接


# 整合局部对比度和尺度感知的多尺度小型肿瘤注意力
class IntegratedSmallTumorAttention(nn.Module):
    """
    整合型小型肿瘤注意力

    结合局部对比度、尺度感知和多尺度特征的优点
    专门针对小型肿瘤的特征进行增强
    """

    def __init__(self, in_channels, reduction_ratio=4):
        super(IntegratedSmallTumorAttention, self).__init__()

        # 局部对比度注意力
        self.local_contrast = TunedLocalContrastAttention(
            in_channels,
            kernel_size=5,
            contrast_weight=1.5,
            edge_weight=1.2
        )

        # 多尺度特征集成 - 捕获不同尺度的特征
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=5, padding=2),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
            )
        ])

        # 尺度特定增强
        self.scale_enhancement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 融合模块 - 整合不同的注意力特征
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # 感受野自适应模块 - 针对不同大小的肿瘤调整感受野
        self.receptive_field_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 3, kernel_size=1),  # 3个不同的感受野权重
            nn.Softmax(dim=1)
        )

        # 不同感受野的卷积
        self.receptive_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        ])

    def forward(self, x):
        # 局部对比度增强
        contrast_feat = self.local_contrast(x)

        # 多尺度特征提取
        multi_scale_feats = []
        for i, layer in enumerate(self.multi_scale):
            feat = layer(x)
            # 对于全局平均池化的分支，需要上采样
            if i == 3:
                feat = feat.expand(-1, -1, x.size(2), x.size(3))
            multi_scale_feats.append(feat)

        # 连接多尺度特征
        multi_scale_feat = torch.cat(multi_scale_feats, dim=1)

        # 尺度增强
        scale_weight = self.scale_enhancement(multi_scale_feat)

        # 感受野自适应
        rf_weights = self.receptive_field_adapter(x)
        rf_feat = torch.zeros_like(x)
        for i, conv in enumerate(self.receptive_convs):
            rf_feat += conv(x) * rf_weights[:, i:i + 1, :, :]

        # 融合不同的注意力特征
        fusion_input = torch.cat([contrast_feat, rf_feat], dim=1)
        fusion_weight = self.fusion(fusion_input)

        # 最终增强
        enhanced_feat = x * fusion_weight * (1.0 + scale_weight)

        return enhanced_feat


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机输入
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试调优的局部对比度注意力
    tlca = TunedLocalContrastAttention(channels).to(device)
    output_tlca = tlca(x)
    print(f"调优的局部对比度注意力输出形状: {output_tlca.shape}")

    # 测试混合局部对比度与小目标注意力
    hlcso = HybridLocalContrastSmallObjectAttention(channels).to(device)
    output_hlcso = hlcso(x)
    print(f"混合局部对比度与小目标注意力输出形状: {output_hlcso.shape}")

    # 测试整合型小型肿瘤注意力
    ista = IntegratedSmallTumorAttention(channels).to(device)
    output_ista = ista(x)
    print(f"整合型小型肿瘤注意力输出形状: {output_ista.shape}")

    # 打印参数数量比较
    tlca_params = sum(p.numel() for p in tlca.parameters())
    hlcso_params = sum(p.numel() for p in hlcso.parameters())
    ista_params = sum(p.numel() for p in ista.parameters())

    print(f"\n参数数量比较:")
    print(f"调优的局部对比度注意力: {tlca_params:,} 参数")
    print(f"混合局部对比度与小目标注意力: {hlcso_params:,} 参数")
    print(f"整合型小型肿瘤注意力: {ista_params:,} 参数")