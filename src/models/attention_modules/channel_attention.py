# src/models/attention_modules/channel_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    通道注意力模块 (SE Block)

    通过挤压和激励操作重新校准各通道的特征响应
    为不同通道的特征赋予不同的权重
    """

    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化通道注意力模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 降维比例，用于减少计算成本
        """
        super(ChannelAttention, self).__init__()

        # 确保降维后的通道数至少为1
        reduced_channels = max(1, in_channels // reduction_ratio)

        # 挤压和激励操作
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        # 共享MLP (FC -> ReLU -> FC -> Sigmoid)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            加权后的特征，形状相同
        """
        # 全局平均池化和最大池化
        # [B, C, H, W] -> [B, C, 1, 1]
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)

        # 通过共享MLP生成通道注意力权重
        # [B, C, 1, 1] -> [B, C, 1, 1]
        avg_attention = self.mlp(avg_pool)
        max_attention = self.mlp(max_pool)

        # 融合两种池化方式的结果
        attention = self.sigmoid(avg_attention + max_attention)

        # 通过注意力权重增强原始特征
        # [B, C, H, W] * [B, C, 1, 1] -> [B, C, H, W]
        return x * attention


class ECABlock(nn.Module):
    """
    高效通道注意力模块 (ECA Block)

    无需降维和升维，直接通过一维卷积学习通道间的相互依赖关系
    比SE Block更高效，同时保持性能
    """

    def __init__(self, in_channels, gamma=2, b=1):
        """
        初始化高效通道注意力模块

        参数:
            in_channels: 输入通道数
            gamma, b: 用于自适应计算核大小的参数
        """
        super(ECABlock, self).__init__()

        # 自适应计算卷积核大小
        kernel_size = int(abs(math.log(in_channels, 2) + b) / gamma)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1  # 确保是奇数

        # 全局平均池化和一维卷积
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            加权后的特征，形状相同
        """
        # 全局平均池化
        # [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)

        # 调整维度，准备进行一维卷积
        # [B, C, 1, 1] -> [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)

        # 一维卷积处理通道间关系
        # [B, 1, C] -> [B, 1, C]
        y = self.conv(y)

        # 调整回原始维度，应用sigmoid
        # [B, 1, C] -> [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        # 通过注意力权重增强原始特征
        # [B, C, H, W] * [B, C, 1, 1] -> [B, C, H, W]
        return x * y


class CBAM_Channel(nn.Module):
    """
    CBAM通道注意力模块

    结合了压缩和空间注意力机制
    先应用通道注意力再应用空间注意力
    """

    def __init__(self, in_channels, reduction_ratio=16):
        """
        初始化CBAM通道注意力模块

        参数:
            in_channels: 输入通道数
            reduction_ratio: 降维比例，用于减少计算成本
        """
        super(CBAM_Channel, self).__init__()

        # 确保降维后的通道数至少为1
        reduced_channels = max(1, in_channels // reduction_ratio)

        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            加权后的特征，形状相同
        """
        # 全局平均池化
        # [B, C, H, W] -> [B, C, 1, 1]
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)

        # 全局最大池化
        # [B, C, H, W] -> [B, C, 1, 1]
        max_pool, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)
        max_pool = max_pool.unsqueeze(-1)

        # 通过共享MLP生成通道注意力权重
        # [B, C, 1, 1] -> [B, C, 1, 1]
        avg_attention = self.mlp(avg_pool)
        max_attention = self.mlp(max_pool)

        # 融合两种池化方式的结果
        attention = self.sigmoid(avg_attention + max_attention)

        # 通过注意力权重增强原始特征
        # [B, C, H, W] * [B, C, 1, 1] -> [B, C, H, W]
        return x * attention


class GCTModule(nn.Module):
    """
    门控通道变换模块 (GCT)

    通过全局上下文建模来增强通道注意力
    特别适合医学图像的细粒度差异捕获
    """

    def __init__(self, in_channels, epsilon=1e-5, mode='l2', after_relu=False):
        """
        初始化门控通道变换模块

        参数:
            in_channels: 输入通道数
            epsilon: 用于数值稳定性的小常数
            mode: 归一化模式 ('l2' 或 'softmax')
            after_relu: 是否在ReLU激活后应用
        """
        super(GCTModule, self).__init__()

        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        # 全局上下文建模
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # 归一化和激活
        self.register_buffer('running_mean', torch.zeros(1, in_channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, in_channels, 1, 1))

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为 [B, C, H, W]

        返回:
            变换后的特征，形状相同
        """
        if self.training:
            # 训练模式：计算当前批次的均值和方差
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            var = ((x - mean) ** 2).mean(dim=[0, 2, 3], keepdim=True)

            # 更新running统计量
            self.running_mean = self.running_mean * 0.9 + mean * 0.1
            self.running_var = self.running_var * 0.9 + var * 0.1
        else:
            # 评估模式：使用running统计量
            mean = self.running_mean
            var = self.running_var

        # 标准化
        x_normalized = (x - mean) / (var + self.epsilon).sqrt()

        # 通过alpha参数调节标准化特征
        gated_x = x_normalized * self.alpha

        # 应用门控变换
        if self.mode == 'l2':
            # L2范数门控
            gate = (gated_x ** 2).mean(dim=[2, 3], keepdim=True)
            gate = 1. + torch.tanh(self.gamma * gate + self.beta)
        else:
            # Softmax门控
            gate = self.gamma * gated_x + self.beta
            gate = gate.mean(dim=[2, 3], keepdim=True)
            gate = F.softmax(gate, dim=1) * gate.shape[1]

        return x * gate


# 请确保导入必要的库
import math

if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机输入
    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 测试通道注意力模块
    ca = ChannelAttention(channels).to(device)
    output_ca = ca(x)
    print(f"通道注意力输出形状: {output_ca.shape}")

    # 测试高效通道注意力模块
    eca = ECABlock(channels).to(device)
    output_eca = eca(x)
    print(f"高效通道注意力输出形状: {output_eca.shape}")

    # 测试CBAM通道注意力模块
    cbam = CBAM_Channel(channels).to(device)
    output_cbam = cbam(x)
    print(f"CBAM通道注意力输出形状: {output_cbam.shape}")

    # 测试门控通道变换模块
    gct = GCTModule(channels).to(device)
    output_gct = gct(x)
    print(f"门控通道变换输出形状: {output_gct.shape}")