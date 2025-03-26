# src/train/train_baseline_subset.py
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parents[2]))

# 导入模型和工具
from src.models.unet import UNet
from src.utils.losses import CombinedLoss
from src.utils.metrics import dice_coefficient, evaluate_model
from dataloader import LiTSDataset, get_transforms
from src.models.unet import DoubleConv, Down, Up, OutConv


def create_dataset_subset(data_dir, subset_fraction=0.2, seed=42):
    """
    创建数据子集

    参数:
        data_dir: 数据目录
        subset_fraction: 子集比例
        seed: 随机种子

    返回:
        子集文件路径
    """
    np.random.seed(seed)

    # 设置路径
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    subset_dir = splits_dir / "subset"
    os.makedirs(subset_dir, exist_ok=True)

    # 训练集子集
    train_slice_file = splits_dir / "train_slices.txt"
    if train_slice_file.exists():
        with open(train_slice_file, 'r') as f:
            train_slices = [line.strip() for line in f if line.strip()]

        # 确保数据的多样性，并保留一些带有小型肿瘤的样本
        train_subset_size = int(len(train_slices) * subset_fraction)
        train_subset = list(np.random.choice(train_slices, train_subset_size, replace=False))

        # 保存子集
        train_subset_file = subset_dir / "train_slices.txt"
        with open(train_subset_file, 'w') as f:
            f.write('\n'.join(train_subset))
        print(f"创建训练集子集: {len(train_subset)}/{len(train_slices)} 切片 ({subset_fraction * 100:.1f}%)")
    else:
        print(f"错误: 找不到训练集文件 {train_slice_file}")
        return None

    # 验证集子集
    val_slice_file = splits_dir / "val_slices.txt"
    if val_slice_file.exists():
        with open(val_slice_file, 'r') as f:
            val_slices = [line.strip() for line in f if line.strip()]

        val_subset_size = int(len(val_slices) * subset_fraction)
        val_subset = list(np.random.choice(val_slices, val_subset_size, replace=False))

        # 保存子集
        val_subset_file = subset_dir / "val_slices.txt"
        with open(val_subset_file, 'w') as f:
            f.write('\n'.join(val_subset))
        print(f"创建验证集子集: {len(val_subset)}/{len(val_slices)} 切片 ({subset_fraction * 100:.1f}%)")
    else:
        print(f"错误: 找不到验证集文件 {val_slice_file}")
        return None

    return {
        'train': train_subset_file,
        'val': val_subset_file
    }


def get_subset_dataloaders(data_dir, subset_files, batch_size=8, num_workers=2, img_size=(256, 256)):
    """
    获取数据子集的加载器

    参数:
        data_dir: 数据目录
        subset_files: 子集文件字典
        batch_size: 批次大小
        num_workers: 工作线程数
        img_size: 图像大小

    返回:
        训练和验证数据加载器
    """
    # 创建数据加载器
    train_transform = get_transforms('train', img_size)
    train_dataset = LiTSDataset(
        data_dir=data_dir,
        slice_list_file=subset_files['train'],
        transform=train_transform,
        phase="train"
    )

    val_transform = get_transforms('val', img_size)
    val_dataset = LiTSDataset(
        data_dir=data_dir,
        slice_list_file=subset_files['val'],
        transform=val_transform,
        phase="val"
    )

    print(f"训练集子集大小: {len(train_dataset)} 切片")
    print(f"验证集子集大小: {len(val_dataset)} 切片")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, scheduler=None):
    """
    训练一个epoch，支持混合精度训练

    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        scaler: 混合精度训练的梯度缩放器
        scheduler: 学习率调度器

    返回:
        平均训练损失和度量指标
    """
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    batch_count = 0

    with tqdm(train_loader, desc="Training", leave=False) as pbar:
        for batch in pbar:
            # 获取数据
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # 确保掩码是long类型
            if masks.dtype != torch.long:
                masks = masks.long()

            # 处理weight_map (如果存在)
            weight_map = batch.get('weight_map')
            if weight_map is not None:
                weight_map = weight_map.to(device)

            # 前向传播 - 使用混合精度
            optimizer.zero_grad()

            if scaler is not None:  # 使用混合精度训练
                with autocast():
                    outputs = model(images)
                    if weight_map is not None:
                        loss = criterion(outputs, masks, weight_map)
                    else:
                        loss = criterion(outputs, masks)

                # 反向传播和优化 - 使用scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # 不使用混合精度
                outputs = model(images)
                if weight_map is not None:
                    loss = criterion(outputs, masks, weight_map)
                else:
                    loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

            # 计算Dice系数
            with torch.no_grad():
                if outputs.size(1) > 1:  # 多类别
                    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                else:  # 二分类
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                dice = dice_coefficient(preds, masks)

            # 更新进度条
            epoch_loss += loss.item()
            epoch_dice += dice.item()
            batch_count += 1
            pbar.set_postfix(loss=loss.item(), dice=dice.item())

    # 学习率调度器步进
    if scheduler is not None:
        scheduler.step()

    # 返回平均损失和Dice
    return epoch_loss / batch_count, epoch_dice / batch_count


def validate(model, val_loader, criterion, device):
    """
    在验证集上评估模型

    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备

    返回:
        验证损失和度量指标
    """
    model.eval()
    val_loss = 0
    val_dice = 0
    batch_count = 0

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", leave=False) as pbar:
            for batch in pbar:
                # 获取数据
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                # 确保掩码是long类型
                if masks.dtype != torch.long:
                    masks = masks.long()

                # 处理weight_map (如果存在)
                weight_map = batch.get('weight_map')
                if weight_map is not None:
                    weight_map = weight_map.to(device)

                # 前向传播
                outputs = model(images)

                # 计算损失
                if weight_map is not None:
                    loss = criterion(outputs, masks, weight_map)
                else:
                    loss = criterion(outputs, masks)

                # 计算Dice系数
                if outputs.size(1) > 1:  # 多类别
                    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                else:  # 二分类
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                dice = dice_coefficient(preds, masks)

                # 更新统计
                val_loss += loss.item()
                val_dice += dice.item()
                batch_count += 1
                pbar.set_postfix(loss=loss.item(), dice=dice.item())

    # 返回平均损失和Dice
    return val_loss / batch_count, val_dice / batch_count


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=30, device='cuda', save_dir='results/models/baseline_subset',
                patience=10, scheduler=None, use_amp=True):
    """
    训练模型

    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练epoch数
        device: 计算设备
        save_dir: 保存目录
        patience: 早停耐心值
        scheduler: 学习率调度器
        use_amp: 是否使用混合精度训练

    返回:
        训练历史
    """
    # 确保保存目录存在
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化混合精度训练的scaler
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

    # 训练历史记录
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    }

    # 最佳模型追踪
    best_val_dice = 0.0
    no_improvement = 0

    print(f"开始训练，总共 {num_epochs} 个epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 训练一个epoch
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, scheduler
        )

        # 验证
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # 打印epoch统计
        print(f"Epoch {epoch + 1}/{num_epochs} - {epoch_time:.1f}s - "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, "
              f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"保存新的最佳模型，验证Dice: {best_val_dice:.4f}")
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"连续{patience}个epoch没有改善，提前停止训练")
                break

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')

    total_time = time.time() - start_time
    print(f"训练完成，总耗时: {total_time / 60:.1f} 分钟")

    # 保存最终模型
    torch.save(model.state_dict(), save_dir / 'final_model.pth')

    # 保存训练历史
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f)

    # 绘制训练历史
    plot_training_history(history, save_dir / 'training_history.png')

    return history


def plot_training_history(history, save_path=None):
    """
    绘制训练历史

    参数:
        history: 训练历史数据
        save_path: 保存路径
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制损失
    ax1.plot(history['train_loss'], label='训练损失')
    ax1.plot(history['val_loss'], label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)

    # 绘制Dice系数
    ax2.plot(history['train_dice'], label='训练Dice')
    ax2.plot(history['val_dice'], label='验证Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice系数')
    ax2.set_title('训练和验证Dice系数')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"训练历史图已保存到 {save_path}")

    plt.close()


# 创建一个更小的UNet模型
class LightUNet(nn.Module):
    """轻量级U-Net架构"""

    def __init__(self, n_channels=3, n_classes=3, init_features=16):
        super(LightUNet, self).__init__()

        # 初始特征数
        features = init_features

        # 编码器路径
        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)

        # 瓶颈层
        factor = 2 if True else 1  # 如果使用双线性上采样，则需要减半特征数
        self.bridge = DoubleConv(features * 8, features * 16 // factor)

        # 解码器路径 - 确保通道数匹配
        self.up1 = Up(features * 16 // factor, features * 8 // factor, bilinear=True)
        self.up2 = Up(features * 8 // factor, features * 4 // factor, bilinear=True)
        self.up3 = Up(features * 4 // factor, features * 2 // factor, bilinear=True)
        self.up4 = Up(features * 2 // factor, features, bilinear=True)

        # 输出层
        self.outc = OutConv(features, n_classes)

    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 瓶颈层
        x5 = self.bridge(x4)

        # 解码器路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出层
        logits = self.outc(x)
        return logits


def main(config=None):
    """
    主函数

    参数:
        config: 配置字典
    """
    if config is None:
        config = {
            # 数据参数
            'data_dir': 'D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\data\\preprocessed',
            'subset_fraction': 0.1,  # 使用10%的数据
            'batch_size': 8,  # 减小批次大小
            'num_workers': 2,  # 减少数据加载线程
            'img_size': (256, 256),  # 降低图像分辨率

            # 模型参数
            'n_channels': 3,
            'n_classes': 3,
            'init_features': 16,  # 减少特征图数量
            'use_light_model': True,  # 使用轻量级模型

            # 训练参数
            'num_epochs': 30,  # 减少训练轮数
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'patience': 7,
            'use_amp': True,  # 使用混合精度训练

            # 损失参数
            'dice_weight': 1.0,
            'focal_weight': 1.0,
            'boundary_weight': 0.5,
            'size_weight': 1.0,

            # 保存参数
            'save_dir': 'results/models/baseline_subset',
            'experiment_name': datetime.now().strftime('%Y%m%d_%H%M%S')
        }

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置保存目录
    save_dir = Path(config['save_dir']) / config['experiment_name']
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 创建数据子集
    subset_files = create_dataset_subset(
        data_dir=config['data_dir'],
        subset_fraction=config['subset_fraction']
    )

    # 获取数据加载器
    train_loader, val_loader = get_subset_dataloaders(
        data_dir=config['data_dir'],
        subset_files=subset_files,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size']
    )

    # 创建模型
    if config['use_light_model']:
        model = LightUNet(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            init_features=config['init_features']
        ).to(device)
    else:
        model = UNet(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            init_features=config['init_features']
        ).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params:,}")

    # 定义损失函数
    criterion = CombinedLoss(
        dice_weight=config['dice_weight'],
        focal_weight=config['focal_weight'],
        boundary_weight=config['boundary_weight'],
        size_weight=config['size_weight']
    )

    # 定义优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # 训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device,
        save_dir=save_dir,
        patience=config['patience'],
        scheduler=scheduler,
        use_amp=config['use_amp']
    )

    print("\n训练完成!")
    print(f"最终训练Dice: {history['train_dice'][-1]:.4f}")
    print(f"最终验证Dice: {history['val_dice'][-1]:.4f}")
    print(f"最佳验证Dice: {max(history['val_dice']):.4f}")

    print(f"\n结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()