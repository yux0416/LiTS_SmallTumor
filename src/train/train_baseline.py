# src/train/train_baseline.py
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
import matplotlib.pyplot as plt
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parents[2]))

# 导入模型和工具
from src.models.unet import UNet
from src.utils.losses import CombinedLoss
from src.utils.metrics import dice_coefficient, evaluate_model
from dataloader import LiTSDataset, get_transforms, get_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """
    训练一个epoch

    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        scheduler: 学习率调度器(可选)

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
            masks = masks.long()  # 确保掩码是LongTensor类型

            # 确保掩码是long类型
            if masks.dtype != torch.long:
                masks = masks.long()

            # 处理weight_map (如果存在)
            weight_map = batch.get('weight_map')
            if weight_map is not None:
                weight_map = weight_map.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)

            # 计算损失
            if weight_map is not None:
                loss = criterion(outputs, masks, weight_map)
            else:
                loss = criterion(outputs, masks)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 计算Dice系数
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
                masks = masks.long()  # 确保掩码是LongTensor类型

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


def detailed_evaluation(model, val_loader, criterion, device, n_classes=3):
    """
    详细评估模型性能

    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        n_classes: 类别数量

    返回:
        详细的评估指标
    """
    print("\n开始详细评估...")
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        n_classes=n_classes,
        tumor_class=2  # 假设肿瘤类别索引为2
    )

    # 打印评估结果
    print(f"\n评估结果:")
    print(f"平均损失: {metrics['loss']:.4f}")
    print(f"平均Dice: {metrics['dice']:.4f}")

    print("\n各类别Dice系数:")
    for cls, dice in metrics['class_dice'].items():
        class_name = "背景" if cls == 0 else "肝脏" if cls == 1 else "肿瘤"
        print(f"  类别 {cls} ({class_name}): {dice:.4f}")

    print("\n肿瘤检测指标:")
    tumor_metrics = metrics['tumor_metrics']
    print(f"  召回率: {tumor_metrics['recall']:.4f}")
    print(f"  精确率: {tumor_metrics['precision']:.4f}")
    print(f"  F1分数: {tumor_metrics['f1']:.4f}")

    print("\n按肿瘤大小的性能:")
    size_metrics = metrics['size_metrics']
    for size, size_metric in size_metrics.items():
        print(f"  {size}型肿瘤:")
        print(f"    召回率: {size_metric['recall']:.4f}")
        print(f"    精确率: {size_metric['precision']:.4f}")
        print(f"    F1分数: {size_metric['f1']:.4f}")

    return metrics


def visualize_results(model, val_loader, device, num_samples=4, save_path=None):
    """
    可视化模型预测结果

    参数:
        model: 模型
        val_loader: 验证数据加载器
        device: 计算设备
        num_samples: 样本数量
        save_path: 保存路径
    """
    model.eval()

    # 获取一批数据
    batch = next(iter(val_loader))
    images = batch['image'][:num_samples].to(device)
    masks = batch['mask'][:num_samples].cpu().numpy()

    # 获取模型预测
    with torch.no_grad():
        outputs = model(images)
        if outputs.size(1) > 1:  # 多类别
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().numpy()
        else:  # 二分类
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()

    # 转换图像格式
    images = images.cpu().numpy()

    # 绘制结果
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    for i in range(num_samples):
        # 显示原始图像
        ax = axes[i, 0] if num_samples > 1 else axes[0]
        img = images[i].transpose(1, 2, 0)  # CHW -> HWC
        ax.imshow(img)
        ax.set_title(f"原始图像 {i + 1}")
        ax.axis('off')

        # 显示真实掩码
        ax = axes[i, 1] if num_samples > 1 else axes[1]
        ax.imshow(masks[i], cmap='viridis', vmin=0, vmax=2)
        ax.set_title(f"真实掩码 (0=背景, 1=肝脏, 2=肿瘤)")
        ax.axis('off')

        # 显示预测掩码
        ax = axes[i, 2] if num_samples > 1 else axes[2]
        ax.imshow(preds[i], cmap='viridis', vmin=0, vmax=2)
        ax.set_title(f"预测掩码 (0=背景, 1=肝脏, 2=肿瘤)")
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"可视化结果已保存到 {save_path}")

    plt.show()


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=100, device='cuda', save_dir='D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\results\\models\\baseline',
                patience=10, scheduler=None):
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
        scheduler: 学习率调度器(可选)

    返回:
        训练历史和详细指标
    """
    # 确保保存目录存在
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 对于ReduceLROnPlateau，使用验证Dice调整学习率
        if scheduler is not None:
            scheduler.step(val_dice)

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

        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
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

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(save_dir / 'best_model.pth'))
    detailed_metrics = detailed_evaluation(model, val_loader, criterion, device)

    # 保存详细评估结果
    with open(save_dir / 'evaluation_metrics.json', 'w') as f:
        # 转换numpy数组为Python列表
        metrics_serializable = {}
        for k, v in detailed_metrics.items():
            if isinstance(v, dict):
                metrics_serializable[k] = {}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, np.ndarray):
                        metrics_serializable[k][sub_k] = sub_v.tolist()
                    else:
                        metrics_serializable[k][sub_k] = sub_v
            elif isinstance(v, np.ndarray):
                metrics_serializable[k] = v.tolist()
            else:
                metrics_serializable[k] = v

        json.dump(metrics_serializable, f, indent=2)

    # 可视化一些预测结果
    visualize_results(
        model, val_loader, device,
        num_samples=4,
        save_path=save_dir / 'prediction_samples.png'
    )

    return history, detailed_metrics

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
    ax1.plot(history['train_loss'], label='train loss')
    ax1.plot(history['val_loss'], label='val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('train and val loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制Dice系数
    ax2.plot(history['train_dice'], label='train Dice')
    ax2.plot(history['val_dice'], label='val Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice coefficient')
    ax2.set_title('train and val Dice coefficients')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"训练历史图已保存到 {save_path}")

    plt.show()


def main(config=None):
    """
    主函数

    参数:
        config: 配置字典(可选)
    """
    if config is None:
        config = {
            # 数据参数
            'data_dir': 'D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\data\\preprocessed',
            'batch_size': 4,
            'num_workers': 2,
            'img_size': (256, 256),

            # 模型参数
            'n_channels': 3,
            'n_classes': 3,
            'bilinear': False,
            'init_features': 32,

            # 训练参数
            'num_epochs': 30,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'patience': 5,

            # 损失参数
            'dice_weight': 1.0,
            'focal_weight': 1.0,
            'boundary_weight': 0.5,
            'size_weight': 1.0,

            # 保存参数
            'save_dir': 'D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\results\\models\\baseline',
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

    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size']
    )

    print(f"训练集大小: {len(train_loader.dataset)} 切片")
    print(f"验证集大小: {len(val_loader.dataset)} 切片")

    # 创建模型
    model = UNet(
        n_channels=config['n_channels'],
        n_classes=config['n_classes'],
        bilinear=config['bilinear'],
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
    history, metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device,
        save_dir=save_dir,
        patience=config['patience'],
        scheduler=scheduler
    )

    print("\n训练完成!")
    print(f"最终训练Dice: {history['train_dice'][-1]:.4f}")
    print(f"最终验证Dice: {history['val_dice'][-1]:.4f}")
    print(f"最佳验证Dice: {max(history['val_dice']):.4f}")

    print(f"\n结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()