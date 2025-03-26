# train_enhanced_model.py
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
import argparse

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parents[2]))

# 导入模型和工具
from src.models.enhanced_attention_unet import get_enhanced_model, RecallFocusedLoss, DeepSupervisionLoss
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

            # 计算Dice系数 (对于主输出，如果是多输出模型)
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs

            if main_output.size(1) > 1:  # 多类别
                preds = torch.argmax(torch.softmax(main_output, dim=1), dim=1)
            else:  # 二分类
                preds = (torch.sigmoid(main_output) > 0.5).float()

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

                # 计算Dice系数 (对于主输出，如果是多输出模型)
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs

                if main_output.size(1) > 1:  # 多类别
                    preds = torch.argmax(torch.softmax(main_output, dim=1), dim=1)
                else:  # 二分类
                    preds = (torch.sigmoid(main_output) > 0.5).float()

                dice = dice_coefficient(preds, masks)

                # 更新统计
                val_loss += loss.item()
                val_dice += dice.item()
                batch_count += 1
                pbar.set_postfix(loss=loss.item(), dice=dice.item())

    # 返回平均损失和Dice
    return val_loss / batch_count, val_dice / batch_count


def detailed_evaluation(model, val_loader, criterion, device, n_classes=3, small_tumor_class_index=2):
    """
    详细评估模型性能，特别关注小型肿瘤

    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        n_classes: 类别数量
        small_tumor_class_index: 肿瘤类别索引

    返回:
        详细的评估指标
    """
    print("\n开始详细评估...")

    # 确保模型处于评估模式
    model.eval()

    # 使用evaluate_model函数进行详细评估
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        n_classes=n_classes,
        tumor_class=small_tumor_class_index
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

    # 特别关注小型肿瘤的性能
    if 'small' in size_metrics:
        print("\n小型肿瘤(<10mm)检测性能总结:")
        small_recall = size_metrics['small']['recall']
        small_precision = size_metrics['small']['precision']
        small_f1 = size_metrics['small']['f1']
        print(f"  召回率: {small_recall:.4f}")
        print(f"  精确率: {small_precision:.4f}")
        print(f"  F1分数: {small_f1:.4f}")

    return metrics


def visualize_model_predictions(model, val_loader, device, save_dir, num_samples=4):
    """
    可视化模型预测结果

    参数:
        model: 模型
        val_loader: 验证数据加载器
        device: 计算设备
        save_dir: 保存目录
        num_samples: 可视化样本数量
    """
    model.eval()

    # 寻找包含小型肿瘤的样本
    samples_with_tumor = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image']
            masks = batch['mask']
            weight_maps = batch.get('weight_map')

            for i in range(len(images)):
                if torch.any(masks[i] == 2) and weight_maps is not None and torch.any(weight_maps[i] > 0.8):
                    # 找到包含小型肿瘤的样本
                    samples_with_tumor.append({
                        'image': images[i:i + 1],
                        'mask': masks[i:i + 1],
                        'weight_map': weight_maps[i:i + 1] if weight_maps is not None else None
                    })

                    if len(samples_with_tumor) >= num_samples:
                        break

            if len(samples_with_tumor) >= num_samples:
                break

    # 如果没有找到足够的小型肿瘤样本，使用普通样本
    if len(samples_with_tumor) < num_samples:
        for batch in val_loader:
            images = batch['image']
            masks = batch['mask']
            weight_maps = batch.get('weight_map')

            for i in range(len(images)):
                if torch.any(masks[i] == 2):  # 包含肿瘤的样本
                    samples_with_tumor.append({
                        'image': images[i:i + 1],
                        'mask': masks[i:i + 1],
                        'weight_map': weight_maps[i:i + 1] if weight_maps is not None else None
                    })

                    if len(samples_with_tumor) >= num_samples:
                        break

            if len(samples_with_tumor) >= num_samples:
                break

    # 可视化每个样本的预测结果
    for i, sample in enumerate(samples_with_tumor):
        image = sample['image'].to(device)
        mask = sample['mask'].cpu()
        weight_map = sample['weight_map'].cpu() if sample['weight_map'] is not None else None

        # 模型预测
        output = model(image)
        if isinstance(output, tuple):
            output = output[0]  # 使用主输出

        pred = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu()

        # 创建可视化图
        fig, axes = plt.subplots(1, 4 if weight_map is not None else 3, figsize=(16, 4))

        # 显示原始图像
        axes[0].imshow(image[0].cpu().permute(1, 2, 0))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 显示真实掩码
        axes[1].imshow(mask[0], cmap='viridis', vmin=0, vmax=2)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        # 显示预测掩码
        axes[2].imshow(pred[0], cmap='viridis', vmin=0, vmax=2)
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')

        # 显示权重图 (如果有)
        if weight_map is not None:
            axes[3].imshow(weight_map[0], cmap='hot')
            axes[3].set_title('Weight Map (Small Tumor)')
            axes[3].axis('off')

        plt.tight_layout()

        # 保存可视化图
        save_path = Path(save_dir) / f'prediction_sample_{i + 1}.png'
        plt.savefig(save_path)
        plt.close()

    print(f"可视化结果已保存到 {save_dir}")


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


def train_enhanced_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=100,
        device='cuda',
        save_dir='results/models/enhanced',
        model_name='enhanced_model',
        patience=10,
        scheduler=None
):
    """
    训练增强型注意力模型

    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练epoch数
        device: 计算设备
        save_dir: 保存目录
        model_name: 模型名称
        patience: 早停耐心值
        scheduler: 学习率调度器(可选)

    返回:
        训练历史和详细指标
    """
    # 确保保存目录存在
    save_dir = Path(save_dir) / model_name
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

    print(f"开始训练 {model_name}，总共 {num_epochs} 个epochs...")
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
    visualize_model_predictions(
        model, val_loader, device,
        save_dir=save_dir,
        num_samples=4
    )

    return history, detailed_metrics


def main(args):
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置保存目录
    results_dir = Path(args.save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 实验时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{args.model_type}_{timestamp}"

    # 保存配置
    config = vars(args)
    with open(results_dir / f'config_{timestamp}.json', 'w') as f:
        json.dump(config, f, indent=2)

    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(map(int, args.img_size.split(',')))
    )

    print(f"训练集大小: {len(train_loader.dataset)} 切片")
    print(f"验证集大小: {len(val_loader.dataset)} 切片")

    # 创建模型
    model = get_enhanced_model(
        model_type=args.model_type,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        init_features=args.init_features,
        small_tumor_focus=True if args.model_type == 'enhanced_attention' else None,
        use_dynamic_attention=args.use_dynamic_attention if args.model_type == 'enhanced_attention' else None,
        use_deep_supervision=args.use_deep_supervision if args.model_type == 'balanced_recall' else None
    ).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params:,}")

    # 定义损失函数
    if args.loss_type == 'recall_focused':
        main_loss = RecallFocusedLoss(
            recall_weight=args.recall_weight,
            size_weight=args.size_weight
        )
    else:
        main_loss = CombinedLoss(
            dice_weight=args.dice_weight,
            focal_weight=args.focal_weight,
            boundary_weight=args.boundary_weight,
            size_weight=args.size_weight
        )

    # 如果使用深度监督，包装主损失函数
    if args.model_type == 'balanced_recall' and args.use_deep_supervision:
        criterion = DeepSupervisionLoss(
            main_loss=main_loss,
            aux_weights=[0.4, 0.3, 0.2]
        )
    else:
        criterion = main_loss

    # 定义优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # 训练模型
    train_enhanced_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=results_dir,
        model_name=model_name,
        patience=args.patience,
        scheduler=scheduler
    )

    print(f"训练完成! 结果已保存至: {results_dir / model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练增强型注意力模型')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\data\\preprocessed',
                        help='预处理数据目录')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    parser.add_argument('--img_size', type=str, default='256,256',
                        help='图像大小，格式为"高,宽"')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='enhanced_attention',
                        choices=['enhanced_attention', 'balanced_recall'],
                        help='使用的模型类型')
    parser.add_argument('--n_channels', type=int, default=3,
                        help='输入通道数')
    parser.add_argument('--n_classes', type=int, default=3,
                        help='输出类别数')
    parser.add_argument('--init_features', type=int, default=64,
                        help='初始特征数量')
    parser.add_argument('--use_dynamic_attention', action='store_true',
                        help='使用动态注意力机制 (仅对enhanced_attention有效)')
    parser.add_argument('--use_deep_supervision', action='store_true',
                        help='使用深度监督 (仅对balanced_recall有效)')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练epochs数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')

    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='recall_focused',
                        choices=['combined', 'recall_focused'],
                        help='损失函数类型')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Dice损失权重 (combined损失)')
    parser.add_argument('--focal_weight', type=float, default=1.0,
                        help='Focal损失权重 (combined损失)')
    parser.add_argument('--boundary_weight', type=float, default=0.5,
                        help='边界损失权重 (combined损失)')
    parser.add_argument('--size_weight', type=float, default=2.0,
                        help='小型肿瘤大小加权系数')
    parser.add_argument('--recall_weight', type=float, default=2.0,
                        help='召回率权重 (recall_focused损失)')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\LiTS_SmallTumor\\results\\models\\enhanced',
                        help='保存目录')

    args = parser.parse_args()

    main(args)