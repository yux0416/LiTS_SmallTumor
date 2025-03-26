# src/train/train_attention.py
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
from src.models.unet import UNet
from src.models.attention_unet import get_attention_unet_model
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


def visualize_attention_maps(model, val_loader, device, save_path=None):
    """
    可视化注意力图

    参数:
        model: 模型
        val_loader: 验证数据加载器
        device: 计算设备
        save_path: 保存路径
    """
    # 检查模型是否有注意力模块
    has_attention = hasattr(model, 'attentions') or hasattr(model, 'skip_attentions')

    if not has_attention:
        print("该模型没有可视化的注意力模块")
        return

    model.eval()

    # 获取一个包含肿瘤的批次
    batch = None
    for b in val_loader:
        if torch.any(b['mask'] == 2):  # 假设2是肿瘤类别
            batch = b
            break

    if batch is None:
        print("在验证集中未找到包含肿瘤的样本")
        return

    # 选择一个包含肿瘤的样本
    has_tumor = torch.any(batch['mask'] == 2, dim=(1, 2))
    tumor_indices = torch.where(has_tumor)[0]

    if len(tumor_indices) == 0:
        print("在批次中未找到包含肿瘤的样本")
        return

    sample_idx = tumor_indices[0].item()
    image = batch['image'][sample_idx:sample_idx + 1].to(device)
    mask = batch['mask'][sample_idx].cpu().numpy()

    # 获取注意力图 - 需要修改模型以保存注意力图
    # 这里我们假设模型有一个hook机制来捕获注意力图
    # 实际实现可能需要根据具体模型架构进行调整

    # 简单的替代方案：使用GradCAM等技术可视化模型的注意区域
    # 这里我们只展示输入图像、真实标签和预测结果

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)[0].cpu().numpy()

        # 创建图
        plt.figure(figsize=(18, 6))

        # 显示原始图像
        plt.subplot(131)
        img = image[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        plt.imshow(img)
        plt.title('原始图像')
        plt.axis('off')

        # 显示真实掩码
        plt.subplot(132)
        plt.imshow(mask, cmap='viridis', vmin=0, vmax=2)
        plt.title('真实掩码 (0=背景, 1=肝脏, 2=肿瘤)')
        plt.axis('off')

        # 显示预测掩码
        plt.subplot(133)
        plt.imshow(pred, cmap='viridis', vmin=0, vmax=2)
        plt.title('预测掩码 (0=背景, 1=肝脏, 2=肿瘤)')
        plt.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"注意力可视化已保存到 {save_path}")

        plt.show()

        return
def visualize_comparison(models_dict, val_loader, device, save_path=None):
        """
        对比不同模型在小型肿瘤上的表现
        参数:
            models_dict: 模型字典 {名称: 模型}
            val_loader: 验证数据加载器
            device: 计算设备
            save_path: 保存路径
        """
        # 寻找包含小型肿瘤的样本
        batch = None
        for b in val_loader:
            # 检查是否有weight_map，并查找包含小型肿瘤的样本
            if 'weight_map' in b and torch.any(b['mask'] == 2):
                # 查找weight_map值较高的区域（对应小型肿瘤）
                high_weight_samples = torch.any(b['weight_map'] > 0.8, dim=(1, 2))
                if torch.any(high_weight_samples):
                    batch = b
                    break

        if batch is None:
            print("在验证集中未找到包含小型肿瘤的样本")
            return

        # 选择一个包含小型肿瘤的样本
        high_weight_indices = torch.where(torch.any(batch['weight_map'] > 0.8, dim=(1, 2)))[0]
        sample_idx = high_weight_indices[0].item()

        image = batch['image'][sample_idx:sample_idx + 1].to(device)
        mask = batch['mask'][sample_idx].cpu().numpy()
        weight_map = batch['weight_map'][sample_idx].cpu().numpy()

        # 创建图
        plt.figure(figsize=(len(models_dict) * 6 + 6, 12))

        # 显示原始图像
        plt.subplot(2, len(models_dict) + 1, 1)
        img = image[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        plt.imshow(img)
        plt.title('原始图像')
        plt.axis('off')

        # 显示真实掩码
        plt.subplot(2, len(models_dict) + 1, 2)
        plt.imshow(mask, cmap='viridis', vmin=0, vmax=2)
        plt.title('真实掩码')
        plt.axis('off')

        # 显示权重图
        plt.subplot(2, len(models_dict) + 1, 3)
        plt.imshow(weight_map, cmap='hot')
        plt.title('小型肿瘤权重图')
        plt.axis('off')

        # 显示各模型预测结果
        col = 3
        for i, (name, model) in enumerate(models_dict.items()):
            col += 1
            model.eval()

            with torch.no_grad():
                output = model(image)
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1)[0].cpu().numpy()

            plt.subplot(2, len(models_dict) + 1, col)
            plt.imshow(pred, cmap='viridis', vmin=0, vmax=2)
            plt.title(f'{name}预测')
            plt.axis('off')

            # 显示肿瘤区域的放大视图
            if col <= len(models_dict) + 3:
                plt.subplot(2, len(models_dict) + 1, col + len(models_dict) + 1)

                # 找到肿瘤区域
                tumor_y, tumor_x = np.where(mask == 2)
                if len(tumor_y) > 0 and len(tumor_x) > 0:
                    # 计算肿瘤中心
                    center_y = int(np.mean(tumor_y))
                    center_x = int(np.mean(tumor_x))

                    # 提取肿瘤周围区域
                    size = 64  # 放大区域大小
                    y_start = max(0, center_y - size // 2)
                    y_end = min(mask.shape[0], center_y + size // 2)
                    x_start = max(0, center_x - size // 2)
                    x_end = min(mask.shape[1], center_x + size // 2)

                    # 显示放大的预测区域
                    plt.imshow(pred[y_start:y_end, x_start:x_end], cmap='viridis', vmin=0, vmax=2)
                    plt.title(f'{name}肿瘤区域放大')
                    plt.axis('off')

                    # 添加真实轮廓
                    tumor_mask = mask[y_start:y_end, x_start:x_end] == 2
                    if np.any(tumor_mask):
                        plt.contour(tumor_mask, colors='r', linewidths=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"模型比较可视化已保存到 {save_path}")

        plt.show()

def train_attention_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=100,
            device='cuda',
            save_dir='results/models/attention',
            model_name='attention_unet',
            patience=10,
            scheduler=None
    ):
        """
        训练注意力增强模型

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

        # 可视化注意力图
        visualize_attention_maps(
            model, val_loader, device,
            save_path=save_dir / 'attention_visualization.png'
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
        ax1.set_ylabel('Loss')
        ax1.set_title('Train and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # 绘制Dice系数
        ax2.plot(history['train_dice'], label='train Dice')
        ax2.plot(history['val_dice'], label='val Dice')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Coefficient')
        ax2.set_title('Train and Validation Dice Coefficients')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"训练历史图已保存到 {save_path}")

        plt.show()

def compare_models(model_results, save_path=None):
        """
        比较不同模型的性能

        参数:
            model_results: 模型结果字典 {模型名称: {指标名称: 指标值}}
            save_path: 保存路径
        """
        # 提取关键指标
        models = list(model_results.keys())
        overall_dice = [results['dice'] for results in model_results.values()]
        liver_dice = [results['class_dice'][1] for results in model_results.values()]
        tumor_dice = [results['class_dice'][2] for results in model_results.values()]

        # 提取小型肿瘤性能 (如果有)
        small_tumor_f1 = []
        for results in model_results.values():
            if 'size_metrics' in results and 'small' in results['size_metrics']:
                small_tumor_f1.append(results['size_metrics']['small']['f1'])
            else:
                small_tumor_f1.append(0)

        # 创建柱状图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 整体性能比较
        bar_width = 0.25
        index = np.arange(len(models))

        ax1.bar(index - bar_width, overall_dice, bar_width, label='Overall Dice')
        ax1.bar(index, liver_dice, bar_width, label='Liver Dice')
        ax1.bar(index + bar_width, tumor_dice, bar_width, label='Tumor Dice')

        ax1.set_xlabel('Model')
        ax1.set_ylabel('Dice Coefficient')
        ax1.set_title('Segmentation Performance Comparison')
        ax1.set_xticks(index)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, axis='y')

        # 小型肿瘤性能比较
        ax2.bar(index, small_tumor_f1, 0.5, label='Small Tumor F1')

        ax2.set_xlabel('Model')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Small Tumor (<10mm) Detection Performance')
        ax2.set_xticks(index)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"模型比较图已保存到 {save_path}")

        plt.show()

def main(args):
        """
        主函数
        """
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 设置保存目录
        results_dir = Path("D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\results\\models\\attention")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 实验时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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

        # 定义损失函数
        criterion = CombinedLoss(
            dice_weight=args.dice_weight,
            focal_weight=args.focal_weight,
            boundary_weight=args.boundary_weight,
            size_weight=args.size_weight
        )

        # 训练指定的模型
        if args.train_all:
            # 训练所有模型类型
            model_types = ['standard', 'attention', 'deep_attention', 'hierarchical', 'small_tumor']
            model_results = {}
            trained_models = {}

            for model_type in model_types:
                model_name = f"{model_type}_{timestamp}"
                print(f"\n开始训练 {model_type} 模型...")

                # 创建模型
                model = get_attention_unet_model(
                    model_type=model_type,
                    n_channels=args.n_channels,
                    n_classes=args.n_classes,
                    init_features=args.init_features,
                    attention_type=args.attention_type if model_type != 'standard' else None
                ).to(device)

                # 打印模型信息
                total_params = sum(p.numel() for p in model.parameters())
                print(f"模型总参数数量: {total_params:,}")

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
                _, metrics = train_attention_model(
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

                # 保存结果和模型
                model_results[model_type] = metrics
                trained_models[model_type] = model

            # 比较所有模型
            compare_models(model_results, save_path=results_dir / f'model_comparison_{timestamp}.png')

            # 可视化不同模型的输出比较
            visualize_comparison(
                trained_models, val_loader, device,
                save_path=results_dir / f'model_outputs_comparison_{timestamp}.png'
            )

        else:
            # 训练单个模型
            model_name = f"{args.model_type}_{args.attention_type}_{timestamp}"

            # 创建模型
            model = get_attention_unet_model(
                model_type=args.model_type,
                n_channels=args.n_channels,
                n_classes=args.n_classes,
                init_features=args.init_features,
                attention_type=args.attention_type if args.model_type != 'standard' else None
            ).to(device)

            # 打印模型信息
            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型总参数数量: {total_params:,}")

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
            train_attention_model(
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

        print(f"\n训练完成! 结果已保存至: {results_dir}")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='训练注意力增强U-Net模型')

        # 数据参数
        parser.add_argument('--data_dir', type=str, default='D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\data\\preprocessed',
                            help='预处理数据目录')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='批次大小')
        parser.add_argument('--num_workers', type=int, default=2,
                            help='数据加载器工作进程数')
        parser.add_argument('--img_size', type=str, default='256,256',
                            help='图像大小，格式为"高,宽"')

        # 模型参数
        parser.add_argument('--model_type', type=str, default='attention',
                            choices=['standard', 'attention', 'deep_attention', 'hierarchical', 'small_tumor',
                                     'multi_scale'],
                            help='使用的模型类型')
        parser.add_argument('--attention_type', type=str, default='cbam',
                            choices=['channel', 'spatial', 'cbam', 'small', 'scale',
                                     'dual', 'tumor_size', 'gct', 'eca',
                                     'multi_scale_enhanced', 'local_contrast',
                                     'tuned_local_contrast', 'hybrid_local_small',
                                     'integrated_small_tumor',
                                     'scale_aware_small', 'enhanced_tumor_size'],
                            help='使用的注意力类型')
        parser.add_argument('--n_channels', type=int, default=3,
                            help='输入通道数')
        parser.add_argument('--n_classes', type=int, default=3,
                            help='输出类别数')
        parser.add_argument('--init_features', type=int, default=32,
                            help='初始特征数量')

        # 训练参数
        parser.add_argument('--num_epochs', type=int, default=30,
                            help='训练epochs数')
        parser.add_argument('--learning_rate', type=float, default=1e-4,
                            help='初始学习率')
        parser.add_argument('--weight_decay', type=float, default=1e-5,
                            help='权重衰减')
        parser.add_argument('--patience', type=int, default=10,
                            help='早停耐心值')

        # 损失函数参数
        parser.add_argument('--dice_weight', type=float, default=1.0,
                            help='Dice损失权重')
        parser.add_argument('--focal_weight', type=float, default=1.0,
                            help='Focal损失权重')
        parser.add_argument('--boundary_weight', type=float, default=0.5,
                            help='边界损失权重')
        parser.add_argument('--size_weight', type=float, default=2.0,
                            help='小型肿瘤大小加权损失权重')

        # 实验选项
        parser.add_argument('--train_all', action='store_true',
                            help='训练所有模型类型并比较')

        args = parser.parse_args()

        main(args)