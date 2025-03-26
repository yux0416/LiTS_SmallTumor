# dataloader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import random
import matplotlib.pyplot as plt


class LiTSDataset(Dataset):
    """LiTS Dataset Loader"""

    def __init__(self, data_dir, slice_list_file=None, case_list_file=None,
                 transform=None, phase="train", small_tumor_focus=True,
                 return_weight_map=True):
        """
        Initialize LiTS Dataset

        Parameters:
            data_dir: Preprocessed data directory
            slice_list_file: Slice list file (priority)
            case_list_file: Case list file
            transform: albumentations transformations
            phase: 'train' or 'val'
            small_tumor_focus: Whether to focus on small tumors
            return_weight_map: Whether to return weight map
        """
        self.data_dir = Path(data_dir)
        self.slices_dir = self.data_dir / "2d_slices"
        self.transform = transform
        self.phase = phase
        self.small_tumor_focus = small_tumor_focus
        self.return_weight_map = return_weight_map

        # Load slice list
        self.slice_files = []

        if slice_list_file and os.path.exists(slice_list_file):
            # Load slice list from file
            with open(slice_list_file, 'r') as f:
                self.slice_files = [line.strip() for line in f if line.strip()]
        elif case_list_file and os.path.exists(case_list_file):
            # Load all slices from case list
            with open(case_list_file, 'r') as f:
                case_ids = [line.strip() for line in f if line.strip()]

            # Find all matching slices
            for npz_file in self.slices_dir.glob("*.npz"):
                case_id = npz_file.stem.split('_')[0]
                if case_id in case_ids:
                    self.slice_files.append(npz_file.name)
        else:
            # Load all slices
            self.slice_files = [f.name for f in self.slices_dir.glob("*.npz")]

        # Read slice statistics
        stats_file = self.data_dir / "slice_stats.csv"
        if os.path.exists(stats_file):
            import pandas as pd
            self.slice_stats = pd.read_csv(stats_file)

            # Filter slices for current set
            filenames = [f for f in self.slice_files]
            self.slice_stats = self.slice_stats[self.slice_stats['filename'].isin(filenames)]

            # Calculate class weights
            self.compute_class_weights()
        else:
            self.slice_stats = None

        print(f"Loaded {len(self.slice_files)} slices for {phase} set")

    def compute_class_weights(self):
        """Calculate class weights to handle class imbalance"""
        if self.slice_stats is None:
            return

        # Count slices containing tumors
        tumor_slices = self.slice_stats[self.slice_stats['has_tumor'] == True]
        small_tumor_slices = self.slice_stats[self.slice_stats['has_small_tumor'] == True]

        total_slices = len(self.slice_stats)
        tumor_ratio = len(tumor_slices) / total_slices
        small_tumor_ratio = len(small_tumor_slices) / total_slices

        # Calculate weights for background, liver and tumor
        # Using inverse frequency weighting strategy
        bg_weight = 0.1  # Lower weight for background
        liver_weight = 0.3  # Medium weight for liver
        tumor_weight = min(1.0 / tumor_ratio if tumor_ratio > 0 else 5.0, 5.0)  # Higher weight for tumors
        small_tumor_weight = min(1.5 / small_tumor_ratio if small_tumor_ratio > 0 else 10.0, 10.0)  # Highest weight for small tumors

        # If not focusing on small tumors, use same weight for all tumors
        if not self.small_tumor_focus:
            small_tumor_weight = tumor_weight

        self.class_weights = {
            'background': bg_weight,
            'liver': liver_weight,
            'tumor': tumor_weight,
            'small_tumor': small_tumor_weight
        }

        print(
            f"Class weights: Background={bg_weight:.2f}, Liver={liver_weight:.2f}, Tumor={tumor_weight:.2f}, Small Tumor={small_tumor_weight:.2f}")

    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, idx):
        """获取一个数据样本"""
        # 获取切片文件名
        slice_file = self.slice_files[idx]
        slice_path = self.slices_dir / slice_file

        try:
            # 加载切片数据
            data = np.load(slice_path)

            # 提取图像和掩码
            image = data['image']
            liver_mask = data['liver_mask']
            tumor_mask = data['tumor_mask']

            # 确保图像有3个通道
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

            # 确保图像数据类型正确
            image = image.astype(np.float32)

            # 创建分割掩码 (0=背景, 1=肝脏, 2=肿瘤)
            # 确保是整数类型，并且还是NumPy数组
            mask = np.zeros_like(liver_mask, dtype=np.int64)
            mask[liver_mask == 1] = 1
            mask[tumor_mask == 1] = 2

            # 准备权重图 (用于小型肿瘤聚焦)
            weight_map = None
            if self.return_weight_map and 'tumor_size_map' in data:
                weight_map = data['tumor_size_map']

                # 增强小型肿瘤权重
                if self.small_tumor_focus and 'small_tumor_mask' in data:
                    small_tumor = data['small_tumor_mask']
                    if np.any(small_tumor):
                        # 如果有小型肿瘤，赋予它们最高权重
                        small_tumor_weight = 1.0
                        weight_map = weight_map.copy()  # 创建副本以避免修改原始数据
                        weight_map[small_tumor == 1] = small_tumor_weight

            # 确保weight_map是二维的并且是浮点型
            if weight_map is not None:
                if len(weight_map.shape) != 2:
                    weight_map = weight_map.squeeze()  # 移除多余的维度
                weight_map = weight_map.astype(np.float32)

            # 应用数据增强 - 一起处理图像、掩码和权重图
            if self.transform:
                # 确保所有输入的都是NumPy数组，而不是张量
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()

                if weight_map is not None:
                    if isinstance(weight_map, torch.Tensor):
                        weight_map = weight_map.numpy()

                    # 转换所有数据
                    transformed = self.transform(
                        image=image,
                        mask=mask,
                        weight_map=weight_map
                    )
                    image = transformed['image']
                    mask = transformed['mask']
                    weight_map = transformed['weight_map']
                else:
                    # 只转换图像和掩码
                    transformed = self.transform(
                        image=image,
                        mask=mask
                    )
                    image = transformed['image']
                    mask = transformed['mask']
            else:
                # 如果没有提供转换，默认转换为PyTorch张量
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                mask = torch.from_numpy(mask).long()
                if weight_map is not None:
                    weight_map = torch.from_numpy(weight_map).float()

            # 构建样本
            sample = {
                'image': image,
                'mask': mask
            }

            # 添加权重图
            if weight_map is not None:
                sample['weight_map'] = weight_map

            # 添加元数据
            sample['file_name'] = slice_file

            return sample

        except Exception as e:
            print(f"数据增强错误，文件: {slice_file}")
            print(f"图像形状: {image.shape}, 掩码形状: {mask.shape}")
            if weight_map is not None:
                print(f"权重图形状: {weight_map.shape}")
            raise e


def get_transforms(phase, img_size=(512, 512)):
    """
    获取数据增强转换

    参数:
        phase: 'train' 或 'val'
        img_size: 目标图像大小
    """
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            ToTensorV2()
        ], additional_targets={'weight_map': 'image'})  # 将weight_map当作image处理，而不是mask
    else:  # 验证集
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            ToTensorV2()
        ], additional_targets={'weight_map': 'image'})  # 将weight_map当作image处理


def get_dataloaders(data_dir, batch_size=16, num_workers=4, img_size=(512, 512)):
    """
    Create training and validation data loaders

    Parameters:
        data_dir: Preprocessed data directory
        batch_size: Batch size
        num_workers: Number of data loading threads
        img_size: Image size
    """
    # Set paths
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"

    # Training set
    train_transform = get_transforms('train', img_size)
    train_dataset = LiTSDataset(
        data_dir=data_dir,
        slice_list_file=splits_dir / "train_slices.txt",
        transform=train_transform,
        phase="train",
        small_tumor_focus=True,
        return_weight_map=True
    )

    # Validation set
    val_transform = get_transforms('val', img_size)
    val_dataset = LiTSDataset(
        data_dir=data_dir,
        slice_list_file=splits_dir / "val_slices.txt",
        transform=val_transform,
        phase="val",
        small_tumor_focus=True,
        return_weight_map=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


def visualize_batch(batch, num_samples=4, save_path=None):
    """Visualize a batch of samples"""
    images = batch['image'][:num_samples].cpu().numpy()
    masks = batch['mask'][:num_samples].cpu().numpy()

    # Check if weight map is included
    has_weights = 'weight_map' in batch
    if has_weights:
        weights = batch['weight_map'][:num_samples].cpu().numpy()

    # Visualization
    fig, axes = plt.subplots(num_samples, 3 if has_weights else 2, figsize=(12, 3 * num_samples))

    for i in range(num_samples):
        # Transpose image channels
        img = images[i].transpose(1, 2, 0)
        mask = masks[i]

        # Display image
        ax = axes[i, 0] if num_samples > 1 else axes[0]
        ax.imshow(img)
        ax.set_title(f"Sample {i + 1}")
        ax.axis('off')

        # Display mask
        ax = axes[i, 1] if num_samples > 1 else axes[1]
        ax.imshow(mask, cmap='viridis', vmin=0, vmax=2)
        ax.set_title(f"Mask (0=Background, 1=Liver, 2=Tumor)")
        ax.axis('off')

        # Display weight map
        if has_weights:
            weight = weights[i]
            # Fix shape issue - if weight map has multiple channels, only take the first one
            if len(weight.shape) == 3 and weight.shape[0] == 1:
                weight = weight[0]  # Take first channel

            ax = axes[i, 2] if num_samples > 1 else axes[2]
            ax.imshow(weight, cmap='hot')
            ax.set_title(f"Weight Map")
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")

    plt.show()


def test_dataloader():
    """Test data loader"""
    # Set data directory
    data_dir = "data/preprocessed"

    # Get data loaders
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        num_workers=2,
        img_size=(512, 512)
    )

    print(f"Training set size: {len(train_loader.dataset)} slices")
    print(f"Validation set size: {len(val_loader.dataset)} slices")

    # Get a batch
    for batch in train_loader:
        print(f"Batch shapes: Image={batch['image'].shape}, Mask={batch['mask'].shape}")
        if 'weight_map' in batch:
            print(f"Weight map shape: {batch['weight_map'].shape}")

        # Visualize
        visualize_batch(batch, num_samples=4, save_path="results/dataloader_samples.png")
        break


if __name__ == "__main__":
    test_dataloader()