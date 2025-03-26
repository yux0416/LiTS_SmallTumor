import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_npz_content(npz_file):
    """可视化NPZ文件中的内容"""
    data = np.load(npz_file)

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 显示原始CT图像
    axes[0, 0].imshow(data['image'], cmap='gray')
    axes[0, 0].set_title('CT Image')
    axes[0, 0].axis('off')

    # 显示肝脏掩码
    axes[0, 1].imshow(data['liver_mask'], cmap='Blues')
    axes[0, 1].set_title('Liver Mask')
    axes[0, 1].axis('off')

    # 显示肿瘤掩码
    axes[0, 2].imshow(data['tumor_mask'], cmap='Reds')
    axes[0, 2].set_title('Tumor Mask')
    axes[0, 2].axis('off')

    # 显示肿瘤大小权重图
    if 'tumor_size_map' in data:
        axes[1, 0].imshow(data['tumor_size_map'], cmap='hot')
        axes[1, 0].set_title('Tumor Size Weight Map')
        axes[1, 0].axis('off')

    # 显示小型肿瘤掩码
    if 'small_tumor_mask' in data:
        axes[1, 1].imshow(data['small_tumor_mask'], cmap='Greens')
        axes[1, 1].set_title('Small Tumor Mask')
        axes[1, 1].axis('off')

    # 叠加显示
    if 'image' in data and 'tumor_size_map' in data:
        axes[1, 2].imshow(data['image'], cmap='gray')
        # 只在非零区域显示权重图
        tumor_size_map = data['tumor_size_map']
        if np.any(tumor_size_map > 0):
            axes[1, 2].imshow(tumor_size_map, cmap='hot', alpha=0.7)
        axes[1, 2].set_title('CT + Tumor Size Weights')
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    return data


# 查找包含肿瘤的切片
def find_slices_with_tumors(data_dir, limit=5):
    """查找包含肿瘤的切片"""
    data_dir = Path(data_dir)

    tumor_slices = []
    for npz_file in data_dir.glob("*.npz"):
        data = np.load(npz_file)
        if 'tumor_mask' in data and np.any(data['tumor_mask']):
            # 如果存在小型肿瘤，优先选择
            if 'small_tumor_mask' in data and np.any(data['small_tumor_mask']):
                tumor_slices.insert(0, npz_file)
            else:
                tumor_slices.append(npz_file)

        if len(tumor_slices) >= limit:
            break

    return tumor_slices


# 使用示例
data_dir = "D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\data\\preprocessed\\2d_slices"
tumor_slices = find_slices_with_tumors(data_dir)

for slice_file in tumor_slices:
    print(f"Visualizing: {slice_file}")
    data = visualize_npz_content(slice_file)

    # 输出数组形状和值范围信息
    for key, array in data.items():
        print(f"{key}: shape={array.shape}, min={array.min()}, max={array.max()}")

    input("Press Enter to continue to next slice...")