# analyze_lits_dataset.py
import os
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class LiTSDatasetAnalyzer:
    def __init__(self, data_dir_b1, data_dir_b2, output_dir="results/statistics"):
        """
        初始化LiTS数据分析器

        参数:
            data_dir_b1: Training_Batch1目录路径
            data_dir_b2: Training_Batch2目录路径
            output_dir: 分析结果输出目录
        """
        self.data_dir_b1 = Path(data_dir_b1)
        self.data_dir_b2 = Path(data_dir_b2)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有文件路径
        self.volume_files = []
        self.segmentation_files = []

        for batch_dir in [self.data_dir_b1, self.data_dir_b2]:
            self.volume_files.extend(sorted(list(batch_dir.glob("volume-*.nii"))))
            self.segmentation_files.extend(sorted(list(batch_dir.glob("segmentation-*.nii"))))

        print(f"找到 {len(self.volume_files)} 个CT体积文件")
        print(f"找到 {len(self.segmentation_files)} 个分割标签文件")

        # 初始化结果DataFrame
        self.dataset_stats = pd.DataFrame()
        self.liver_stats = pd.DataFrame()
        self.tumor_stats = pd.DataFrame()

    def analyze_dataset(self):
        """分析整个数据集，收集统计信息"""
        print("开始分析LiTS数据集...")

        # 收集每个CT扫描的统计信息
        ct_stats = []
        liver_vols = []
        all_tumors = []

        # 遍历所有文件
        for vol_path, seg_path in tqdm(zip(self.volume_files, self.segmentation_files),
                                       total=len(self.volume_files),
                                       desc="分析CT扫描"):
            try:
                # 读取CT和分割
                ct_img = sitk.ReadImage(str(vol_path))
                seg_img = sitk.ReadImage(str(seg_path))

                # 提取基本信息
                case_id = vol_path.stem.split('-')[1]
                spacing = ct_img.GetSpacing()  # (x, y, z)

                # 转换为NumPy数组
                ct_array = sitk.GetArrayFromImage(ct_img)  # (z, y, x)
                seg_array = sitk.GetArrayFromImage(seg_img)

                # CT统计
                ct_stats.append({
                    'case_id': case_id,
                    'batch': '1' if vol_path.parent == self.data_dir_b1 else '2',
                    'shape': ct_array.shape,
                    'spacing_x': spacing[0],
                    'spacing_y': spacing[1],
                    'spacing_z': spacing[2],
                    'volume_dims': f"{ct_array.shape[0]}×{ct_array.shape[1]}×{ct_array.shape[2]}",
                    'min_hu': np.min(ct_array),
                    'max_hu': np.max(ct_array),
                    'mean_hu': np.mean(ct_array),
                    'std_hu': np.std(ct_array)
                })

                # 肝脏统计
                liver_mask = (seg_array == 1) | (seg_array == 2)  # 肝脏和肿瘤都属于肝脏
                liver_voxels = np.sum(liver_mask)
                liver_volume_ml = (liver_voxels * spacing[0] * spacing[1] * spacing[2]) / 1000  # 转换为毫升

                liver_vols.append({
                    'case_id': case_id,
                    'liver_voxels': liver_voxels,
                    'liver_volume_ml': liver_volume_ml,
                    'liver_slices': np.sum(np.any(liver_mask, axis=(1, 2))),
                    'percent_liver': liver_voxels / ct_array.size * 100
                })

                # 肿瘤分析
                tumor_mask = (seg_array == 2)
                if np.any(tumor_mask):
                    # 标记连通肿瘤区域
                    labeled_tumors, num_tumors = ndimage.label(tumor_mask)

                    # 收集每个肿瘤的统计数据
                    for tumor_id in range(1, num_tumors + 1):
                        tumor = (labeled_tumors == tumor_id)
                        tumor_voxels = np.sum(tumor)

                        # 计算肿瘤体积和直径
                        volume_mm3 = tumor_voxels * spacing[0] * spacing[1] * spacing[2]
                        diameter_mm = 2 * ((3 * volume_mm3) / (4 * np.pi)) ** (1 / 3)  # 假设球形

                        # 获取肿瘤在CT中的位置
                        z_indices, y_indices, x_indices = np.where(tumor)
                        center_z = np.mean(z_indices)
                        center_y = np.mean(y_indices)
                        center_x = np.mean(x_indices)

                        # 相对位置（归一化到[0,1]）
                        rel_z = center_z / ct_array.shape[0]
                        rel_y = center_y / ct_array.shape[1]
                        rel_x = center_x / ct_array.shape[2]

                        # 肿瘤内部HU值
                        tumor_hu_values = ct_array[tumor]
                        tumor_mean_hu = np.mean(tumor_hu_values)
                        tumor_min_hu = np.min(tumor_hu_values)
                        tumor_max_hu = np.max(tumor_hu_values)

                        # 肿瘤与周围肝实质的对比度
                        # 获取肿瘤边界周围的肝实质（不包括肿瘤）
                        dilated_tumor = ndimage.binary_dilation(tumor, iterations=3)
                        liver_around = (dilated_tumor & liver_mask & ~tumor)

                        # 如果周围有肝实质，计算对比度
                        if np.any(liver_around):
                            liver_hu_values = ct_array[liver_around]
                            liver_mean_hu = np.mean(liver_hu_values)
                            contrast = tumor_mean_hu - liver_mean_hu
                        else:
                            liver_mean_hu = None
                            contrast = None

                        # 确定肿瘤大小类别
                        if diameter_mm < 10:
                            size_category = 'small'
                        elif diameter_mm < 20:
                            size_category = 'medium'
                        else:
                            size_category = 'large'

                        # 收集肿瘤统计
                        all_tumors.append({
                            'case_id': case_id,
                            'tumor_id': tumor_id,
                            'volume_mm3': volume_mm3,
                            'diameter_mm': diameter_mm,
                            'size_category': size_category,
                            'voxel_count': tumor_voxels,
                            'center_z': center_z,
                            'center_y': center_y,
                            'center_x': center_x,
                            'rel_z': rel_z,
                            'rel_y': rel_y,
                            'rel_x': rel_x,
                            'tumor_mean_hu': tumor_mean_hu,
                            'tumor_min_hu': tumor_min_hu,
                            'tumor_max_hu': tumor_max_hu,
                            'liver_mean_hu': liver_mean_hu,
                            'contrast': contrast
                        })
                else:
                    # 记录没有肿瘤的情况
                    print(f"案例 {case_id} 没有肿瘤")

            except Exception as e:
                print(f"处理 {vol_path.name} 时出错: {e}")

        # 转换为DataFrame
        self.dataset_stats = pd.DataFrame(ct_stats)
        self.liver_stats = pd.DataFrame(liver_vols)
        self.tumor_stats = pd.DataFrame(all_tumors)

        # 保存统计数据
        self.dataset_stats.to_csv(self.output_dir / "dataset_statistics.csv", index=False)
        self.liver_stats.to_csv(self.output_dir / "liver_statistics.csv", index=False)
        self.tumor_stats.to_csv(self.output_dir / "tumor_statistics.csv", index=False)

        print(f"分析完成，统计结果已保存到 {self.output_dir}")

        return self.dataset_stats, self.liver_stats, self.tumor_stats

    def generate_summary_report(self):
        """生成摘要报告和统计数据"""
        if self.dataset_stats.empty or self.liver_stats.empty or self.tumor_stats.empty:
            print("请先运行analyze_dataset()以收集统计数据")
            return

        # 创建结果摘要文件
        summary_file = self.output_dir / "summary_report.txt"

        with open(summary_file, 'w') as f:
            # 标题
            f.write("=" * 80 + "\n")
            f.write("LiTS数据集统计报告\n")
            f.write("=" * 80 + "\n\n")

            # 1. 整体数据集统计
            f.write("-" * 80 + "\n")
            f.write("1. 整体数据集统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"总CT扫描数量: {len(self.dataset_stats)}\n")

            # 体积尺寸统计
            f.write("\n体积尺寸统计:\n")
            shapes = np.array([eval(s.replace('×', ',')) for s in self.dataset_stats['volume_dims']])
            f.write(
                f"  平均尺寸 (Z×Y×X): {np.mean(shapes[:, 0]):.1f}×{np.mean(shapes[:, 1]):.1f}×{np.mean(shapes[:, 2]):.1f}\n")
            f.write(f"  最小尺寸 (Z×Y×X): {np.min(shapes[:, 0])}×{np.min(shapes[:, 1])}×{np.min(shapes[:, 2])}\n")
            f.write(f"  最大尺寸 (Z×Y×X): {np.max(shapes[:, 0])}×{np.max(shapes[:, 1])}×{np.max(shapes[:, 2])}\n")

            # 体素间距统计
            f.write("\n体素间距统计 (mm):\n")
            f.write(
                f"  平均间距 (X,Y,Z): ({self.dataset_stats['spacing_x'].mean():.3f}, {self.dataset_stats['spacing_y'].mean():.3f}, {self.dataset_stats['spacing_z'].mean():.3f})\n")
            f.write(
                f"  最小间距 (X,Y,Z): ({self.dataset_stats['spacing_x'].min():.3f}, {self.dataset_stats['spacing_y'].min():.3f}, {self.dataset_stats['spacing_z'].min():.3f})\n")
            f.write(
                f"  最大间距 (X,Y,Z): ({self.dataset_stats['spacing_x'].max():.3f}, {self.dataset_stats['spacing_y'].max():.3f}, {self.dataset_stats['spacing_z'].max():.3f})\n")

            # HU值统计
            f.write("\nHU值统计:\n")
            f.write(
                f"  平均HU值范围: [{self.dataset_stats['min_hu'].mean():.1f}, {self.dataset_stats['max_hu'].mean():.1f}]\n")
            f.write(f"  全局最小HU值: {self.dataset_stats['min_hu'].min():.1f}\n")
            f.write(f"  全局最大HU值: {self.dataset_stats['max_hu'].max():.1f}\n")
            f.write(f"  平均HU标准差: {self.dataset_stats['std_hu'].mean():.1f}\n")

            # 2. 肝脏统计
            f.write("\n" + "-" * 80 + "\n")
            f.write("2. 肝脏统计\n")
            f.write("-" * 80 + "\n")

            # 检查是否有至少一个扫描中存在肝脏
            if len(self.liver_stats) > 0:
                # 肝脏体积统计
                f.write(
                    f"存在肝脏的扫描: {len(self.liver_stats)}/{len(self.dataset_stats)} ({len(self.liver_stats) / len(self.dataset_stats) * 100:.1f}%)\n")
                f.write(
                    f"平均肝脏体积: {self.liver_stats['liver_volume_ml'].mean():.2f} mL (标准差: {self.liver_stats['liver_volume_ml'].std():.2f} mL)\n")
                f.write(f"最小肝脏体积: {self.liver_stats['liver_volume_ml'].min():.2f} mL\n")
                f.write(f"最大肝脏体积: {self.liver_stats['liver_volume_ml'].max():.2f} mL\n")
                f.write(f"中位肝脏体积: {self.liver_stats['liver_volume_ml'].median():.2f} mL\n")

                # 肝脏占比统计
                f.write(f"平均肝脏占比: {self.liver_stats['percent_liver'].mean():.2f}%\n")
                f.write(f"最小肝脏占比: {self.liver_stats['percent_liver'].min():.2f}%\n")
                f.write(f"最大肝脏占比: {self.liver_stats['percent_liver'].max():.2f}%\n")
            else:
                f.write("数据集中未找到肝脏分割\n")

            # 3. 肿瘤统计
            f.write("\n" + "-" * 80 + "\n")
            f.write("3. 肿瘤统计\n")
            f.write("-" * 80 + "\n")

            # 检查是否有肿瘤数据
            if len(self.tumor_stats) > 0:
                # 计算肿瘤表相关统计
                cases_with_tumors = len(self.tumor_stats['case_id'].unique())
                tumors_per_case = self.tumor_stats.groupby('case_id').size()

                f.write(
                    f"存在肿瘤的扫描: {cases_with_tumors}/{len(self.dataset_stats)} ({cases_with_tumors / len(self.dataset_stats) * 100:.1f}%)\n")
                f.write(f"总肿瘤数量: {len(self.tumor_stats)}\n")
                f.write(f"每例平均肿瘤数量: {tumors_per_case.mean():.2f} (标准差: {tumors_per_case.std():.2f})\n")
                f.write(f"单例最大肿瘤数量: {tumors_per_case.max()}\n")

                # 肿瘤大小分类
                size_counts = self.tumor_stats['size_category'].value_counts()
                f.write("\n肿瘤大小分布:\n")
                for category, count in size_counts.items():
                    percentage = count / len(self.tumor_stats) * 100
                    f.write(f"  {category}: {count} ({percentage:.1f}%)\n")

                # 小型肿瘤统计
                small_tumors = self.tumor_stats[self.tumor_stats['size_category'] == 'small']
                if len(small_tumors) > 0:
                    f.write("\n小型肿瘤(<10mm)统计:\n")
                    f.write(f"  数量: {len(small_tumors)}\n")
                    f.write(f"  平均直径: {small_tumors['diameter_mm'].mean():.2f} mm\n")
                    f.write(f"  最小直径: {small_tumors['diameter_mm'].min():.2f} mm\n")
                    f.write(f"  最大直径: {small_tumors['diameter_mm'].max():.2f} mm\n")
                    f.write(f"  平均体积: {small_tumors['volume_mm3'].mean():.2f} mm³\n")

                # 肿瘤体积统计
                f.write("\n肿瘤体积统计:\n")
                f.write(f"  平均体积: {self.tumor_stats['volume_mm3'].mean():.2f} mm³\n")
                f.write(f"  最小体积: {self.tumor_stats['volume_mm3'].min():.2f} mm³\n")
                f.write(f"  最大体积: {self.tumor_stats['volume_mm3'].max():.2f} mm³\n")
                f.write(f"  中位体积: {self.tumor_stats['volume_mm3'].median():.2f} mm³\n")

                # 肿瘤HU值统计
                f.write("\n肿瘤HU值统计:\n")
                f.write(f"  平均HU值: {self.tumor_stats['tumor_mean_hu'].mean():.2f}\n")
                f.write(f"  最小平均HU值: {self.tumor_stats['tumor_mean_hu'].min():.2f}\n")
                f.write(f"  最大平均HU值: {self.tumor_stats['tumor_mean_hu'].max():.2f}\n")

                # 肿瘤对比度统计
                valid_contrast = self.tumor_stats['contrast'].dropna()
                if len(valid_contrast) > 0:
                    f.write("\n肿瘤-肝脏对比度统计:\n")
                    f.write(f"  平均对比度: {valid_contrast.mean():.2f} HU\n")
                    f.write(f"  最小对比度: {valid_contrast.min():.2f} HU\n")
                    f.write(f"  最大对比度: {valid_contrast.max():.2f} HU\n")
                    f.write(f"  中位对比度: {valid_contrast.median():.2f} HU\n")

                    # 检查对比度分布
                    hyper_dense = len(valid_contrast[valid_contrast > 0])
                    hypo_dense = len(valid_contrast[valid_contrast < 0])
                    f.write(f"  高密度肿瘤(对比度>0): {hyper_dense} ({hyper_dense / len(valid_contrast) * 100:.1f}%)\n")
                    f.write(f"  低密度肿瘤(对比度<0): {hypo_dense} ({hypo_dense / len(valid_contrast) * 100:.1f}%)\n")

                # 4. 小型肿瘤特征分析
                if len(small_tumors) > 0:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("4. 小型肿瘤(<10mm)特征分析\n")
                    f.write("-" * 80 + "\n")

                    # 小型肿瘤对比度
                    small_valid_contrast = small_tumors['contrast'].dropna()
                    if len(small_valid_contrast) > 0:
                        f.write("\n小型肿瘤对比度统计:\n")
                        f.write(f"  平均对比度: {small_valid_contrast.mean():.2f} HU\n")
                        f.write(f"  最小对比度: {small_valid_contrast.min():.2f} HU\n")
                        f.write(f"  最大对比度: {small_valid_contrast.max():.2f} HU\n")

                        # 小型肿瘤对比度分布
                        small_hyper_dense = len(small_valid_contrast[small_valid_contrast > 0])
                        small_hypo_dense = len(small_valid_contrast[small_valid_contrast < 0])
                        f.write(
                            f"  高密度小型肿瘤: {small_hyper_dense} ({small_hyper_dense / len(small_valid_contrast) * 100:.1f}%)\n")
                        f.write(
                            f"  低密度小型肿瘤: {small_hypo_dense} ({small_hypo_dense / len(small_valid_contrast) * 100:.1f}%)\n")

                    # 小型肿瘤位置分布
                    f.write("\n小型肿瘤位置分布 (相对位置0-1):\n")
                    f.write(
                        f"  平均Z位置(上下): {small_tumors['rel_z'].mean():.2f} (标准差: {small_tumors['rel_z'].std():.2f})\n")
                    f.write(
                        f"  平均Y位置(前后): {small_tumors['rel_y'].mean():.2f} (标准差: {small_tumors['rel_y'].std():.2f})\n")
                    f.write(
                        f"  平均X位置(左右): {small_tumors['rel_x'].mean():.2f} (标准差: {small_tumors['rel_x'].std():.2f})\n")
            else:
                f.write("数据集中未找到肿瘤\n")

            # 5. 结论与挑战
            f.write("\n" + "-" * 80 + "\n")
            f.write("5. 结论与挑战\n")
            f.write("-" * 80 + "\n")

            if len(self.tumor_stats) > 0:
                f.write("\n主要发现:\n")
                f.write(f"1. 数据集包含 {len(self.dataset_stats)} 个CT扫描，其中 {cases_with_tumors} 个包含肿瘤\n")

                # 肿瘤大小分布
                size_percentages = {cat: count / len(self.tumor_stats) * 100 for cat, count in size_counts.items()}
                f.write(
                    f"2. 肿瘤大小分布: 小型(<10mm): {size_percentages.get('small', 0):.1f}%, 中型(10-20mm): {size_percentages.get('medium', 0):.1f}%, 大型(>20mm): {size_percentages.get('large', 0):.1f}%\n")

                if len(valid_contrast) > 0:
                    contrast_stats = f"平均对比度: {valid_contrast.mean():.1f}HU, 范围: [{valid_contrast.min():.1f}, {valid_contrast.max():.1f}]HU"
                    f.write(f"3. 肿瘤对比度: {contrast_stats}\n")

                f.write("\n挑战:\n")
                if 'small' in size_counts and size_counts['small'] > 0:
                    small_percentage = size_counts['small'] / len(self.tumor_stats) * 100
                    f.write(f"1. 小型肿瘤占比高 ({small_percentage:.1f}%)，检测难度大\n")

                if len(valid_contrast) > 0:
                    low_contrast_count = len(valid_contrast[abs(valid_contrast) < 20])
                    low_contrast_percentage = low_contrast_count / len(valid_contrast) * 100
                    f.write(f"2. 低对比度肿瘤占比 ({low_contrast_percentage:.1f}%)，增加检测难度\n")

                # 数据不平衡问题
                f.write(f"3. 每个扫描的肿瘤数量差异大 (范围: 0-{tumors_per_case.max()})，存在数据不平衡问题\n")

                # 体素大小不一致
                if self.dataset_stats['spacing_z'].max() / self.dataset_stats['spacing_z'].min() > 2:
                    f.write(
                        f"4. 体素尺寸不一致，特别是切片厚度差异大 ({self.dataset_stats['spacing_z'].min():.2f}-{self.dataset_stats['spacing_z'].max():.2f}mm)\n")

                f.write("\n针对小型肿瘤的建议:\n")
                f.write("1. 使用注意力机制聚焦于小型目标特征\n")
                f.write("2. 设计基于肿瘤大小的加权损失函数\n")
                f.write("3. 考虑多尺度特征融合，增强对小结构的感知\n")
                f.write("4. 针对肿瘤与肝脏对比度差异设计特殊的预处理策略\n")

        print(f"统计报告已生成: {summary_file}")

        return summary_file

    def visualize_statistics(self):
        """生成统计可视化"""
        if self.dataset_stats.empty or self.liver_stats.empty or self.tumor_stats.empty:
            print("请先运行analyze_dataset()以收集统计数据")
            return

        # 设置绘图风格
        plt.style.use('ggplot')
        sns.set(style="whitegrid")

        # 1. 肿瘤大小分布
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        sns.histplot(data=self.tumor_stats, x='diameter_mm', bins=30, kde=True)
        plt.axvline(x=10, color='r', linestyle='--', alpha=0.7)
        plt.axvline(x=20, color='r', linestyle='--', alpha=0.7)
        plt.title('肿瘤直径分布')
        plt.xlabel('直径 (mm)')
        plt.ylabel('数量')
        plt.annotate('小型肿瘤\n(<10mm)', xy=(5, plt.ylim()[1] * 0.9), ha='center')
        plt.annotate('中型肿瘤\n(10-20mm)', xy=(15, plt.ylim()[1] * 0.9), ha='center')
        plt.annotate('大型肿瘤\n(>20mm)', xy=(30, plt.ylim()[1] * 0.9), ha='center')

        plt.subplot(2, 2, 2)
        size_counts = self.tumor_stats['size_category'].value_counts().reindex(['small', 'medium', 'large'])
        ax = sns.barplot(x=size_counts.index, y=size_counts.values)
        plt.title('肿瘤大小分类')
        plt.xlabel('大小类别')
        plt.ylabel('数量')
        # 添加百分比标签
        total = len(self.tumor_stats)
        for i, p in enumerate(ax.patches):
            percentage = 100 * p.get_height() / total
            ax.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom')

        # 2. 肿瘤体积分布 (对数尺度)
        plt.subplot(2, 2, 3)
        sns.histplot(data=self.tumor_stats, x='volume_mm3', bins=30, log_scale=True, kde=True)
        plt.title('肿瘤体积分布 (对数尺度)')
        plt.xlabel('体积 (mm³)')
        plt.ylabel('数量')

        # 3. 肿瘤-肝脏对比度分布
        plt.subplot(2, 2, 4)
        valid_contrast = self.tumor_stats.dropna(subset=['contrast'])
        sns.histplot(data=valid_contrast, x='contrast', bins=30, kde=True)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        plt.title('肿瘤-肝脏对比度分布')
        plt.xlabel('对比度 (HU)')
        plt.ylabel('数量')
        plt.annotate('低密度肿瘤', xy=(-50, plt.ylim()[1] * 0.9), ha='center')
        plt.annotate('高密度肿瘤', xy=(50, plt.ylim()[1] * 0.9), ha='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / "tumor_size_distribution.png")

        # 4. 肿瘤直径与对比度的关系
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.scatterplot(data=valid_contrast, x='diameter_mm', y='contrast', hue='size_category',
                        palette={'small': 'red', 'medium': 'green', 'large': 'blue'}, alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('肿瘤直径与对比度关系')
        plt.xlabel('直径 (mm)')
        plt.ylabel('对比度 (HU)')

        plt.subplot(1, 2, 2)
        sns.boxplot(data=valid_contrast, x='size_category', y='contrast',
                    order=['small', 'medium', 'large'])
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('不同大小肿瘤的对比度分布')
        plt.xlabel('肿瘤大小类别')
        plt.ylabel('对比度 (HU)')

        plt.tight_layout()
        plt.savefig(self.output_dir / "tumor_size_contrast.png")

        # 5. 每个CT扫描的肿瘤数量分布
        plt.figure(figsize=(10, 6))
        tumors_per_case = self.tumor_stats.groupby('case_id').size().reset_index(name='tumor_count')
        sns.histplot(data=tumors_per_case, x='tumor_count',
                     bins=range(0, int(tumors_per_case['tumor_count'].max()) + 2))
        plt.title('每个CT扫描的肿瘤数量分布')
        plt.xlabel('肿瘤数量')
        plt.ylabel('CT扫描数量')
        plt.savefig(self.output_dir / "tumors_per_case.png")

        # 6. 小型肿瘤的空间分布
        plt.figure(figsize=(15, 5))
        small_tumors = self.tumor_stats[self.tumor_stats['size_category'] == 'small']

        plt.subplot(1, 3, 1)
        sns.kdeplot(data=small_tumors, x='rel_x', y='rel_y', fill=True, cmap='viridis')
        plt.title('小型肿瘤在X-Y平面上的分布')
        plt.xlabel('相对X位置 (左-右)')
        plt.ylabel('相对Y位置 (前-后)')

        plt.subplot(1, 3, 2)
        sns.kdeplot(data=small_tumors, x='rel_x', y='rel_z', fill=True, cmap='viridis')
        plt.title('小型肿瘤在X-Z平面上的分布')
        plt.xlabel('相对X位置 (左-右)')
        plt.ylabel('相对Z位置 (下-上)')

        plt.subplot(1, 3, 3)
        sns.kdeplot(data=small_tumors, x='rel_y', y='rel_z', fill=True, cmap='viridis')
        plt.title('小型肿瘤在Y-Z平面上的分布')
        plt.xlabel('相对Y位置 (前-后)')
        plt.ylabel('相对Z位置 (下-上)')

        plt.tight_layout()
        plt.savefig(self.output_dir / "small_tumor_distribution.png")

        # 7. 小型肿瘤与大型肿瘤的对比度比较
        plt.figure(figsize=(12, 5))

        # 对比度分布比较
        plt.subplot(1, 2, 1)
        for category, color, label in zip(['small', 'large'], ['red', 'blue'], ['小型肿瘤', '大型肿瘤']):
            category_data = self.tumor_stats[(self.tumor_stats['size_category'] == category) &
                                             self.tumor_stats['contrast'].notna()]
            if len(category_data) > 0:
                sns.kdeplot(data=category_data, x='contrast', color=color, label=label, fill=True, alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('小型vs大型肿瘤的对比度分布')
        plt.xlabel('对比度 (HU)')
        plt.ylabel('密度')
        plt.legend()

        # HU值分布比较
        plt.subplot(1, 2, 2)
        for category, color, label in zip(['small', 'large'], ['red', 'blue'], ['小型肿瘤', '大型肿瘤']):
            category_data = self.tumor_stats[self.tumor_stats['size_category'] == category]
            if len(category_data) > 0:
                sns.kdeplot(data=category_data, x='tumor_mean_hu', color=color, label=label, fill=True, alpha=0.3)
        plt.title('小型vs大型肿瘤的HU值分布')
        plt.xlabel('平均HU值')
        plt.ylabel('密度')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "tumor_size_comparison.png")

        print(f"统计可视化已保存到 {self.output_dir}")

        # 返回所有图像路径作为列表
        return [
            self.output_dir / "tumor_size_distribution.png",
            self.output_dir / "tumor_size_contrast.png",
            self.output_dir / "tumors_per_case.png",
            self.output_dir / "small_tumor_distribution.png",
            self.output_dir / "tumor_size_comparison.png"
        ]


def main():
    # 设置数据路径
    data_dir_b1 = r"data/raw/Training_Batch1"
    data_dir_b2 = r"data/raw/Training_Batch2"
    output_dir = r"results/statistics"

    # 创建分析器并运行分析
    analyzer = LiTSDatasetAnalyzer(data_dir_b1, data_dir_b2, output_dir)

    # 分析数据集
    analyzer.analyze_dataset()

    # 生成报告和可视化
    analyzer.generate_summary_report()
    analyzer.visualize_statistics()

    print("数据集分析完成!")


if __name__ == "__main__":
    main()