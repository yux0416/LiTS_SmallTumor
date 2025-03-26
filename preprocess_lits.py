# preprocess_lits.py
import os
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
import h5py
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import multiprocessing
from functools import partial

warnings.filterwarnings("ignore", category=UserWarning)


class LiTSPreprocessor:
    def __init__(self, data_dir_b1, data_dir_b2, output_dir,
                 window_min=-100, window_max=400,
                 liver_label=1, tumor_label=2,
                 slice_thickness=None,
                 small_tumor_diameter=10):
        """
        LiTS数据集预处理器

        参数:
            data_dir_b1: Training_Batch1目录路径
            data_dir_b2: Training_Batch2目录路径
            output_dir: 预处理数据输出目录
            window_min: 窗口化下限HU值
            window_max: 窗口化上限HU值
            liver_label: 肝脏标签值
            tumor_label: 肿瘤标签值
            slice_thickness: 如果不为None，则重采样到此切片厚度
            small_tumor_diameter: 小型肿瘤直径阈值(mm)
        """
        self.data_dir_b1 = Path(data_dir_b1)
        self.data_dir_b2 = Path(data_dir_b2)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 预处理参数
        self.window_min = window_min
        self.window_max = window_max
        self.liver_label = liver_label
        self.tumor_label = tumor_label
        self.slice_thickness = slice_thickness
        self.small_tumor_diameter = small_tumor_diameter

        # 收集所有文件路径
        self.volume_files = []
        self.segmentation_files = []

        for batch_dir in [self.data_dir_b1, self.data_dir_b2]:
            self.volume_files.extend(sorted(list(batch_dir.glob("volume-*.nii"))))
            self.segmentation_files.extend(sorted(list(batch_dir.glob("segmentation-*.nii"))))

        print(f"找到 {len(self.volume_files)} 个CT体积文件")
        print(f"找到 {len(self.segmentation_files)} 个分割标签文件")

        # 创建子目录
        (self.output_dir / "2d_slices").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)

        # 收集预处理统计信息
        self.preprocessing_stats = []

    def numpy_to_python_types(self, obj):
        """将NumPy类型转换为标准Python类型"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.numpy_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.numpy_to_python_types(i) for i in obj]
        else:
            return obj

    def window_and_normalize(self, ct_array):
        """
        窗口化和标准化CT数据
        """
        # 窗口化
        ct_array = np.clip(ct_array, self.window_min, self.window_max)

        # 标准化到[0,1]
        ct_array = (ct_array - self.window_min) / (self.window_max - self.window_min)

        return ct_array

    def process_case(self, vol_path, seg_path, visualize=False):
        """
        处理单个病例

        参数:
            vol_path: CT体积文件路径
            seg_path: 分割标签文件路径
            visualize: 是否生成可视化

        返回:
            处理结果统计
        """
        case_id = vol_path.stem.split('-')[1]
        print(f"处理案例 {case_id}...")

        start_time = time.time()

        # 读取CT和分割
        try:
            ct_img = sitk.ReadImage(str(vol_path))
            seg_img = sitk.ReadImage(str(seg_path))

            # 确保分割与CT对齐
            seg_img.SetDirection(ct_img.GetDirection())
            seg_img.SetOrigin(ct_img.GetOrigin())

            # 获取原始间距和尺寸
            original_spacing = ct_img.GetSpacing()
            original_size = ct_img.GetSize()

            # 如果需要重采样到指定切片厚度
            if self.slice_thickness is not None and self.slice_thickness != original_spacing[2]:
                # 计算新尺寸
                new_spacing = (original_spacing[0], original_spacing[1], self.slice_thickness)
                new_size = [
                    int(round(original_size[0])),
                    int(round(original_size[1])),
                    int(round(original_size[2] * original_spacing[2] / self.slice_thickness))
                ]

                # 重采样CT
                resample = sitk.ResampleImageFilter()
                resample.SetOutputSpacing(new_spacing)
                resample.SetSize(new_size)
                resample.SetOutputDirection(ct_img.GetDirection())
                resample.SetOutputOrigin(ct_img.GetOrigin())
                resample.SetTransform(sitk.Transform())
                resample.SetDefaultPixelValue(ct_img.GetPixelIDValue())
                resample.SetInterpolator(sitk.sitkLinear)
                ct_img = resample.Execute(ct_img)

                # 重采样分割（使用最近邻插值）
                resample.SetInterpolator(sitk.sitkNearestNeighbor)
                seg_img = resample.Execute(seg_img)

                # 更新间距和尺寸
                spacing = new_spacing
                size = new_size
            else:
                spacing = original_spacing
                size = original_size

            # 转换为NumPy数组
            ct_array = sitk.GetArrayFromImage(ct_img)
            seg_array = sitk.GetArrayFromImage(seg_img)

            # 窗口化和标准化
            ct_array_processed = self.window_and_normalize(ct_array)

            # 提取肝脏和肿瘤掩码
            liver_mask = (seg_array == self.liver_label)
            tumor_mask = (seg_array == self.tumor_label)
            combined_mask = liver_mask | tumor_mask  # 肝脏和肿瘤的并集

            # 识别连通肿瘤区域并分类大小
            labeled_tumors, num_tumors = ndimage.label(tumor_mask)

            # 计算每个肿瘤的信息
            tumor_info = []

            for tumor_id in range(1, num_tumors + 1):
                tumor = (labeled_tumors == tumor_id)
                tumor_voxels = np.sum(tumor)

                # 计算体积和直径
                volume_mm3 = tumor_voxels * spacing[0] * spacing[1] * spacing[2]
                diameter_mm = 2 * ((3 * volume_mm3) / (4 * np.pi)) ** (1 / 3)  # 假设球形

                # 确定肿瘤大小类别
                if diameter_mm < self.small_tumor_diameter:
                    size_category = 'small'
                elif diameter_mm < 20:
                    size_category = 'medium'
                else:
                    size_category = 'large'

                # 获取肿瘤所在的切片
                tumor_slices = np.unique(np.where(tumor)[0])

                tumor_info.append({
                    'id': tumor_id,
                    'volume_mm3': volume_mm3,
                    'diameter_mm': diameter_mm,
                    'size_category': size_category,
                    'voxels': tumor_voxels,
                    'slices': tumor_slices.tolist()
                })

            # 创建肿瘤大小标记图
            tumor_size_map = np.zeros_like(seg_array, dtype=np.float32)
            small_tumor_mask = np.zeros_like(seg_array, dtype=bool)

            for t_info in tumor_info:
                tumor = (labeled_tumors == t_info['id'])
                # 根据肿瘤大小设置权重:
                # 小型=1.0, 中型=0.67, 大型=0.33
                if t_info['size_category'] == 'small':
                    weight = 1.0
                    small_tumor_mask |= tumor
                elif t_info['size_category'] == 'medium':
                    weight = 0.67
                else:
                    weight = 0.33

                tumor_size_map[tumor] = weight

            # 找到包含肝脏的切片
            liver_slices = np.where(np.any(combined_mask, axis=(1, 2)))[0]

            if len(liver_slices) == 0:
                print(f"警告: 案例 {case_id} 未检测到肝脏!")
                return None

            # 处理每个包含肝脏的切片
            slice_data = []

            for z in liver_slices:
                # 提取当前切片
                ct_slice = ct_array_processed[z]
                liver_slice = liver_mask[z]
                tumor_slice = tumor_mask[z]
                size_map_slice = tumor_size_map[z]
                small_tumor_slice = small_tumor_mask[z]

                # 计算切片中的肿瘤体素数量
                tumor_pixels = np.sum(tumor_slice)
                small_tumor_pixels = np.sum(small_tumor_slice)

                # 保存切片数据
                slice_filename = f"{case_id}_{z:03d}"

                # 保存为.npy文件
                slice_data_path = self.output_dir / "2d_slices" / f"{slice_filename}.npz"
                np.savez_compressed(
                    slice_data_path,
                    image=ct_slice.astype(np.float32),
                    liver_mask=liver_slice.astype(np.uint8),
                    tumor_mask=tumor_slice.astype(np.uint8),
                    tumor_size_map=size_map_slice.astype(np.float32),
                    small_tumor_mask=small_tumor_slice.astype(np.uint8)
                )

                # 收集切片信息
                slice_data.append({
                    'case_id': case_id,
                    'slice_idx': z,
                    'filename': f"{slice_filename}.npz",
                    'has_liver': np.any(liver_slice),
                    'has_tumor': np.any(tumor_slice),
                    'has_small_tumor': np.any(small_tumor_slice),
                    'tumor_pixels': int(tumor_pixels),
                    'small_tumor_pixels': int(small_tumor_pixels),
                    'liver_pixels': int(np.sum(liver_slice))
                })

            # 保存可视化
            if visualize and num_tumors > 0:
                # 选择一个包含小型肿瘤的切片进行可视化
                small_tumor_slices = []
                for t_info in tumor_info:
                    if t_info['size_category'] == 'small':
                        small_tumor_slices.extend(t_info['slices'])

                if small_tumor_slices:
                    # 选择包含小型肿瘤的中间切片
                    viz_slice = small_tumor_slices[len(small_tumor_slices) // 2]
                else:
                    # 如果没有小型肿瘤，选择中间切片
                    viz_slice = liver_slices[len(liver_slices) // 2]

                plt.figure(figsize=(15, 5))

                plt.subplot(131)
                plt.imshow(ct_array_processed[viz_slice], cmap='gray')
                plt.title('预处理后CT')
                plt.axis('off')

                plt.subplot(132)
                plt.imshow(ct_array_processed[viz_slice], cmap='gray')
                plt.contour(liver_mask[viz_slice], colors='b', linewidths=0.5)
                plt.contour(tumor_mask[viz_slice], colors='r', linewidths=0.5)
                plt.title('肝脏(蓝)和肿瘤(红)轮廓')
                plt.axis('off')

                plt.subplot(133)
                plt.imshow(tumor_size_map[viz_slice], cmap='hot', alpha=0.7)
                plt.imshow(ct_array_processed[viz_slice], cmap='gray', alpha=0.3)
                plt.title('肿瘤大小权重图')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(self.output_dir / "visualizations" / f"{case_id}_visualization.png")
                plt.close()

            # 收集病例统计信息
            stats = {
                'case_id': case_id,
                'original_spacing': original_spacing,
                'processed_spacing': spacing,
                'original_size': original_size,
                'processed_size': size,
                'num_slices': len(liver_slices),
                'num_tumors': num_tumors,
                'num_small_tumors': sum(1 for t in tumor_info if t['size_category'] == 'small'),
                'processing_time': time.time() - start_time
            }

            # 保存病例元数据
            metadata = {
                'case_info': self.numpy_to_python_types(stats),
                'tumor_info': self.numpy_to_python_types(tumor_info),
                'slice_info': self.numpy_to_python_types(slice_data)
            }

            # 保存为JSON
            import json
            with open(self.output_dir / "metadata" / f"{case_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"案例 {case_id} 处理完成，耗时 {stats['processing_time']:.2f} 秒")

            return stats, slice_data

        except Exception as e:
            print(f"处理案例 {case_id} 时出错: {e}")
            return None

    def preprocess_dataset(self, num_workers=4, visualize_samples=5):
        """
        预处理整个数据集

        参数:
            num_workers: 并行处理的工作进程数
            visualize_samples: 要可视化的样本数量
        """
        print(f"使用 {num_workers} 个进程开始预处理 LiTS 数据集...")

        # 准备路径对
        cases = []
        for vol_path, seg_path in zip(self.volume_files, self.segmentation_files):
            vol_id = vol_path.stem.split('-')[1]
            seg_id = seg_path.stem.split('-')[1]
            if vol_id == seg_id:
                cases.append((vol_path, seg_path, vol_id))

        # 选择要可视化的样本
        if visualize_samples > 0:
            import random
            visualize_ids = set(random.sample([c[2] for c in cases], min(visualize_samples, len(cases))))
        else:
            visualize_ids = set()

        # 顺序处理或并行处理
        all_stats = []
        all_slice_data = []

        if num_workers <= 1:
            # 顺序处理
            for vol_path, seg_path, case_id in tqdm(cases, desc="预处理案例"):
                visualize = case_id in visualize_ids
                result = self.process_case(vol_path, seg_path, visualize)
                if result:
                    stats, slice_data = result
                    all_stats.append(stats)
                    all_slice_data.extend(slice_data)
        else:
            # 并行处理（不包括可视化）
            with multiprocessing.Pool(num_workers) as pool:
                process_func = partial(self._process_case_wrapper, visualize_ids=visualize_ids)
                results = list(tqdm(
                    pool.imap(process_func, cases),
                    total=len(cases),
                    desc="预处理案例"
                ))

                # 收集结果
                for result in results:
                    if result:
                        stats, slice_data = result
                        all_stats.append(stats)
                        all_slice_data.extend(slice_data)

        # 创建数据集CSV
        df_stats = pd.DataFrame(all_stats)
        df_slice = pd.DataFrame(all_slice_data)

        # 保存统计信息
        df_stats.to_csv(self.output_dir / "preprocessing_stats.csv", index=False)
        df_slice.to_csv(self.output_dir / "slice_stats.csv", index=False)

        # 生成数据集摘要
        with open(self.output_dir / "preprocessing_summary.txt", 'w') as f:
            f.write("LiTS数据集预处理摘要\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"处理的案例数: {len(df_stats)}\n")
            f.write(f"生成的切片总数: {len(df_slice)}\n")
            f.write(f"包含肿瘤的切片数: {len(df_slice[df_slice['has_tumor']])}\n")
            f.write(f"包含小型肿瘤的切片数: {len(df_slice[df_slice['has_small_tumor']])}\n\n")

            # 数据分布
            tumor_slice_percent = len(df_slice[df_slice['has_tumor']]) / len(df_slice) * 100
            small_tumor_slice_percent = len(df_slice[df_slice['has_small_tumor']]) / len(df_slice) * 100
            f.write(f"包含肿瘤的切片比例: {tumor_slice_percent:.2f}%\n")
            f.write(f"包含小型肿瘤的切片比例: {small_tumor_slice_percent:.2f}%\n\n")

            # 肿瘤大小统计
            if 'num_small_tumors' in df_stats.columns:
                total_tumors = df_stats['num_tumors'].sum()
                total_small = df_stats['num_small_tumors'].sum()
                if total_tumors > 0:
                    small_percent = total_small / total_tumors * 100
                    f.write(f"小型肿瘤比例: {small_percent:.2f}% ({total_small}/{total_tumors})\n\n")

            # 处理时间统计
            f.write(f"平均每个案例处理时间: {df_stats['processing_time'].mean():.2f} 秒\n")
            f.write(f"总处理时间: {df_stats['processing_time'].sum():.2f} 秒\n\n")

            # 预处理参数
            f.write("预处理参数:\n")
            f.write(f"  窗口化范围: [{self.window_min}, {self.window_max}]\n")
            f.write(f"  小型肿瘤直径阈值: {self.small_tumor_diameter} mm\n")
            if self.slice_thickness:
                f.write(f"  标准化切片厚度: {self.slice_thickness} mm\n")
            else:
                f.write("  保留原始切片厚度\n")

        print(f"预处理完成! 结果保存在 {self.output_dir}")

        return df_stats, df_slice

    def _process_case_wrapper(self, case_tuple, visualize_ids):
        """并行处理的包装器函数"""
        vol_path, seg_path, case_id = case_tuple
        visualize = case_id in visualize_ids
        return self.process_case(vol_path, seg_path, visualize)

    def split_dataset(self, train_ratio=0.8, stratify_by_small_tumor=True, seed=42):
        """
        划分训练集和验证集

        参数:
            train_ratio: 训练集比例
            stratify_by_small_tumor: 是否按小型肿瘤存在与否进行分层
            seed: 随机种子
        """
        import json
        import random
        random.seed(seed)

        # 读取所有病例元数据
        case_metadata = {}
        for meta_file in (self.output_dir / "metadata").glob("*_metadata.json"):
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                case_id = metadata['case_info']['case_id']
                case_metadata[case_id] = metadata

        # 获取所有案例ID
        all_cases = list(case_metadata.keys())

        if stratify_by_small_tumor:
            # 根据是否包含小型肿瘤进行分层
            cases_with_small = []
            cases_without_small = []

            for case_id, metadata in case_metadata.items():
                if metadata['case_info']['num_small_tumors'] > 0:
                    cases_with_small.append(case_id)
                else:
                    cases_without_small.append(case_id)

            # 分别对两组进行采样
            train_with_small = random.sample(
                cases_with_small,
                int(len(cases_with_small) * train_ratio)
            )
            train_without_small = random.sample(
                cases_without_small,
                int(len(cases_without_small) * train_ratio)
            )

            # 合并训练集
            train_cases = train_with_small + train_without_small

            # 验证集是剩余的案例
            val_cases = [c for c in all_cases if c not in train_cases]
        else:
            # 直接随机划分
            train_size = int(len(all_cases) * train_ratio)
            train_cases = random.sample(all_cases, train_size)
            val_cases = [c for c in all_cases if c not in train_cases]

        # 计算每个集合中的切片数量
        train_slices = []
        val_slices = []

        for case_id, metadata in case_metadata.items():
            slice_files = [s['filename'] for s in metadata['slice_info']]
            if case_id in train_cases:
                train_slices.extend(slice_files)
            else:
                val_slices.extend(slice_files)

        # 保存划分结果
        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        with open(splits_dir / "train_cases.txt", 'w') as f:
            f.write('\n'.join(train_cases))

        with open(splits_dir / "val_cases.txt", 'w') as f:
            f.write('\n'.join(val_cases))

        with open(splits_dir / "train_slices.txt", 'w') as f:
            f.write('\n'.join(train_slices))

        with open(splits_dir / "val_slices.txt", 'w') as f:
            f.write('\n'.join(val_slices))

        # 生成划分摘要
        splits_summary = {
            'total_cases': len(all_cases),
            'train_cases': len(train_cases),
            'val_cases': len(val_cases),
            'train_slices': len(train_slices),
            'val_slices': len(val_slices),
            'train_ratio': train_ratio,
            'stratify_by_small_tumor': stratify_by_small_tumor,
            'seed': seed
        }

        with open(splits_dir / "splits_summary.json", 'w') as f:
            json.dump(splits_summary, f, indent=2)

        print(f"数据集划分完成:")
        print(f"  训练集: {len(train_cases)} 案例, {len(train_slices)} 切片")
        print(f"  验证集: {len(val_cases)} 案例, {len(val_slices)} 切片")

        return splits_summary




def main():
    # 设置数据路径
    data_dir_b1 = r"data/raw/Training_Batch1"
    data_dir_b2 = r"data/raw/Training_Batch2"
    output_dir = r"data/preprocessed"

    # 创建预处理器
    preprocessor = LiTSPreprocessor(
        data_dir_b1=data_dir_b1,
        data_dir_b2=data_dir_b2,
        output_dir=output_dir,
        window_min=-100,
        window_max=400,
        small_tumor_diameter=10,
        slice_thickness=None  # 设为None保留原始厚度，或指定统一厚度
    )

    # 预处理数据集
    # 使用多进程加速，并对5个样本进行可视化
    preprocessor.preprocess_dataset(num_workers=4, visualize_samples=5)

    # 划分数据集
    preprocessor.split_dataset(train_ratio=0.8, stratify_by_small_tumor=True)

    print("数据预处理完成!")


if __name__ == "__main__":
    main()