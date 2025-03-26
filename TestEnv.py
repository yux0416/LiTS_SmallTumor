# test_env.py - 环境与项目设置验证脚本
import os
import sys
import time
import platform
import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import cv2
import albumentations as A  # 如果已安装


def check_cuda():
    """检查CUDA是否可用并测试基本操作"""
    print("\n" + "=" * 50)
    print("CUDA/GPU状态检查")
    print("=" * 50)

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - 总内存: {props.total_memory / 1024 ** 3:.2f} GB")
            print(f"  - 多处理器数量: {props.multi_processor_count}")

        # 测试GPU计算性能
        print("\n执行简单GPU性能测试...")

        # 测试矩阵乘法
        size = 5000
        start_time = time.time()
        x = torch.randn(size, size, device='cuda')
        y = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize()  # 确保GPU操作完成
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        matrix_mul_time = time.time() - start_time
        print(f"矩阵乘法 ({size}x{size}): {matrix_mul_time:.4f} 秒")

        # 测试神经网络前向传播
        batch_size = 16
        channels = 3
        height = 512
        width = 512

        try:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).cuda()
            model.eval()

            inputs = torch.randn(batch_size, channels, height, width, device='cuda')

            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                outputs = model(inputs)
            torch.cuda.synchronize()

            inference_time = time.time() - start_time
            print(f"ResNet-18前向传播 (批次大小={batch_size}): {inference_time:.4f} 秒")
            print(f"每张图像平均推理时间: {inference_time / batch_size * 1000:.2f} 毫秒")
        except Exception as e:
            print(f"ResNet测试失败: {e}")
    else:
        print("警告: CUDA不可用! 将使用CPU进行训练和推理，速度会显著降低。")


def check_env():
    """检查系统环境和Python库"""
    print("\n" + "=" * 50)
    print("系统和Python环境检查")
    print("=" * 50)

    print(f"操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"当前工作目录: {os.getcwd()}")

    # 检查重要库的版本
    libraries = {
        "NumPy": np.__version__,
        "PyTorch": torch.__version__,
        "SimpleITK": sitk.Version(),
        "Matplotlib": matplotlib.__version__,
        "NiBabel": nib.__version__,
        "Pandas": pd.__version__,
        "OpenCV": cv2.__version__
    }

    # 尝试检查albumentations (如果安装)
    try:
        libraries["Albumentations"] = A.__version__
    except (ImportError, AttributeError):
        libraries["Albumentations"] = "未安装"

    print("\n已安装库版本:")
    for lib, version in libraries.items():
        print(f"- {lib}: {version}")


def check_project_structure():
    """检查项目目录结构"""
    print("\n" + "=" * 50)
    print("项目结构检查")
    print("=" * 50)

    # 项目根目录应该是当前目录的父级目录
    project_root = Path("D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor")

    expected_dirs = [
        "data/raw/Training_Batch1",
        "data/raw/Training_Batch2",
        "data/preprocessed",
        "data/splits",
        "notebooks",
        "results/models",
        "results/visualizations",
        "src/data",
        "src/models",
        "src/utils"
    ]

    print("检查必要目录:")
    all_exist = True
    for dir_path in expected_dirs:
        dir_full_path = project_root / dir_path
        exists = dir_full_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}")
        if not exists:
            all_exist = False

    if all_exist:
        print("\n目录结构检查通过!")
    else:
        print("\n警告: 部分目录不存在，请确保创建所有必要的目录!")


def check_data():
    """检查数据文件并加载样本"""
    print("\n" + "=" * 50)
    print("数据检查")
    print("=" * 50)

    project_root = Path("D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor")
    data_dir1 = project_root / "data" / "raw" / "Training_Batch1"
    data_dir2 = project_root / "data" / "raw" / "Training_Batch2"

    # 检查数据集大小
    if data_dir1.exists() and data_dir2.exists():
        volume_files1 = list(data_dir1.glob("volume-*.nii"))
        volume_files2 = list(data_dir2.glob("volume-*.nii"))
        seg_files1 = list(data_dir1.glob("segmentation-*.nii"))
        seg_files2 = list(data_dir2.glob("segmentation-*.nii"))

        print(f"Training_Batch1: {len(volume_files1)} CT体积, {len(seg_files1)} 分割标签")
        print(f"Training_Batch2: {len(volume_files2)} CT体积, {len(seg_files2)} 分割标签")
        print(f"总计: {len(volume_files1) + len(volume_files2)} CT体积, {len(seg_files1) + len(seg_files2)} 分割标签")

        # 尝试加载一个样本文件
        try:
            if volume_files1:
                sample_path = volume_files1[0]
                print(f"\n加载样本CT: {sample_path.name}")
                start_time = time.time()

                # 使用SimpleITK加载
                ct_img = sitk.ReadImage(str(sample_path))
                ct_array = sitk.GetArrayFromImage(ct_img)

                load_time = time.time() - start_time

                print(f"加载时间: {load_time:.2f} 秒")
                print(f"图像形状: {ct_array.shape}")
                print(f"数据类型: {ct_array.dtype}")
                print(f"像素值范围: [{ct_array.min()}, {ct_array.max()}]")
                print(f"体素间距: {ct_img.GetSpacing()}")

                # 如果有对应的分割文件，也加载它
                seg_path = sample_path.parent / sample_path.name.replace("volume", "segmentation")
                if seg_path.exists():
                    seg_img = sitk.ReadImage(str(seg_path))
                    seg_array = sitk.GetArrayFromImage(seg_img)
                    print(f"\n分割标签形状: {seg_array.shape}")
                    unique_labels = np.unique(seg_array)
                    print(f"标签值: {unique_labels}")

                    liver_voxels = np.sum(seg_array == 1)
                    tumor_voxels = np.sum(seg_array == 2)

                    print(f"肝脏体素数量: {liver_voxels}")
                    print(f"肿瘤体素数量: {tumor_voxels}")

                    if tumor_voxels > 0:
                        # 简单统计肿瘤大小信息
                        from scipy import ndimage
                        labeled_tumors, num_tumors = ndimage.label(seg_array == 2)
                        print(f"肿瘤区域数量: {num_tumors}")

                        # 计算每个肿瘤的大小
                        tumor_sizes = []
                        for i in range(1, num_tumors + 1):
                            tumor = (labeled_tumors == i)
                            voxel_count = np.sum(tumor)
                            spacing = seg_img.GetSpacing()
                            volume_mm3 = voxel_count * spacing[0] * spacing[1] * spacing[2]
                            diameter_mm = 2 * ((3 * volume_mm3) / (4 * np.pi)) ** (1 / 3)  # 假设肿瘤近似球形

                            tumor_sizes.append({
                                "id": i,
                                "voxels": voxel_count,
                                "volume_mm3": volume_mm3,
                                "diameter_mm": diameter_mm
                            })

                        # 按大小排序
                        tumor_sizes.sort(key=lambda x: x["volume_mm3"])

                        print("\n肿瘤大小统计(按体积从小到大):")
                        for i, tumor in enumerate(tumor_sizes[:5]):  # 只显示前5个
                            print(
                                f"  肿瘤 {tumor['id']}: {tumor['volume_mm3']:.2f} mm³, 直径: {tumor['diameter_mm']:.2f} mm")

                        # 分析小型肿瘤数量(直径<10mm)
                        small_tumors = [t for t in tumor_sizes if t["diameter_mm"] < 10]
                        print(f"\n小型肿瘤(<10mm)数量: {len(small_tumors)} / {num_tumors}")
            else:
                print("未找到样本CT文件进行加载测试")
        except Exception as e:
            print(f"加载样本文件时出错: {e}")
    else:
        print("数据目录不存在或为空:")
        print(f"Training_Batch1目录存在: {data_dir1.exists()}")
        print(f"Training_Batch2目录存在: {data_dir2.exists()}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("LiTS小型肝脏肿瘤检测项目 - 环境测试")
    print("=" * 50)

    # 运行各项检查
    check_env()
    check_cuda()
    check_project_structure()
    check_data()

    print("\n" + "=" * 50)
    print("环境测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()