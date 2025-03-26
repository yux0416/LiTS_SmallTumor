# 肝脏小型肿瘤检测研究项目

## 项目概述
本项目旨在解决医学图像分析中小型肝脏肿瘤（直径<10mm）的自动检测挑战。通过使用注意力机制增强的卷积神经网络，提高对小型肿瘤的检测能力。

## 关键特性
- 多种注意力机制的实现与比较
- 针对小型肿瘤的特殊损失函数设计
- 基于尺寸的层次化评估指标
- 多种U-Net架构变体

## 项目结构
LiTS_SmallTumor/
├─ data/                        # 数据目录(不包含在仓库中)
├─ notebooks/                   # 实验分析笔记本
├─ results/                     # 实验结果(不包含在仓库中)
├─ src/                         # 源代码
│  ├─ data/                     # 数据处理代码
│  ├─ models/                   # 模型定义
│  │  ├─ attention_modules/     # 注意力模块实现
│  │  └─ ...
│  ├─ train/                    # 训练脚本
│  └─ utils/                    # 工具函数

## 数据集
本项目使用LiTS (Liver Tumor Segmentation)数据集。由于数据集大小限制，未包含在代码仓库中。请从[官方网站](https://competitions.codalab.org/competitions/17094)下载。

## 安装与使用
### 环境配置
```bash
# 克隆仓库
git clone https://github.com/yux0416/LiTS_SmallTumor.git
cd LiTS_SmallTumor

# 安装依赖
pip install -r requirements.txt

# 数据预处理
python preprocess_lits.py

# 训练模型
python src/train/train_attention.py --model_type attention --attention_type tuned_local_contrast

# 评估模型
python src/train/evaluate_models.py --models_dir results/models/attention

主要发现

针对小型肿瘤优化的注意力机制显著提高了检测性能
局部对比度增强注意力在小型肿瘤检测中表现最佳
小型肿瘤检测的F1分数较基线模型提升了。。。？

