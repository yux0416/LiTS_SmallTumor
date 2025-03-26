# compare_models.py
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_metrics(models_dir):
    """
    加载所有模型的评估指标

    参数:
        models_dir: 模型目录

    返回:
        模型评估指标字典 {model_name: metrics}
    """
    models_dir = Path(models_dir)
    model_metrics = {}

    # 遍历模型目录
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        metrics_file = model_dir / 'evaluation_metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                # 提取模型类型和名称
                model_name = model_dir.name

                # 保存指标
                model_metrics[model_name] = metrics
                print(f"已加载模型 {model_name} 的评估指标")
            except Exception as e:
                print(f"加载模型 {model_dir.name} 的评估指标时出错: {e}")

    return model_metrics


def create_comparison_table(model_metrics):
    """
    创建模型比较表

    参数:
        model_metrics: 模型评估指标字典

    返回:
        pandas DataFrame
    """
    # 提取关键指标
    comparison_data = []

    for model_name, metrics in model_metrics.items():
        # 提取总体性能
        overall_dice = metrics.get('dice', 0)

        # 提取各类别Dice
        class_dice = metrics.get('class_dice', {})
        background_dice = class_dice.get('0', 0)
        liver_dice = class_dice.get('1', 0)
        tumor_dice = class_dice.get('2', 0)

        # 提取肿瘤检测指标
        tumor_metrics = metrics.get('tumor_metrics', {})
        tumor_recall = tumor_metrics.get('recall', 0)
        tumor_precision = tumor_metrics.get('precision', 0)
        tumor_f1 = tumor_metrics.get('f1', 0)

        # 提取小型肿瘤性能
        size_metrics = metrics.get('size_metrics', {})
        small_tumor_metrics = size_metrics.get('small', {})
        small_tumor_recall = small_tumor_metrics.get('recall', 0)
        small_tumor_precision = small_tumor_metrics.get('precision', 0)
        small_tumor_f1 = small_tumor_metrics.get('f1', 0)

        # 简化模型名称（去除时间戳）
        simplified_name = model_name.split('_')[0]
        if len(model_name.split('_')) > 1:
            # 如果有注意力类型，加上它
            attention_type = model_name.split('_')[1]
            if attention_type not in ['20', '202']:  # 排除时间戳部分
                simplified_name += f"_{attention_type}"

        # 添加到比较数据
        comparison_data.append({
            'Model': simplified_name,
            'Overall Dice': overall_dice,
            'Background Dice': background_dice,
            'Liver Dice': liver_dice,
            'Tumor Dice': tumor_dice,
            'Tumor Recall': tumor_recall,
            'Tumor Precision': tumor_precision,
            'Tumor F1': tumor_f1,
            'Small Tumor Recall': small_tumor_recall,
            'Small Tumor Precision': small_tumor_precision,
            'Small Tumor F1': small_tumor_f1
        })

    # 创建DataFrame
    df = pd.DataFrame(comparison_data)

    # 对模型性能进行排序，按小型肿瘤的F1分数降序
    df = df.sort_values(by='Small Tumor F1', ascending=False)

    return df


def plot_overall_performance(df, save_path=None):
    """
    绘制总体性能比较图
    """
    plt.figure(figsize=(12, 6))

    # 提取模型名称和Dice系数
    models = df['Model']
    overall_dice = df['Overall Dice']
    liver_dice = df['Liver Dice']
    tumor_dice = df['Tumor Dice']

    # 设置柱状图
    x = np.arange(len(models))
    width = 0.25

    # 绘制柱状图
    plt.bar(x - width, overall_dice, width, label='Overall Dice')
    plt.bar(x, liver_dice, width, label='Liver Dice')
    plt.bar(x + width, tumor_dice, width, label='Tumor Dice')

    plt.xlabel('Model')
    plt.ylabel('Dice Coefficient')
    plt.title('Overall Segmentation Performance')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"总体性能比较图已保存到 {save_path}")

    plt.show()


def plot_tumor_performance(df, save_path=None):
    """
    绘制肿瘤检测性能比较图
    """
    plt.figure(figsize=(12, 6))

    # 提取模型名称和肿瘤检测指标
    models = df['Model']
    tumor_recall = df['Tumor Recall']
    tumor_precision = df['Tumor Precision']
    tumor_f1 = df['Tumor F1']

    # 设置柱状图
    x = np.arange(len(models))
    width = 0.25

    # 绘制柱状图
    plt.bar(x - width, tumor_recall, width, label='Recall')
    plt.bar(x, tumor_precision, width, label='Precision')
    plt.bar(x + width, tumor_f1, width, label='F1 Score')

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Tumor Detection Performance')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"肿瘤检测性能比较图已保存到 {save_path}")

    plt.show()


def plot_small_tumor_performance(df, save_path=None):
    """
    绘制小型肿瘤检测性能比较图
    """
    plt.figure(figsize=(12, 6))

    # 提取模型名称和小型肿瘤检测指标
    models = df['Model']
    small_tumor_recall = df['Small Tumor Recall']
    small_tumor_precision = df['Small Tumor Precision']
    small_tumor_f1 = df['Small Tumor F1']

    # 设置柱状图
    x = np.arange(len(models))
    width = 0.25

    # 绘制柱状图
    plt.bar(x - width, small_tumor_recall, width, label='Recall')
    plt.bar(x, small_tumor_precision, width, label='Precision')
    plt.bar(x + width, small_tumor_f1, width, label='F1 Score')

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Small Tumor Detection Performance')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"小型肿瘤检测性能比较图已保存到 {save_path}")

    plt.show()


def plot_comparative_heatmap(df, metrics, save_path=None):
    """
    绘制性能指标热图

    参数:
        df: 比较数据DataFrame
        metrics: 要包含的指标列表
        save_path: 保存路径
    """
    # 提取指定的指标
    heatmap_data = df[['Model'] + metrics].set_index('Model')

    plt.figure(figsize=(10, len(df) * 0.7))

    # 创建热图
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".3f",
                linewidths=.5, cbar_kws={'label': 'Score'})

    plt.title('Model Performance Comparison')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"性能指标热图已保存到 {save_path}")

    plt.show()


def analyze_best_model(df):
    """
    分析最佳模型
    """
    # 获取小型肿瘤F1分数最高的模型
    best_small_tumor_model = df.loc[df['Small Tumor F1'].idxmax()]

    print("\n最佳小型肿瘤检测模型分析:")
    print(f"模型: {best_small_tumor_model['Model']}")
    print(f"小型肿瘤 F1 分数: {best_small_tumor_model['Small Tumor F1']:.4f}")
    print(f"小型肿瘤召回率: {best_small_tumor_model['Small Tumor Recall']:.4f}")
    print(f"小型肿瘤精确率: {best_small_tumor_model['Small Tumor Precision']:.4f}")

    # 与基线模型比较
    if 'standard' in df['Model'].values:
        baseline_model = df[df['Model'] == 'standard'].iloc[0]

        small_tumor_f1_improvement = (best_small_tumor_model['Small Tumor F1'] - baseline_model['Small Tumor F1']) / \
                                     baseline_model['Small Tumor F1'] * 100

        print("\n与基线模型比较:")
        print(f"小型肿瘤 F1 分数提升: {small_tumor_f1_improvement:.2f}%")

    # 分析最佳模型在各方面的表现
    print("\n最佳模型在各指标上的表现:")
    for metric in ['Overall Dice', 'Liver Dice', 'Tumor Dice', 'Tumor F1']:
        print(f"{metric}: {best_small_tumor_model[metric]:.4f}")

    return best_small_tumor_model


def main(args):
    # 加载模型评估指标
    model_metrics = load_metrics(args.models_dir)

    if not model_metrics:
        print("未找到任何模型评估指标")
        return

    # 创建比较表
    comparison_df = create_comparison_table(model_metrics)

    # 保存比较表
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"模型比较表已保存到 {output_dir / 'model_comparison.csv'}")

    # 打印比较表
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print("\n模型性能比较表:")
    print(comparison_df)

    # 绘制性能比较图
    plot_overall_performance(comparison_df, output_dir / 'overall_performance.png')
    plot_tumor_performance(comparison_df, output_dir / 'tumor_performance.png')
    plot_small_tumor_performance(comparison_df, output_dir / 'small_tumor_performance.png')

    # 绘制性能指标热图
    performance_metrics = [
        'Overall Dice', 'Tumor Dice',
        'Tumor F1', 'Small Tumor F1'
    ]
    plot_comparative_heatmap(comparison_df, performance_metrics,
                             output_dir / 'performance_heatmap.png')

    # 分析最佳模型
    best_model = analyze_best_model(comparison_df)

    # 保存分析结果
    with open(output_dir / 'best_model_analysis.txt', 'w') as f:
        f.write(f"最佳小型肿瘤检测模型: {best_model['Model']}\n")
        f.write(f"小型肿瘤 F1 分数: {best_model['Small Tumor F1']:.4f}\n")
        f.write(f"小型肿瘤召回率: {best_model['Small Tumor Recall']:.4f}\n")
        f.write(f"小型肿瘤精确率: {best_model['Small Tumor Precision']:.4f}\n\n")

        f.write("各指标表现:\n")
        for metric in comparison_df.columns[1:]:
            f.write(f"{metric}: {best_model[metric]:.4f}\n")

    print(f"\n分析结果已保存到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='比较不同模型的性能')
    parser.add_argument('--models_dir', type=str, default='D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\results\\models\\attention',
                        help='模型目录')
    parser.add_argument('--output_dir', type=str, default='D:\\Documents\\NUS\\BN5207\\20250315 - FinalProject\\LiTS_SmallTumor\\results\\analysis',
                        help='输出目录')

    args = parser.parse_args()
    main(args)