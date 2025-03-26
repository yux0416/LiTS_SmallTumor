# analyze_small_tumor.py
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_metrics(models_dirs):
    """
    加载所有模型的评估指标，专注于小型肿瘤性能

    参数:
        models_dirs: 模型目录列表
    """
    small_tumor_metrics = {}

    # 确保models_dirs是列表
    if isinstance(models_dirs, str):
        models_dirs = [models_dirs]

    # 遍历所有目录
    for base_dir in models_dirs:
        base_dir = Path(base_dir)
        print(f"查找目录: {base_dir}")

        # 如果目录不存在，跳过
        if not base_dir.exists():
            print(f"目录不存在: {base_dir}")
            continue

        # 遍历目录中的所有子目录
        for model_dir in base_dir.iterdir():
            if not model_dir.is_dir():
                continue

            metrics_file = model_dir / 'evaluation_metrics.json'
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)

                    # 提取模型名称
                    model_name = model_dir.name

                    # 提取小型肿瘤指标
                    if 'size_metrics' in metrics and 'small' in metrics['size_metrics']:
                        small_tumor_data = metrics['size_metrics']['small']
                        # 添加总体肿瘤Dice和整体Dice作为参考
                        if 'class_dice' in metrics and '2' in metrics['class_dice']:
                            small_tumor_data['tumor_dice'] = metrics['class_dice']['2']
                        small_tumor_data['overall_dice'] = metrics.get('dice', 0)

                        # 简化模型名称（去除时间戳）
                        simplified_name = model_name.split('_')[0]
                        if len(model_name.split('_')) > 1:
                            # 如果有注意力类型，加上它
                            attention_type = model_name.split('_')[1]
                            if attention_type not in ['20', '202']:  # 排除时间戳部分
                                simplified_name += f"_{attention_type}"

                        small_tumor_metrics[simplified_name] = small_tumor_data
                        print(f"已加载模型 {simplified_name} 的小型肿瘤评估指标")
                except Exception as e:
                    print(f"加载模型 {model_dir.name} 的评估指标时出错: {e}")

    return small_tumor_metrics


def create_small_tumor_comparison(metrics):
    """
    创建小型肿瘤性能比较表
    """
    comparison_data = []

    for model_name, data in metrics.items():
        comparison_data.append({
            'Model': model_name,
            'Recall': data.get('recall', 0),
            'Precision': data.get('precision', 0),
            'F1 Score': data.get('f1', 0),
            'Tumor Dice': data.get('tumor_dice', 0),
            'Overall Dice': data.get('overall_dice', 0)
        })

    # 创建DataFrame
    df = pd.DataFrame(comparison_data)

    # 对模型性能进行排序，按F1分数降序
    df = df.sort_values(by='F1 Score', ascending=False)

    return df


def plot_small_tumor_metrics(df, save_path=None):
    """
    绘制小型肿瘤检测指标比较图
    """
    plt.figure(figsize=(14, 8))

    # 提取模型名称和检测指标
    models = df['Model']
    recall = df['Recall']
    precision = df['Precision']
    f1 = df['F1 Score']

    # 设置柱状图
    x = np.arange(len(models))
    width = 0.25

    # 绘制柱状图
    plt.bar(x - width, recall, width, label='Recall')
    plt.bar(x, precision, width, label='Precision')
    plt.bar(x + width, f1, width, label='F1 Score')

    # 增加文本标签
    for i, v in enumerate(f1):
        plt.text(i + width / 2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Small Tumor (<10mm) Detection Performance')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(max(recall), max(precision), max(f1)) * 1.15)  # 设置y轴上限，留出空间显示标签
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"小型肿瘤检测指标比较图已保存到 {save_path}")

    plt.show()


def plot_radar_chart(df, save_path=None):
    """
    绘制雷达图比较不同模型在各方面的性能
    """
    # 选择前5个最佳模型和基线模型
    if len(df) > 6:
        if 'standard' in df['Model'].values:
            # 获取基线模型
            baseline = df[df['Model'] == 'standard']
            # 获取前5个非基线模型
            top_models = df[df['Model'] != 'standard'].head(5)
            # 合并
            radar_df = pd.concat([top_models, baseline])
        else:
            radar_df = df.head(6)
    else:
        radar_df = df

    # 准备雷达图数据
    categories = ['Recall', 'Precision', 'F1 Score', 'Tumor Dice', 'Overall Dice']
    N = len(categories)

    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合

    # 创建图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 绘制外部圆和标签
    plt.xticks(angles[:-1], categories, size=12)

    # 绘制y轴标签
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], size=10)
    plt.ylim(0, 1)

    # 为每个模型绘制雷达图
    for i, (idx, row) in enumerate(radar_df.iterrows()):
        model_name = row['Model']
        values = row[categories].values.flatten().tolist()
        values += values[:1]  # 闭合

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)

    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance Comparison (Radar Chart)', size=15, y=1.1)

    if save_path:
        plt.savefig(save_path)
        print(f"性能雷达图已保存到 {save_path}")

    plt.show()


def analyze_improvement(df):
    """
    分析注意力模型相对于基线模型的改进
    """
    # 检查是否有基线模型
    if 'standard' not in df['Model'].values:
        print("未找到基线模型 (standard)，无法计算改进")
        return None

    # 获取基线模型
    baseline = df[df['Model'] == 'standard'].iloc[0]

    # 计算每个模型相对于基线的改进
    improvements = []

    for idx, row in df.iterrows():
        if row['Model'] == 'standard':
            continue

        model_name = row['Model']
        f1_improvement = (row['F1 Score'] - baseline['F1 Score']) / baseline['F1 Score'] * 100
        recall_improvement = (row['Recall'] - baseline['Recall']) / baseline['Recall'] * 100
        precision_improvement = (row['Precision'] - baseline['Precision']) / baseline['Precision'] * 100

        improvements.append({
            'Model': model_name,
            'F1 Improvement (%)': f1_improvement,
            'Recall Improvement (%)': recall_improvement,
            'Precision Improvement (%)': precision_improvement
        })

    # 创建改进DataFrame
    improvements_df = pd.DataFrame(improvements)

    # 按F1改进排序
    improvements_df = improvements_df.sort_values(by='F1 Improvement (%)', ascending=False)

    return improvements_df


def plot_improvements(improvements_df, save_path=None):
    """
    绘制相对于基线模型的改进图
    """
    if improvements_df is None or len(improvements_df) == 0:
        return

    plt.figure(figsize=(14, 8))

    # 提取模型名称和改进数据
    models = improvements_df['Model']
    f1_improvement = improvements_df['F1 Improvement (%)']
    recall_improvement = improvements_df['Recall Improvement (%)']
    precision_improvement = improvements_df['Precision Improvement (%)']

    # 设置柱状图
    x = np.arange(len(models))
    width = 0.25

    # 绘制柱状图
    plt.bar(x - width, f1_improvement, width, label='F1 Score')
    plt.bar(x, recall_improvement, width, label='Recall')
    plt.bar(x + width, precision_improvement, width, label='Precision')

    # 增加文本标签
    for i, v in enumerate(f1_improvement):
        plt.text(i - width, v + (5 if v > 0 else -10), f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Model')
    plt.ylabel('Improvement over Baseline (%)')
    plt.title('Small Tumor Detection Improvement Compared to Baseline')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 添加零线
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"改进比较图已保存到 {save_path}")

    plt.show()


def main(args):
    # 加载模型评估指标
    models_dirs = args.models_dirs.split(',')
    print(f"将分析以下目录的模型: {models_dirs}")

    small_tumor_metrics = load_metrics(models_dirs)

    if not small_tumor_metrics:
        print("未找到任何模型的小型肿瘤评估指标")
        return

    # 创建比较表
    comparison_df = create_small_tumor_comparison(small_tumor_metrics)

    # 保存比较表
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_dir / 'small_tumor_comparison.csv', index=False)
    print(f"小型肿瘤性能比较表已保存到 {output_dir / 'small_tumor_comparison.csv'}")

    # 打印比较表
    pd.set_option('display.max_columns', None)
    print("\n小型肿瘤检测性能比较表:")
    print(comparison_df)

    # 绘制小型肿瘤检测指标比较图
    plot_small_tumor_metrics(comparison_df, output_dir / 'small_tumor_metrics.png')

    # 绘制雷达图
    plot_radar_chart(comparison_df, output_dir / 'radar_chart.png')

    # 分析相对于基线模型的改进
    improvements_df = analyze_improvement(comparison_df)

    if improvements_df is not None:
        # 保存改进表
        improvements_df.to_csv(output_dir / 'improvement_over_baseline.csv', index=False)
        print(f"改进数据已保存到 {output_dir / 'improvement_over_baseline.csv'}")

        # 打印改进表
        print("\n相对于基线模型的改进:")
        print(improvements_df)

        # 绘制改进比较图
        plot_improvements(improvements_df, output_dir / 'improvements.png')

    # 输出最佳模型
    best_model = comparison_df.iloc[0]
    print("\n最佳小型肿瘤检测模型:")
    print(f"模型: {best_model['Model']}")
    print(f"F1 分数: {best_model['F1 Score']:.4f}")
    print(f"召回率: {best_model['Recall']:.4f}")
    print(f"精确率: {best_model['Precision']:.4f}")

    # 如果有基线模型，计算改进
    if 'standard' in comparison_df['Model'].values:
        baseline = comparison_df[comparison_df['Model'] == 'standard'].iloc[0]
        f1_improvement = (best_model['F1 Score'] - baseline['F1 Score']) / baseline['F1 Score'] * 100
        print(f"相对于基线模型的F1分数提升: {f1_improvement:.2f}%")

    # 保存分析摘要
    with open(output_dir / 'small_tumor_analysis_summary.txt', 'w') as f:
        f.write("小型肿瘤检测性能分析摘要\n")
        f.write("=========================\n\n")

        f.write("最佳模型:\n")
        f.write(f"- 模型: {best_model['Model']}\n")
        f.write(f"- F1 分数: {best_model['F1 Score']:.4f}\n")
        f.write(f"- 召回率: {best_model['Recall']:.4f}\n")
        f.write(f"- 精确率: {best_model['Precision']:.4f}\n")
        f.write(f"- 肿瘤 Dice: {best_model['Tumor Dice']:.4f}\n")
        f.write(f"- 整体 Dice: {best_model['Overall Dice']:.4f}\n\n")

        # 如果有基线模型，添加与基线的比较
        if 'standard' in comparison_df['Model'].values:
            baseline = comparison_df[comparison_df['Model'] == 'standard'].iloc[0]
            f1_improvement = (best_model['F1 Score'] - baseline['F1 Score']) / baseline['F1 Score'] * 100
            recall_improvement = (best_model['Recall'] - baseline['Recall']) / baseline['Recall'] * 100
            precision_improvement = (best_model['Precision'] - baseline['Precision']) / baseline['Precision'] * 100

            f.write("与基线模型比较:\n")
            f.write(f"- F1 分数提升: {f1_improvement:.2f}%\n")
            f.write(f"- 召回率提升: {recall_improvement:.2f}%\n")
            f.write(f"- 精确率提升: {precision_improvement:.2f}%\n\n")

        # 添加所有模型的排名
        f.write("模型排名 (按小型肿瘤F1分数):\n")
        for i, (idx, row) in enumerate(comparison_df.iterrows()):
            f.write(
                f"{i + 1}. {row['Model']}: F1={row['F1 Score']:.4f}, 召回率={row['Recall']:.4f}, 精确率={row['Precision']:.4f}\n")

        f.write("\n关键发现与建议:\n")

        # 分析最有效的注意力类型
        attention_types = []
        for model in comparison_df['Model']:
            if '_' in model:
                parts = model.split('_')
                if len(parts) > 1 and parts[1] not in ['20', '202']:
                    attention_types.append(parts[1])

        if attention_types:
            most_common = max(set(attention_types), key=attention_types.count)
            f.write(f"1. 最有效的注意力类型是 '{most_common}', 多个模型中表现优秀\n")

        # 分析小型肿瘤检测中的关键因素
        f.write(f"2. 小型肿瘤检测受益于专门针对性设计，如 'small_tumor' 模型架构\n")

        # 分析精确率与召回率的平衡
        recall_precision_diff = abs(best_model['Recall'] - best_model['Precision'])
        if recall_precision_diff > 0.1:
            if best_model['Recall'] > best_model['Precision']:
                f.write(f"3. 最佳模型的召回率显著高于精确率，表明该模型倾向于检测更多候选区域，但可能产生更多假阳性\n")
            else:
                f.write(f"3. 最佳模型的精确率显著高于召回率，表明该模型更保守，可能会漏检一些小型肿瘤\n")
        else:
            f.write(f"3. 最佳模型在精确率和召回率之间取得了良好平衡\n")

        # 推荐下一步工作
        f.write("\n推荐下一步工作:\n")
        f.write(f"1. 对最佳模型 '{best_model['Model']}' 进行进一步优化，可尝试调整其注意力机制参数\n")
        f.write(f"2. 探索组合多种注意力机制的可能性，特别是将表现最好的几种注意力类型融合\n")
        f.write(f"3. 对精确率和召回率进行平衡调整，根据实际临床需求确定最佳权衡点\n")
        f.write(f"4. 考虑针对小型肿瘤设计更专门化的损失函数，进一步提高检测性能\n")

    print(f"分析摘要已保存到 {output_dir / 'small_tumor_analysis_summary.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析小型肿瘤检测性能')
    parser.add_argument('--models_dirs', type=str,
                        default='results/models/attention,results/models/baseline',
                        help='模型目录列表，用逗号分隔')
    parser.add_argument('--output_dir', type=str,
                        default='results/analysis/small_tumor',
                        help='输出目录')

    args = parser.parse_args()
    main(args)