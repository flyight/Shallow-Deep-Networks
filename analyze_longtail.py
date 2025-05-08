# analyze_longtail.py
import os
import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

CONFIG = {
    "longtail_dir": "data/longtail/",
    "output_dir": "outputs/longtail_analysis/"
}

from datasets import Dataset  # 确保导入 Dataset

def load_longtail_dataset(dataset_name):
    """加载生成的长尾数据集"""
    dataset_path = os.path.join(CONFIG["longtail_dir"], dataset_name, "train.parquet")
    metadata_path = os.path.join(CONFIG["longtail_dir"], dataset_name, "metadata.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"长尾数据集 {dataset_name} 不存在于路径: {dataset_path}")
    
    try:
        # 使用 from_parquet 加载 Parquet 格式数据集
        dataset = Dataset.from_parquet(dataset_path)
        with open(metadata_path) as f:
            metadata = json.load(f)
        return dataset, metadata
    except Exception as e:
        raise RuntimeError(f"加载数据集失败: {str(e)}")
def analyze_longtail(dataset, metadata, dataset_name):
    """完整分析长尾数据集"""
    labels = dataset['label']
    counts = Counter(labels)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # 基础统计
    stats = {
        "dataset": dataset_name,
        "total_samples": len(dataset),
        "num_classes": len(counts),
        "max_samples": max(counts.values()),
        "min_samples": min(counts.values()),
        "median_samples": int(np.median(list(counts.values()))),
        "mean_samples": round(np.mean(list(counts.values())), 1),
        "imbalance_ratio": round(metadata["imbalance_ratio"], 1),
        "generation_params": metadata["parameters"]
    }
    
    # 类别分布可视化
    class_names = dataset.features['label'].names if 'label' in dataset.features else None
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_counts)), [c[1] for c in sorted_counts], color='salmon')
    plt.xticks([])  # 隐藏类别标签（通常较多）
    plt.title(f"Longtail Distribution - {dataset_name}\nImbalance Ratio: {stats['imbalance_ratio']}")
    plt.xlabel("Classes (Sorted by Sample Count)")
    plt.ylabel("Sample Count")
    plt.grid(axis='y', linestyle='--')
    plt.savefig(os.path.join(CONFIG["output_dir"], f"{dataset_name}_longtail_dist.png"))
    plt.close()
    
    return stats

def generate_report(stats_list):
    """生成分析报告"""
    report_path = os.path.join(CONFIG["output_dir"], "analysis_report.md")
    
    with open(report_path, "w") as f:
        f.write("# 长尾数据集分析报告\n\n")
        for stats in stats_list:
            f.write(f"## {stats['dataset']}\n")
            f.write(f"- **总样本量**: {stats['total_samples']}\n")
            f.write(f"- **类别数量**: {stats['num_classes']}\n")
            f.write(f"- **最大类别样本量**: {stats['max_samples']}\n")
            f.write(f"- **最小类别样本量**: {stats['min_samples']}\n")
            f.write(f"- **样本量中位数**: {stats['median_samples']}\n")
            f.write(f"- **样本量平均值**: {stats['mean_samples']}\n")
            f.write(f"- **不平衡系数**: {stats['imbalance_ratio']}\n")
            f.write(f"- **生成参数**: mu={stats['generation_params']['mu']}, seed={stats['generation_params']['seed']}\n\n")
            f.write(f"![Distribution]({stats['dataset']}_longtail_dist.png)\n\n")

def main():
    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 获取所有生成的长尾数据集
    datasets = [d for d in os.listdir(CONFIG["longtail_dir"]) 
               if os.path.isdir(os.path.join(CONFIG["longtail_dir"], d))]
    
    analysis_results = []
    
    for dataset_name in datasets:
        print(f"\n正在分析: {dataset_name}...")
        try:
            # 加载数据集和元数据
            dataset, metadata = load_longtail_dataset(dataset_name)
            
            # 执行分析
            stats = analyze_longtail(dataset, metadata, dataset_name)
            analysis_results.append(stats)
            
            # 打印关键指标
            print(f"分析结果:")
            print(f"  最大样本量: {stats['max_samples']}")
            print(f"  最小样本量: {stats['min_samples']}")
            print(f"  不平衡系数: {stats['imbalance_ratio']}")
            
        except Exception as e:
            print(f"分析失败: {str(e)}")
            continue
    
    # 生成最终报告
    generate_report(analysis_results)
    print(f"\n分析完成！结果已保存至 {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()