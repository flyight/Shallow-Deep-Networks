# === 文件2: generate_longtail.py (长尾生成) ===
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datasets import load_dataset, Dataset
import os
import json

# 配置区域
CONFIG = {
    "datasets": [
        {"name": "cifar-10-100", "path": "wasifis/cifar-10-100"},
        {"name": "tiny-imagenet", "path": "zh-plus/tiny-imagenet"}
    ],
    "longtail_params": {
        "mu": 0.01,      # 不平衡系数（0-1，越小越不平衡）
        "seed": 42       # 随机种子
    },
    "output_dir": "data/longtail/"
}

def create_longtail(dataset, mu=0.1, seed=42):
    """创建指数衰减长尾数据集"""
    np.random.seed(seed)
    labels = np.array(dataset['label'])
    classes = np.unique(labels)
    num_classes = len(classes)
    
    # 指数衰减采样
    target_counts = [int(len(dataset) * (mu ** (i / (num_classes - 1)))) for i in range(num_classes)]
    
    # 分层采样
    selected_indices = []
    for cls in classes:
        indices = np.where(labels == cls)[0]
        np.random.shuffle(indices)
        selected = indices[:target_counts[cls]]
        selected_indices.extend(selected)
    
    return dataset.select(selected_indices)

def save_longtail(dataset, dataset_name, config):
    """保存长尾数据集"""
    output_path = os.path.join(config["output_dir"], dataset_name)
    os.makedirs(output_path, exist_ok=True)
    
    # 保存数据集
    dataset.to_parquet(os.path.join(output_path, "train.parquet"))
    
    # 保存元数据
    metadata = {
        "dataset": dataset_name,
        "original_samples": len(dataset),
        "longtail_samples": len(dataset),
        "imbalance_ratio": calc_imbalance_ratio(Counter(dataset['label'])),
        "parameters": config["longtail_params"]
    }
    with open(os.path.join(output_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 遍历所有数据集
    for cfg in CONFIG["datasets"]:
        print(f"\n正在处理: {cfg['name']}...")
        try:
            # 加载原始数据集
            dataset = load_dataset(cfg["path"])
            
            # 生成长尾版本
            longtail_dataset = create_longtail(
                dataset['train'],
                mu=CONFIG["longtail_params"]["mu"],
                seed=CONFIG["longtail_params"]["seed"]
            )
            
            # 保存结果
            save_longtail(longtail_dataset, cfg['name'], CONFIG)
            print(f"已保存至: {os.path.join(CONFIG['output_dir'], cfg['name'])}")
            
        except Exception as e:
            print(f"处理数据集 {cfg['name']} 失败: {str(e)}")

if __name__ == "__main__":
    # 复用分析文件的工具函数
    from analyze_dataset import plot_distribution, calc_imbalance_ratio
    
    main()
    print("\n长尾数据集生成完成！")