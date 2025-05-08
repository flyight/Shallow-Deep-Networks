import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def visualize_class_distribution(dataset, title="Class Distribution"):
    # 检查是否为 Subset 对象
    if isinstance(dataset, torch.utils.data.Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices
        labels = [original_dataset.targets[i] for i in indices]
    else:
        labels = dataset.targets if hasattr(dataset, 'targets') else dataset.labels

    # 计算每个类的数量
    class_counts = Counter(labels)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.show()

# 加载 CIFAR-LT 测试集
testset = torch.load('./data/longtail/cifar-10/cifar10_longtail_test.pth')

# 可视化测试集的类分布
visualize_class_distribution(testset, title="CIFAR-LT Test Set Class Distribution")