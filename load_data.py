# from datasets import load_dataset

# # 使用指定的缓存目录
# ds = load_dataset("zh-plus/tiny-imagenet", cache_dir="./data")


# # ds = load_dataset("wasifis/cifar-10-100", cache_dir="./data")
# # 打印数据集的前几个项，以确保成功加载
# print(ds)

from datasets import load_dataset

# 加载CIFAR-10-100
cifar = load_dataset("wasifis/cifar-10-100", cache_dir="./data")  # 根据你的路径调整
print(cifar['train'].features['label'].names)  # 查看类别名称
print(cifar['train'].shape)  # 查看数据量

# 加载Tiny ImageNet
tiny_imagenet = load_dataset("zh-plus/tiny-imagenet", cache_dir="./data")
print(tiny_imagenet['train'].features)


import matplotlib.pyplot as plt
from collections import Counter

def plot_distribution(dataset, title):
    labels = [sample['label'] for sample in dataset]
    counts = Counter(labels)
    plt.bar(counts.keys(), counts.values())
    plt.title(title)
    plt.show()

plot_distribution(cifar['train'], "CIFAR-10-100 Original Distribution")
plot_distribution(tiny_imagenet['train'], "Tiny ImageNet Original Distribution")

def calc_imbalance_ratio(counts):
    max_count = max(counts.values())
    min_count = min(counts.values())
    return max_count / min_count

cifar_counts = Counter([sample['label'] for sample in cifar['train']])
tiny_counts = Counter([sample['label'] for sample in tiny_imagenet['train']])

print(f"CIFAR imbalance ratio: {calc_imbalance_ratio(cifar_counts)}")
print(f"Tiny ImageNet imbalance ratio: {calc_imbalance_ratio(tiny_counts)}")