import torch
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_longtail(dataset, mu=0.01):
    """Create long-tailed version of the dataset"""
    np.random.seed(42)
    labels = np.array(dataset.targets)
    classes = np.unique(labels)
    num_classes = len(classes)
    
    # 获取每个类别的样本数量
    class_counts = {cls: (labels == cls).sum() for cls in classes}
    
    # 计算每个类别的目标样本数量（长尾分布）
    max_count = max(class_counts.values())  # 最大类别的样本数量
    target_counts = [int(max_count * (mu ** (i / (num_classes - 1)))) for i in range(num_classes)]
    print("Target counts per class:", target_counts)  # 打印每个类别的目标样本数量
    
    selected_indices = []
    for cls, target_count in zip(classes, target_counts):
        indices = np.where(labels == cls)[0]
        np.random.shuffle(indices)
        selected = indices[:min(target_count, len(indices))]  # 确保不超过该类别的样本数量
        selected_indices.extend(selected)
    
    return dataset, selected_indices

def save_longtail(train_dataset, test_dataset, train_selected_indices, test_selected_indices, output_dir, dataset_name="cifar10"):
    """Save the longtail dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the long-tail version of CIFAR-10
    longtail_train_data = torch.utils.data.Subset(train_dataset, train_selected_indices)
    longtail_test_data = torch.utils.data.Subset(test_dataset, test_selected_indices)

    # Saving train and test datasets
    torch.save(longtail_train_data, os.path.join(output_dir, f"{dataset_name}_longtail_train.pth"))
    torch.save(longtail_test_data, os.path.join(output_dir, f"{dataset_name}_longtail_test.pth"))
    
    print(f"Long-tail train dataset saved to {output_dir}/{dataset_name}_longtail_train.pth")
    print(f"Long-tail test dataset saved to {output_dir}/{dataset_name}_longtail_test.pth")

def main():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create long-tail dataset for both train and test sets
    longtail_train_dataset, train_selected_indices = create_longtail(cifar10_train, mu=0.1)
    longtail_test_dataset, test_selected_indices = create_longtail(cifar10_test, mu=0.1)
    
    # Save the long-tail version
    save_longtail(longtail_train_dataset, longtail_test_dataset, train_selected_indices, test_selected_indices, 'data/longtail/cifar-10')

if __name__ == "__main__":
    main()
