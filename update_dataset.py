import torchvision
from torchvision import datasets

# 下载 CIFAR-10 数据集
datasets.CIFAR10(root='./data', train=True, download=True)
datasets.CIFAR10(root='./data', train=False, download=True)

# 下载 CIFAR-100 数据集
datasets.CIFAR100(root='./data', train=True, download=True)
datasets.CIFAR100(root='./data', train=False, download=True)
