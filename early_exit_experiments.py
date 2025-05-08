# early_exit_experiments.py
# runs the experiments in section 5.1 

import torch
import numpy as np

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs

import matplotlib.pyplot as plt
from collections import Counter


import os

from profiler import profile_sdn, profile


# 定义每个类别的阈值
# category_thresholds = {i: 0.99 for i in range(10)}

category_thresholds = {
    0: 0.8,  # 类别0 - 高频类别，较低的阈值
    1: 0.7,
    2: 0.75,
    3: 0.75,
    4: 0.85,  # 类别4 - 低频类别，较高的阈值
    5: 0.85,
    6: 0.8,
    7: 0.8,
    8: 0.75,
    9: 0.5  # 类别9 
}



def visualize_class_distribution(dataset, title="Class Distribution"):
    """可视化数据集的类分布"""
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

def get_plot_data_and_auc(layer_cumul_dist, ic_costs):
    layers = sorted(list(ic_costs.keys()))

    c_i = {layer: ic_costs[layer] / ic_costs[layers[-1]] for layer in layers}
    c_i = [c_i[layer] for layer in layers]
    c_i.insert(0, 0)
    plot_data = [c_i, layer_cumul_dist]

    area_under_curve = np.trapz(layer_cumul_dist, x=c_i)

    return plot_data, area_under_curve
def convert_num_early_exits_at_each_ic_to_cumulative_dis(ic_exits, total_samples):
    num_exits = len(ic_exits)

    layer_cumul_dist = [0]

    running_total = 0
    for cur_exit in range(num_exits):
        running_total += ic_exits[cur_exit]
        layer_cumul_dist.append(running_total)

    layer_cumul_dist[-1] = total_samples
    layer_cumul_dist = [val / total_samples for val in layer_cumul_dist]
    return layer_cumul_dist
def Tsne(models_path, device='cpu'):
    features = []
    labels_list = []
    sdn_training_type = 'ic_only' # IC-only training
    #sdn_training_type = 'sdn_training' # SDN training


    task = 'cifar10'
    #task = 'cifar100'
    #task = 'tinyimagenet'

    #sdn_names = ['vgg16bn_sdn', 'resnet56_sdn', 'wideresnet32_4_sdn', 'mobilenet_sdn']; add_trigger = False
    
    sdn_names = ['vgg16bn_sdn']; add_trigger = False
    sdn_names = [task + '_' + sdn_name + '_' + sdn_training_type for sdn_name in sdn_names]

    
    for sdn_name in sdn_names:
        cnn_name = sdn_name.replace('sdn', 'cnn')
        cnn_name = cnn_name.replace('_ic_only', '')
        cnn_name = cnn_name.replace('_sdn_training', '')

        print(sdn_name)
        print(cnn_name)

        sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
        sdn_model.to(device)

        dataset = af.get_dataset(sdn_params['task'])
        def hook(module, input, output):
            features.append(output)
        for name, module in sdn_model.named_modules():
            if name == 'layers.9.output':
                hook_handle = module.register_forward_hook(hook)
        
        sdn_model.eval()
        with torch.no_grad():
            for batch in dataset.test_loader:
                b_x = batch[0].to(device)
                b_y = batch[1].to(device)
                labels_list.append(b_y.cpu().numpy())
                output = sdn_model(b_x)        
        hook_handle.remove()
        all_labels = np.concatenate(labels_list, axis=0)
        af.visualize_tsne(features, all_labels, perplexity=50, n_iter=3000, title="CIFAR-10 t-SNE Visualization", num_classes=10)


def early_exit_experiments(models_path, device='cpu'):
    # 定义任务和模型
    # task = 'cifar10'
    task = 'longtail/cifar10'
    # sdn_training_type = 'sdn_training'
    # sdn_names = ['vgg16bn_sdn']
    # sdn_names = [task + '_' + sdn_name + '_' + sdn_training_type for sdn_name in sdn_names]

    # for sdn_name in sdn_names:
    #     cnn_name = sdn_name.replace('sdn', 'cnn').replace('_ic_only', '').replace('_sdn_training', '')
    #     sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)

    # 仅筛选包含 'sdn' 的模型路径
    for sdn_name in os.listdir(os.path.join(models_path, 'longtail')):
        if 'sdn' in sdn_name:
            full_path = os.path.join(models_path, 'longtail', sdn_name)
            print(f"[INFO] Testing {full_path}")

            params_path = os.path.join(full_path, 'parameters_last')
            if not os.path.exists(params_path):
                print(f"[WARN] parameters_last not found for {sdn_name}, skipping...")
                continue  # 跳过这个模型

            # 加载 SDN 模型
            sdn_model, sdn_params = arcs.load_model(os.path.join(models_path, 'longtail'), sdn_name, epoch=-1)
            sdn_model.to(device)

            dataset = af.get_dataset(sdn_params['task'])

            print(f"Task in sdn_params before override: {sdn_params['task']}")
            sdn_params['task'] = 'longtail/cifar10'  # 强制覆盖任务名称
            print(f"Task in sdn_params after override: {sdn_params['task']}")

            one_batch_dataset = af.get_dataset(sdn_params['task'], 1)

            print('Get SDN early exit results')
            total_ops, total_params = profile_sdn(sdn_model, sdn_model.input_size, device)
            print("#Ops (GOps): {}".format(total_ops))
            print("#Params (mil): {}".format(total_params))

            # 仅对 SDN 模型应用 early_exit
            sdn_model.forward = sdn_model.early_exit

            # 遍历每个类别的阈值
            for class_id, threshold in category_thresholds.items():
                print(f"Using threshold for class {class_id}: {threshold}")

                # 设置当前类别的置信度阈值
                sdn_model.confidence_threshold = threshold

                # 调用测试函数
                try:
                    top1_test, top5_test, early_exit_counts, total_time, per_acc = mf.sdn_test_class_count(
                        sdn_model, one_batch_dataset.test_loader, device
                    )
                except ValueError as e:
                    print(f"Error during testing with threshold {threshold}: {e}")
                    continue

                # 计算并打印每个类别的早期退出效率
                class_auc = []
                total_exit_counts = [sum(col) for col in zip(*early_exit_counts)]
                per_class_number = [sum(col) for col in early_exit_counts]

                for i in range(10):
                    layer_cumul_dist = convert_num_early_exits_at_each_ic_to_cumulative_dis(early_exit_counts[i], per_class_number[i])
                    plot_data, early_exit_auc = get_plot_data_and_auc(layer_cumul_dist, total_ops)
                    class_auc.append(early_exit_auc)

                print('Per class efficiency:')
                print(class_auc)

                # 计算所有类别的早期退出效率
                total_layer_cumul_dist = convert_num_early_exits_at_each_ic_to_cumulative_dis(total_exit_counts, sum(per_class_number))
                total_plot_data, total_early_exit_auc = get_plot_data_and_auc(total_layer_cumul_dist, total_ops)
                print('Total class efficiency:')
                print(total_early_exit_auc)

                print('Top1 Test accuracy: {}'.format(top1_test))
                print('Top5 Test accuracy: {}'.format(top5_test))
                print('SDN cascading took {} seconds.'.format(total_time))

def main():
    torch.manual_seed(af.get_random_seed())    # reproducible
    np.random.seed(af.get_random_seed())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # 模型路径列表（你可以按需调整）
    model_paths = [
        'networks/longtail_cifar10_balanced',
        'networks/longtail_cifar10_ce',
        'networks/longtail_cifar10_weighted'
    ]

    for path in model_paths:
        print(f"\n===== Testing path: {path} =====")
        early_exit_experiments(path, device)


if __name__ == '__main__':
    main() 