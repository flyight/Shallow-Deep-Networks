# early_exit_experiments.py
import torch
import numpy as np
import os
import csv
from collections import Counter
import matplotlib.pyplot as plt

import aux_funcs as af
import model_funcs as mf
import network_architectures as arcs
from profiler import profile_sdn

# category_thresholds = {
#     0: 0.8, 1: 0.7, 2: 0.75, 3: 0.75, 4: 0.85,
#     5: 0.85, 6: 0.8, 7: 0.8, 8: 0.75, 9: 0.5
# }

category_thresholds = {i: 0.99 for i in range(10)}


def convert_num_early_exits_to_cumulative(ic_exits, total_samples):
    layer_cumul_dist = [0]
    running_total = 0
    for val in ic_exits:
        running_total += val
        layer_cumul_dist.append(running_total)
    layer_cumul_dist[-1] = total_samples
    return [v / total_samples for v in layer_cumul_dist]

def get_plot_data_and_auc(layer_cumul_dist, ic_costs):
    layers = sorted(ic_costs.keys())
    c_i = {layer: ic_costs[layer] / ic_costs[layers[-1]] for layer in layers}
    c_i = [c_i[layer] for layer in layers]
    c_i.insert(0, 0)
    return [c_i, layer_cumul_dist], np.trapz(layer_cumul_dist, x=c_i)

def early_exit_experiments(models_path, task, device='cpu'):
    results = []
    for sdn_name in os.listdir(models_path):
        if 'sdn' not in sdn_name:
            continue

        full_path = os.path.join(models_path, sdn_name)
        print(f"[INFO] Testing {full_path}")

        if not os.path.exists(os.path.join(full_path, 'parameters_last')):
            print(f"[WARN] parameters_last not found for {sdn_name}, skipping...")
            continue

        sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
        sdn_model.to(device)

        dataset = af.get_dataset(task)
        one_batch_dataset = af.get_dataset(task, 1)

        print(f"Task: {task}")
        print("Get SDN early exit results")
        total_ops, total_params = profile_sdn(sdn_model, sdn_model.input_size, device)

        sdn_model.forward = sdn_model.early_exit

        for class_id, threshold in category_thresholds.items():
            print(f"Using threshold for class {class_id}: {threshold}")
            sdn_model.confidence_threshold = threshold

            try:
                top1_test, top5_test, early_exit_counts, total_time, _ = mf.sdn_test_class_count(
                    sdn_model, one_batch_dataset.test_loader, device
                )
            except ValueError as e:
                print(f"Error during testing with threshold {threshold}: {e}")
                continue

            total_exit_counts = [sum(col) for col in zip(*early_exit_counts)]
            per_class_number = [sum(col) for col in early_exit_counts]

            total_cumul = convert_num_early_exits_to_cumulative(total_exit_counts, sum(per_class_number))
            _, total_auc = get_plot_data_and_auc(total_cumul, total_ops)

            results.append({
                'model': sdn_name,
                'class_id': class_id,
                'threshold': threshold,
                'top1_acc': top1_test,
                'top5_acc': top5_test,
                'total_efficiency_auc': total_auc,
                'inference_time': total_time
            })

    csv_path = os.path.join(models_path, f"{task.replace('/', '_')}_efficiency_results.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Results saved to {csv_path}")

def main():
    torch.manual_seed(af.get_random_seed())
    np.random.seed(af.get_random_seed())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    model_groups = {
        "cifar10_normal": "networks/1221",
        "cifar10_augmented": "networks/1221_normal_loss_agumented",
        "longtail_normal": "networks/1221/longtail"
    }

    for group_name, path in model_groups.items():
        print(f"\n===== Testing group: {group_name} =====")
        # task = "cifar10" if "cifar10" in group_name and "longtail" not in group_name else "longtail/cifar10"
        task = "longtail/cifar10"

        early_exit_experiments(path, task=task, device=device)

if __name__ == '__main__':
    main()
