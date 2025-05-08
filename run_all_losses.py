# run_all_losses.py
import os
import subprocess

loss_modes = {
    'ce': 'CrossEntropyLoss (original)',
    'balanced': 'Balanced Softmax Loss',
    'weighted': 'Weighted CrossEntropyLoss'
}

for loss_mode, description in loss_modes.items():
    print(f"\n=== Running experiment: {description} (loss_mode = '{loss_mode}') ===")
    
    # Set environment variable if needed (optional, can be omitted)
    os.environ['LOSS_MODE'] = loss_mode

    # Build and run command
    command = f'python train_networks.py --loss_mode {loss_mode}'
    exit_code = subprocess.call(command, shell=True)

    if exit_code != 0:
        print(f"[Warning] Experiment failed: {loss_mode}")
    else:
        print(f"[Success] Training finished for: {description}")
