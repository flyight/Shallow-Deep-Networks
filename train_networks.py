# train_networks.py
# For training CNNs and SDNs via IC-only and SDN-training strategies
# It trains and save the resulting models to an output directory specified in the main function

import copy
import torch
import time
import os
import random
import numpy as np
import csv

import aux_funcs  as af
import network_architectures as arcs

from architectures.CNNs.VGG import VGG




def train(models_path, untrained_models, sdn=False, device='cpu'):
    print('Training models...')

    for base_model in untrained_models:
        trained_model, model_params = arcs.load_model(models_path, base_model, 0)
        dataset = af.get_dataset(model_params['task'])

        learning_rate = model_params['learning_rate']
        momentum = model_params['momentum']
        weight_decay = model_params['weight_decay']
        milestones = model_params['milestones']
        gammas = model_params['gammas']
        num_epochs = model_params['epochs']

        model_params['optimizer'] = 'SGD'
        trained_model.augment_training = True  # ✅ 强制使用增强数据

        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
        trained_model_name = base_model + ('_sdn_training' if sdn else '')

        print('Training: {}...'.format(trained_model_name))
        trained_model.to(device)
        metrics = trained_model.train_func(trained_model, dataset, num_epochs, optimizer, scheduler, device=device)

        model_params.update({
            'train_top1_acc': metrics['train_top1_acc'],
            'test_top1_acc': metrics['test_top1_acc'],
            'train_top5_acc': metrics['train_top5_acc'],
            'test_top5_acc': metrics['test_top5_acc'],
            'epoch_times': metrics['epoch_times'],
            'lrs': metrics['lrs'],
            'total_time': sum(metrics['epoch_times'])
        })

        print('Training took {} seconds...'.format(model_params['total_time']))
        arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)

        # Save training log
        log_file = os.path.join(models_path, trained_model_name + '_log.csv')
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Top1 Acc', 'Train Top5 Acc', 'Test Top1 Acc', 'Test Top5 Acc', 'LR', 'Time(s)'])
            for i in range(num_epochs):
                writer.writerow([
                    i + 1,
                    metrics['train_top1_acc'][i],
                    metrics['train_top5_acc'][i],
                    metrics['test_top1_acc'][i],
                    metrics['test_top5_acc'][i],
                    metrics['lrs'][i],
                    metrics['epoch_times'][i]
                ])

def train_sdns(models_path, networks, device='cpu'):
    for sdn_name in networks:
        cnn_to_tune = sdn_name.replace('sdn', 'cnn')
        sdn_params = arcs.load_params(models_path, sdn_name)
        sdn_params = arcs.get_net_params(sdn_params['network_type'], sdn_params['task'])
        sdn_model, _ = af.cnn_to_sdn(models_path, cnn_to_tune, sdn_params, epoch=0)
        arcs.save_model(sdn_model, sdn_params, models_path, sdn_name, epoch=0)

    train(models_path, networks, sdn=True, device=device)

def train_models(models_path, device='cpu'):
    tasks = ['cifar10', "longtail/cifar10"]
    cnns = []
    sdns = []

    for task in tasks:
        af.extend_lists(cnns, sdns, arcs.create_vgg16bn(models_path, task, save_type='cd'))
        af.extend_lists(cnns, sdns, arcs.create_resnet56(models_path, task, save_type='cd'))
        af.extend_lists(cnns, sdns, arcs.create_wideresnet32_4(models_path, task, save_type='cd'))
        af.extend_lists(cnns, sdns, arcs.create_mobilenet(models_path, task, save_type='cd'))

    train(models_path, cnns, sdn=False, device=device)
    train_sdns(models_path, sdns, device=device)  # full SDN training only

# for backdoored models, load a backdoored CNN and convert it to an SDN via IC-only strategy
def sdn_ic_only_backdoored(device):
    params = arcs.create_vgg16bn(None, 'cifar10', None, True)

    path = 'backdoored_models'
    backdoored_cnn_name = 'VGG16_cifar10_backdoored'
    save_sdn_name = 'VGG16_cifar10_backdoored_SDN'

    # Use the class VGG
    backdoored_cnn = VGG(params)
    backdoored_cnn.load_state_dict(torch.load('{}/{}'.format(path, backdoored_cnn_name), map_location='cpu'), strict=False)

    # convert backdoored cnn into a sdn
    backdoored_sdn, sdn_params = af.cnn_to_sdn(None, backdoored_cnn, params, preloaded=backdoored_cnn) # load the CNN and convert it to a sdn
    arcs.save_model(backdoored_sdn, sdn_params, path, save_sdn_name, epoch=0) # save the resulting sdn

    networks = [save_sdn_name]

    train(path, networks, sdn=True, ic_only_sdn=True, device=device)

    
def main():
    # ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    print(f"[INFO] Using device: {device}") #打印当前设备

    # 添加标识符到路径中
    loss_identifier = "normal_loss_agumented" 
    models_path = 'networks/{}_{}'.format(af.get_random_seed(), loss_identifier)
    af.create_path(models_path)
    af.set_logger('outputs/train_models_{}'.format(loss_identifier))

    train_models(models_path, device = device)
    # sdn_ic_only_backdoored(device)

if __name__ == '__main__':
    main()