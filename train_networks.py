import torch
import aux_funcs as af
import network_architectures as arcs
from architectures.CNNs.VGG import VGG
import os


def train_cifar10(models_root, device='cuda', loss_mode='ce'):

    print('Training CIFAR-10 models...')

    task = 'longtail/cifar10'  # 使用长尾数据集

    model_tag = f'{task.replace("/", "_")}_{loss_mode}'
    models_path = os.path.join(models_root, model_tag)
    af.create_path(models_path)

    cnns = []
    sdns = []

    af.extend_lists(cnns, sdns, arcs.create_vgg16bn(models_path, task, save_type='cd'))

    train(models_path, cnns, sdn=False, device=device, loss_mode=loss_mode)
    train_sdns(models_path, sdns, device=device, loss_mode=loss_mode)

def train(models_path, untrained_models, sdn=False, device='cuda', loss_mode='ce'):
    print('Training models...')

    for base_model in untrained_models:
        trained_model, model_params = arcs.load_model(models_path, base_model, 0)
        print(f"Preparing to load dataset for task: {model_params['task']}")
        dataset = af.get_dataset(model_params['task'])
        print(f"Task: {model_params['task']}")
        print(f"Number of training samples: {len(dataset.train_loader.dataset)}")
        print(f"Number of testing samples: {len(dataset.test_loader.dataset)}")

        # 配置参数写入 model_params
        model_params['optimizer'] = 'AdamW'
        model_params['scheduler'] = 'CosineAnnealingLR'
        model_params['loss_mode'] = loss_mode
        model_params['amsgrad'] = False
        model_params['betas'] = (0.9, 0.999)
        model_params['eps'] = 1e-8
        model_params['T_max'] = model_params.get('epochs', 100)

        num_epochs = model_params['epochs']

        # 使用 af 封装的 AdamW 优化器创建函数
        optimizer = af.get_adamw_optimizer(trained_model, model_params)

        # 使用 CosineAnnealingLR 调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=model_params['T_max']
        )

        # 将 loss_mode 信息附加到模型供训练函数使用
        trained_model.train_func_loss_mode = loss_mode

        trained_model_name = base_model + '_sdn_training' if sdn else base_model

        print('Training: {}...'.format(trained_model_name))
        trained_model.to(device)

        # 调用模型的训练函数
        metrics = trained_model.train_func(
            trained_model, dataset, num_epochs, optimizer, scheduler, device=device
        )

        # 保存训练结果
        model_params['train_top1_acc'] = metrics['train_top1_acc']
        model_params['test_top1_acc'] = metrics['test_top1_acc']
        model_params['train_top5_acc'] = metrics['train_top5_acc']
        model_params['test_top5_acc'] = metrics['test_top5_acc']
        model_params['epoch_times'] = metrics['epoch_times']
        model_params['lrs'] = metrics['lrs']
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time))

        arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)

def train_sdns(models_path, networks, device='cuda', loss_mode='ce'):
    for sdn_name in networks:
        cnn_to_tune = sdn_name.replace('sdn', 'cnn')
        sdn_params = arcs.load_params(models_path, sdn_name)
        sdn_params = arcs.get_net_params(sdn_params['network_type'], sdn_params['task'])
        sdn_model, _ = af.cnn_to_sdn(models_path, cnn_to_tune, sdn_params)
        arcs.save_model(sdn_model, sdn_params, models_path, sdn_name, epoch=0)

    train(models_path, networks, sdn=True, device=device, loss_mode=loss_mode)





def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_mode', type=str, default='ce', choices=['ce', 'weighted', 'balanced'])
    args = parser.parse_args()
    loss_mode = args.loss_mode

    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()

    models_root = 'networks'
    af.create_path(models_root)
    af.set_logger(f'outputs/train_models_{random_seed}.log')
    print("CUDA Available:", torch.cuda.is_available(), "| Device:", device)

    train_cifar10(models_root, device, loss_mode)


if __name__ == '__main__':
    main()
