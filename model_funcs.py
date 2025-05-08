# model_funcs.py
# Implements training, testing for SDNs and CNNs with support for AdamW and cosine LR schedule.

import torch
import math
import copy
import time
import random

import torch.nn as nn
import numpy as np

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from random import choice, shuffle
from collections import Counter

import aux_funcs as af
import data

def sdn_training_step(optimizer, model, coeffs, batch, device, loss_type='ce'):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()
    total_loss = 0.0

    weight = torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32).to(device)

    for ic_id in range(model.num_output - 1):
        cur_output = output[ic_id]

        if loss_type == 'balanced':
            cur_loss = float(coeffs[ic_id]) * af.get_BalancedSoftmaxLoss(weight)(cur_output, b_y)
        elif loss_type == 'weighted':
            cur_loss = float(coeffs[ic_id]) * af.get_weight_loss(weight)(cur_output, b_y)
        else:
            cur_loss = float(coeffs[ic_id]) * af.get_loss_criterion()(cur_output, b_y)

        total_loss += cur_loss

    if loss_type == 'balanced':
        total_loss += af.get_BalancedSoftmaxLoss(weight)(output[-1], b_y)
    elif loss_type == 'weighted':
        total_loss += af.get_weight_loss(weight)(output[-1], b_y)
    else:
        total_loss += af.get_loss_criterion()(output[-1], b_y)

    total_loss.backward()
    optimizer.step()
    return total_loss

def get_loader(data, augment):
    return data.aug_train_loader if augment else data.train_loader

def sdn_train(model, data, epochs, optimizer, scheduler, device='cpu', loss_type='ce'):
    augment = model.augment_training
    metrics = {'epoch_times': [], 'test_top1_acc': [], 'test_top5_acc': [], 'train_top1_acc': [], 'train_top5_acc': [], 'lrs': []}
    max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])

    print('sdn will be trained from scratch...(The SDN training)')

    for epoch in range(1, epochs + 1):
        scheduler.step()
        cur_lr = af.get_lr(optimizer)
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        cur_coeffs = 0.01 + epoch * (max_coeffs / epochs)
        cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
        print('Cur coeffs: {}'.format(cur_coeffs))

        start_time = time.time()
        model.train()
        loader = get_loader(data, augment)
        for i, batch in enumerate(loader):
            total_loss = sdn_training_step(optimizer, model, cur_coeffs, batch, device, loss_type)
            if i % 100 == 0:
                print('Loss: {}: '.format(total_loss))

        top1_test, top5_test = sdn_test(model, data.test_loader, device)
        print('Top1 Test accuracies: {}'.format(top1_test))
        print('Top5 Test accuracies: {}'.format(top5_test))
        end_time = time.time()

        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = sdn_test(model, get_loader(data, augment), device)
        print('Top1 Train accuracies: {}'.format(top1_train))
        print('Top5 Train accuracies: {}'.format(top1_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)

        epoch_time = int(end_time - start_time)
        metrics['epoch_times'].append(epoch_time)
        print('Epoch took {} seconds.'.format(epoch_time))

        metrics['lrs'].append(cur_lr)

    return metrics

def sdn_test(model, loader, device='cpu'):
    model.eval()
    top1 = [data.AverageMeter() for _ in range(model.num_output)]
    top5 = [data.AverageMeter() for _ in range(model.num_output)]

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            for output_id in range(model.num_output):
                cur_output = output[output_id]
                prec1, prec5 = data.accuracy(cur_output, b_y, topk=(1, 5))
                top1[output_id].update(prec1[0], b_x.size(0))
                top5[output_id].update(prec5[0], b_x.size(0))

    top1_accs = [m.avg.data.cpu().numpy()[()] for m in top1]
    top5_accs = [m.avg.data.cpu().numpy()[()] for m in top5]
    return top1_accs, top5_accs

def sdn_get_detailed_results(model, loader, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]
            for output_id in outputs:
                cur_output = output[output_id]
                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]

                pred = cur_output.max(1, keepdim=True)[1]
                is_correct = pred.eq(b_y.view_as(pred))
                for test_id in range(len(b_x)):
                    cur_instance_id = test_id + cur_batch_id*loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, layer_predictions, layer_confidence


def sdn_get_confusion(model, loader, confusion_stats, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    instance_confusion = {}
    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, confusion_stats, device)
            
            for test_id in range(len(b_x)):
                cur_instance_id = test_id + cur_batch_id*loader.batch_size
                instance_confusion[cur_instance_id] = cur_confusion[test_id].cpu().numpy()
                for output_id in outputs:
                    cur_output = output[output_id]
                    pred = cur_output.max(1, keepdim=True)[1]
                    is_correct = pred.eq(b_y.view_as(pred))
                    correct = is_correct[test_id]
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, instance_confusion

# to normalize the confusion scores
def sdn_confusion_stats(model, loader, device='cpu'):
    model.eval()
    outputs = list(range(model.num_output))
    confusion_scores = []

    total_num_instances = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            total_num_instances += len(b_x)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, None, device)
            for test_id in range(len(b_x)):
                confusion_scores.append(cur_confusion[test_id].cpu().numpy())

    confusion_scores = np.array(confusion_scores)
    mean_con = float(np.mean(confusion_scores))
    std_con = float(np.std(confusion_scores))
    return (mean_con, std_con)

def sdn_test_early_exits(model, loader, device='cpu'):
    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output

    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            end_time = time.time()
            total_time+= (end_time - start_time)
            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1

            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, early_output_counts, non_conf_output_counts, total_time
def sdn_test_class_count(model, loader, device='cpu'):
    model.eval()
    correct_per_class = [0] * model.num_classes 
    total_per_class = [0] * model.num_classes
    early_output_counts = [ [0] * model.num_output for _ in range(model.num_classes) ]

    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            end_time = time.time()
            total_time+= (end_time - start_time)
            early_output_counts[b_y.item()][output_id] += 1


            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
            pred = output.argmax(dim=1)  # 获取每个样本的预测类别
            for i in range(b_x.size(0)):
                label = b_y[i].item()
                total_per_class[label] += 1  # 增加该类别的样本总数
                if pred[i] == label:
                    correct_per_class[label] += 1
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]
    accuracy_per_class = [a / b for a, b in zip(correct_per_class, total_per_class)]
    return top1_acc, top5_acc, early_output_counts, total_time, accuracy_per_class

def cnn_training_step(model, optimizer, data, labels, device='cpu'):
    b_x = data.to(device)   # batch x
    b_y = labels.to(device)   # batch y
    output = model(b_x)            # cnn final output
    criterion = af.get_loss_criterion()
    loss = criterion(output, b_y)   # cross entropy loss
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients


def cnn_train(model, data, epochs, optimizer, scheduler, device='cpu'):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

    for epoch in range(1, epochs+1):
        scheduler.step()

        cur_lr = af.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for x, y in train_loader:
            cnn_training_step(model, optimizer, x, y, device)
        
        end_time = time.time()
    
        top1_test, top5_test = cnn_test(model, data.test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top5 Test accuracy: {}'.format(top5_test))
        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = cnn_test(model, train_loader, device)
        print('Top1 Train accuracy: {}'.format(top1_train))
        print('Top5 Train accuracy: {}'.format(top5_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)
        epoch_time = int(end_time-start_time)
        print('Epoch took {} seconds.'.format(epoch_time))
        metrics['epoch_times'].append(epoch_time)

        metrics['lrs'].append(cur_lr)

    return metrics
    

def cnn_test_time(model, loader, device='cpu'):
    model.eval()
    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output = model(b_x)
            end_time = time.time()
            total_time += (end_time - start_time)
            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, total_time


def cnn_test(model, loader, device='cpu'):
    model.eval()
    top1 = data.AverageMeter()
    top5 = data.AverageMeter()

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc


def cnn_get_confidence(model, loader, device='cpu'):
    model.eval()
    correct = set()
    wrong = set()
    instance_confidence = {}
    correct_cnt = 0

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = nn.functional.softmax(output, dim=1)
            model_pred = output.max(1, keepdim=True)
            pred = model_pred[1].to(device)
            pred_prob = model_pred[0].to(device)

            is_correct = pred.eq(b_y.view_as(pred))
            correct_cnt += pred.eq(b_y.view_as(pred)).sum().item()

            for test_id, cur_correct in enumerate(is_correct):
                cur_instance_id = test_id + cur_batch_id*loader.batch_size
                instance_confidence[cur_instance_id] = pred_prob[test_id].cpu().numpy()[0]

                if cur_correct == 1:
                    correct.add(cur_instance_id)
                else:
                    wrong.add(cur_instance_id)

   
    return correct, wrong, instance_confidence