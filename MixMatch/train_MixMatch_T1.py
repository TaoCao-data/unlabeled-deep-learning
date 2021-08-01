# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 21:46:44 2021

"""

# The code is created following the pytorch implementation: https://github.com/YU1ut/MixMatch-pytorch

from __future__ import print_function

#import argparse
import os
#import shutil
#import time
#import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.wideresnet as models
import dataset.cifar10 as dataset
#from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
#from tensorboardX import SummaryWriter

# Setting Parameters
epochs = 10
batch_size = 64
lr = 0.002
set_seed = 0
n_label = 250
n_iter = 1024
alpha = 0.75
lambda_u = 75
T = 1
decay = 0.999
ema = True

# GPU Check
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if_cuda = torch.cuda.is_available()
print(if_cuda)

# Other Initialization
best_acc = 0
np.random.seed(set_seed)

# Read Data
data_aug_train = transforms.Compose([
    dataset.RandomPadandCrop(32),
    dataset.RandomFlip(),
    dataset.ToTensor()])

data_aug_val = transforms.Compose([dataset.ToTensor()])

train_labeled, train_unlabeled, valid, test = dataset.get_cifar10('./data', n_label, transform_train=data_aug_train, transform_val=data_aug_val)

batched_train_labeled = data.DataLoader(train_labeled, batch_size=batch_size, drop_last = True)
batched_train_unlabeled = data.DataLoader(train_unlabeled, batch_size=batch_size, drop_last = True)
batched_valid = data.DataLoader(valid, batch_size=batch_size)
batched_test = data.DataLoader(test, batch_size=batch_size)

model = models.WideResNet(num_classes=10)
ema_model = models.WideResNet(num_classes=10)

if if_cuda:
    model = model.cuda()
    ema_model = ema_model.cuda()
    
if ema:
    for paramter in ema_model.parameters():
        paramter.detach_()

# Training Setup
class SemiLoss(object):
    def __call__(self, pred_x, label_x, pred_u, label_u, epoch, lambda_u, epochs):
        probs_u = torch.softmax(pred_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(pred_x, dim=1) * label_x, dim=1))
        Lu = torch.mean((probs_u - label_u) ** 2)
        
        loss = Lx + Lu * lambda_u * np.minimum(1.0, np.maximum(epoch/epochs, 0.0))
        
        return loss

class EMA_OP(object):
    def __init__(self, model, ema_model, alpha=0.999, lr=lr):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * (1.0 - self.alpha))
                param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

criterion = SemiLoss()
criterion2 = nn.CrossEntropyLoss()    
optimizer = optim.Adam(model.parameters(), lr=lr)
ema_optimizer= EMA_OP(model, ema_model, alpha=decay, lr=lr)

# Training & Validation
for epoch in range(epochs):
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, epochs, lr))
    
    epoch_train_loss_avg = 0
    epoch_valid_loss_avg = 0
    epoch_valid_acc1_avg = 0
    epoch_test_loss_avg = 0
    epoch_test_acc1_avg = 0
    
    #train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, use_cuda)
    
    batched_train_labeled_iter = iter(batched_train_labeled)
    batched_train_unlabeled_iter = iter(batched_train_unlabeled)
    
    model.train()
    
    for i in range(n_iter):
        
        try:
            X_x, X_y = batched_train_labeled_iter.next()
        except:
            batched_train_labeled_iter = iter(batched_train_labeled)
            X_x, X_y = batched_train_labeled_iter.next()
            
        try:
            (U_x1, U_x2), _ = batched_train_unlabeled_iter.next()
        except:
            batched_train_unlabeled_iter = iter(batched_train_unlabeled)
            (U_x1, U_x2), _ = batched_train_unlabeled_iter.next()
            
     
        X_y = torch.zeros(batch_size, 10).scatter_(1, X_y.view(-1,1).long(), 1)
        
        if if_cuda:
            X_x = X_x.cuda()
            X_y = X_y.cuda(non_blocking=True)
            U_x1 = U_x1.cuda()
            U_x2 = U_x2.cuda()
        
        with torch.no_grad():
            U_pred1 = model(U_x1)
            U_pred2 = model(U_x2)
            p = (torch.softmax(U_pred1, dim=1) + torch.softmax(U_pred2, dim=1)) / 2
            U_q = p**(1/T) / (p**(1/T)).sum(dim=1, keepdim=True)
            U_q = U_q.detach()
            
        W_x = torch.cat([X_x, U_x1, U_x2], dim=0)
        W_y = torch.cat([X_y, U_q, U_q], dim=0)
        
        shuffle_index = torch.randperm(W_x.size(0))
        A_x, B_x = W_x, W_x[shuffle_index]
        A_y, B_y = W_y, W_y[shuffle_index]

        beta_lambda = np.random.beta(alpha, alpha)
        beta_lambda = max(beta_lambda, 1-beta_lambda)
        
        Mix_x = beta_lambda * A_x + (1 - beta_lambda) * B_x
        Mix_y = beta_lambda * A_y + (1 - beta_lambda) * B_y
        
        Mix_x = list(torch.split(Mix_x, batch_size))
        Mix_x = interleave(Mix_x, batch_size)
        
        pred = [model(Mix_x[0])]
        for input in Mix_x[1:]:
            pred.append(model(input))
            
        pred = interleave(pred, batch_size)
        pred_x = pred[0]
        pred_u = torch.cat(pred[1:], dim=0)
        

        loss = criterion(pred_x, Mix_y[:batch_size], pred_u, Mix_y[batch_size:], epoch + i/n_iter, lambda_u, epochs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        
        epoch_train_loss_avg += loss
        
        #print("Epoch:{}, Iternation:{}, Train Loss:{}".format(epoch, i, loss))
        
    # Validation
    #_, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
    
    ema_model.eval()

    with torch.no_grad():
        for j, (X, y) in enumerate(batched_valid):
            if if_cuda:
                X, y = X.cuda(), y.cuda(non_blocking=True)

            outputs = ema_model(X)
            y = y.type(torch.LongTensor)
            
            if if_cuda:
                outputs = outputs.cuda()
                y = y.cuda(non_blocking=True)
             
            #print(outputs.dtype)
            #print(y.dtype)
            
            loss = criterion2(outputs, y)           
            acc1 = torch.sum(y == outputs.max(1)[1]) / y.size(0)
            
            epoch_valid_loss_avg += loss
            epoch_valid_acc1_avg += acc1
            
            #print("Epoch:{}, Iternation:{}, Valid Loss:{}, Valid Acc:{}".format(epoch, j, loss, acc1))

    with torch.no_grad():
        for k, (X, y) in enumerate(batched_test):
            if if_cuda:
                X, y = X.cuda(), y.cuda(non_blocking=True)

            outputs = ema_model(X)
            y = y.type(torch.LongTensor)
            
            if if_cuda:
                outputs = outputs.cuda()
                y = y.cuda(non_blocking=True)     
            
            #print(outputs.dtype)
            #print(y.dtype)
            
            loss = criterion2(outputs, y)           
            acc1 = torch.sum(y == outputs.max(1)[1]) / y.size(0)
            
            epoch_test_loss_avg += loss
            epoch_test_acc1_avg += acc1
            
            #print("Epoch:{}, Iternation:{}, Test Loss:{}, Test Acc:{}".format(epoch, k, loss, acc1))
            
    print('---------------------------')
    print("Epoch:{}, Train Avg Loss:{}".format(epoch, epoch_train_loss_avg / (i + 1)))
    print("Epoch:{}, Valid Avg Loss:{}".format(epoch, epoch_valid_loss_avg / (j + 1)))
    print("Epoch:{}, Valid Avg Acc1:{}".format(epoch, epoch_valid_acc1_avg / (j + 1)))
    print("Epoch:{}, Test Avg Loss:{}".format(epoch, epoch_test_loss_avg / (k + 1)))
    print("Epoch:{}, Test Avg Acc1:{}".format(epoch, epoch_test_acc1_avg / (k + 1)))           
            
    #if best_acc < epoch_test_acc1_avg / (k + 1):
        #best_acc = epoch_test_acc1_avg / (k + 1)

#print('Best Test Acc1:{}'.format(best_acc))

# =============================================================================
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict()
#             }, 'Model_Checkpoint')
# 
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': ema_model.state_dict(),
#             }, 'EMA_Model_Checkpoint')
# =============================================================================
