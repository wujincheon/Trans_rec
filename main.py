#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 19 Sep, 2019

@author: wangshuo
"""

import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

import metric
from utils import collate_fn
from narm import NARM
from dataset import load_data, RecSysDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/gowalla/', help='dataset directory path: datasets/gowalla/movie1m/movie20m')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=50, help='the dimension of item embedding')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0, help='split the portion of training set as validation set')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' )

def main():
    print('Loading data...')
    train, valid, test = load_data(args.dataset_path, valid_portion=0)
    
    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    
    ## Load Transition Matrix
    M2=pd.read_csv('datasets/transition/final2_transition_gowalla_narm.csv')
    M2=M2.T[1:].T
    M2.index=M2.columns
    
    n_items= 38575 #  38575, 3271, 8487
    model = NARM(n_items, M2, args.hidden_size, args.embed_dim, args.batch_size).to(device)

    

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch = epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 512)

        recall10, mrr10, recall20, mrr20, recall50, mrr50 = validate(test_loader, model)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f}  \n'.format(epoch, 10, recall10, 10, mrr10, 20, recall20, 20, mrr20, 50, recall50, 50, mrr50))

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if epoch % 10 ==0:
            torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')


def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=512):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)//512, miniters=1000):# //512, miniters=1000
        seq = seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(seq) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model):
    model.eval()
    recalls10 = []
    mrrs10 = []
    recalls20 = []
    mrrs20 = []
    recalls50 = []
    mrrs50 = []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader, total=len(valid_loader), miniters=1000):
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim = 1)
            recall10, mrr10 = metric.evaluate(logits, target, k = 10)
            recall20, mrr20 = metric.evaluate(logits, target, k = 20)
            recall50, mrr50 = metric.evaluate(logits, target, k = 50)
            
            recalls10.append(recall10)
            mrrs10.append(mrr10)
            recalls20.append(recall20)
            mrrs20.append(mrr20)
            recalls50.append(recall50)
            mrrs50.append(mrr50)
    
    mean_recall10 = np.mean(recalls10)
    mean_mrr10 = np.mean(mrrs10)
    mean_recall20 = np.mean(recalls20)
    mean_mrr20 = np.mean(mrrs20)
    mean_recall50 = np.mean(recalls50)
    mean_mrr50 = np.mean(mrrs50)
    
    return mean_recall10, mean_mrr10, mean_recall20, mean_mrr20, mean_recall50, mean_mrr50


if __name__ == '__main__':
    main()
