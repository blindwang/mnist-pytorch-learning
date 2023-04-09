import tensorflow as tf
import torch.nn as nn
import torch
import torch.optim as optim
import shutil
import random
import numpy as np

from train import train
from evaluate import evaluate
from dataloader import build_dataloader
from model import Net, LeNet, NetDeep, NetSigmoid
from utils import bulid_tensorboard_writer

"""随机种子"""
seed = 2023

# Python随机数生成器的种子
random.seed(seed)

# Numpy随机数生成器的种子
np.random.seed(seed)

# Pytorch随机数生成器的种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model_types = [
    {'name': 'ori_model', 'lr': 0.001},
    {'name': 'Sigmoid', 'epoch': 20, 'lr': 0.01},
    {'name': 'LeNet', 'lr': 0.001},
    {'name': 'add-conv-fc', 'epoch': 15, 'lr': 0.01},
    {'name': 'Adam', 'lr': 0.001},
    {'name': 'batchsize-128', 'batch_size': 128, 'lr': 0.001},
]

# 判断是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

for model_type in model_types[4:]:
    print(f"Training {model_type['name']} model...")

    """默认超参数"""
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    num_workers = 0  # CPU中为0，GPU可以不为0

    """tensorboard writer"""
    train_summary_writer, test_summary_writer = bulid_tensorboard_writer("compare_base", model_type['name'])

    """默认dataloader"""
    trainloader, testloader, trainset, testset = build_dataloader(batch_size, num_workers)

    """定义默认网络"""
    net = Net().to(device)

    """定义默认损失函数和优化器"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=model_type['lr'])

    if model_type['name'] == 'Sigmoid':
        net = NetSigmoid().to(device)
        num_epochs = model_type['epoch']
        optimizer = optim.SGD(net.parameters(), lr=model_type['lr'])
    elif model_type['name'] == 'LeNet':
        net = LeNet().to(device)
        optimizer = optim.SGD(net.parameters(), lr=model_type['lr'])
    elif model_type['name'] == 'add-conv-fc':
        net = NetDeep().to(device)
        num_epochs = model_type['epoch']
        optimizer = optim.SGD(net.parameters(), lr=model_type['lr'])
    elif model_type['name'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=model_type['lr'])
    elif model_type['name'] == 'batchsize-128':
        batch_size = model_type['batch_size']
        trainloader, testloader, trainset, testset = build_dataloader(batch_size, num_workers)

    # 训练模型
    for epoch in range(num_epochs):
        train(trainloader, net, criterion, optimizer, epoch, device, train_summary_writer)
        evaluate(testloader, net, epoch, device, test_summary_writer)

    print('Finished Training')
