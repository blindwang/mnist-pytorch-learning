import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np

from train import train
from evaluate import evaluate
from dataloader import build_dataloader
from model import Net
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

# 判断是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""定义优化器列表"""
optimizers = [
    {'name': 'SGD', 'optimizer': optim.SGD, 'lr': 0.01},
    {'name': 'Adam', 'optimizer': optim.Adam, 'lr': 0.01},
    {'name': 'RMSprop', 'optimizer': optim.RMSprop, 'lr': 0.01},
    {'name': 'Adadelta', 'optimizer': optim.Adadelta, 'lr': 0.01},
    {'name': 'Adagrad', 'optimizer': optim.Adagrad, 'lr': 0.01},
    {'name': 'SparseAdam', 'optimizer': optim.SparseAdam, 'lr': 0.01},
]

"""默认超参数"""
batch_size = 64
learning_rate = 0.001
num_epochs = 10
num_workers = 0  # CPU中为0，GPU可以不为0

"""默认dataloader"""
trainloader, testloader, trainset, testset = build_dataloader(batch_size, num_workers)

for opt in optimizers:
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt['optimizer'](net.parameters(), lr=opt['lr'])

    print(f"Training model... w/ {opt['name']}")

    """tensorboard writer"""
    train_summary_writer, test_summary_writer = bulid_tensorboard_writer("compare_optim", opt['name'])

    # 训练模型
    for epoch in range(num_epochs):
        train(trainloader, net, criterion, optimizer, epoch, device, train_summary_writer)
        evaluate(testloader, net, epoch, device, test_summary_writer)

    print('Finished Training')