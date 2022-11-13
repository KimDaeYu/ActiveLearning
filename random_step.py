import sys

import sklearn.datasets
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

from model import NeuralNetwork
from loop import train_loop, vaild_loop, test_loop
from dataset import get_np_dataset, get_dataset_size
from train import train_loop, vaild_loop, test_loop
from logger import logger_init

rtn = get_np_dataset()
valid_size, test_size = get_dataset_size()

train_total_image_set = rtn[0][0]
train_total_label_set = rtn[0][1]

vaild_image_set = rtn[1][0]
vaild_label_set = rtn[1][1]
test_image_set = rtn[2][0]
test_label_set = rtn[2][1]

batch_size = 64

vaild_image_set = torch.from_numpy(vaild_image_set).to(torch.float32)
vaild_label_set = torch.from_numpy(vaild_label_set).to(torch.long).squeeze(dim=1)
test_image_set = torch.from_numpy(test_image_set).to(torch.float32)
test_label_set = torch.from_numpy(test_label_set).to(torch.long).squeeze(dim=1)

valid_dataloader = DataLoader(TensorDataset(vaild_image_set, vaild_label_set), batch_size=batch_size, num_workers=0)
test_dataloader = DataLoader(TensorDataset(test_image_set, test_label_set), batch_size=batch_size, num_workers=0)


model = NeuralNetwork().to(device)
learning_rate = 1e-5
batch_size = 64
epochs = 10

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

logger_init("active-learning")

for i in range(30):
    print(f'{i+1} STEP::')
    # Make Train Set Init
    if i == 0:
        idx = np.arange(0, 5000)
        train_image_set = train_total_image_set[idx]
        train_label_set = train_total_label_set[idx].squeeze()
    else:
        if len(train_total_image_set) > 1:
            idx = np.random.choice(len(train_total_image_set), 5000, replace=False)
            # if i == 5:
            #     import pdb; pdb.set_trace()
            train_image_set = np.concatenate((train_image_set.numpy(), train_total_image_set[idx]), axis=0)
            train_label_set = np.concatenate((train_label_set.numpy(), train_total_label_set[idx].squeeze()))
    
    if len(train_total_image_set) > 1:
        train_total_image_set = np.delete(train_total_image_set, idx, axis=0)
        train_total_label_set = np.delete(train_total_label_set, idx, axis=0)

        train_image_set = torch.from_numpy(train_image_set).to(torch.float32)
        train_label_set = torch.from_numpy(train_label_set).to(torch.long)
        train_dataloader = DataLoader(TensorDataset(train_image_set, train_label_set), batch_size=batch_size, num_workers=0)


    train_loop(train_dataloader, model, loss_fn, optimizer,logger=True)
    vaild_loop(valid_dataloader, model, loss_fn, logger=True)
    test_loop(test_dataloader, model, loss_fn, logger=True)