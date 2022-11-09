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


rtn = get_np_dataset()
valid_size, test_size = get_dataset_size()

train_image_set = rtn[0][0]
train_label_set = rtn[0][1]
vaild_image_set = rtn[1][0]
vaild_label_set = rtn[1][1]
test_image_set = rtn[2][0]
test_label_set = rtn[2][1]


batch_size = 64

valid_dataloader = DataLoader(TensorDataset(vaild_image_set, vaild_label_set), batch_size=batch_size, num_workers=0)
test_dataloader = DataLoader(TensorDataset(test_image_set, test_label_set), batch_size=batch_size, num_workers=0)


for i in range(10):
    model = NeuralNetwork().to(device)

    learning_rate = 1e-5
    batch_size = 64
    epochs = 10

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    

    # Make Train Set Init 5000
    train_image_set = torch.from_numpy(np_train_image_set[:5000]).to(torch.float32)
    train_label_set = torch.from_numpy(np_train_label_set[:5000]).to(torch.long).squeeze(dim=1)

    train_dataloader = DataLoader(TensorDataset(train_image_set, train_label_set), batch_size=batch_size, num_workers=0)

    

    # train_image_set = torch.from_numpy(np_train_image_set).to(torch.float32)
    # train_label_set = torch.from_numpy(np_train_label_set).to(torch.long).squeeze(dim=1)
    # vaild_image_set = torch.from_numpy(np_vaild_image_set).to(torch.float32)
    # vaild_label_set = torch.from_numpy(np_vaild_label_set).to(torch.long).squeeze(dim=1)
    # test_image_set = torch.from_numpy(np_test_image_set).to(torch.float32)
    # test_label_set = torch.from_numpy(np_test_label_set).to(torch.long).squeeze(dim=1)
