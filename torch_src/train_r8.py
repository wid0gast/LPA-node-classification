import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from data_loader import load_data, load_npz, load_random
from LPA import LPA
from utils import *

adj, adj_n, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus('r8')

train_size = y_train.sum()
val_size = y_val.sum()
test_size = y_test.sum()

adj2 = np.load('../data/R8_adj_2.npy')
adj3 = adj2.dot('../data/R8_adj_3.npy')
adj4 = adj3.dot('../data/R8_adj_4.npy')

alpha = 0.5
adj_n = adj + alpha * adj2 + alpha * alpha * adj3 + alpha * alpha * alpha * adj4

model = LPA(adj_n)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

epochs = 10

for epoch in range(epochs):
    model.train()
    print(f'\nEpoch {epoch}: ')
    outputs = model(torch.tensor(y_train, dtype=torch.float64))
    loss = criterion(outputs[:train_size], torch.tensor(y_train[:train_size], dtype=torch.long))
    # train_acc = np.sum(torch.argmax(outputs, dim=1) == train_target)

    preds = torch.argmax(outputs, dim=1)
    train_acc = torch.eq(preds[:train_size], torch.tensor(y_train[:train_size])).sum() / train_size
    print(f'Training Loss: {loss}\tTraining Accuracy: {train_acc}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    preds = model(torch.tensor(y_val, dtype=torch.float64))
    loss = criterion(preds[train_size: train_size+val_size], torch.tensor(y_val[train_size: train_size+val_size]))
    preds = torch.argmax(preds, dim=1)
    val_acc = torch.eq(preds[train_size: train_size+val_size], torch.tensor(y_val[train_size: train_size+val_size])).sum() / val_size
    print(f'Validation Loss: {loss}\tValidation Accuracy: {val_acc}')

torch.save(model, 'LPA_R8')
