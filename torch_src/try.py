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
dataset = 'R8'
adj, adj_n, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
train_size = int(y_train.sum())
val_size = int(y_val.sum())

vocab_size = adj.shape[0] - train_size - val_size - test_size
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("CPU")
adj_final = torch.load(dataset + '_adj_final.pt', map_location=device)
epochs = 10
def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.float64)
rows = np.concatenate([np.arange(train_size + val_size), np.arange(-test_size, 0)])
rows.shape
cols = rows.copy()
cols.shape
adj_subset = adj_final[rows][:, cols]
# adj_subset = adj_final
adj_rand = np.random.rand(adj_subset.shape[0], adj_subset.shape[0])
adj_rand
model = LPA(adj_rand)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e+6)
all_train = torch.tensor(y_train[rows], dtype=torch.float64).to(device)
val_input = torch.tensor(y_train[rows], dtype=torch.float64).to(device)
test_input = torch.tensor(y_train[rows] + y_val[rows], dtype=torch.float64).to(device)
train_labels = torch.argmax(torch.tensor(y_train[rows]), dim=1)
val_labels = torch.argmax(torch.tensor(y_val[rows]), dim=1)
test_labels = torch.argmax(torch.tensor(y_test[rows]), dim=1)
for epoch in range(epochs):
    model.train()
    # print(model)
    # for p in model.parameters():
    #     print(p.name, p.data, p.requires_grad)
    train_indices = np.random.choice(list(range(train_size)), size=int(train_size * 0.8), replace=False)
    train_mask = get_mask(train_indices, adj_subset.shape[0])
    train_input = all_train * train_mask[:, None]
    print(f'\nEpoch {epoch}: ')
    outputs = model(train_input)
    loss = criterion(outputs[:train_size], train_labels[:train_size])
    # train_acc = np.sum(torch.argmax(outputs, dim=1) == train_target)

    preds = torch.argmax(outputs, dim=1)
    train_acc = torch.eq(preds[:train_size], torch.tensor(train_labels[:train_size])).sum() / train_size
    print(f'Training Loss: {loss}\tTraining Accuracy: {train_acc}')

    optimizer.zero_grad()
    # print(model.lpa_layer.adj.grad)
    loss.backward()
    plt.hist(model.lpa_layer.adj.grad)
    plt.show()
    # print(model.lpa_layer.adj.grad)
    optimizer.step()

    model.eval()
    preds = model(val_input)
    loss = criterion(preds[train_size: train_size+val_size], val_labels[train_size: train_size+val_size])
    preds = torch.argmax(preds, dim=1)
    val_acc = torch.eq(preds[train_size: train_size+val_size], val_labels[train_size: train_size+val_size]).sum() / val_size
    print(f'Validation Loss: {loss}\tValidation Accuracy: {val_acc}')
print(model.lpa_layer.adj.grad)
test_preds = model(test_input)
test_preds = torch.argmax(test_preds, dim=1)
test_acc = torch.eq(test_preds[-1 * test_size:], test_labels[-1 * test_size:]).sum() / test_size
print("Test Accuracy: ", test_acc)
