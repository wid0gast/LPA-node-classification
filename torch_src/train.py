import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from data_loader import load_data, load_npz, load_random
from LPA import LPA

features, labels, adj, len_train, len_val, len_test = load_data('pubmed')
labels = np.array(labels)
# print(labels)
# print(adj)
# print(train_mask)
# print(val_mask)
# print(test_mask)

def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.float64)


train_indices = np.random.choice(list(range(len_train)), size=int(len_train * 0.8), replace=False)
train_mask = get_mask(train_indices, len_train + len_val + len_test)

train_inputs = labels * train_mask[:, None]
val_inputs = labels * (np.concatenate([np.ones(len_train),np.zeros(len_val + len_test)]))[:, None]
test_inputs = labels * (np.concatenate([np.ones(len_train + len_val),np.zeros(len_test)]))[:, None]

adj2 = adj.dot(adj)
adj3 = adj2.dot(adj)
adj4 = adj3.dot(adj)

alpha = 0.5
adj_n = adj + alpha * adj2 + alpha * alpha * adj3 + alpha * alpha * alpha * adj4

with open('../data/pubmed/pubmed_adj_4norm.npy', 'wb') as f:
    np.save(f, adj_n)

# adj_n = np.load('../data/cora/cora_adj_4norm.npy')

model = LPA(adj_n)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

epochs = 10
# plt.hist(np.argmax(labels, axis=1)[:len_train])
# plt.show()

train_target = np.concatenate([np.argmax(labels, axis=1)[:len_train], (np.ones(len_val + len_test) * (-100))])

val_target = np.argmax(labels[len_train:len_train + len_val], axis=1)
test_target = np.argmax(labels[len_train + len_val:])

# plt.hist(train_target[:len_train])
# plt.show()

# exit()

for epoch in range(epochs):
    model.train()
    print(f'\nEpoch {epoch}: ')
    outputs = model(torch.tensor(train_inputs, dtype=torch.float64))
    loss = criterion(outputs[:len_train], torch.tensor(train_target[:len_train], dtype=torch.long))
    # train_acc = np.sum(torch.argmax(outputs, dim=1) == train_target)

    preds = torch.argmax(outputs, dim=1)
    train_acc = torch.eq(preds[:len_train], torch.tensor(train_target[:len_train])).sum() / len_train
    print(f'Training Loss: {loss}\tTraining Accuracy: {train_acc}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    preds = model(torch.tensor(val_inputs, dtype=torch.float64))
    loss = criterion(preds[len_train: len_train+len_val], torch.tensor(val_target))
    preds = torch.argmax(preds, dim=1)
    val_acc = torch.eq(preds[len_train: len_train + len_val], torch.tensor(val_target)).sum() / len_val
    print(f'Validation Loss: {loss}\tValidation Accuracy: {val_acc}')

torch.save(model, 'LPA_cora')

print('\n\nTesting...')

preds = model(torch.tensor(test_inputs, dtype=torch.float64))
preds = torch.argmax(preds, dim=1)
test_acc = torch.eq(preds[len_train + len_val:], torch.tensor(test_target)).sum() / len_test
print('Test Accuracy: ', test_acc)
