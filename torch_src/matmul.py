#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from LPA import LPA
from tqdm import tqdm, trange
#%%
from utils import *
#%%
adj, adj_n, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus('r8')
#%%
vocab_size = adj.shape[0] - train_size - test_size
#%%
torch.cuda.is_available()
#%%
adj_n = torch.tensor(adj_n.A)
#%%
adj_n = adj_n.to(torch.device('cuda'))
#%%
adj_2n = adj_n.mm(adj_n)
#%%
adj_2n[:train_size, train_size:train_size+vocab_size] = adj_n[:train_size, train_size:train_size+vocab_size]
adj_2n[train_size+vocab_size:, train_size:train_size+vocab_size] = adj_n[train_size+vocab_size:, train_size:train_size+vocab_size]
adj_2n[train_size:train_size+vocab_size, :] = adj_n[train_size:train_size+vocab_size, :]
#%%
torch.save(adj_n, 'R8_adj_1.pt')
torch.save(adj_2n, 'R8_adj_2.pt')
#%%
adj_3n = adj_2n.mm(adj_n)
#%%
adj_3n[:train_size, train_size:train_size+vocab_size] = adj_n[:train_size, train_size:train_size+vocab_size]
adj_3n[train_size+vocab_size:, train_size:train_size+vocab_size] = adj_n[train_size+vocab_size:, train_size:train_size+vocab_size]
adj_3n[train_size:train_size+vocab_size, :] = adj_n[train_size:train_size+vocab_size, :]
#%%
torch.save(adj_3n, 'R8_adj_3.pt')
#%%
adj_4n = adj_3n.mm(adj_n)
#%%
adj_4n[:train_size, train_size:train_size+vocab_size] = adj_n[:train_size, train_size:train_size+vocab_size]
adj_4n[train_size+vocab_size:, train_size:train_size+vocab_size] = adj_n[train_size+vocab_size:, train_size:train_size+vocab_size]
adj_4n[train_size:train_size+vocab_size, :] = adj_n[train_size:train_size+vocab_size, :]
#%%
torch.save(adj_4n, 'R8_adj_4.pt')
#%%
alpha = 0.5
#%%
adj_final = adj_n + alpha * adj_2n + alpha * alpha * adj_3n + alpha * alpha * alpha * adj_4n
#%%
torch.save(adj_final, 'R8_adj_final.pt')