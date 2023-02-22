import torch
import torch.nn as nn

class LPA_layer(nn.Module):
    def __init__(self, adj):
        super(LPA_layer, self).__init__()
        self.adj = nn.Parameter(adj)
    def forward(self, x):
        output = torch.matmul(self.adj, x)
        return output

class LPA(nn.Module):
    def __init__(self, adj):
        # self.labels = torch.tensor(labels, dtype=torch.float64, requires_grad=False)
        # self.label_mask = torch.ones(labels.shape[0])
        # self.label_mask = torch.tensor(label_mask, dtype=torch.int32)
        super(LPA, self).__init__()
        # self.label_inputs = torch.tensor(labels * label_mask[:, None], dtype=torch.float64, requires_grad=False)
        # self.label_list = [self.label_inputs]
        self.adj = torch.tensor(adj, dtype=torch.float64, requires_grad=True)
        # self.adj_n = torch.nn.Softmax(self.adj)
        self.lpa_layer = LPA_layer(adj=self.adj)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # for _ in range(self.iters):
        # lpa_layer = LPA_layer(adj=self.adj_n)
        output = self.lpa_layer(x)
        output = self.softmax(output)
        return output