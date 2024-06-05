import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from torch.utils.data import Dataset


class CommunicationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Retrieve the graph and its corresponding data
        graph, user_power_allocations, channel_links = self.data_list[idx]

        return graph, user_power_allocations, channel_links


class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GNNModel, self).__init__()
        self.conv1 = dgl.nn.pytorch.conv.GraphConv(in_feats,
                                                   hidden_feats)
        self.conv2 = dgl.nn.pytorch.conv.GraphConv(hidden_feats,
                                                   hidden_feats)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, 2 * hidden_feats),
            nn.ReLU(),
            nn.Linear(2 * hidden_feats, out_feats)  
        )

    def forward(self, graph):
        # Perform message passing using GraphConv
        power = self.conv1(graph, graph.ndata['feat'])
        power = self.conv2(graph, power)
        power = self.proj(self.mlp(power))

        return power

    def proj(self, x):
        eps = 1e-6
        if x.size()[0] == 1:
            row_norms = torch.norm(abs(x))
        else:
            row_norms = torch.norm(abs(x), dim=1)

        x_proj = torch.max(torch.full_like(x, eps), abs(x / row_norms.unsqueeze(-1)))

        return x_proj
