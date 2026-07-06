import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear

"""
Homogeneous base encoder converted to heterogeneous with to_hetero().
Returns node embeddings.
"""
class GATEncoder(torch.nn.Module):

    def __init__(self, hidden_channels=64, dropout=0.3, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    (-1, -1),
                    hidden_channels,
                    add_self_loops=False,
                    dropout=dropout
                )
            )
            self.lins.append(Linear(-1, hidden_channels))

    def forward(self, x, edge_index):
        for conv, lin in zip(self.convs, self.lins):
            x = conv(x, edge_index) + lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

