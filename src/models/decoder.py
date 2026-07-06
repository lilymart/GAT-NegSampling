import torch
from torch_geometric.nn import Linear


"""
Binary link decoder.
Given z_src and z_dst, returns one logit per edge.
"""
class EdgeDecoder(torch.nn.Module):

    def __init__(self, hidden_channels=64):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, 1)
        )

    def forward(self, z_src, z_dst, edge_label_index):
        src, dst = edge_label_index
        edge_feat = torch.cat([z_src[src], z_dst[dst]], dim=-1)
        return self.mlp(edge_feat).view(-1)

