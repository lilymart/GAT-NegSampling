import torch

""" Wrapper around heterogeneous encoder + edge decoder """
class HeteroGATLinkPrediction(torch.nn.Module):

    def __init__(self, encoder, decoder, edge_type):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.edge_type = edge_type

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)

        src_type, _, dst_type = self.edge_type
        pred = self.decoder(
            z_dict[src_type],
            z_dict[dst_type],
            edge_label_index
        )

        return pred, z_dict