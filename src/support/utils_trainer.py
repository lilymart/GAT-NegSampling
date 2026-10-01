import torch


"""Combine positive and negative links and create binary labels."""
def build_edge_labels(positive_edge_index, negative_edge_index):

    edge_label_index = torch.cat([positive_edge_index, negative_edge_index], dim=1)
    edge_label = torch.cat(
        [
            torch.ones(positive_edge_index.size(1), dtype=torch.float, device=positive_edge_index.device),
            torch.zeros(negative_edge_index.size(1), dtype=torch.float, device=positive_edge_index.device)
        ],
        dim=0,
    )
    permutation = torch.randperm(edge_label.size(0), device=edge_label.device)

    return edge_label_index[:, permutation], edge_label[permutation]
