import torch
from torch_geometric.utils import negative_sampling


"""
Sample random non-existing links from the bipartite source-target interaction space. 

all_positive_edge_index must contain all known positive links,
including training, validation, and test positives.
"""
def sample_random_negative_edges(all_positive_edge_index, num_source_nodes, num_target_nodes, num_negative_samples, device):

    if num_negative_samples < 0:
        raise ValueError("num_negative_samples must be non-negative.")

    if num_negative_samples == 0:
        return torch.empty((2, 0),dtype=torch.long,device=device)

    positive_edge_index_cpu = all_positive_edge_index.detach().to(dtype=torch.long, device="cpu").contiguous()

    negative_edge_index = negative_sampling(
        edge_index=positive_edge_index_cpu,
        num_nodes=(num_source_nodes, num_target_nodes),
        num_neg_samples=num_negative_samples,
        method="sparse",
    )

    if negative_edge_index.size(1) != num_negative_samples:
        raise RuntimeError(f"Requested={num_negative_samples} negatives, returned={negative_edge_index.size(1)}.")

    return negative_edge_index.to(device)


""" score candidate links while preserving the model training state. """
@torch.no_grad()
def score_candidate_logits(model, train_data, candidate_edge_index):

    was_training = model.training
    model.eval()
    logits, _ = model(train_data.x_dict,  train_data.edge_index_dict, candidate_edge_index)
    logits = logits.detach().view(-1).cpu()

    if was_training:
        model.train()

    return logits


""" Deduplicate bipartite source-target pairs. """
def deduplicate_edge_index(edge_index, num_target_nodes):

    edge_index = edge_index.detach().to(dtype=torch.long, device="cpu").contiguous()

    if edge_index.numel() == 0:
        return torch.empty((2, 0),dtype=torch.long)

    pair_ids = torch.unique(edge_index[0] * num_target_nodes + edge_index[1])

    return torch.stack([pair_ids//num_target_nodes, pair_ids%num_target_nodes],dim=0).contiguous()


"""
Construct a compact CSR-like neighbor representation.
Returns:
    row_pointer: shape [num_rows + 1]
    column_indices: neighbors sorted by source row
"""
def build_csr_neighbors(edge_index, num_rows, num_columns,):

    edge_index = deduplicate_edge_index(edge_index=edge_index, num_target_nodes=num_columns)

    rows = edge_index[0]
    columns = edge_index[1]

    if rows.numel() == 0:
        return torch.zeros(num_rows + 1, dtype=torch.long), torch.empty(0, dtype=torch.long)

    sorting_indices = torch.argsort(rows,stable=True)
    rows = rows[sorting_indices]
    columns = columns[sorting_indices]
    row_degrees = torch.bincount(rows, minlength=num_rows)
    row_pointer = torch.zeros(num_rows + 1, dtype=torch.long)
    row_pointer[1:] = torch.cumsum(row_degrees, dim=0)

    return row_pointer, columns


""" Uniformly sample one stored neighbor for every requested row """
def sample_one_neighbor_per_row(rows, row_pointer, column_indices):

    starts = row_pointer[rows]
    degrees = row_pointer[rows + 1] - starts
    offsets = (torch.rand(rows.numel()) * degrees.to(torch.float)).to(torch.long)

    return column_indices[starts + offsets]
