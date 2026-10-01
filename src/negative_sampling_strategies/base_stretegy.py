import torch

from src.support.utils_sampling import sample_random_negative_edges

"""
Abstract interface shared by all negative-sampling strategies.
The trainer only depends on this class contract and does not need to know how individual strategies construct their negative links.
"""
class NegativeSamplingStrategy:

    strategy_name = "base"
    uses_warmup = False


    def __init__(self, all_positive_edge_index, num_source_nodes, num_target_nodes, num_negative_samples, device):

        self.all_positive_edge_index = all_positive_edge_index.detach().to(dtype=torch.long, device="cpu").contiguous()
        self.num_source_nodes = num_source_nodes
        self.num_target_nodes = num_target_nodes
        self.num_negative_samples = num_negative_samples
        self.device = device
        self.start_epoch = 1 #non-mixed strategies are active from the first epoch


    """Return a [2, num_negative_samples] negative edge index."""
    def sample(self, model=None, train_data=None, epoch=1):
        pass


    """Shared uniform fallback used by all strategies."""
    def sample_random(self, num_samples=None, excluded_edge_index=None):

        if num_samples is None:
            num_samples = self.num_negative_samples

        if excluded_edge_index is None:
            excluded_edge_index = self.all_positive_edge_index

        return sample_random_negative_edges(
            all_positive_edge_index=excluded_edge_index,
            num_source_nodes=self.num_source_nodes,
            num_target_nodes=self.num_target_nodes,
            num_negative_samples=num_samples,
            device=self.device,
        )