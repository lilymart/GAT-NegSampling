import torch

from src.negative_sampling_strategies.base_stretegy import NegativeSamplingStrategy

"""
Common implementation shared by all mixed strategies.
A mixed batch contains:
- a configurable informative component;
- a uniform-random component.
Before start_epoch, every mixed strategy behaves as random sampling.
"""
class MixedNegativeSamplingStrategy(NegativeSamplingStrategy):

    uses_warmup = True

    def __init__(self, *args, informative_fraction=0.5, candidate_multiplier=5.0, start_epoch=20, **kwargs):
        super().__init__(*args, **kwargs)

        self.informative_fraction = informative_fraction
        self.candidate_multiplier = candidate_multiplier
        self.start_epoch = start_epoch

    @property
    def num_informative_samples(self):
        return int(round(self.num_negative_samples * self.informative_fraction))

    @property
    def candidate_pool_size(self):
        return max(self.num_informative_samples, int(round(self.num_informative_samples * self.candidate_multiplier)))

    def uses_random_warmup(self, epoch):
        return epoch < self.start_epoch or self.num_informative_samples == 0

    """
    Merge informative and random negatives without duplicates.
    All known positives and already selected informative negatives are excluded from the random completion step.
    """
    def combine_with_random(self, informative_edge_index):

        informative_edge_index = informative_edge_index.detach().to(dtype=torch.long, device=self.device).contiguous()
        num_random_samples = self.num_negative_samples - informative_edge_index.size(1)

        if num_random_samples < 0:
            raise ValueError("The informative component contains more edges than the requested negative batch.")

        excluded_edge_index = torch.cat([self.all_positive_edge_index,informative_edge_index.cpu()], dim=1)
        random_edge_index = self.sample_random(
            num_samples=num_random_samples,
            excluded_edge_index=excluded_edge_index
        )
        negative_edge_index = torch.cat([random_edge_index, informative_edge_index], dim=1)
        permutation = torch.randperm(negative_edge_index.size(1), device=self.device)

        return negative_edge_index[:, permutation]