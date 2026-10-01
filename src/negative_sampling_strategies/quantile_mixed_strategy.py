import torch

from src.negative_sampling_strategies.mixed_base import MixedNegativeSamplingStrategy
from src.support.utils_sampling import score_candidate_logits

"""
Informative candidates are selected from the configured score-quantile
interval. The implementation intentionally preserves the old sigmoid-based
candidate scoring to keep previous results comparable.
"""
class QuantileMixedNegativeSamplingStrategy(MixedNegativeSamplingStrategy):

    strategy_name = "mixed"

    def __init__(self, *args, lower_quantile=0.70, upper_quantile=0.95, **kwargs):
        super().__init__(*args, **kwargs)

        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def sample(self, model=None, train_data=None, epoch=1):

        if self.uses_random_warmup(epoch):
            return self.sample_random()

        if model is None or train_data is None:
            raise ValueError("The quantile mixed strategy requires model and train_data.")

        candidate_edge_index = self.sample_random(num_samples=self.candidate_pool_size)

        candidate_logits = score_candidate_logits(model=model, train_data=train_data, candidate_edge_index=(candidate_edge_index))

        # retained from the original implementation.
        candidate_scores = torch.sigmoid(candidate_logits)

        lower_score = torch.quantile(candidate_scores, self.lower_quantile)
        upper_score = torch.quantile(candidate_scores, self.upper_quantile)

        eligible_indices = torch.where((candidate_scores >= lower_score) & (candidate_scores <= upper_score))[0]

        if eligible_indices.numel() >= self.num_informative_samples:
            permutation = torch.randperm(eligible_indices.numel())
            selected_indices = eligible_indices[permutation[:self.num_informative_samples]]
        else:
            selected_indices = torch.argsort(candidate_scores, descending=True)[:self.num_informative_samples]

        informative_edge_index = candidate_edge_index[:,selected_indices.to(candidate_edge_index.device)]

        return self.combine_with_random(informative_edge_index)
