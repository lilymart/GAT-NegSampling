import torch

from src.negative_sampling_strategies.mixed_base import MixedNegativeSamplingStrategy
from src.support.utils_sampling import score_candidate_logits

"""
Informative candidates are sampled without replacement with probability:
p_i proportional to exp(alpha * logit_i)
"""
class SelfAdversarialMixedNegativeSamplingStrategy(MixedNegativeSamplingStrategy):

    strategy_name = "mixed_self_adversarial"

    def __init__(self, *args, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def sample(self, model=None, train_data=None, epoch=1):
        if self.uses_random_warmup(epoch):
            return self.sample_random()

        if model is None or train_data is None:
            raise ValueError("The self-adversarial strategy requires model and train_data.")

        candidate_edge_index = self.sample_random(num_samples=self.candidate_pool_size)

        candidate_logits = score_candidate_logits(model=model,train_data=train_data,candidate_edge_index=candidate_edge_index)

        # subtracting the maximum improves numerical stability.
        stabilized_logits = candidate_logits - candidate_logits.max()

        sampling_probabilities = torch.softmax( self.alpha * stabilized_logits, dim=0)

        # avoid too few non-zero entries for sampling without replacement when candidate scores are extremely concentrated.
        sampling_probabilities = torch.clamp(sampling_probabilities, min=torch.finfo(sampling_probabilities.dtype).tiny)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        selected_indices = torch.multinomial(sampling_probabilities, num_samples=self.num_informative_samples,replacement=False)

        informative_edge_index = candidate_edge_index[:,selected_indices.to(candidate_edge_index.device)]

        return self.combine_with_random(informative_edge_index)
