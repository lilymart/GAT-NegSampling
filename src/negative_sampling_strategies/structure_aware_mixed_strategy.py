import torch

from src.negative_sampling_strategies.mixed_base import MixedNegativeSamplingStrategy
from src.support.utils_sampling import build_csr_neighbors, sample_one_neighbor_per_row, deduplicate_edge_index

"""
Informative candidates are generated only from training interactions via:

user u -> target n1 <- user u2 -> target n2

The candidate negative is (u, n2), provided it is not a known positive.

Validation and test positives are used only for exclusion from the negative space, never to generate structural paths.
"""
class StructureAwareMixedNegativeSamplingStrategy(MixedNegativeSamplingStrategy):


    strategy_name = "mixed_structure_aware"

    def __init__(self,*args,training_positive_edge_index,max_attempts_multiplier=20,**kwargs):
        super().__init__(*args, **kwargs)

        self.max_attempts_multiplier = max_attempts_multiplier

        training_positive_edge_index = deduplicate_edge_index(edge_index=training_positive_edge_index, num_target_nodes=self.num_target_nodes)


        self.training_positive_edge_index = training_positive_edge_index

        self.user_row_pointer, self.user_targets = build_csr_neighbors(
            edge_index=training_positive_edge_index,
            num_rows=self.num_source_nodes,
            num_columns=self.num_target_nodes,
        )

        self.target_row_pointer, self.target_users = build_csr_neighbors(
            edge_index=training_positive_edge_index.flip(0).contiguous(),
            num_rows=self.num_target_nodes,
            num_columns=self.num_source_nodes,
        )

        user_degrees = self.user_row_pointer[1:] - self.user_row_pointer[:-1]

        self.active_users = torch.where(user_degrees > 0)[0]

        self.all_positive_pair_ids = torch.unique(self.all_positive_edge_index[0] * self.num_target_nodes + self.all_positive_edge_index[1])

    def _draw_candidate_pair_ids(self,batch_size):
        active_user_indices = torch.randint(low=0,high=self.active_users.numel(), size=(batch_size,))

        source_users = self.active_users[active_user_indices]

        first_targets = sample_one_neighbor_per_row(
                rows=source_users,
                row_pointer=self.user_row_pointer,
                column_indices=self.user_targets
            )

        related_users = sample_one_neighbor_per_row(
                rows=first_targets,
                row_pointer=self.target_row_pointer,
                column_indices=self.target_users
            )

        candidate_targets = sample_one_neighbor_per_row(
                rows=related_users,
                row_pointer=self.user_row_pointer,
                column_indices=self.user_targets
        )

        return source_users* self.num_target_nodes + candidate_targets

    def _sample_structure_aware_edges(self):
        desired_pool_size = self.candidate_pool_size
        max_draws = max(desired_pool_size, desired_pool_size* self.max_attempts_multiplier)

        candidate_pair_ids = set()
        draws = 0

        while len(candidate_pair_ids) < desired_pool_size and draws < max_draws:
            remaining_draws = max_draws - draws
            batch_size = min(max(1024,2* (desired_pool_size-len(candidate_pair_ids))),remaining_draws)
            drawn_pair_ids = torch.unique(self._draw_candidate_pair_ids(batch_size=batch_size))
            known_positive_mask = torch.isin(drawn_pair_ids,self.all_positive_pair_ids)
            valid_pair_ids = drawn_pair_ids[~known_positive_mask]
            candidate_pair_ids.update(valid_pair_ids.tolist())
            draws += batch_size

        if not candidate_pair_ids:
            return torch.empty((2, 0),dtype=torch.long,device=self.device)

        candidate_pair_ids = torch.tensor(list(candidate_pair_ids),dtype=torch.long)

        if candidate_pair_ids.numel() > self.num_informative_samples:
            permutation = torch.randperm(candidate_pair_ids.numel())

            candidate_pair_ids = candidate_pair_ids[permutation[:self.num_informative_samples]]

        return torch.stack([candidate_pair_ids//self.num_target_nodes, candidate_pair_ids%self.num_target_nodes], dim=0).to(self.device)


    def sample(self, model=None, train_data=None, epoch=1):
        del model, train_data

        if self.uses_random_warmup(epoch):
            return self.sample_random()

        informative_edge_index = self._sample_structure_aware_edges()

        # if the graph cannot provide enough structural candidates,
        # the remaining positions are filled with random negatives.
        return self.combine_with_random(informative_edge_index)
