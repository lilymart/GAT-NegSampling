from src.negative_sampling_strategies.base_stretegy import NegativeSamplingStrategy

"""Uniformly sample non-existing source-target links."""
class RandomNegativeSamplingStrategy(NegativeSamplingStrategy):

    strategy_name = "random"

    def sample(self, model=None, train_data=None, epoch=1):
        del model, train_data, epoch
        return self.sample_random()