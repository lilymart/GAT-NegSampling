"""Negative-sampling strategies for heterogeneous link prediction."""

from .base_stretegy import NegativeSamplingStrategy
from .factory import MIXED_STRATEGY_NAMES, SUPPORTED_STRATEGY_NAMES, build_negative_sampling_strategy
from .quantile_mixed_strategy import QuantileMixedNegativeSamplingStrategy
from .random_strategy import RandomNegativeSamplingStrategy
from .self_adversarial_mixed_strategy import (
    SelfAdversarialMixedNegativeSamplingStrategy,
)
from .structure_aware_mixed_strategy import (
    StructureAwareMixedNegativeSamplingStrategy,
)

__all__ = [
    "NegativeSamplingStrategy",
    "RandomNegativeSamplingStrategy",
    "QuantileMixedNegativeSamplingStrategy",
    "SelfAdversarialMixedNegativeSamplingStrategy",
    "StructureAwareMixedNegativeSamplingStrategy",
    "MIXED_STRATEGY_NAMES",
    "SUPPORTED_STRATEGY_NAMES",
    "build_negative_sampling_strategy"
]
