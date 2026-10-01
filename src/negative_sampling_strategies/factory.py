from src.negative_sampling_strategies.quantile_mixed_strategy import QuantileMixedNegativeSamplingStrategy
from src.negative_sampling_strategies.random_strategy import RandomNegativeSamplingStrategy
from src.negative_sampling_strategies.self_adversarial_mixed_strategy import SelfAdversarialMixedNegativeSamplingStrategy
from src.negative_sampling_strategies.structure_aware_mixed_strategy import StructureAwareMixedNegativeSamplingStrategy

MIXED_STRATEGY_NAMES = {"mixed", "mixed_self_adversarial", "mixed_structure_aware"}
SUPPORTED_STRATEGY_NAMES = {"random", *MIXED_STRATEGY_NAMES}

""" single construction point used by trainer.py """
def build_negative_sampling_strategy(
    strategy_name,
    all_positive_edge_index,
    num_source_nodes,
    num_target_nodes,
    num_negative_samples,
    device,
    training_positive_edge_index=None,
    informative_fraction=0.5,
    candidate_multiplier=5.0,
    lower_quantile=0.70,
    upper_quantile=0.95,
    start_epoch=20,
    self_adversarial_alpha=1.0,
    structure_max_attempts_multiplier=20,
):

    if strategy_name not in SUPPORTED_STRATEGY_NAMES:
        raise ValueError(
            "Unsupported negative strategy: "
            f"{strategy_name}. Supported values are "
            f"{sorted(SUPPORTED_STRATEGY_NAMES)}."
        )

    common_arguments = {
        "all_positive_edge_index": all_positive_edge_index,
        "num_source_nodes": num_source_nodes,
        "num_target_nodes": num_target_nodes,
        "num_negative_samples": num_negative_samples,
        "device": device
    }

    if strategy_name == "random":
        return RandomNegativeSamplingStrategy(**common_arguments)

    mixed_arguments = {
        **common_arguments,
        "informative_fraction": informative_fraction,
        "candidate_multiplier": candidate_multiplier,
        "start_epoch": start_epoch
    }

    if strategy_name == "mixed":
        return QuantileMixedNegativeSamplingStrategy(**mixed_arguments,lower_quantile=lower_quantile,upper_quantile=upper_quantile)

    if strategy_name == "mixed_self_adversarial":
        return SelfAdversarialMixedNegativeSamplingStrategy(**mixed_arguments,alpha=self_adversarial_alpha)

    return StructureAwareMixedNegativeSamplingStrategy(**mixed_arguments,training_positive_edge_index=training_positive_edge_index,
            max_attempts_multiplier=structure_max_attempts_multiplier)
