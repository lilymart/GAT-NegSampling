import torch
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

from src.support.utils_trainer import build_edge_labels

from src.negative_sampling_strategies import MIXED_STRATEGY_NAMES, build_negative_sampling_strategy

"""
negative_mode:
        "none"    -> positive-only training;
        "static"  -> random negatives sampled once;
        "dynamic" -> negatives periodically resampled.

negative_strategy:
        "random" -> uniformly sampled non-edges;
        "mixed"                     -> random + quantile semi-hard negatives;
        "mixed_self_adversarial"    -> random + score-weighted negatives;
        "mixed_structure_aware"     -> random + structural negatives.
"""
def train_link_predictor(model, criterion, train_data, val_data, optimizer, edge_type,
            all_positive_edge_index=None,
            negative_mode="static", negative_strategy="random",
            neg_sampling_ratio=1.0,
            negative_resample_every=1,
            semi_hard_fraction=0.5, candidate_multiplier=5,
            semi_hard_lower_quantile=0.70, semi_hard_upper_quantile=0.95, semi_hard_start_epoch=20,
            self_adversarial_alpha=1.0,
            structure_max_attempts_multiplier=20,
            n_epochs=500, patience=50, epsilon=1e-5):

    positive_edge_index = train_data[edge_type].edge_label_index
    positive_labels = torch.ones(positive_edge_index.size(1),dtype=torch.float,device=positive_edge_index.device)

    source_node_type, _, target_node_type = edge_type
    num_source_nodes = train_data[source_node_type].num_nodes
    num_target_nodes = train_data[target_node_type].num_nodes

    num_negative_samples = int(positive_edge_index.size(1) * neg_sampling_ratio)

    static_edge_label_index = None
    static_edge_label = None
    dynamic_edge_label_index = None
    dynamic_edge_label = None

    negative_sampling_strategy = None # one strategy object is built before the training loop

    if negative_mode != "none":
        # structure-aware paths use only training target interactions.
        # RandomLinkSplit separates message-passing and supervision positives,
        # therefore both sets are merged here.
        training_positive_edge_index = torch.cat([train_data[edge_type].edge_index.detach().cpu(), positive_edge_index.detach().cpu()],dim=1)

        negative_sampling_strategy = build_negative_sampling_strategy(
                strategy_name=negative_strategy,
                all_positive_edge_index=all_positive_edge_index,
                num_source_nodes=num_source_nodes,
                num_target_nodes=num_target_nodes,
                num_negative_samples=num_negative_samples,
                device=positive_edge_index.device,
                training_positive_edge_index=training_positive_edge_index,
                informative_fraction=semi_hard_fraction,
                candidate_multiplier=candidate_multiplier,
                lower_quantile=semi_hard_lower_quantile,
                upper_quantile=semi_hard_upper_quantile,
                start_epoch=semi_hard_start_epoch,
                self_adversarial_alpha=self_adversarial_alpha,
                structure_max_attempts_multiplier=structure_max_attempts_multiplier
        )

    if negative_mode == "static":
        static_negative_edge_index = negative_sampling_strategy.sample(epoch=1)
        static_edge_label_index, static_edge_label = build_edge_labels(
            positive_edge_index=positive_edge_index,
            negative_edge_index=static_negative_edge_index
        )

    best_val_auc = 0.0
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, n_epochs + 1):

        epochs_trained = epoch

        if negative_mode == "none":
            edge_label_index = positive_edge_index
            edge_label = positive_labels

        elif negative_mode == "static":
            edge_label_index = static_edge_label_index
            edge_label = static_edge_label

        else: # negative_mode == "dynamic":
            # the trainer decides when to resample, while the strategy object decides how to construct the negative batch
            activation_resample = negative_strategy in MIXED_STRATEGY_NAMES and epoch == semi_hard_start_epoch

            should_resample = (
                    dynamic_edge_label_index is None
                    or (epoch - 1) % negative_resample_every == 0
                    or activation_resample
            )

            if should_resample:
                negative_edge_index = negative_sampling_strategy.sample(model=model, train_data=train_data, epoch=epoch)

                dynamic_edge_label_index, dynamic_edge_label = build_edge_labels(
                    positive_edge_index=positive_edge_index,
                    negative_edge_index=negative_edge_index
                )

            edge_label_index = dynamic_edge_label_index
            edge_label = dynamic_edge_label

        model.train()
        optimizer.zero_grad()

        logits, _ = model(
            train_data.x_dict,
            train_data.edge_index_dict,
            edge_label_index
        )

        loss = criterion(logits, edge_label)
        loss.backward()
        optimizer.step()

        # During training, threshold-dependent metrics are displayed at 0.5.
        # Early stopping depends only on AUC and is therefore threshold-free.
        val_metrics = eval_link_predictor(
            model=model,
            data=val_data,
            edge_type=edge_type,
            threshold=0.5
        )

        val_auc = val_metrics["auc"]

        if val_auc > best_val_auc + epsilon:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {
                name: parameter.detach().cpu().clone()
                for name, parameter
                in model.state_dict().items()
            } # Stored on CPU to avoid additional GPU memory usage.
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0:
            print(
                f"Epoch: {epoch:03d}, "
                f"Train Loss: {loss.item():.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val AP: {val_metrics['ap']:.4f}, "
                f"Val F1-binary: {val_metrics['f1-binary']:.4f}, "
                f"Val F1-macro: {val_metrics['f1-macro']:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}. "
                f"Best Val AUC: {best_val_auc:.4f}"
            )
            break

    if best_state_dict is None:
        raise RuntimeError(
            "No valid model state was stored during training."
        )

    # The returned model is the one associated with the best validation AUC
    model.load_state_dict(best_state_dict)

    training_info = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "epochs_trained": epochs_trained,
    }

    return model, training_info



@torch.no_grad()
def eval_link_predictor(model, data, edge_type, threshold=0.5, cached_scores=None, return_scores=False):

    if cached_scores is None:

        model.eval()

        edge_label_index = data[edge_type].edge_label_index
        edge_label = data[edge_type].edge_label.detach().view(-1).cpu().numpy().astype(int)

        logits, _ = model(
            data.x_dict,
            data.edge_index_dict,
            edge_label_index,
        )

        probs = torch.sigmoid(logits).detach().view(-1).cpu().numpy()

    else:
        edge_label, probs = cached_scores

    metrics = _compute_link_metrics(
        edge_label=edge_label,
        probs=probs,
        threshold=threshold
    )

    if return_scores:
        return metrics, (
            edge_label,
            probs,
        )

    return metrics


"""
Compute both threshold-free and threshold-dependent metrics.
"""
def _compute_link_metrics(edge_label, probs, threshold):

    pred = (probs >= threshold).astype(int)

    return {
        "auc": roc_auc_score(edge_label, probs),
        "ap": average_precision_score(edge_label, probs),
        "f1-binary": f1_score(edge_label,pred,zero_division=0),
        "f1-macro": f1_score(edge_label,pred,average="macro",zero_division=0),
        "precision": precision_score(edge_label,pred,zero_division=0),
        "recall": recall_score(edge_label,pred,zero_division=0)
    }

"""
Selects the global decision threshold on validation data.

Supported optimization metrics:
- "f1-binary": F1-score of the positive interaction class;
- "f1-macro": average F1-score across the negative and positive classes.
"""
@torch.no_grad()
def find_best_threshold(model,data,edge_type,optimization_metric="f1-macro"):

    if optimization_metric not in {"f1-binary", "f1-macro"}:
        raise ValueError(
            "optimization_metric must be either "
            "'f1-binary' or 'f1-macro'."
        )

    model.eval()

    edge_label_index = data[edge_type].edge_label_index
    edge_label = data[edge_type].edge_label.detach().view(-1).cpu().numpy().astype(int)

    logits, _ = model(
        data.x_dict,
        data.edge_index_dict,
        edge_label_index,
    )

    probs = torch.sigmoid(logits).detach().view(-1).cpu().numpy()

    # Sort scores in descending order.
    sorting_indices = np.argsort(-probs, kind="mergesort")

    sorted_probs = probs[sorting_indices]
    sorted_labels = edge_label[sorting_indices]

    positive_mask = sorted_labels == 1
    negative_mask = sorted_labels == 0

    # For each position k, all examples from 0 to k are predicted positive.
    cumulative_tp = np.cumsum(positive_mask)
    cumulative_fp = np.cumsum(negative_mask)

    total_positives = cumulative_tp[-1]
    total_negatives = cumulative_fp[-1]

    # A threshold only changes the predictions when the score changes.
    last_index_per_score = np.concatenate(
        [np.where(sorted_probs[:-1] != sorted_probs[1:])[0],np.array([len(sorted_probs) - 1])]
    )

    candidate_thresholds = sorted_probs[last_index_per_score]

    tp = cumulative_tp[last_index_per_score]
    fp = cumulative_fp[last_index_per_score]
    fn = total_positives - tp
    tn = total_negatives - fp

    # F1 of the positive interaction class.
    positive_denominator = 2 * tp + fp + fn

    f1_binary_values = np.divide(
        2 * tp,
        positive_denominator,
        out=np.zeros_like(tp, dtype=float),
        where=positive_denominator != 0
    )

    # For class 0:
    # TP_0 = TN, FP_0 = FN, FN_0 = FP.
    negative_denominator = 2 * tn + fn + fp

    f1_negative_values = np.divide(
        2 * tn,
        negative_denominator,
        out=np.zeros_like(tn, dtype=float),
        where=negative_denominator != 0
    )

    f1_macro_values = (
        f1_binary_values + f1_negative_values
    ) / 2

    if optimization_metric == "f1-binary":
        optimization_values = f1_binary_values
    else:
        optimization_values = f1_macro_values

    best_index = int(np.argmax(optimization_values))
    best_threshold = float(candidate_thresholds[best_index])

    # Only now calculate the complete metrics for the selected threshold.
    best_metrics = _compute_link_metrics(
        edge_label=edge_label,
        probs=probs,
        threshold=best_threshold
    )

    # Compute the 0.5 baseline using the same exact scores.
    fixed_metrics = _compute_link_metrics(
        edge_label=edge_label,
        probs=probs,
        threshold=0.5
    )

    return best_threshold, best_metrics, fixed_metrics