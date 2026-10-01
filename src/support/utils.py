import torch
import os
import random
import numpy as np
import pandas as pd
import time



def get_base_dir():
    #return "/home/jovyan/data"
    return "/home/martirano/data"

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def get_time_in_millis():
    return int(round(time.time() * 1000))

def get_target_type(dataset_name):
    if dataset_name == "mumin":
        return "claim"
    elif dataset_name == "politifact":
        return "news"
    else:
        raise Exception("dataset not found")

def get_metapaths_depth(dataset_name): #sto usando 3 per entrambi
    if dataset_name == "mumin":
        return 3
    elif dataset_name == "politifact":
        return 2
    else:
        raise Exception("dataset not found")


"""
def validate_heterodata_indices(data, data_name="data"):
    print(f"\nValidating {data_name}...")

    # PyG built-in validation.
    data.validate(raise_on_error=True)

    for edge_type in data.edge_types:
        source_type, _, target_type = edge_type

        num_source_nodes = data[source_type].num_nodes
        num_target_nodes = data[target_type].num_nodes

        edge_store = data[edge_type]

        tensors_to_check = []

        if hasattr(edge_store, "edge_index"):
            tensors_to_check.append(
                ("edge_index", edge_store.edge_index)
            )

        if hasattr(edge_store, "edge_label_index"):
            tensors_to_check.append(
                (
                    "edge_label_index",
                    edge_store.edge_label_index,
                )
            )

        for tensor_name, edge_index in tensors_to_check:
            if edge_index.numel() == 0:
                print(
                    f"{edge_type} - {tensor_name}: empty"
                )
                continue

            if edge_index.dtype != torch.long:
                raise TypeError(
                    f"{edge_type} - {tensor_name} has dtype "
                    f"{edge_index.dtype}, expected torch.long."
                )

            source_min = int(edge_index[0].min())
            source_max = int(edge_index[0].max())
            target_min = int(edge_index[1].min())
            target_max = int(edge_index[1].max())

            source_valid = (
                source_min >= 0
                and source_max < num_source_nodes
            )

            target_valid = (
                target_min >= 0
                and target_max < num_target_nodes
            )

            print(
                f"{edge_type} - {tensor_name}: "
                f"shape={tuple(edge_index.shape)}, "
                f"source=[{source_min}, {source_max}]"
                f"/{num_source_nodes}, "
                f"target=[{target_min}, {target_max}]"
                f"/{num_target_nodes}"
            )

            if not source_valid or not target_valid:
                invalid_mask = (
                    (edge_index[0] < 0)
                    | (edge_index[0] >= num_source_nodes)
                    | (edge_index[1] < 0)
                    | (edge_index[1] >= num_target_nodes)
                )

                invalid_edges = edge_index[
                    :,
                    invalid_mask,
                ]

                raise ValueError(
                    f"Invalid indices in {data_name}, "
                    f"relation {edge_type}, "
                    f"tensor {tensor_name}.\n"
                    f"Number of invalid edges: "
                    f"{invalid_edges.size(1)}\n"
                    f"First invalid edges:\n"
                    f"{invalid_edges[:, :10]}"
                )

    print(f"{data_name} is valid.")
"""


""" Aggregate experimental results across seeds. """
def process_results(results_file, output_path=None,metric_columns=None, decimals=4):

    results_df = pd.read_csv(results_file)

    # Convert workload and epoch columns to numeric values.
    numeric_columns = [
        "epochs_trained",
        "FLOPs_fwd",
        "FLOPs_bwd",
        "FLOPs_tot",
    ]

    for column in numeric_columns:
        if column in results_df.columns:
            results_df[column] = pd.to_numeric(
                results_df[column],
                errors="coerce",
            )

    # Compute the average computational workload per training epoch.
    if "epochs_trained" in results_df.columns:
        valid_epochs = results_df["epochs_trained"].replace(0,np.nan)

        for flop_column in ["FLOPs_fwd","FLOPs_bwd","FLOPs_tot"]:
            if flop_column in results_df.columns:
                results_df[f"{flop_column}_per_epoch"] = results_df[flop_column] / valid_epochs

    if metric_columns is None:
        default_metrics = [
            # Predictive performance
            "test_auc",
            "test_ap",
            "test_fixed_f1_binary",
            "test_fixed_f1_macro",
            "test_fixed_precision",
            "test_fixed_recall",
            "test_selected_f1_binary",
            "test_selected_f1_macro",
            "test_selected_precision",
            "test_selected_recall",

            # Training behavior
            "epochs_trained",
            "training_time_seconds",

            # Cumulative computational workload
            "FLOPs_fwd",
            "FLOPs_bwd",
            "FLOPs_tot",

            # Average computational workload per epoch
            "FLOPs_fwd_per_epoch",
            "FLOPs_bwd_per_epoch",
            "FLOPs_tot_per_epoch",
        ]

        metric_columns = [
            metric
            for metric in default_metrics
            if metric in results_df.columns
        ]

    group_columns = [
        "dataset_name",
        "interaction_mode",
        "negative_mode",
        "negative_strategy",
        "train_neg_ratio",
        "eval_neg_ratio",
        "negative_resample_every",
        "semi_hard_fraction",
        "candidate_multiplier",
        "semi_hard_lower_quantile",
        "semi_hard_upper_quantile",
        "semi_hard_start_epoch",
        "self_adversarial_alpha",
        "structure_max_attempts_multiplier",
        "disjoint_train_ratio",
        "val_ratio",
        "test_ratio",
        "hidden_channels",
        "num_layers",
        "dropout",
        "learning_rate",
        "weight_decay"
    ]

    grouped_results = results_df.groupby(group_columns, dropna=False, sort=False)

    summary_df = grouped_results.size().rename("n_seeds").reset_index()

    for metric in metric_columns:
        metric_summary = (
            grouped_results[metric].agg(["mean", "std"]).reset_index().rename(
                columns={
                    "mean": f"{metric}_mean",
                    "std": f"{metric}_std",
                }
            )
        )

        metric_summary[f"{metric}_std"] = metric_summary[f"{metric}_std"].fillna(0.0)

        summary_df = summary_df.merge(metric_summary, on=group_columns, how="left")

        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"

        if metric.startswith("FLOPs"):
            summary_df[f"{metric}_mean_std"] = (
                    summary_df[mean_column].map(lambda value: f"{value:.4e}")
                    + " ± "
                    + summary_df[std_column].map(lambda value: f"{value:.4e}")
            )
        else:
            summary_df[f"{metric}_mean_std"] = (
                    summary_df[mean_column].map(lambda value: (f"{value:.{decimals}f}"))
                    + " ± "
                    + summary_df[std_column].map(lambda value: (f"{value:.{decimals}f}"))
            )

    if output_path is not None:

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        summary_df.to_csv(output_path,index=False)
        print(f"Summary saved to {os.path.abspath(output_path)}")

    return summary_df


if __name__ == '__main__':

    base_dir = get_base_dir()
    dataset_names = ["mumin"] #, "politifact"
    for datataset_name in dataset_names:
        res_dir = os.path.join(base_dir, f"{datataset_name}", "results", "link_prediction")
        results_file = os.path.join(res_dir, "all_runs_path_masked.csv")
        process_results(results_file, output_path=os.path.join(res_dir, "results_path_masked.csv"))
