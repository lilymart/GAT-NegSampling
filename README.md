# GAT-NegSampling

Official implementation of the paper:

**Learning from the Unobserved: How Negative Sampling Shapes User Engagement Prediction in Heterogeneous Social Networks**

The paper has been accepted at **Complex Networks & Their Applications 2026**.  
The final bibliographic reference will be added after publication of the proceedings.

> This repository investigates how negative supervision affects user–news engagement prediction in heterogeneous social networks. It compares static, dynamic, model-informed, and structure-aware negative sampling strategies, together with different negative-to-positive ratios, computational cost analysis, and interaction-specific path-masking experiments.

## Requirements
The required dependencies can be installed with:
```
pip install -r requirements.txt
```

The main dependencies include PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, and FLOPpy.
Depending on the installed PyTorch and CUDA versions, some PyTorch Geometric extension packages may require compatible pre-built wheels. Please ensure that the PyG installation matches your local PyTorch/CUDA configuration.

## Data
The experiments consider two heterogeneous misinformation-related social networks:
- MuMiN, where the prediction target is the user–claim relation;
- PolitiFact, where the prediction target is the user–news relation.
The target user–content relation is not assumed to be explicitly available in the original graph. Instead, it is automatically constructed from user-to-target interaction meta-paths discovered from the heterogeneous schema.
The current data loader expects each dataset to follow the structure:

```
data/
└── dataset_name/
    └── heterodata/
        ├── features/
        │   ├── user.pt
            ├── tweet.pt
        │   ├── ...
        │   └── target_type.pt
        ├── edgelists/
        │   ├── user_posts_tweet.pt
        │   └── ...
        └── processed/
            └── metapaths_exact_v2/
```

Node features and heterogeneous graph connectivity are stored as PyTorch tensors and assembled into a PyTorch Geometric *HeteroData* object. Edge-list filenames are expected to follow *sourceType_relationType_targetType.pt*. The *processed/* directory is used to cache automatically generated user-to-target interaction meta-paths. The root data directory is currently configured through 'get_base_dir()' in 'src/support/utils.py' and should be adapted to the local environment before running the experiments.


## Training and evaluation
Multiple configurations can be executed through:
```
bash src/run_experiments.sh
```

The main parameters are:
- *dataset-name*: dataset to process (mumin or politifact).
- *interaction-mode: target interaction relation. Use 'any' for the union of all discovered meta-paths or a '1-based meta-path ID' (1, 2, ...).
- *seed*: random seed used for splitting, initialization, sampling, and training (default: 42).
- *hidden-channels*: dimensionality of hidden node representations (default: 64).
- *num-layers*: number of GATv2 layers (default: 3).
- *dropout*: dropout probability (default: 0.3).
- *learning-rate*: optimizer learning rate (default: 0.005).
- *weight-decay*: optimizer weight decay (default: 0.0005).
- *n-epoch*s: maximum number of training epochs (default: 500).
- *patience*: early-stopping patience (default: 50).
- *epsilon*: minimum improvement required by early stopping (default: 1e-5).
- *threshold-metric*: validation metric used for threshold selection (f1-binary or f1-macro, default: f1-macro).
- *val-ratio*: fraction of positive links assigned to validation (default: 0.15).
- *test-ratio*: fraction of positive links assigned to test (default: 0.25).
- *train-neg-ratio*: training negative-to-positive ratio (default: 1.0).
- *eval-neg-ratio*: validation/test negative-to-positive ratio (default: 1.0).
- *disjoint-train-ratio*: fraction of training positive links reserved for decoder supervision (default: 0.3).
- *negative-mode*: negative-supervision mode (none, static, or dynamic).
- *negative-strategy*: negative-sampling strategy (random, mixed, mixed_self_adversarial, or mixed_structure_aware).
- *path-mask*: optional flag enabling the path-masking experiment.
  
The implemented negative-sampling settings correspond to:
- *none*: positive-only sanity-check ablation;
- *static* + random: one random negative set retained throughout training;
- *dynamic* + *random*: dynamically resampled random negatives;
- *dynamic* + *mixed*: quantile-based mixed sampling;
- *dynamic* + *mixed_self_adversarial*: self-adversarial mixed sampling;
- *dynamic* + *mixed_structure_aware*: structure-aware mixed sampling.

    Note: A single experiment can be launched with, e.g., 'python src/main.py --dataset-name mumin --interaction-mode any --negative-mode dynamic --negative-strategy mixed --train-neg-ratio 1.0 --disjoint-train-ratio 0.2 --seed 42

## Results
Experimental results are stored under:
```
data/
└── dataset_name/
    └── results/
        └── link_prediction/
```
Standard experiments are appended to 'all_runs.csv' while path-masked experiments are stored in 'all_runs_path_masked.csv'. The output includes the experimental configuration, predictive metrics, selected validation threshold, training time, number of trained epochs, and computational-cost measurements.
Re-running the same experimental configuration and seed replaces the corresponding previous result.



## Reference

> Coming soon...
