#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/projects/GNNLinkPrediction


# experiment setting parameters
dataset_names=("mumin" "politifact")
seeds=(42 96 132 123 2026)
interaction_modes=("any" "1" "2" "3" "4" "5")
negative_modes=("none" "static" "dynamic")
negative_strategies=("random" "mixed" "mixed_self_adversarial" "mixed_structure_aware"")
train_neg_ratios=(1.0 2.0 3.0 5.0)
disjoint_train_ratios=(0.0 0.2 0.3)


for dataset_name in "${dataset_names[@]}"; do
  for interaction_mode in "${interaction_modes[@]}"; do
    for negative_mode in "${negative_modes[@]}"; do
      for negative_strategy in "${negative_strategies[@]}"; do
        for disjoint_train_ratio in "${disjoint_train_ratios[@]}"; do

          # "none" and "static" do not use semi-hard negatives.
          if [[ "$negative_mode" != "dynamic" \
                && "$negative_strategy" != "random" ]]; then
            continue
          fi

          for train_neg_ratio in "${train_neg_ratios[@]}"; do

            if [[ "$negative_mode" == "none" ]]; then
              effective_train_neg_ratio="0.0"
            else
              effective_train_neg_ratio="$train_neg_ratio"
            fi

            for seed in "${seeds[@]}"; do
              echo
              echo "============================================================"
              echo "Dataset: ${dataset_name}"
              echo "Interaction mode: ${interaction_mode}"
              echo "Negative mode: ${negative_mode}"
              echo "Negative strategy: ${negative_strategy}"
              echo "Train negative ratio: ${effective_train_neg_ratio}"
              echo "Disjoint train ratio: ${disjoint_train_ratio}"
              echo "Seed: ${seed}"
              echo "============================================================"

              python src/main.py \
                --dataset-name "$dataset_name" \
                --interaction-mode "$interaction_mode" \
                --negative-mode "$negative_mode" \
                --negative-strategy "$negative_strategy" \
                --train-neg-ratio "$effective_train_neg_ratio" \
                --disjoint-train-ratio "$disjoint_train_ratio" \
                --seed "$seed" \
                --path-mask

            done
          done
        done
      done
    done
  done
done
