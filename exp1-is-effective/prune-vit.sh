#!/bin/bash
WANDB_PROJECT="selective-pruning"
DTYPE="fp32" # small model, should be good on all systems

# VISION TRANSFORMER TESTS

# Tuples of form ["retain_set forget_set", ...]
tuples=(
    "imagenet-1k-birdless imagenet-1k-birds"
    "imagenet-1k-birds imagenet-1k-birdless"
)

models=(
    "google/vit-base-patch16-224"
)

# Iterate through the array
for tuple in "${tuples[@]}"; do
    # Split each tuple into variables
    read -r retain_set forget_set <<< "$tuple"

    # run for each model
    for model_repo in "${models[@]}"; do
        # Note: collection_sample_size is number of 16x16 "tokens"
        # .     eval_sample_size is number of tested images
        poetry run python prune.py $model_repo \
            --dtype $DTYPE \
            --wandb_project $WANDB_PROJECT \
            --focus   $retain_set \
            --cripple $forget_set \
            --token_limit 1000 \
            --ff_frac 0.02 --attn_frac 0.00 \
            --collection_sample_size 1e5 \
            --eval_sample_size 1e3 \
	        --recalculate_activations false \
            --name "$model_repo $retain_set $forget_set" "$@"
    done
done