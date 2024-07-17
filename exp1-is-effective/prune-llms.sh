#!/bin/bash
WANDB_PROJECT="selective-pruning"
DTYPE="fp16" # "bfp16" or "int8" if CPU or Low VRAM

# LANGUAGE MODELS TESTS

# Tuples of form ["retain_set forget_set", ...]
tuples=(
    "pile_codeless code"
    "code pile_codeless"
    "python code"
    "code python"
)

models=(
    "facebook/opt-125m"
    "facebook/opt-1.3b"
    "facebook/opt-6.7b"
    "facebook/galactica-125m"
    "facebook/galactica-1.3b"
    "facebook/galactica-6.7b"
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-6.9b"
    "FacebookAI/roberta-large"
)

# Iterate through the array
for tuple in "${tuples[@]}"; do
    # Split each tuple into variables
    read -r retain_set forget_set <<< "$tuple"

    # run for each model
    for model_repo in "${models[@]}"; do
        poetry run python prune.py $model_repo --dtype $DTYPE \
            --wandb_project $WANDB_PROJECT \
            --focus $retain_set --cripple $forget_set \
            --token_limit 1000 \
            --ff_frac 0.02 --attn_frac 0.00 \
            --collection_sample_size 1e5 \
            --eval_sample_size 1e5 \
            --run_pre_test True \
            --name "$model_repo $retain_set $forget_set"
    done
done
