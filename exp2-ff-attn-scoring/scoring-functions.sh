#!/bin/bash
WANDB_PROJECT="selective-pruning"
DTYPE="fp16" # "bfp16" or "int8" if CPU or Low VRAM

# LANGUAGE MODELS TESTS
retain_set="pile_codeless"
forget_set="code"

scoring_functions=(
    "random"
    "abs"
    "std"
    "rms"
    "freq"
)

models=(
    "facebook/opt-1.3b"
    "facebook/galactica-1.3b"
    "EleutherAI/pythia-1.4b"
    "FacebookAI/roberta-large"
)

# run each scoring function for each model
for model_repo in "${models[@]}"; do
    for fn in "${scoring_functions[@]}"; do
        # Run for FF Only
        poetry run python prune.py $model_repo --dtype $DTYPE \
            --wandb_project $WANDB_PROJECT
            --focus $retain_set --cripple $forget_set \
            --token_limit 1000 \
            --ff_frac 0.02 --attn_frac 0.00 \
            --collection_sample_size 1e5 \
            --eval_sample_size 1e5 \
            --run_pre_test True \
            --name "$model_repo $retain_set $forget_set"

        # Run for ATTN Only
        poetry run python prune.py $model_repo --dtype $DTYPE \
            --wandb_project $WANDB_PROJECT
            --focus $retain_set --cripple $forget_set \
            --token_limit 1000 \
            --ff_frac 0.00 --attn_frac 0.02 \
            --collection_sample_size 1e5 \
            --eval_sample_size 1e5 \
            --run_pre_test True \
            --name "$model_repo $retain_set $forget_set"
    done
done
