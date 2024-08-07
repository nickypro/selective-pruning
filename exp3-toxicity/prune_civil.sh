# PRUNE CIVIL TOXIC

prune_gpt2() {
    poetry run python prune_30.py "gpt2-large" \
        --wandb_project civil-toxic \
        --focus civil --cripple toxic --additional_datasets wiki,toxicity \
        --recalculate_activations True \
        --run_pre_test True \
        --ff_scoring abs --attn_scoring abs \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}

prune_llama2() {
    poetry run python prune_30.py "meta-llama/Llama-2-7b-hf" \
        --wandb_project civil-toxic \
        --focus civil --cripple toxic --additional_datasets wiki,toxicity2,mmlu:all \
        --recalculate_activations True \
        --run_pre_test True \
        --ff_scoring abs --attn_scoring abs \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}

prune_gpt2 "gpt2-large 0% 0.5%" --ff_frac 0.0 --attn_frac 0.005 --n_steps 10
prune_llama2 "llama 7b 0% 0.5%" --ff_frac 0.005 --attn_frac 0.005 --n_steps 1
