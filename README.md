# Selective Pruning Experiments
---
Experiments based on paper "Dissecting Language	Models: Machine Unlearning via Selective Pruning" [https://arxiv.org/pdf/2403.01267](https://arxiv.org/pdf/2403.01267).

Install repository and packages with poetry:
```
git clone https://github.com/nickypro/selective-pruning
curl -sSL https://install.python-poetry.org | python -
poetry install
```

The main packages required are:
- taker
- torchvision

[Taker](https://github.com/nickypro/taker) is a library for working with HuggingFace transformers.

### Run experiment with bash:
to run experiments, simply:
```
cd exp1-is-effective
bash prune-llms.sh
```

See a simplified of the procedure in the example:
`exp4-comparison/sp.py`

### Credits
Most files for expriment 4 were taker from [https://github.com/if-loops/selective-synaptic-dampening](selective-synaptic-dampening)

