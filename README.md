# Sheaf-ADMM

![Sheaf-ADMM overview](assets/fig2.png)

Official JAX/Flax implementation of **Learning Multi-Agent Coordination via
Sheaf-ADMM** (ICML 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2605.31005-b31b1b?style=flat-square)](https://arxiv.org/abs/2605.31005)
[![Blog](https://img.shields.io/badge/Blog-Sakana%20AI-1f6feb?style=flat-square)](https://pub.sakana.ai/sheaf-admm/)

Sheaf-ADMM decomposes an input into overlapping local views, each processed by an
agent that solves a small convex subproblem parameterized by a neural encoder.
Agents coordinate through the Alternating Direction Method of Multipliers (ADMM),
with the inter-agent constraints specified by a *cellular sheaf* — which aspects
of neighboring solutions must agree. The optimization is unrolled for a fixed
number of iterations, so the whole pipeline is differentiable and every component
is trained end-to-end.

## Installation

This repository is intended to be run from a source checkout. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
```

For CUDA experiments, install a JAX build matching your driver:

```bash
uv pip install -U "jax[cuda12]"
```

## Data

```bash
uv run python -m sheaf_admm.data.build_maze \
  --height 19 \
  --width 19 \
  --train-size 10000 \
  --test-size 1000 \
  --min-path-length 18 \
  --ood-sizes \
  --output-dir datasets/maze_std3_19px_10k

uv run python -m sheaf_admm.data.build_mnist \
  --output-dir datasets/mnist

uv run python -m sheaf_admm.data.build_sudoku \
  --output-dir datasets/sudoku_easy
```

The MNIST and Sudoku builders download their source datasets on first run.

## Training

Each task has a Sheaf-ADMM model and a recurrent-MPNN baseline:

```bash
# Maze
uv run python scripts/train.py +experiment=maze_sheaf
uv run python scripts/train.py +experiment=maze_mpnn

# MNIST
uv run python scripts/train.py +experiment=mnist_sheaf
uv run python scripts/train.py +experiment=mnist_mpnn

# Sudoku
uv run python scripts/train.py +experiment=sudoku_sheaf
uv run python scripts/train.py +experiment=sudoku_sheaf_lora
uv run python scripts/train.py +experiment=sudoku_mpnn
```

Set `training.seed=42`, `123`, or `456` for the paper seeds. Set
`wandb.mode=online` to enable Weights & Biases logging. Checkpoints and
`history.json` are written to Hydra's run directory under `outputs/`.

## Visualization

The visualization script expects a Sheaf-ADMM checkpoint:

```bash
uv run python -m scripts.visualize \
  --checkpoint outputs/<date>/<time>/checkpoint.pkl \
  --out-dir /tmp/sheaf_admm_viz
```

## Checks

```bash
uv run python -m ruff check .
uv run python -m pytest -q
```

## Citation

```bibtex
@inproceedings{sheafadmm2026,
  title     = {Learning Multi-Agent Coordination via Sheaf-ADMM},
  author    = {Seely, Jeffrey and Cupia{\l}, Bart{\l}omiej and Jones, Llion},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
