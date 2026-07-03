"""Model configuration.

A single frozen dataclass holding every architectural / solver hyperparameter.
It is hashable, so it can be a static field on a Flax module (no recompiles).
The trainer builds it from the Hydra config via :func:`model_config_from_dict`.

Field names follow the paper notation
(``d_v``, ``d_e``, ``gamma``, ``cg_iters``, ``K_train``/``K_eval``, ...).
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    # --- output ---
    num_classes: int = 6  # per-agent output channels (grid: vocab; sudoku: 10; cls: #classes)

    # --- stalk dims ---
    d_v: int = 64  # vertex stalk (comm_dim)
    d_e: int = 32  # edge stalk

    # --- encoder ---
    encoder_arch: str = "mlp_v2"  # "mlp_v2" (maze) | "mlp" (mnist) | "sudoku"
    enc_hidden_dim: int = 256  # mlp_v2 hidden width
    enc_d_model: int = 128  # sudoku token width
    enc_num_blocks: int = 2  # sudoku mixer blocks
    comm_norm_type: str = "layernorm"
    dropout_rate: float = 0.0

    # --- local objective (x-update) ---
    objective_mode: str = "l1box_diag"  # simple|quadratic|lasso|non_negative|l1box_diag
    x_solver: str = "diagonal_prox"  # simple|diagonal_prox (dense_quadratic is test/extension-only)
    l1_weight: float = 0.0  # scalar L1 for objective_mode="lasso"
    l1_init: float = 0.01  # per-dim L1 init for "l1box_diag"
    upper_init: float = 1.0  # per-dim box upper init for "l1box_diag"
    beta_init: float = 1.0  # tether stiffness for "simple"
    q_epsilon: float = 1e-4

    # --- consensus (z-update) ---
    z_solver: str = "unrolled_cg"  # unrolled_cg | gd
    z_mode: str = "prox"  # project (hard Fz=0) | prox (soft gamma)
    gamma: float = 1.0  # soft-consensus weight (paper gamma)
    cg_iters: int = 5  # T (unrolled CG steps)
    tikhonov_eps: float = 1e-5  # project-mode L+eps*I
    prox_init: str = "legacy"  # legacy | warm
    gd_steps: int = 10  # GD inner diffusion steps
    gd_momentum: float = 0.9

    # --- restriction maps / geometry ---
    rm_sharing: str = "directional"  # global | directional | sudoku
    rm_init: str = "orthonormal"  # orthonormal | default | identity | soft_slice (sudoku)
    rm_mode: str = "fixed"  # fixed | context (LoRA)
    lora_rank: int = 4
    lora_alpha: float = 1.0
    lora_use_gate: bool = False
    lora_init_style: str = "legacy"  # legacy (B=0) | standard (A=0)
    num_directions: int = 4  # grid 4/8; sudoku LoRA uses 9

    # --- ADMM penalty / step ---
    rho_init: float = 0.1
    rho_learnable: bool = True
    eta_init: float = 0.01  # GD step size
    eta_min: float | None = None  # bounded-log eta leash (with eta_max)
    eta_max: float | None = None
    eta_learnable: bool = True
    relaxation_alpha: float = 1.0
    z_init: str = "h"  # z^0 seed: "h" (encoder embedding, default) | "zeros" (Algorithm 1)

    # --- recurrent-MPNN baseline (ignored by SheafADMMModel) ---
    mpnn_message_dim: int | None = None  # message width (defaults to d_e)
    mpnn_aggregation: str = "max"  # add | mean | symnorm | max
    mpnn_edge_type_mode: str = "spatial"  # shared (1) | spatial (4/8) | slot (9, sudoku)
    mpnn_graph_readout: str = "per_node"  # per_node (maze/sudoku) | graph (MNIST)

    # --- decoder ---
    decoder_arch: str = "mlp_concat_v2"  # mlp_concat_v2 | sudoku | classification
    dec_hidden_dim: int = 256  # mlp_concat_v2
    dec_hidden_dims: tuple[int, ...] = (128,)  # sudoku / classification
    dec_linear_head: bool = False  # classification linear head (MNIST)
    dec_readout_mode: str = "concat"  # classification: concat | x_only
    dec_norm_type: str = "rmsnorm"


def model_config_from_dict(d: dict[str, Any]) -> ModelConfig:
    """Build a :class:`ModelConfig`, coercing list fields to tuples and dropping ``None`` overrides.

    Unknown keys raise (no silent typos). Intended to be fed the merged
    ``model`` + ``solver`` Hydra subtrees.
    """
    out = dict(d)
    # Legacy checkpoints/configs may carry pre-public no-op fields. They never
    # selected code paths in the public model, so ignore them on load.
    for key in ("task", "mpnn_num_rounds"):
        out.pop(key, None)
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    unknown = set(out) - fields
    if unknown:
        raise KeyError(f"unknown ModelConfig fields: {sorted(unknown)}")
    for k in ("dec_hidden_dims",):
        if k in out and out[k] is not None:
            out[k] = tuple(out[k])
    return ModelConfig(**out)
