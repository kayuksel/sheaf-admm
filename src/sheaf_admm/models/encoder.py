"""Per-agent encoders: patch -> consensus vector ``h`` + local-objective params.

The encoder is the only learned map from raw local views to the convex
subproblem each agent solves. It is **shared across agents**: the parent model
flattens the agent and batch axes (``[N, B, ...] -> [N*B, ...]``), applies the
encoder once, and reshapes the outputs back to ``[N, B, ...]``. So every module
here operates on a single leading batch axis ``B' = N*B``; the ``[N, B, ...]``
contract in the solvers is the parent's responsibility.

Three architectures cover the tasks:

* :class:`MLPEncoderV2` (``arch="mlp_v2"``) — Maze. A one-hidden-layer MLP over
  the flattened patch.
* :class:`MLPEncoder` (``arch="mlp"``) — MNIST. A residual-MLP over the patch.
* :class:`SudokuEncoder` (``arch="sudoku"``) — Sudoku. An MLP-Mixer over the 9
  cells of an agent's row/col/box view, with a per-cell projection that lays out
  the 9 cell embeddings contiguously in the ``comm_dim`` stalk.

Objective heads (see :mod:`sheaf_admm.solvers.x_solvers`) turn the encoder
features into the local-objective parameters. The diagonal curvature is emitted
as a ``q_diag`` **vector** ``[B', d]`` (not a dense ``[B', d, d]`` matrix),
matching the unified ``diagonal_prox`` x-solver.

Emitted ``encoder_output`` keys by ``objective_mode``:

* ``simple``: ``h``, ``beta`` (scalar learned tether stiffness).
* ``quadratic``: ``h``, ``q_diag``, ``q``.
* ``lasso`` (MNIST): ``h``, ``q_diag``, ``q``, ``l1_weight`` (scalar config).
* ``non_negative`` (Sudoku): ``h``, ``q_diag``, ``q``, ``lower=0``.
* ``l1box_diag`` (Maze): ``h``, ``q_diag``, ``q``, ``l1_weight`` (per-dim),
  ``upper`` (per-dim), ``lower=0``.

LoRA modulation (``rm_mode="context"``) additionally emits ``A`` ``[B',K,d_e,r]``,
``B`` ``[B',K,d_v,r]`` (legacy init: ``A=lecun_normal``, ``B=zeros`` so the
initial ``A B^T = 0``), and an optional ``gate`` ``[B',K]``. These feed
``create_lora_geometry`` / ``create_sudoku_lora_geometry`` after the parent
reshapes them to ``[N, B, K, ...]``. ``K`` is the number of grid directions
(4/8) or 9 (Sudoku cell-slots).
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from sheaf_admm.admm import inverse_softplus

from .layers import CommHead, MLPBlock, MLPMixerBlock, RMSNorm


def _use_standard_lora_init(init_style: str) -> bool:
    """``legacy`` (A=random, B=0) vs ``standard`` (A=0, B=random, Hu et al. 2021)."""
    style = (init_style or "").lower()
    if style not in ("legacy", "standard"):
        raise ValueError(f"unknown lora_init_style={init_style!r} (legacy|standard)")
    return style == "standard"


# ---------------------------------------------------------------------------
# Objective heads: encoder features -> local-objective parameters.
# Each takes pooled/flattened features [B', feat] and emits a parameter tensor.
# ---------------------------------------------------------------------------


def diag_curvature(
    x: jnp.ndarray, comm_dim: int, q_epsilon: float, softplus: bool, name: str
) -> jnp.ndarray:
    """Per-coordinate curvature ``q_diag`` ``[B', d]``, strictly positive.

    ``softplus(raw) + q_epsilon`` (default) keeps ``Q`` PSD; ``softplus=False``
    allows negative diagonals (only ``Q + rho I`` needs to be PSD).
    """
    raw = nn.Dense(comm_dim, name=name)(x)
    return (jax.nn.softplus(raw) if softplus else raw) + q_epsilon


def linear_term(x: jnp.ndarray, comm_dim: int, name: str) -> jnp.ndarray:
    """Linear objective term ``q`` ``[B', d]`` (the paper uses a dense vector)."""
    return nn.Dense(comm_dim, name=name)(x)


def softplus_head(x: jnp.ndarray, out_dim: int, init_value: float, name: str) -> jnp.ndarray:
    """Positive per-dim head ``softplus(Dense(x))`` with bias seeded to ``init_value``.

    Kernel is zero-init so the output starts uniformly at ``init_value``
    (``softplus(inverse_softplus(init_value)) = init_value``). Used for per-dim
    L1 weights (init 0.01) and box upper bounds (init 1.0).
    """
    raw = nn.Dense(
        out_dim,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.constant(inverse_softplus(init_value)),
        name=name,
    )(x)
    return jax.nn.softplus(raw)


def emit_lora_factors(
    x: jnp.ndarray,
    *,
    num_slots: int,
    d_e: int,
    d_v: int,
    rank: int,
    init_style: str,
    use_gate: bool,
) -> dict[str, jnp.ndarray]:
    """LoRA factors for restriction-map modulation ``F = R + (alpha/r) A B^T``.

    ``A`` ``[B', K, d_e, r]`` up-projects into edge space; ``B`` ``[B', K, d_v, r]``
    down-projects from vertex space. Legacy init (``B=0``) gives initial
    ``A B^T = 0`` so training starts from the fixed base maps. ``x`` should be the
    LoRA-prenorm features; ``K = num_slots`` (grid directions or Sudoku slots).
    """
    B = x.shape[0]
    use_standard = _use_standard_lora_init(init_style)
    zero_init = nn.initializers.zeros
    default_init = nn.initializers.lecun_normal()

    A_flat = nn.Dense(
        num_slots * d_e * rank,
        kernel_init=zero_init if use_standard else default_init,
        name="lora_A_dense",
    )(x)
    B_flat = nn.Dense(
        num_slots * d_v * rank,
        kernel_init=default_init if use_standard else zero_init,
        name="lora_B_dense",
    )(x)
    out = {
        "A": A_flat.reshape((B, num_slots, d_e, rank)),
        "B": B_flat.reshape((B, num_slots, d_v, rank)),
    }
    if use_gate:
        gate_logits = nn.Dense(
            num_slots, bias_init=nn.initializers.constant(-2.0), name="lora_gate_dense"
        )(x)
        out["gate"] = jax.nn.sigmoid(gate_logits)
    return out


class _ObjectiveHeads(nn.Module):
    """Objective heads shared by both encoders (a submodule so heads own params).

    Reads pooled/flattened features and emits the local-objective parameters for
    the chosen ``objective_mode``; the ``simple`` mode emits a single learned
    scalar tether stiffness ``beta = softplus(beta_raw)``. See the module
    docstring for the key contract.
    """

    objective_mode: str
    comm_dim: int
    q_epsilon: float = 1e-4
    q_diag_softplus: bool = True
    l1_weight: float = 0.0
    l1_init: float = 0.01
    upper_init: float = 1.0
    beta_init: float = 1.0

    @nn.compact
    def __call__(self, h: jnp.ndarray, feats: jnp.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {"h": h}

        if self.objective_mode == "simple":
            beta_raw = self.param(
                "beta_raw",
                lambda k, s, d=jnp.float32: jnp.full(s, inverse_softplus(self.beta_init), d),
                (),
            )
            out["beta"] = jax.nn.softplus(beta_raw)
            return out

        if self.objective_mode in ("quadratic", "lasso", "non_negative", "l1box_diag"):
            out["q_diag"] = diag_curvature(
                feats, self.comm_dim, self.q_epsilon, self.q_diag_softplus, "q_diag_dense"
            )
            out["q"] = linear_term(feats, self.comm_dim, "q_dense")
        else:
            raise ValueError(
                f"unknown objective_mode={self.objective_mode!r} "
                "(simple|quadratic|lasso|non_negative|l1box_diag)"
            )

        if self.objective_mode == "lasso":
            # Scalar L1 weight from config (MNIST); no per-dim head.
            out["l1_weight"] = jnp.asarray(self.l1_weight, dtype=feats.dtype)
        elif self.objective_mode == "non_negative":
            out["lower"] = 0.0
        elif self.objective_mode == "l1box_diag":
            out["l1_weight"] = softplus_head(feats, self.comm_dim, self.l1_init, "l1_weight_dense")
            out["upper"] = softplus_head(feats, self.comm_dim, self.upper_init, "upper_bound_dense")
            out["lower"] = 0.0
        return out


class MLPEncoderV2(nn.Module):
    """Single-hidden-layer MLP encoder (``arch="mlp_v2"``). Maze + MNIST.

    Architecture: ``flatten -> RMSNorm -> Dense(hidden_dim) -> GELU -> [Dropout]
    -> CommHead(comm_dim)``, then objective and (optional) LoRA heads.
    """

    hidden_dim: int = 256
    comm_dim: int = 64
    edge_stalk_dim: int | None = None
    comm_norm_type: str = "layernorm_zeros"
    dropout_rate: float = 0.0

    objective_mode: str = "simple"
    q_epsilon: float = 1e-4
    q_diag_softplus: bool = True
    l1_weight: float = 0.0  # scalar L1 weight for objective_mode="lasso"
    l1_init: float = 0.01  # per-dim L1 init for objective_mode="l1box_diag"
    upper_init: float = 1.0  # per-dim box upper-bound init for "l1box_diag"
    beta_init: float = 1.0  # tether stiffness init for objective_mode="simple"

    rm_mode: str = "fixed"  # "fixed" | "context" (LoRA)
    lora_rank: int = 4
    lora_use_gate: bool = False
    lora_init_style: str = "legacy"
    num_directions: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> dict[str, Any]:
        """Encode patches ``[B', H, W, C]`` -> ``encoder_output`` dict (see module doc)."""
        B = x.shape[0]
        x = x.reshape((B, -1))
        x = RMSNorm(name="input_norm")(x)
        x = nn.Dense(self.hidden_dim, name="dense")(x)
        x = jax.nn.gelu(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        h = CommHead(
            communication_dim=self.comm_dim,
            comm_norm_type=self.comm_norm_type,
            name="comm_head",
        )(x)

        output = _ObjectiveHeads(
            objective_mode=self.objective_mode,
            comm_dim=self.comm_dim,
            q_epsilon=self.q_epsilon,
            q_diag_softplus=self.q_diag_softplus,
            l1_weight=self.l1_weight,
            l1_init=self.l1_init,
            upper_init=self.upper_init,
            beta_init=self.beta_init,
            name="objective_heads",
        )(h, x)

        if self.rm_mode == "context":
            d_e = self.edge_stalk_dim if self.edge_stalk_dim is not None else self.comm_dim
            x_lora = nn.LayerNorm(name="lora_pre_ln")(x)
            output.update(
                emit_lora_factors(
                    x_lora,
                    num_slots=self.num_directions,
                    d_e=d_e,
                    d_v=self.comm_dim,
                    rank=self.lora_rank,
                    init_style=self.lora_init_style,
                    use_gate=self.lora_use_gate,
                )
            )
        return output


class SudokuEncoder(nn.Module):
    """MLP-Mixer encoder over the 9 cells of an agent's view (``arch="sudoku"``).

    Architecture: token embed (std ``1/sqrt(d_model)``) + optional global
    cell-position ``Embed(81)`` + learned relative position, scaled by
    ``sqrt(d_model)/sqrt(2)``; ``num_blocks`` mixer blocks (``mlp_ratio=2``,
    SwiGLU); RMSNorm; per-cell shared ``Dense(comm_dim//9)``; RMSNorm; flatten the
    9 cell blocks contiguously into ``[B', comm_dim]`` (requires ``comm_dim % 9 ==
    0``); CommHead. With ``comm_dim=288`` each cell occupies 32 dims.

    The objective/LoRA heads read the mean-pooled post-mixer tokens ``x_pooled``.
    LoRA ``K = num_directions`` (set to 9 for Sudoku cell-slots).
    """

    d_model: int = 128
    num_blocks: int = 2
    mlp_ratio: float = 2.0
    mlp_type: str = "swiglu"
    comm_dim: int = 288
    edge_stalk_dim: int | None = None
    comm_norm_type: str = "layernorm_zeros"

    objective_mode: str = "simple"
    q_epsilon: float = 1e-4
    q_diag_softplus: bool = True
    l1_weight: float = 0.0
    l1_init: float = 0.01
    upper_init: float = 1.0
    beta_init: float = 1.0

    rm_mode: str = "fixed"
    lora_rank: int = 4
    lora_use_gate: bool = False
    lora_init_style: str = "legacy"
    num_directions: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, cell_ids: jnp.ndarray | None = None) -> dict[str, Any]:
        """Encode cell views ``[B', 9, 10]`` -> ``encoder_output`` dict.

        ``cell_ids`` ``[B', 9]`` (optional) are global board-cell indices 0..80 for
        the absolute-position ``Embed(81)``.
        """
        B, T, _ = x.shape
        embed_scale = math.sqrt(self.d_model)

        x_embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.normal(stddev=1.0 / embed_scale),
            name="token_embed",
        )(x)

        if cell_ids is not None:
            x_embed = x_embed + nn.Embed(
                num_embeddings=81,
                features=self.d_model,
                embedding_init=nn.initializers.normal(stddev=1.0 / embed_scale),
                name="global_pos_embed",
            )(cell_ids)

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=1.0 / embed_scale),
            (1, T, self.d_model),
        )
        x_embed = (x_embed + pos_embed) * (embed_scale / math.sqrt(2))

        token_mlp_dim = int(T * self.mlp_ratio)
        channel_mlp_dim = int(self.d_model * self.mlp_ratio)
        y = x_embed
        for i in range(self.num_blocks):
            y = MLPMixerBlock(
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim,
                mlp_type=self.mlp_type,
                name=f"mixer_block_{i}",
            )(y)
        y = RMSNorm(name="pre_flat_norm")(y)

        if self.comm_dim % 9 != 0:
            raise ValueError(f"comm_dim must be divisible by 9, got {self.comm_dim}")
        cell_dim = self.comm_dim // 9

        # Per-cell projection shared across the 9 tokens, then RMSNorm, then a
        # contiguous flatten so cell k occupies stalk block [k*cell_dim, ...].
        y_cells = nn.Dense(cell_dim, name="cell_proj")(y)
        y_cells = RMSNorm(name="cell_norm")(y_cells)
        h = y_cells.reshape(B, 9 * cell_dim)

        h = CommHead(
            communication_dim=self.comm_dim,
            comm_norm_type=self.comm_norm_type,
            name="comm_head",
        )(h)

        x_pooled = jnp.mean(y, axis=1)

        output = _ObjectiveHeads(
            objective_mode=self.objective_mode,
            comm_dim=self.comm_dim,
            q_epsilon=self.q_epsilon,
            q_diag_softplus=self.q_diag_softplus,
            l1_weight=self.l1_weight,
            l1_init=self.l1_init,
            upper_init=self.upper_init,
            beta_init=self.beta_init,
            name="objective_heads",
        )(h, x_pooled)

        if self.rm_mode == "context":
            d_e = self.edge_stalk_dim if self.edge_stalk_dim is not None else self.comm_dim
            x_lora = RMSNorm(name="lora_pre_norm")(x_pooled)
            output.update(
                emit_lora_factors(
                    x_lora,
                    num_slots=self.num_directions,
                    d_e=d_e,
                    d_v=self.comm_dim,
                    rank=self.lora_rank,
                    init_style=self.lora_init_style,
                    use_gate=self.lora_use_gate,
                )
            )
        return output


class MLPEncoder(nn.Module):
    """Residual-MLP encoder (``arch="mlp"``). The camera-ready MNIST encoder.

    Architecture: ``flatten -> Dense(hidden_dims[0]) -> MLPBlock x len(hidden_dims)``
    (each a pre-norm + GELU + residual block) ``-> CommHead(comm_dim)``, then the
    objective and (optional) LoRA heads. Differs from :class:`MLPEncoderV2` (which
    is a single norm->Dense->GELU) by the input projection + residual blocks.
    """

    hidden_dims: tuple[int, ...] = (256,)
    comm_dim: int = 32
    edge_stalk_dim: int | None = None
    comm_norm_type: str = "layernorm_zeros"
    norm_type: str = "rmsnorm"
    dropout_rate: float = 0.0

    objective_mode: str = "simple"
    q_epsilon: float = 1e-4
    q_diag_softplus: bool = True
    l1_weight: float = 0.0
    l1_init: float = 0.01
    upper_init: float = 1.0
    beta_init: float = 1.0

    rm_mode: str = "fixed"
    lora_rank: int = 4
    lora_use_gate: bool = False
    lora_init_style: str = "legacy"
    num_directions: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> dict[str, Any]:
        B = x.shape[0]
        x = x.reshape((B, -1))
        x = nn.Dense(self.hidden_dims[0], name="input_proj")(x)
        for i, dim in enumerate(self.hidden_dims):
            x = MLPBlock(out_dim=dim, norm_type=self.norm_type, name=f"block_{i}")(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        h = CommHead(
            communication_dim=self.comm_dim, comm_norm_type=self.comm_norm_type, name="comm_head"
        )(x)
        output = _ObjectiveHeads(
            objective_mode=self.objective_mode,
            comm_dim=self.comm_dim,
            q_epsilon=self.q_epsilon,
            q_diag_softplus=self.q_diag_softplus,
            l1_weight=self.l1_weight,
            l1_init=self.l1_init,
            upper_init=self.upper_init,
            beta_init=self.beta_init,
            name="objective_heads",
        )(h, x)
        if self.rm_mode == "context":
            d_e = self.edge_stalk_dim if self.edge_stalk_dim is not None else self.comm_dim
            x_lora = nn.LayerNorm(name="lora_pre_ln")(x)
            output.update(
                emit_lora_factors(
                    x_lora,
                    num_slots=self.num_directions,
                    d_e=d_e,
                    d_v=self.comm_dim,
                    rank=self.lora_rank,
                    init_style=self.lora_init_style,
                    use_gate=self.lora_use_gate,
                )
            )
        return output


def create_encoder(arch: str = "mlp_v2", **kwargs: Any) -> nn.Module:
    """Build an encoder module by name.

    ``arch`` in ``{"mlp_v2", "mlp", "sudoku"}``: ``mlp_v2`` (Maze, and the simple
    single-layer MLP), ``mlp`` (the camera-ready MNIST residual-MLP encoder),
    ``sudoku`` (the MLP-Mixer cell encoder). Unknown kwargs for the chosen arch
    raise (no silent drops).
    """
    if arch == "mlp_v2":
        return MLPEncoderV2(**kwargs)
    if arch == "mlp":
        return MLPEncoder(**kwargs)
    if arch == "sudoku":
        return SudokuEncoder(**kwargs)
    raise ValueError(f"unknown encoder arch={arch!r} (mlp_v2|mlp|sudoku)")
