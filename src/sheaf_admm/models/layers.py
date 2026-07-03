"""Shared neural building blocks for the encoders, decoders, and MPNN baseline.

Everything here is purely a network layer; the ADMM/sheaf machinery lives
elsewhere.

Conventions matched exactly to the paper runs:

* ``RMSNorm`` — learnable per-channel scale, ``eps = 1e-6`` under the sqrt.
* ``MLPBlock`` — pre-norm residual MLP ``x + Dense(GELU(Dense(Norm(x))))``, with a
  learned residual projection when in/out widths differ.
* ``MLPMixerBlock`` — TRM-style *post-norm* token/channel mixing with SwiGLU (or
  GELU) sub-MLPs.
* ``CommHead`` — final linear projection to the ``comm_dim`` consensus vector with
  an optional normalization (the shipped configs use ``layernorm``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers


def rms_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """RMS normalization without learnable scale: ``x / sqrt(mean(x^2) + eps)``."""
    variance = jnp.mean(x**2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps)


class RMSNorm(nn.Module):
    """RMS layer normalization with a learnable per-channel scale (init 1)."""

    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        return rms_norm(x, self.eps) * scale


class MLPBlock(nn.Module):
    """Pre-norm residual MLP block: ``x + Dense(GELU(Dense(Norm(x))))``.

    Uses a learned linear projection on the residual path when the input and
    output widths differ. ``hidden_dim`` defaults to ``out_dim``.
    """

    out_dim: int
    hidden_dim: int | None = None  # defaults to out_dim
    norm_type: str = "rmsnorm"  # 'rmsnorm' | 'layernorm'
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_dim = x.shape[-1]
        hidden_dim = self.hidden_dim if self.hidden_dim is not None else self.out_dim

        if self.norm_type == "rmsnorm":
            h = RMSNorm(eps=self.eps, name="norm")(x)
        else:
            h = nn.LayerNorm(epsilon=self.eps, name="norm")(x)

        h = nn.Dense(hidden_dim, name="dense1")(h)
        h = jax.nn.gelu(h)
        h = nn.Dense(self.out_dim, name="dense2")(h)

        if in_dim != self.out_dim:
            x = nn.Dense(self.out_dim, name="residual_proj")(x)
        return x + h


class SwiGLU(nn.Module):
    """SwiGLU MLP: ``Dense_down(silu(gate) * up)`` with a fused gate/up projection.

    From Shazeer (2020), "GLU Variants Improve Transformer"; used by the Sudoku
    mixer. The fused ``[gate, up]`` projection is a single matmul.
    """

    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate_up = nn.Dense(2 * self.hidden_dim, name="gate_up")(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        hidden = jax.nn.silu(gate) * up
        return nn.Dense(self.out_dim, name="down")(hidden)


class GeLUMLP(nn.Module):
    """Two-layer GELU MLP ``Dense_down(gelu(Dense_up(x)))`` (no gating pathway)."""

    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, name="up")(x)
        x = jax.nn.gelu(x)
        return nn.Dense(self.out_dim, name="down")(x)


class MLPMixerBlock(nn.Module):
    """TRM-style post-norm MLP-Mixer block over ``[B, T, C]``.

    Token mixing: ``x = RMSNorm(x + TokenMLP(x^T)^T)``;
    channel mixing: ``x = RMSNorm(x + ChannelMLP(x))``. ``mlp_type`` selects the
    SwiGLU or GELU sub-MLP (the paper Sudoku encoder uses ``swiglu``).
    """

    token_mlp_dim: int
    channel_mlp_dim: int
    mlp_type: str = "swiglu"  # "swiglu" | "gelu"
    rms_eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        T, C = x.shape[1], x.shape[2]

        def make_mlp(hidden_dim: int, out_dim: int, name: str) -> nn.Module:
            if self.mlp_type == "swiglu":
                return SwiGLU(hidden_dim=hidden_dim, out_dim=out_dim, name=name)
            if self.mlp_type == "gelu":
                return GeLUMLP(hidden_dim=hidden_dim, out_dim=out_dim, name=name)
            raise ValueError(f"unknown mlp_type={self.mlp_type!r} (swiglu|gelu)")

        # Token mixing (post-norm): mix across the T axis.
        y = jnp.swapaxes(x, 1, 2)  # [B, C, T]
        y = make_mlp(self.token_mlp_dim, T, "token_mlp")(y)
        y = jnp.swapaxes(y, 1, 2)  # [B, T, C]
        x = RMSNorm(eps=self.rms_eps, name="token_norm")(x + y)

        # Channel mixing (post-norm): mix across the C axis.
        y = make_mlp(self.channel_mlp_dim, C, "channel_mlp")(x)
        x = RMSNorm(eps=self.rms_eps, name="channel_norm")(x + y)
        return x


class CommHead(nn.Module):
    """Project encoder features to the ``comm_dim`` consensus vector.

    ``comm_norm_type`` post-processes the projection:

    * ``identity`` — no normalization;
    * ``tanh`` — squash to ``[-1, 1]``;
    * ``layernorm`` — standard LayerNorm (the shipped configs' setting);
    * ``layernorm_zeros`` — LayerNorm initialized near-off (scale ``1e-4``, bias
      0), so the consensus vector starts close to zero.
    """

    communication_dim: int
    comm_norm_type: str = "identity"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        comm = nn.Dense(features=self.communication_dim, name="comm_dense")(x)

        if self.comm_norm_type == "tanh":
            comm = nn.tanh(comm)
        elif self.comm_norm_type == "layernorm":
            comm = nn.LayerNorm(name="comm_norm")(comm)
        elif self.comm_norm_type == "layernorm_zeros":
            comm = nn.LayerNorm(
                scale_init=initializers.constant(1e-4),
                bias_init=initializers.zeros,
                name="comm_norm",
            )(comm)
        elif self.comm_norm_type != "identity":
            raise ValueError(
                f"unknown comm_norm_type={self.comm_norm_type!r} "
                "(identity|tanh|layernorm|layernorm_zeros)"
            )
        return comm
