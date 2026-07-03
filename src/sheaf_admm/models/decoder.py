"""Per-agent decoders: final agent state ``x`` (+ raw patch) -> logits.

Like the encoder, decoders are **shared across agents**: the parent flattens
``[N, B, d_v] -> [N*B, d_v]``, decodes once, and reshapes the logits back. So
each module here takes a single leading batch axis ``B' = N*B``.

Three decoders cover the paper tasks:

* :class:`ConcatMLPDecoderV2` (``arch="mlp_concat_v2"``) — Maze and the MNIST
  reconstruction path. A one-hidden-layer MLP over ``concat(flatten(patch), x)``,
  reshaped to the output patch grid.
* :class:`SudokuDecoder` (``arch="sudoku"``) — Sudoku. Reshapes ``[B', 288] ->
  [B', 9, 32]``, runs shared MLP blocks over the 9 cells, and projects each to 10
  digit logits.
* :class:`ClassificationDecoder` (``arch="classification"``) — MNIST
  classification. A linear or MLP head over ``x`` alone (``readout_mode="x_only"``,
  the MNIST path) or ``concat([x, patch])``.

All decoders share the ``(x, patches, *, training)`` call signature so the parent
can dispatch them uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from .layers import MLPBlock, RMSNorm


class ConcatMLPDecoderV2(nn.Module):
    """Single-hidden-layer MLP decoder over ``concat(flatten(patch), x)``.

    Architecture: ``RMSNorm(concat([flatten(patch), x])) -> Dense(hidden_dim) ->
    GELU -> [Dropout] -> Dense(prod(output_shape))``, reshaped to ``output_shape``.
    Maze (and the MNIST reconstruction path).
    """

    hidden_dim: int = 256
    output_shape: tuple[int, ...] = (5, 5, 6)  # (H, W, C) or (H*W, C)
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,  # [B', comm_dim] final agent state
        patches: jnp.ndarray,  # [B', H, W, C] input patch
        training: bool = True,
    ) -> jnp.ndarray:
        B = x.shape[0]
        patch_flat = patches.reshape((B, -1))

        h = jnp.concatenate([patch_flat, x], axis=-1)
        h = RMSNorm(name="input_norm")(h)
        h = nn.Dense(self.hidden_dim, name="dense")(h)
        h = jax.nn.gelu(h)
        if self.dropout_rate > 0:
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)

        out_size = int(np.prod(self.output_shape))
        h = nn.Dense(out_size, name="output_dense")(h)
        return h.reshape((B, *self.output_shape))


class SudokuDecoder(nn.Module):
    """Decode the consensus vector to per-cell digit logits (``arch="sudoku"``).

    ``[B', 9*cell_dim] -> [B', 9, cell_dim]``, then shared :class:`MLPBlock` s over
    the 9 cells (Flax applies Dense over the last axis, sharing across cells), then
    ``Dense(output_channels)`` -> ``[B', 9, output_channels]``.
    """

    hidden_dims: Sequence[int] = (128,)
    output_channels: int = 10
    norm_type: str = "rmsnorm"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,  # [B', comm_dim]
        patches: jnp.ndarray | None = None,
        training: bool = True,
    ) -> jnp.ndarray:
        B = x.shape[0]
        x = x.reshape(B, 9, -1)
        for i, dim in enumerate(self.hidden_dims):
            x = MLPBlock(out_dim=dim, norm_type=self.norm_type, name=f"block_{i}")(x)
        return nn.Dense(self.output_channels, name="output_dense")(x)


class ClassificationDecoder(nn.Module):
    """Per-agent classification head (``arch="classification"``).

    ``readout_mode`` selects the head input: ``x_only`` (the final agent state
    alone — the MNIST path) or ``concat`` (``[x, flatten(patch)]``).
    ``linear_head=True`` is a bare ``Dense(num_classes)``; otherwise the input runs
    through ``Dense(hidden_dims[0])`` then :class:`MLPBlock` s before the output.
    """

    hidden_dims: Sequence[int] = (128,)
    output_channels: int = 10
    norm_type: str = "rmsnorm"
    linear_head: bool = False
    readout_mode: str = "concat"  # "concat" | "x_only"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,  # [B', comm_dim]
        patches: jnp.ndarray,  # [B', H, W, C]
        training: bool = True,
    ) -> jnp.ndarray:
        B = x.shape[0]
        if self.readout_mode == "concat":
            readout = jnp.concatenate([x, patches.reshape((B, -1))], axis=-1)
        elif self.readout_mode == "x_only":
            readout = x
        else:
            raise ValueError(f"unknown readout_mode={self.readout_mode!r} (concat|x_only)")

        if self.linear_head:
            return nn.Dense(features=self.output_channels, name="cls_output")(readout)

        h = nn.Dense(self.hidden_dims[0], name="input_proj")(readout)
        for i, dim in enumerate(self.hidden_dims):
            h = MLPBlock(out_dim=dim, norm_type=self.norm_type, name=f"block_{i}")(h)
        return nn.Dense(features=self.output_channels, name="cls_output")(h)


def create_decoder(arch: str = "mlp_concat_v2", **kwargs) -> nn.Module:
    """Build a decoder module by name.

    ``arch`` in ``{"mlp_concat_v2", "sudoku", "classification"}``. Unknown kwargs
    for the chosen arch raise (no silent drops).
    """
    if arch == "mlp_concat_v2":
        return ConcatMLPDecoderV2(**kwargs)
    if arch == "sudoku":
        return SudokuDecoder(**kwargs)
    if arch == "classification":
        return ClassificationDecoder(**kwargs)
    raise ValueError(f"unknown decoder arch={arch!r} (mlp_concat_v2|sudoku|classification)")
