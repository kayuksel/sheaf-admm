"""Directional gated-GNN baseline: the MPNN ablation of the ADMM coordinator.

This is the message-passing counterpart to the sheaf-ADMM loop. It reuses the
**same** shared encoder (``objective_mode="simple"``, ``rm_mode="fixed"``) and
decoder, but replaces the unrolled ADMM core with a recurrent gated graph net:
agents exchange typed messages over the agent graph for ``num_rounds`` rounds,
with weights shared across rounds (``nn.scan(variable_broadcast="params")``).

Direction typing matches the geometry: ``shared`` (1 type), ``spatial``
(connectivity 4/8 grid directions), or ``slot`` (9 Sudoku cell-slots).

States carry the agent/batch layout ``[N, B, d]`` throughout (the segment ops in
:class:`DirectionalGGNNCell` reduce over the N node axis; the B and d axes ride
along). The parent supplies the encoder context ``[N, B, comm_dim]`` and the
directed edge list / direction ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from .layers import MLPBlock, RMSNorm


class GraphClassificationHead(nn.Module):
    """Graph-level readout: pooled node state -> class logits (MNIST MPNN).

    ``linear_head=True`` is a bare ``Dense(output_channels)``; otherwise the
    pooled vector runs through ``Dense(hidden_dims[0])`` then :class:`MLPBlock` s.
    Consumes a mean/sum pool of the final node states ``[B', comm_dim]``.
    """

    hidden_dims: Sequence[int] = (128,)
    output_channels: int = 10
    norm_type: str = "rmsnorm"
    linear_head: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.linear_head:
            return nn.Dense(features=self.output_channels, name="cls_output")(x)
        h = nn.Dense(self.hidden_dims[0], name="input_proj")(x)
        for i, dim in enumerate(self.hidden_dims):
            h = MLPBlock(out_dim=dim, norm_type=self.norm_type, name=f"block_{i}")(h)
        return nn.Dense(features=self.output_channels, name="cls_output")(h)


class DirectionalGGNNCell(nn.Module):
    """One gated message-passing round with direction-typed linear messages.

    Per edge ``(src -> dst)`` with direction ``k``: message
    ``W_k h_src + b_k`` (``W_dir`` ``[K, message_dim, hidden_dim]``), aggregated
    over incoming edges (``add`` / ``mean`` / ``symnorm`` / ``max``), projected
    ``message_dim -> hidden_dim`` if they differ, RMS/LayerNorm-ed, concatenated
    with the (norm-ed) encoder ``context`` injected **every round**, then folded
    into the hidden state by a GRU-like reset/update gate. Weights are per-cell;
    the round-sharing happens in :class:`DirectionalGGNN`.
    """

    hidden_dim: int
    message_dim: int
    num_directions: int
    aggregation: str = "add"  # "add" | "mean" | "symnorm" | "max"
    norm_type: str = "rmsnorm"
    eps: float = 1e-6

    def _norm(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        if self.norm_type == "layernorm":
            return nn.LayerNorm(epsilon=self.eps, name=name)(x)
        return RMSNorm(eps=self.eps, name=name)(x)

    def _aggregate(
        self,
        messages: jnp.ndarray,  # [E, B, m]
        src: jnp.ndarray,  # [E]
        dst: jnp.ndarray,  # [E]
        num_nodes: int,
        self_messages: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Reduce per-edge messages onto destination nodes -> ``[N, B, m]``."""
        if messages.shape[0] == 0:
            agg = jnp.zeros((num_nodes, messages.shape[1], messages.shape[2]), messages.dtype)
            if self.aggregation == "symnorm" and self_messages is not None:
                return agg + self_messages
            return agg

        if self.aggregation == "add":
            return jax.ops.segment_sum(messages, dst, num_segments=num_nodes)

        if self.aggregation == "mean":
            sums = jax.ops.segment_sum(messages, dst, num_segments=num_nodes)
            ones = jnp.ones((messages.shape[0], 1, 1), messages.dtype)
            counts = jax.ops.segment_sum(ones, dst, num_segments=num_nodes)
            return sums / jnp.maximum(counts, 1.0)

        if self.aggregation == "symnorm":
            if self_messages is None:
                raise ValueError("symnorm aggregation requires self_messages")
            ones = jnp.ones((messages.shape[0],), messages.dtype)
            src_deg = jax.ops.segment_sum(ones, src, num_segments=num_nodes) + 1.0
            dst_deg = jax.ops.segment_sum(ones, dst, num_segments=num_nodes) + 1.0
            edge_scale = jax.lax.rsqrt(src_deg[src] * dst_deg[dst])[:, None, None]
            edge_sum = jax.ops.segment_sum(messages * edge_scale, dst, num_segments=num_nodes)
            self_scale = (1.0 / dst_deg)[:, None, None]
            return edge_sum + self_messages * self_scale

        if self.aggregation == "max":
            neg_inf = jnp.full(
                (num_nodes, messages.shape[1], messages.shape[2]), -jnp.inf, messages.dtype
            )
            maxed = neg_inf.at[dst].max(messages)
            return jnp.where(jnp.isfinite(maxed), maxed, 0.0)

        raise ValueError(f"unknown aggregation={self.aggregation!r} (add|mean|symnorm|max)")

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,  # [N, B, hidden_dim]
        context: jnp.ndarray,  # [N, B, comm_dim] encoder context
        edge_indices: jnp.ndarray,  # [E, 2] directed (src, dst)
        direction_ids: jnp.ndarray,  # [E] direction type per edge
    ) -> jnp.ndarray:
        num_nodes = hidden.shape[0]
        src, dst = edge_indices[:, 0], edge_indices[:, 1]

        dir_kernel = self.param(
            "dir_kernel",
            nn.initializers.xavier_uniform(),
            (self.num_directions, self.message_dim, self.hidden_dim),
        )
        dir_bias = self.param(
            "dir_bias", nn.initializers.zeros, (self.num_directions, self.message_dim)
        )
        edge_kernel = dir_kernel[direction_ids]  # [E, m, d]
        edge_bias = dir_bias[direction_ids]  # [E, m]
        messages = jnp.einsum("emd,ebd->ebm", edge_kernel, hidden[src])
        messages = messages + edge_bias[:, None, :]

        if self.aggregation == "symnorm":
            if self.message_dim == self.hidden_dim:
                self_messages = hidden
            else:
                self_messages = nn.Dense(
                    self.message_dim, use_bias=False, name="self_message_to_edge"
                )(hidden)
        else:
            self_messages = None

        aggregated = self._aggregate(messages, src, dst, num_nodes, self_messages)
        if self.message_dim != self.hidden_dim:
            aggregated = nn.Dense(self.hidden_dim, use_bias=False, name="message_to_hidden")(
                aggregated
            )

        aggregated = self._norm(aggregated, name="aggregated_norm")
        context = self._norm(context, name="context_norm")
        update_inputs = jnp.concatenate([aggregated, context], axis=-1)
        update_inputs = nn.Dense(self.hidden_dim, name="update_input")(update_inputs)

        gate_inputs = self._norm(
            jnp.concatenate([hidden, update_inputs], axis=-1), name="gate_norm"
        )
        reset_gate = nn.sigmoid(nn.Dense(self.hidden_dim, name="reset_gate")(gate_inputs))
        update_gate = nn.sigmoid(nn.Dense(self.hidden_dim, name="update_gate")(gate_inputs))

        candidate_inputs = self._norm(
            jnp.concatenate([reset_gate * hidden, update_inputs], axis=-1),
            name="candidate_norm",
        )
        candidate = nn.tanh(nn.Dense(self.hidden_dim, name="candidate")(candidate_inputs))
        return (1.0 - update_gate) * candidate + update_gate * hidden


class DirectionalGGNN(nn.Module):
    """Round-recurrent directional GGNN processor (weights shared across rounds).

    Runs :class:`DirectionalGGNNCell` for ``num_rounds`` rounds via
    ``nn.scan(variable_broadcast="params")`` so a single set of cell weights is
    reused every round (the recurrent ablation of the ADMM loop). Returns the
    final node hidden state ``[N, B, hidden_dim]``; with ``return_history=True``
    also returns the stacked per-round states ``[num_rounds, N, B, hidden_dim]``.
    """

    hidden_dim: int
    message_dim: int
    num_directions: int
    aggregation: str = "add"
    norm_type: str = "rmsnorm"
    eps: float = 1e-6

    @nn.compact
    def __call__(
        self,
        hidden0: jnp.ndarray,  # [N, B, hidden_dim]
        context: jnp.ndarray,  # [N, B, comm_dim]
        edge_indices: jnp.ndarray,  # [E, 2] directed
        direction_ids: jnp.ndarray,  # [E]
        num_rounds: int,
        return_history: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")

        cell = DirectionalGGNNCell(
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim,
            num_directions=self.num_directions,
            aggregation=self.aggregation,
            norm_type=self.norm_type,
            eps=self.eps,
            name="cell",
        )

        def body(cell: DirectionalGGNNCell, hidden: jnp.ndarray, _step: Any):
            next_hidden = cell(hidden, context, edge_indices, direction_ids)
            return next_hidden, (next_hidden if return_history else None)

        scan = nn.scan(
            body,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=num_rounds,
        )
        steps = jnp.arange(num_rounds, dtype=jnp.int32)
        hidden_final, history = scan(cell, hidden0, steps)
        if return_history:
            return hidden_final, history
        return hidden_final
