"""The recurrent-MPNN baseline model.

The baseline uses the same task decompositions and decoder interfaces as
:class:`~sheaf_admm.models.sheaf_model.SheafADMMModel`, but replaces the
unrolled ADMM loop with a directional gated-GNN
(:class:`~sheaf_admm.models.mpnn.DirectionalGGNN`).

The encoder is used in ``objective_mode="simple"`` / ``rm_mode="fixed"`` (the GGNN
consumes only the consensus seed ``h`` as its per-round context, not the convex
objective params). Readout is per-agent decode + overlap-average (Maze/Sudoku) or
graph-level mean-pool (MNIST), matching the Sheaf model's readout per task.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from sheaf_admm.geometry import compute_direction_index

from .config import ModelConfig
from .decoder import create_decoder
from .encoder import create_encoder
from .mpnn import DirectionalGGNN, GraphClassificationHead


class MPNNModel(nn.Module):
    """Directional-GGNN baseline (see module docstring)."""

    config: ModelConfig

    def _directed_edges(self, edge_indices, node_positions, map_u, map_v):
        """Double the undirected edges and assign a direction/type id per directed edge."""
        c = self.config
        u, v = edge_indices[:, 0], edge_indices[:, 1]
        directed = jnp.concatenate([edge_indices, jnp.stack([v, u], axis=1)], axis=0)

        if c.mpnn_edge_type_mode == "shared":
            dir_ids = jnp.zeros((directed.shape[0],), dtype=jnp.int32)
            num_types = 1
        elif c.mpnn_edge_type_mode == "spatial":
            dy = node_positions[v, 0] - node_positions[u, 0]
            dx = node_positions[v, 1] - node_positions[u, 1]
            fwd = compute_direction_index(dy, dx, c.num_directions)
            bwd = compute_direction_index(-dy, -dx, c.num_directions)
            dir_ids = jnp.concatenate([fwd, bwd]).astype(jnp.int32)
            num_types = c.num_directions
        elif c.mpnn_edge_type_mode == "slot":
            dir_ids = jnp.concatenate([map_u, map_v]).astype(jnp.int32)
            num_types = 9
        else:
            raise ValueError(f"unknown mpnn_edge_type_mode={c.mpnn_edge_type_mode!r}")
        return directed, dir_ids, num_types

    @nn.compact
    def __call__(
        self,
        patches: jnp.ndarray,  # [N,B,...]
        edge_indices: jnp.ndarray,  # [E, 2] undirected
        *,
        num_rounds: int,
        node_positions: jnp.ndarray | None = None,
        map_u: jnp.ndarray | None = None,
        map_v: jnp.ndarray | None = None,
        cell_ids: jnp.ndarray | None = None,
        training: bool = True,
    ):
        c = self.config
        N, B = patches.shape[:2]

        # Shared encoder (simple/fixed) -> per-agent context h.
        enc_kwargs = dict(
            comm_dim=c.d_v,
            edge_stalk_dim=c.d_e,
            comm_norm_type=c.comm_norm_type,
            objective_mode="simple",
            rm_mode="fixed",
            beta_init=c.beta_init,
        )
        if c.encoder_arch == "mlp_v2":
            enc_kwargs.update(hidden_dim=c.enc_hidden_dim, dropout_rate=c.dropout_rate)
            encoder = create_encoder("mlp_v2", **enc_kwargs)
            enc = encoder(patches.reshape((N * B, *patches.shape[2:])), training=training)
        elif c.encoder_arch == "mlp":
            enc_kwargs.update(hidden_dims=(c.enc_hidden_dim,), dropout_rate=c.dropout_rate)
            encoder = create_encoder("mlp", **enc_kwargs)
            enc = encoder(patches.reshape((N * B, *patches.shape[2:])), training=training)
        elif c.encoder_arch == "sudoku":
            enc_kwargs.update(d_model=c.enc_d_model, num_blocks=c.enc_num_blocks)
            encoder = create_encoder("sudoku", **enc_kwargs)
            cids = None
            if cell_ids is not None:
                cids = jnp.broadcast_to(cell_ids[:, None, :], (N, B, cell_ids.shape[-1]))
                cids = cids.reshape((N * B, cell_ids.shape[-1]))
            enc = encoder(patches.reshape((N * B, *patches.shape[2:])), cids)
        else:
            raise ValueError(f"unknown encoder_arch={c.encoder_arch!r}")
        context = enc["h"].reshape((N, B, c.d_v))

        hidden0 = nn.Dense(c.d_v, name="init_hidden")(context)
        directed, dir_ids, num_types = self._directed_edges(
            edge_indices, node_positions, map_u, map_v
        )
        ggnn = DirectionalGGNN(
            hidden_dim=c.d_v,
            message_dim=c.mpnn_message_dim or c.d_e,
            num_directions=num_types,
            aggregation=c.mpnn_aggregation,
            name="ggnn",
        )
        hidden_final = ggnn(hidden0, context, directed, dir_ids, num_rounds)

        patches_flat = patches.reshape((N * B, *patches.shape[2:]))
        if c.mpnn_graph_readout == "graph":
            pooled = jnp.mean(hidden_final, axis=0)  # [B, d_v]
            head = GraphClassificationHead(
                hidden_dims=c.dec_hidden_dims,
                output_channels=c.num_classes,
                norm_type=c.dec_norm_type,
                linear_head=c.dec_linear_head,
            )
            return head(pooled), hidden_final

        # per-agent decode (shared decoder), then [N,B,*out]
        dec_kwargs: dict = {}
        if c.decoder_arch == "mlp_concat_v2":
            dec_kwargs = dict(
                hidden_dim=c.dec_hidden_dim,
                output_shape=(*patches.shape[2:-1], c.num_classes),
                dropout_rate=c.dropout_rate,
            )
        elif c.decoder_arch == "sudoku":
            dec_kwargs = dict(
                hidden_dims=c.dec_hidden_dims,
                output_channels=c.num_classes,
                norm_type=c.dec_norm_type,
            )
        elif c.decoder_arch == "classification":
            dec_kwargs = dict(
                hidden_dims=c.dec_hidden_dims,
                output_channels=c.num_classes,
                norm_type=c.dec_norm_type,
                linear_head=c.dec_linear_head,
                readout_mode=c.dec_readout_mode,
            )
        decoder = create_decoder(c.decoder_arch, **dec_kwargs)
        logits = decoder(hidden_final.reshape((N * B, c.d_v)), patches_flat, training=training)
        return logits.reshape((N, B, *logits.shape[1:])), hidden_final
