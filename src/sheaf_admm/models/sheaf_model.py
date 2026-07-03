"""The full Sheaf-ADMM model: encode -> coordinate (ADMM) -> decode.

This ties the pieces together as one Flax module:

1. a **shared encoder** maps each agent's local view to its convex-subproblem
   parameters and a consensus seed ``h``;
2. **restriction maps** (shared by direction / globally / by Sudoku cell-slot,
   optionally LoRA-modulated by the encoder) define the sheaf geometry;
3. the **unrolled ADMM loop** (:func:`sheaf_admm.admm.run_admm`) coordinates the
   agents with learned penalty ``rho`` (and step ``eta`` for the GD z-solver);
4. a **shared decoder** maps the final per-agent states to per-agent logits.

The model returns per-agent logits over the final ``loss_window`` ADMM iterates;
the trainer handles the loss-window averaging and (for eval) the overlap
reassembly into a global prediction. The agent graph is passed in per call, so
the same trained weights apply to larger inputs (size generalization).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from sheaf_admm.admm import inverse_softplus, run_admm, run_admm_history
from sheaf_admm.geometry import (
    DirectionalRestrictionMaps,
    FixedGeometry,
    SharedRestrictionMap,
    SudokuRestrictionMaps,
    build_directional_restriction_maps,
    build_shared_restriction_maps,
    build_sudoku_restriction_maps,
    create_lora_geometry,
    create_sudoku_lora_geometry,
)
from sheaf_admm.solvers.x_solvers import make_x_solver
from sheaf_admm.solvers.z_solvers import GDParams, UnrolledCGParams, make_z_solver

from .config import ModelConfig
from .decoder import create_decoder
from .encoder import create_encoder

_ARRAY_KEYS = ("h", "q_diag", "q", "l1_weight", "upper", "A", "B", "gate")


def _logit(p: float) -> float:
    return float(jnp.log(p / (1.0 - p)))


class SheafADMMModel(nn.Module):
    """End-to-end Sheaf-ADMM model (see module docstring)."""

    config: ModelConfig

    # ---- learned ADMM scalars (offset-softplus so the init is exact) ----
    def _learned_scalar(self, name: str, init: float, learnable: bool) -> jnp.ndarray:
        delta = self.param(name, nn.initializers.zeros, ())
        if not learnable:
            delta = jax.lax.stop_gradient(delta)
        return jax.nn.softplus(delta + inverse_softplus(init))

    def _learned_eta(self) -> jnp.ndarray:
        c = self.config
        if c.eta_min is not None and c.eta_max is not None:
            # Bounded log-space leash: keeps the GD step size inside [eta_min, eta_max]
            # so the inner diffusion cannot diverge.
            lo, hi = jnp.log(c.eta_min), jnp.log(c.eta_max)
            t0 = jnp.clip((jnp.log(c.eta_init) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
            raw = self.param("eta_raw", nn.initializers.zeros, ()) + _logit(float(t0))
            if not c.eta_learnable:
                raw = jax.lax.stop_gradient(raw)
            return jnp.exp(lo + jax.nn.sigmoid(raw) * (hi - lo))
        return self._learned_scalar("eta_raw", c.eta_init, c.eta_learnable)

    # ---- encoder application (shared across agents) ----
    def _encode(self, patches, cell_ids, training):
        c = self.config
        N, B = patches.shape[:2]
        enc_kwargs = dict(
            comm_dim=c.d_v,
            edge_stalk_dim=c.d_e,
            comm_norm_type=c.comm_norm_type,
            objective_mode=c.objective_mode,
            q_epsilon=c.q_epsilon,
            l1_weight=c.l1_weight,
            l1_init=c.l1_init,
            upper_init=c.upper_init,
            beta_init=c.beta_init,
            rm_mode=c.rm_mode,
            lora_rank=c.lora_rank,
            lora_use_gate=c.lora_use_gate,
            lora_init_style=c.lora_init_style,
            num_directions=c.num_directions,
        )
        if c.encoder_arch == "mlp_v2":
            enc_kwargs.update(hidden_dim=c.enc_hidden_dim, dropout_rate=c.dropout_rate)
            encoder = create_encoder("mlp_v2", **enc_kwargs)
            flat = encoder(patches.reshape((N * B, *patches.shape[2:])), training=training)
        elif c.encoder_arch == "mlp":
            enc_kwargs.update(hidden_dims=(c.enc_hidden_dim,), dropout_rate=c.dropout_rate)
            encoder = create_encoder("mlp", **enc_kwargs)
            flat = encoder(patches.reshape((N * B, *patches.shape[2:])), training=training)
        elif c.encoder_arch == "sudoku":
            enc_kwargs.update(d_model=c.enc_d_model, num_blocks=c.enc_num_blocks)
            encoder = create_encoder("sudoku", **enc_kwargs)
            cids = None
            if cell_ids is not None:
                cids = jnp.broadcast_to(cell_ids[:, None, :], (N, B, cell_ids.shape[-1]))
                cids = cids.reshape((N * B, cell_ids.shape[-1]))
            flat = encoder(patches.reshape((N * B, *patches.shape[2:])), cids)
        else:
            raise ValueError(f"unknown encoder_arch={c.encoder_arch!r}")

        out = dict(flat)
        for k in _ARRAY_KEYS:
            if k in out and jnp.ndim(out[k]) >= 1 and out[k].shape[0] == N * B:
                out[k] = out[k].reshape((N, B, *out[k].shape[1:]))
        return out

    # ---- geometry assembly ----
    def _build_geometry(self, enc_out, edge_indices, node_positions, map_u, map_v):
        c = self.config
        E = edge_indices.shape[0]
        if c.rm_sharing == "directional":
            R_dict = DirectionalRestrictionMaps(
                stalk_dim=c.d_v,
                edge_stalk_dim=c.d_e,
                init_method=c.rm_init,
                num_directions=c.num_directions,
                name="rm",
            )()
            base = build_directional_restriction_maps(
                R_dict, edge_indices, node_positions, c.num_directions
            )
        elif c.rm_sharing == "global":
            R = SharedRestrictionMap(
                stalk_dim=c.d_v, edge_stalk_dim=c.d_e, init_method=c.rm_init, name="rm"
            )()
            base = build_shared_restriction_maps(R, E)
        elif c.rm_sharing == "sudoku":
            R_stack = SudokuRestrictionMaps(
                stalk_dim=c.d_v, edge_stalk_dim=c.d_e, init_method=c.rm_init, name="rm"
            )()
            base = build_sudoku_restriction_maps(R_stack, map_u, map_v)
        else:
            raise ValueError(f"unknown rm_sharing={c.rm_sharing!r} (directional|global|sudoku)")

        if c.rm_mode == "fixed":
            return FixedGeometry(edge_indices=edge_indices, restriction_maps=base)
        if c.rm_mode == "context":
            A, Bf, gate = enc_out["A"], enc_out["B"], enc_out.get("gate")
            if c.rm_sharing == "sudoku":
                return create_sudoku_lora_geometry(
                    edge_indices, map_u, map_v, base, A, Bf, c.lora_alpha, gate=gate
                )
            return create_lora_geometry(
                edge_indices, node_positions, base, A, Bf, c.lora_alpha, c.num_directions, gate=gate
            )
        raise ValueError(f"unknown rm_mode={c.rm_mode!r} (fixed|context)")

    # ---- z-solver params (rho/eta baked in per call) ----
    def _z_params(self, rho):
        c = self.config
        if c.z_solver == "unrolled_cg":
            return UnrolledCGParams(
                mode=c.z_mode,
                gamma=c.gamma,
                num_iters=c.cg_iters,
                tikhonov_eps=c.tikhonov_eps,
                prox_init=c.prox_init,
            )
        if c.z_solver == "gd":
            return GDParams(
                mode=c.z_mode,
                gamma=c.gamma,
                num_steps=c.gd_steps,
                momentum=c.gd_momentum,
                eta=self._learned_eta(),
            )
        raise ValueError(f"unknown z_solver={c.z_solver!r} (unrolled_cg|gd)")

    def _decode_window(self, x_window, patches, training):
        """Decode each of the W windowed agent states -> per-agent logits [W, N, B, *out]."""
        c = self.config
        W, N, B = x_window.shape[:3]
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
        else:
            raise ValueError(f"unknown decoder_arch={c.decoder_arch!r}")
        decoder = create_decoder(c.decoder_arch, **dec_kwargs)

        patches_flat = patches.reshape((N * B, *patches.shape[2:]))
        outs = []
        for w in range(W):
            xw = x_window[w].reshape((N * B, c.d_v))
            logits = decoder(xw, patches_flat, training=training)
            outs.append(logits.reshape((N, B, *logits.shape[1:])))
        return jnp.stack(outs, axis=0)

    def _setup_admm(self, patches, edge_indices, node_positions, map_u, map_v, cell_ids, training):
        """Build everything the ADMM loop needs (encoder out, geometry, solvers, rho, z_init).

        Shared by :meth:`__call__` and :meth:`coordinate_history` so the two
        entry points see identical parameters and setup.
        """
        c = self.config
        enc_out = self._encode(patches, cell_ids, training)
        geometry = self._build_geometry(enc_out, edge_indices, node_positions, map_u, map_v)
        rho = self._learned_scalar("rho_raw", c.rho_init, c.rho_learnable)
        x_solver, x_params = make_x_solver(c.x_solver)
        z_solver, _ = make_z_solver(c.z_solver)
        z_params = self._z_params(rho)
        z_init = enc_out["h"] if c.z_init == "h" else jnp.zeros_like(enc_out["h"])
        return enc_out, geometry, rho, x_solver, x_params, z_solver, z_params, z_init

    @nn.compact
    def __call__(
        self,
        patches: jnp.ndarray,  # grid: [N,B,ph,pw,C]; sudoku: [N,B,9,10]
        edge_indices: jnp.ndarray,  # [E, 2]
        *,
        num_iters: int,
        loss_window: int = 1,
        grad_window: int | None = None,
        node_positions: jnp.ndarray | None = None,  # grid [N,2]
        map_u: jnp.ndarray | None = None,  # sudoku [E]
        map_v: jnp.ndarray | None = None,  # sudoku [E]
        cell_ids: jnp.ndarray | None = None,  # sudoku [N,9]
        training: bool = True,
    ):
        c = self.config
        enc_out, geometry, rho, x_solver, x_params, z_solver, z_params, z_init = self._setup_admm(
            patches, edge_indices, node_positions, map_u, map_v, cell_ids, training
        )
        final_state, x_window = run_admm(
            enc_out,
            geometry,
            x_solver,
            x_params,
            z_solver,
            z_params,
            rho,
            z_init,
            num_iters,
            relaxation_alpha=c.relaxation_alpha,
            loss_window=loss_window,
            grad_window=grad_window,
        )
        logits_window = self._decode_window(x_window, patches, training)
        return logits_window, final_state, geometry

    @nn.compact
    def coordinate_history(
        self,
        patches: jnp.ndarray,
        edge_indices: jnp.ndarray,
        *,
        num_iters: int,
        node_positions: jnp.ndarray | None = None,
        map_u: jnp.ndarray | None = None,
        map_v: jnp.ndarray | None = None,
        cell_ids: jnp.ndarray | None = None,
        training: bool = False,
    ):
        """Forward-only path returning the full per-iteration ADMM trajectory.

        Mirrors :meth:`__call__` but records every iterate and its residuals (see
        :class:`sheaf_admm.admm.ADMMHistory`) and decodes each local proposal
        ``x^k`` to per-agent logits ``[K, N, B, *out]``. Invoke via
        ``model.apply(params, ..., method=SheafADMMModel.coordinate_history)``;
        because the submodule names match :meth:`__call__`, it reuses the trained
        parameters without re-initialization.
        """
        c = self.config
        enc_out, geometry, rho, x_solver, x_params, z_solver, z_params, z_init = self._setup_admm(
            patches, edge_indices, node_positions, map_u, map_v, cell_ids, training
        )
        final_state, history = run_admm_history(
            enc_out,
            geometry,
            x_solver,
            x_params,
            z_solver,
            z_params,
            rho,
            z_init,
            num_iters,
            relaxation_alpha=c.relaxation_alpha,
        )
        # Decode each local proposal x^k -> per-agent logits [K, N, B, *out]
        # (the prediction is read off the x-iterate, matching the training loss).
        logits_per_iter = self._decode_window(history.x, patches, training)
        return history, logits_per_iter, final_state, geometry, rho
