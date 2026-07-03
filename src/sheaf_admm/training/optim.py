"""Optimizer, learning-rate schedule, train state, and EMA.

Matches the paper's shared optimization setup: AdamW, a 200-step linear warmup
to a constant learning rate, global-norm gradient clipping at 1.0, and an
exponential moving average of parameters (decay 0.999) used at evaluation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state


def make_schedule(learning_rate: float, warmup_steps: int) -> optax.Schedule:
    """Linear warmup ``0 -> lr`` over ``warmup_steps``, then constant ``lr``."""
    if warmup_steps <= 0:
        return optax.constant_schedule(learning_rate)
    warmup = optax.linear_schedule(0.0, learning_rate, warmup_steps)
    return optax.join_schedules(
        [warmup, optax.constant_schedule(learning_rate)], boundaries=[warmup_steps]
    )


def make_optimizer(
    learning_rate: float,
    weight_decay: float = 0.0,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
) -> optax.GradientTransformation:
    """AdamW with warmup→constant LR; global-norm clip applied *before* the update."""
    tx = optax.adamw(make_schedule(learning_rate, warmup_steps), weight_decay=weight_decay)
    if grad_clip and grad_clip > 0:
        tx = optax.chain(optax.clip_by_global_norm(grad_clip), tx)
    return tx


class TrainState(train_state.TrainState):
    """Flax train state carrying an EMA shadow of the parameters.

    ``ema_decay`` is a static field (so jit can branch on it). The EMA shadow is
    initialized eagerly as a *copy* of the params in :func:`init_ema` — the copy
    matters so the shadow never aliases a (possibly donated) live param buffer.
    """

    ema_params: optax.Params | None = None
    ema_decay: float = struct.field(pytree_node=False, default=0.999)


def init_ema(state: TrainState) -> TrainState:
    """Eagerly seed the EMA shadow with a copy of the current params (if EMA on)."""
    if state.ema_decay <= 0:
        return state
    return state.replace(ema_params=jax.tree_util.tree_map(jnp.copy, state.params))


def ema_update(state: TrainState) -> TrainState:
    """Update the EMA shadow: ``ema <- decay*ema + (1-decay)*params``."""
    if state.ema_decay <= 0 or state.ema_params is None:
        return state
    d = state.ema_decay
    new_ema = jax.tree_util.tree_map(
        lambda e, p: d * e + (1.0 - d) * p, state.ema_params, state.params
    )
    return state.replace(ema_params=new_ema)


def eval_params(state: TrainState):
    """Parameters to evaluate with: the EMA shadow if active, else the live params."""
    if state.ema_decay > 0 and state.ema_params is not None:
        return state.ema_params
    return state.params
