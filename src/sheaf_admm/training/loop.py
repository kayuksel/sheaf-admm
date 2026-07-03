"""Training loop: state creation, jitted steps, and the fit/eval driver.

Handles both model families (Sheaf-ADMM and the MPNN baseline) and all tasks via
the :mod:`sheaf_admm.training.tasks` hooks. The ADMM iteration count ``K`` is a
static argument to the jitted step (so the per-step ``K ~ U[k_min, k_max]``
resampling used for Maze training triggers one compile per distinct ``K``, then
reuses the cache). Evaluation runs at ``K_eval`` on the EMA parameters.

Tasks return ``(fwd, targets, aux)``; only ``fwd``/``targets`` (pure arrays) cross
the ``jit`` boundary, ``aux`` (centers/image size) is used eager-side at eval.
"""

from __future__ import annotations

from functools import partial

import jax
import numpy as np

from sheaf_admm.models import ModelConfig, MPNNModel, SheafADMMModel

from .optim import TrainState, ema_update, eval_params, init_ema, make_optimizer


def build_model(config: ModelConfig, model_type: str):
    if model_type == "sheaf":
        return SheafADMMModel(config=config)
    if model_type == "mpnn":
        return MPNNModel(config=config)
    raise ValueError(f"unknown model_type={model_type!r} (sheaf|mpnn)")


def _forward(apply_fn, params, fwd, *, n_iter, loss_window, model_type, training, rng):
    """Run the model to task logits.

    Sheaf returns windowed per-agent logits ``[W,N,B,*out]``; MPNN returns
    ``[N,B,*out]`` for per-node readouts or ``[B,C]`` for graph readouts.
    """
    rngs = {"dropout": rng} if (training and rng is not None) else None
    if model_type == "sheaf":
        logits_window, _state, _geom = apply_fn(
            params,
            fwd["patches"],
            fwd["edge_indices"],
            num_iters=n_iter,
            loss_window=loss_window,
            **fwd["model_kwargs"],
            training=training,
            rngs=rngs,
        )
        return logits_window
    logits, _hidden = apply_fn(
        params,
        fwd["patches"],
        fwd["edge_indices"],
        num_rounds=n_iter,
        **fwd["model_kwargs"],
        training=training,
        rngs=rngs,
    )
    return logits


def create_train_state(
    model,
    sample_fwd,
    *,
    model_type,
    lr,
    weight_decay,
    warmup_steps,
    grad_clip,
    ema_decay,
    k_init,
    loss_window,
    seed,
):
    init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(seed))
    rngs = {"params": init_rng, "dropout": dropout_rng}
    kw = (
        dict(num_iters=k_init, loss_window=loss_window)
        if model_type == "sheaf"
        else dict(num_rounds=k_init)
    )
    variables = model.init(
        rngs,
        sample_fwd["patches"],
        sample_fwd["edge_indices"],
        **kw,
        **sample_fwd["model_kwargs"],
        training=False,
    )
    tx = make_optimizer(lr, weight_decay, warmup_steps, grad_clip)
    state = TrainState.create(apply_fn=model.apply, params=variables, tx=tx, ema_decay=ema_decay)
    return init_ema(state)


def make_train_step(task, model_type, graph_readout):
    @partial(jax.jit, static_argnames=("n_iter", "loss_window"))
    def train_step(state: TrainState, fwd, targets, dropout_rng, *, n_iter, loss_window):
        def loss_fn(params):
            logits = _forward(
                state.apply_fn,
                params,
                fwd,
                n_iter=n_iter,
                loss_window=loss_window,
                model_type=model_type,
                training=True,
                rng=dropout_rng,
            )
            if model_type == "mpnn" and graph_readout == "graph":
                return task.loss_graph(logits, targets)
            if model_type == "mpnn":
                return task.loss(logits[None], targets)
            return task.loss(logits, targets)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return ema_update(state), loss

    return train_step


def evaluate(state, task, batches, *, model_type, graph_readout, k_eval, max_batches=None):
    """Eval at ``K_eval`` on EMA params; average task metrics over the split."""
    params = eval_params(state)

    @partial(jax.jit, static_argnames=("n_iter",))
    def fwd_fn(params, fwd, *, n_iter):
        return _forward(
            state.apply_fn,
            params,
            fwd,
            n_iter=n_iter,
            loss_window=1,
            model_type=model_type,
            training=False,
            rng=None,
        )

    agg: dict[str, float] = {}
    n = 0
    for batch in batches:
        fwd, targets, aux = task.prepare(batch)
        logits = fwd_fn(params, fwd, n_iter=k_eval)
        final = logits if model_type == "mpnn" else logits[-1]
        if model_type == "mpnn" and graph_readout == "graph":
            m = task.evaluate_graph(final, targets, aux)
        else:
            m = task.evaluate(final, targets, aux)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    return {k: v / max(n, 1) for k, v in agg.items()}


def sample_k(rng_np: np.random.Generator, dist: str, k_min: int, k_max: int) -> int:
    if dist != "uniform":
        return k_max
    return int(rng_np.integers(min(k_min, k_max), k_max + 1))
