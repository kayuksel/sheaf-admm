"""Training: optimizer/EMA, task hooks, and the fit/eval loop."""

from __future__ import annotations

from .loop import (
    build_model,
    create_train_state,
    evaluate,
    make_train_step,
    sample_k,
)
from .optim import TrainState, ema_update, eval_params, make_optimizer
from .tasks import MazeTask, MNISTTask, SudokuTask, make_task

__all__ = [
    "build_model",
    "create_train_state",
    "make_train_step",
    "evaluate",
    "sample_k",
    "TrainState",
    "ema_update",
    "eval_params",
    "make_optimizer",
    "make_task",
    "MazeTask",
    "MNISTTask",
    "SudokuTask",
]
