"""Global float32 matmul precision.

Sheaf-ADMM runs many chained matmuls inside the unrolled z-update (conjugate
gradient / gradient descent on the sheaf Laplacian ``F^T F``). On GPU/TPU, JAX
defaults to lower-precision matmuls (TF32-style, ~10-bit mantissa), which
corrupts the CG residual orthogonality over the fixed ``T`` steps and injects
noise into the unrolled gradients. The paper sets matmul precision to
``highest`` (true float32) for every run; reproducing the reported numbers
depends on it.

We enforce this once, at import time, unless the user has already pinned the
precision via the ``JAX_DEFAULT_MATMUL_PRECISION`` environment variable.
"""

from __future__ import annotations

import os

import jax


def set_high_precision() -> None:
    """Set JAX default matmul precision to ``highest`` unless already configured.

    Respects a pre-existing ``JAX_DEFAULT_MATMUL_PRECISION`` env var so callers
    can opt out (e.g. ``JAX_DEFAULT_MATMUL_PRECISION=default``).
    """
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"):
        return
    jax.config.update("jax_default_matmul_precision", "highest")
