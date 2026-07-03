"""Shared test fixtures and path setup.

Adds ``src/`` to ``sys.path`` so ``import sheaf_admm`` works whether or not the
package is installed (the smoke command uses ``PYTHONPATH=src``; this makes a
bare ``pytest`` work too). Importing :mod:`sheaf_admm` also enforces ``highest``
float32 matmul precision, which the unrolled-CG equivalence tests rely on.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import jax  # noqa: E402

# CPU-only, deterministic: the tests are tiny and must not depend on a GPU.
jax.config.update("jax_platform_name", "cpu")
