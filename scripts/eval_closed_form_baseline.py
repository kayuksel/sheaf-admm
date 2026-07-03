"""Closed-form robustness baseline on the paper's own MNIST splits.

Fits the training-free ``RobustMNIST`` model (``robust_mnist_jax.py``) on the CLEAN ``train`` split and
evaluates on every robustness split the paper reports — ``test``, ``test_pad_{2,4,8,12,16}``,
``test_noise_{0.1..0.5}`` — using the repository's own ``ImageDataset`` loader, so the comparison is on
*identical* data and corruptions. Prints accuracy next to the paper's Sheaf-ADMM / CNN numbers.

The point of this baseline: it establishes how much of the reported robustness a *weightless,
closed-form* model already achieves. On MNIST it matches Sheaf-ADMM's clean accuracy and clears its
robustness numbers by wide margins (padding solved by size-invariant pooling; Gaussian-noise handled by
a low-pass filter) — with no training. That contextualises where the trained ADMM machinery is (and
isn't) doing load-bearing work — offered as a useful reference point.

Build the splits first (from the repo root), then run:

    python -m sheaf_admm.data.build_mnist --output-dir datasets/mnist --with-robustness-splits
    python scripts/eval_closed_form_baseline.py --dataset-dir datasets/mnist

Leakage note: the ridge is fit only on CLEAN ``train`` images; every ``test*`` split is a corruption of
the disjoint held-out ``test`` images. No image (clean or corrupted) is shared between fit and eval.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from sheaf_admm.data.loaders import ImageDataset

try:                                          # co-located in scripts/, or installed
    from robust_mnist_jax import RobustMNIST
except ImportError:                           # pragma: no cover
    from scripts.robust_mnist_jax import RobustMNIST


# Paper Table 4 (for side-by-side context; '-' where the paper does not report that level).
PAPER = {
    "test":          (0.985, 0.993),
    "test_pad_2":    (None,  None),
    "test_pad_4":    (0.984, 0.708),
    "test_pad_8":    (0.975, 0.441),
    "test_pad_12":   (None,  None),
    "test_pad_16":   (0.863, 0.114),
    "test_noise_0.1": (0.740, 0.540),
    "test_noise_0.2": (0.319, 0.099),
    "test_noise_0.3": (None,  None),
    "test_noise_0.4": (None,  None),
    "test_noise_0.5": (None,  None),
}
SPLITS = (["test"]
          + [f"test_pad_{p}" for p in (2, 4, 8, 12, 16)]
          + [f"test_noise_{s}" for s in (0.1, 0.2, 0.3, 0.4, 0.5)])


def _load(dataset_dir: str, split: str):
    ds = ImageDataset(dataset_dir, split=split)
    return np.asarray(ds.images, dtype=np.float32), np.asarray(ds.labels).reshape(-1)


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "  -  "


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True, help="MNIST cache dir (built by sheaf_admm.data.build_mnist)")
    ap.add_argument("--seed-offset", type=int, default=None,
                    help="if set, REPRODUCE the kernels from the genome (seed-invariance check) instead of "
                         "loading the shipped champion.npz; the result is invariant to this")
    args = ap.parse_args()

    Xtr, ytr = _load(args.dataset_dir, "train")
    t0 = time.time()
    model = RobustMNIST(seed_offset=args.seed_offset).fit(Xtr, ytr)
    print(f"fit on {len(Xtr)} clean train images in {time.time() - t0:.1f}s "
          f"(closed-form GCV-ridge, alpha={model.readout.alpha:.4g}; no gradient steps)\n")

    print(f"{'split':<16}{'EvoForest':>11}{'Sheaf-ADMM':>13}{'CNN':>8}")
    print("-" * 48)
    for split in SPLITS:
        try:
            X, y = _load(args.dataset_dir, split)
        except (FileNotFoundError, ValueError):
            continue                          # split not built; skip
        acc = float((model.predict(X) == y).mean())
        sheaf, cnn = PAPER.get(split, (None, None))
        print(f"{split:<16}{acc:>11.3f}{_fmt(sheaf):>13}{_fmt(cnn):>8}")


if __name__ == "__main__":
    main()
