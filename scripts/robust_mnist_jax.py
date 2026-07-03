"""Training-free closed-form robustness baseline for MNIST (JAX).

A *weightless* random-convolution feature map followed by a *closed-form* ridge classifier. There is
no gradient descent: the convolution kernels are fixed and the only fitted quantity is a linear
read-out solved in closed form. Fit on clean MNIST-train, it matches a strong CNN's clean accuracy and
**substantially exceeds trained models under distribution shift** (padding, additive Gaussian noise) —
see `eval_closed_form_baseline.py` for the head-to-head on the paper's splits.

The 12-"gene" architecture was discovered by EvoForest (arXiv:2604.19761), an evolutionary feature-map
search whose fitness is held-out robustness. The exact evolved kernels are embedded (float16, zlib+base64)
in ``WEIGHTS_B64`` at the bottom of this file and loaded by default — self-contained, no binary, no RNG
regeneration. The ``GENES`` table records the architecture for provenance and the seed-invariance check.

Feature map (per gene):

    image --[Gaussian blur, sigma]--> [random conv, 48 kernels, size k, dilation d] --> adaptive-max-pool(2x2)

concatenated (2,304-D), standardised, then a ridge read-out whose strength is chosen by **generalised
cross-validation** (closed-form leave-one-out) on the *training set only*.

    m = RobustMNIST().fit(Xtr_clean, ytr)     # embedded evolved kernels + closed-form read-out; X:(N,H,W,1) in [0,1]
    acc = (m.predict(Xte) == yte).mean()

Requires: jax, numpy. CPU is fine.
"""
from __future__ import annotations

import functools
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# Architecture discovered by evolutionary search: (kernel_seed, ksize, dilation, pool_grid, blur_sigma).
GENES: tuple[tuple[int, int, int, int, float], ...] = (
    (1922646, 5, 1, 2, 1.2), (15844910, 7, 1, 2, 1.2), (19958281, 9, 1, 2, 0.7),
    (20518863, 7, 2, 2, 1.2), (28084017, 9, 1, 2, 1.2), (30719058, 5, 1, 2, 1.2),
    (37603597, 9, 2, 2, 1.2), (47660464, 9, 1, 2, 1.2), (64776218, 5, 4, 2, 1.2),
    (72963421, 7, 1, 2, 1.2), (74778293, 5, 1, 2, 2.2), (81201489, 5, 1, 2, 0.7),
)
N_KERNELS = 48
_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion.npz")


class Readout(NamedTuple):
    W: jnp.ndarray       # (n_features, n_classes)
    mu: jnp.ndarray      # (n_features,)
    sd: jnp.ndarray      # (n_features,)
    alpha: float         # GCV-selected ridge strength


# ---------------------------------------------------------------- fixed feature-map kernels
def _gaussian(sigma: float, k: int = 5) -> np.ndarray:
    ax = np.arange(k) - k // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2.0 * sigma * sigma))
    return (g / g.sum()).astype(np.float32)


def _random_kernels(seed: int, ks: int, nk: int = N_KERNELS) -> np.ndarray:
    W = np.random.RandomState(seed).randn(nk, ks, ks).astype(np.float32)
    return W - W.mean((1, 2), keepdims=True)                 # zero-mean


def kernels_from_genes(seed_offset: int = 0):
    """Reproduce the feature-map kernels from the architecture spec (for provenance / seed-invariance)."""
    convs = [jnp.asarray(_random_kernels(s + seed_offset, k)) for (s, k, _, _, _) in GENES]
    blurs = [jnp.asarray(_gaussian(sig)) for (*_, sig) in GENES]
    return convs, blurs


def _load_kernels(path: str):
    d = np.load(path)
    convs = [jnp.asarray(d[f"conv{i}"].astype(np.float32)) for i in range(len(GENES))]
    blurs = [jnp.asarray(d[f"blur{i}"].astype(np.float32)) for i in range(len(GENES))]
    return convs, blurs


def load_embedded_kernels():
    """Decode the exact evolved kernels embedded (float16, zlib+base64) at the bottom of this file
    (``WEIGHTS_B64``). Self-contained — no binary file, no RNG regeneration."""
    import base64, io, zlib
    d = np.load(io.BytesIO(zlib.decompress(base64.b64decode(WEIGHTS_B64))))
    convs = [jnp.asarray(d[f"conv{i}"].astype(np.float32)) for i in range(len(GENES))]
    blurs = [jnp.asarray(d[f"blur{i}"].astype(np.float32)) for i in range(len(GENES))]
    return convs, blurs


# ---------------------------------------------------------------- feature map (NHWC, size-agnostic)
def _blur(x, kern):
    p = kern.shape[-1] // 2
    return jax.lax.conv_general_dilated(x, kern[:, :, None, None], (1, 1), [(p, p), (p, p)],
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"))


def _conv(x, kern, dilation):
    p = dilation * (kern.shape[-1] // 2)
    w = jnp.transpose(kern, (1, 2, 0))[:, :, None, :]        # (nk,k,k) -> HWIO (I=1)
    return jax.lax.conv_general_dilated(x, w, (1, 1), [(p, p), (p, p)],
                                        rhs_dilation=(dilation, dilation),
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"))


def _adaptive_max_2x2(c):
    """Adaptive max-pool to a 2x2 grid (== torch adaptive_max_pool2d(x,2) for even H,W); size-invariant."""
    _, H, W, _ = c.shape
    h, w = H // 2, W // 2
    quads = (c[:, :h, :w], c[:, :h, w:], c[:, h:, :w], c[:, h:, w:])
    return jnp.concatenate([q.max(axis=(1, 2)) for q in quads], axis=1)


@jax.jit
def features(convs, blurs, x):
    """x: (N,H,W,1) in [0,1] -> (N,2304) feature matrix. Size-agnostic (handles padded splits)."""
    out = []
    for (s, ks, dil, pg, sig), wk, bk in zip(GENES, convs, blurs):
        out.append(_adaptive_max_2x2(_conv(_blur(x, bk), wk, dil)))
    return jnp.concatenate(out, axis=1)


def _features_batched(convs, blurs, X, bs=4000):
    return jnp.concatenate([features(convs, blurs, jnp.asarray(X[i:i + bs])) for i in range(0, len(X), bs)], 0)


# ---------------------------------------------------------------- closed-form GCV ridge read-out
def fit_readout(Z, y, n_classes=10) -> Readout:
    """Ridge onto one-hot labels; alpha by generalised cross-validation (LOO) on the training features Z."""
    mu = Z.mean(0); sd = jnp.clip(Z.std(0), 1e-6, None); Zs = (Z - mu) / sd
    Y = jax.nn.one_hot(jnp.asarray(y), n_classes).astype(Z.dtype)
    lam, V = jnp.linalg.eigh(Zs.T @ Zs); lam = jnp.clip(lam, 1e-9, None)
    P = V.T @ (Zs.T @ Y); B2 = (P ** 2).sum(1) / lam; sumB2 = B2.sum(); Yn = float((Y ** 2).sum()); n = Zs.shape[0]

    def gcv(a):
        f = lam / (lam + a)
        return (((1 - f) ** 2 * B2).sum() + (Yn - sumB2)) / n / jnp.clip(1 - f.sum() / n, 1e-6, None) ** 2

    alphas = jnp.logspace(-2, 4, 12)
    a = float(alphas[jnp.argmin(jax.vmap(gcv)(alphas))])
    return Readout(W=V @ (P / (lam + a)[:, None]), mu=mu, sd=sd, alpha=a)


class RobustMNIST:
    """Weightless feature map (materialised in ``champion.npz``) + closed-form ridge read-out.

    ``RobustMNIST()`` loads the shipped kernels; ``seed_offset=k`` instead *reproduces* them from the
    architecture spec (used only for the seed-invariance check)."""
    def __init__(self, weights: str | None = None, seed_offset: int | None = None):
        if seed_offset is not None:
            self.convs, self.blurs = kernels_from_genes(seed_offset)     # reproduce from genome (invariance check)
        elif weights is not None:
            self.convs, self.blurs = _load_kernels(weights)              # explicit .npz path
        else:
            self.convs, self.blurs = load_embedded_kernels()            # DEFAULT: exact evolved kernels (embedded)
        self.readout: Readout | None = None

    def fit(self, X, y):
        self.readout = fit_readout(_features_batched(self.convs, self.blurs, np.asarray(X)), np.asarray(y))
        return self

    def predict(self, X):
        assert self.readout is not None, "fit() first, or use RobustMNIST.from_pretrained(...)"
        Z = (_features_batched(self.convs, self.blurs, np.asarray(X)) - self.readout.mu) / self.readout.sd
        return np.asarray((Z @ self.readout.W).argmax(1))

    def save(self, path):
        """Materialise the full model — every conv/blur kernel + the fitted read-out — to one .npz."""
        assert self.readout is not None, "fit() before save()"
        arr = {f"conv{i}": np.asarray(c) for i, c in enumerate(self.convs)}
        arr.update({f"blur{i}": np.asarray(b) for i, b in enumerate(self.blurs)})
        arr.update(W=np.asarray(self.readout.W), mu=np.asarray(self.readout.mu),
                   sd=np.asarray(self.readout.sd), alpha=np.float32(self.readout.alpha))
        np.savez(path, **arr)

    @classmethod
    def from_pretrained(cls, path=_WEIGHTS):
        m = cls(weights=path); d = np.load(path)
        m.readout = Readout(W=jnp.asarray(d["W"]), mu=jnp.asarray(d["mu"]), sd=jnp.asarray(d["sd"]), alpha=float(d["alpha"]))
        return m


# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Seed-invariance check: the robustness is a property of the ARCHITECTURE, not the RNG draw.
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = (X.astype("float32") / 255.0).reshape(-1, 28, 28, 1); y = y.astype("int64")
    Xtr, ytr, Xte, yte = X[:50000], y[:50000], X[50000:60000], y[50000:60000]
    rs = np.random.RandomState(0)
    for off in (0, 12345, 67890):
        m = RobustMNIST(seed_offset=off).fit(Xtr, ytr)
        clean = (m.predict(Xte) == yte).mean()
        n2 = (m.predict(np.clip(Xte + rs.normal(0, 0.2, Xte.shape).astype("float32"), 0, 1)) == yte).mean()
        print(f"seed_offset={off:>6}: clean={clean:.4f}  noise_sigma0.2={n2:.4f}  alpha={m.readout.alpha:.4g}")


# ---------------------------------------------------------------------------------------------------- 
# Embedded evolved feature-map kernels: 12 genes x 48 conv kernels + Gaussian blurs, float16,
# zlib-compressed and base64-encoded (~72 KB). Decoded by load_embedded_kernels(). Self-contained.
WEIGHTS_B64 = (
"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"
)
