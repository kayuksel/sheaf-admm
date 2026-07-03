"""Materialize MNIST into NumPy caches for the :class:`ImageDataset` loader.

Images are reshaped to ``[N, 28+2p, 28+2p, 1]`` and normalized to ``[0, 1]``
(divide by 255). ``--gen-robustness`` additionally writes the generalization
suite used in the paper: padding splits (``test_pad_{2,4,8,12,16}``, which change
the spatial shape — hence ``split_shapes`` in the metadata) and additive-Gaussian
noise splits (``test_noise_{0.1..0.5}``, clipped back to ``[0, 1]``).

Usage:
    python -m sheaf_admm.data.build_mnist --output-dir datasets/mnist --gen-robustness
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import save_json, save_npy, shuffle_in_unison

PAD_LEVELS = (2, 4, 8, 12, 16)
NOISE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)


@dataclass
class MNISTConfig:
    padding: int = 0
    seed: int = 0
    normalize: bool = True
    gen_robustness: bool = False
    output_dir: Path = Path("datasets/mnist")

    def to_dict(self):
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _load_mnist_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import ml_datasets

    data = ml_datasets.mnist()
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], tuple):
        (train_images, train_labels), (test_images, test_labels) = data
    elif isinstance(data, tuple) and len(data) == 4:
        train_images, train_labels, test_images, test_labels = data
    else:
        raise ValueError("Unexpected ml_datasets.mnist() return format.")
    return tuple(np.asarray(x) for x in (train_images, train_labels, test_images, test_labels))


def _prep_images(images: np.ndarray, padding: int, normalize: bool) -> np.ndarray:
    """Reshape to ``[N, H, W, 1]`` float32, normalize to ``[0, 1]``, zero-pad."""
    images = images.astype(np.float32)
    if normalize and images.max() > 1.0:
        images = images / 255.0
    images = images.reshape((-1, 28, 28, 1))
    if padding > 0:
        images = np.pad(
            images,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    return images


def _save_split(root: Path, name: str, images: np.ndarray, labels: np.ndarray) -> None:
    save_npy(root / name / "images.npy", images)
    save_npy(root / name / "labels.npy", labels)


def build(cfg: MNISTConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    raw_train_img, raw_train_lbl, raw_test_img, raw_test_lbl = _load_mnist_arrays()

    train_labels = raw_train_lbl.astype(np.int64)
    test_labels = raw_test_lbl.astype(np.int64)
    if train_labels.ndim > 1:
        train_labels = np.argmax(train_labels, axis=-1)
    if test_labels.ndim > 1:
        test_labels = np.argmax(test_labels, axis=-1)

    splits_meta: dict[str, int] = {}
    split_shapes: dict[str, list[int]] = {}

    train_images = _prep_images(raw_train_img, cfg.padding, cfg.normalize)
    test_images = _prep_images(raw_test_img, cfg.padding, cfg.normalize)
    train_images, train_labels = shuffle_in_unison(rng, train_images, train_labels)

    _save_split(cfg.output_dir, "train", train_images, train_labels)
    _save_split(cfg.output_dir, "test", test_images, test_labels)
    for name, imgs in (("train", train_images), ("test", test_images)):
        splits_meta[name] = int(imgs.shape[0])
        split_shapes[name] = list(imgs.shape[1:])

    if cfg.gen_robustness:
        for pad in PAD_LEVELS:
            total_pad = cfg.padding + pad
            name = f"test_pad_{total_pad}"
            images_pad = _prep_images(raw_test_img, total_pad, cfg.normalize)
            _save_split(cfg.output_dir, name, images_pad, test_labels)
            splits_meta[name] = int(images_pad.shape[0])
            split_shapes[name] = list(images_pad.shape[1:])

        for sigma in NOISE_LEVELS:
            name = f"test_noise_{sigma}"
            noise = rng.normal(0.0, sigma, size=test_images.shape).astype(np.float32)
            images_noisy = np.clip(test_images + noise, 0.0, 1.0)
            _save_split(cfg.output_dir, name, images_noisy, test_labels)
            splits_meta[name] = int(images_noisy.shape[0])
            split_shapes[name] = list(images_noisy.shape[1:])

    save_json(
        cfg.output_dir / "metadata.json",
        {
            "task": "mnist",
            "image_shape": list(train_images.shape[1:]),
            "num_classes": 10,
            "padding": cfg.padding,
            "seed": cfg.seed,
            "normalize": cfg.normalize,
            "splits": splits_meta,
            "split_shapes": split_shapes,
            "config": cfg.to_dict(),
        },
    )
    print(f"MNIST cached to {cfg.output_dir}; splits: {list(splits_meta.keys())}")


def parse_args() -> MNISTConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padding", type=int, default=MNISTConfig.padding)
    parser.add_argument("--seed", type=int, default=MNISTConfig.seed)
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip division by 255 (keep range 0..255)."
    )
    parser.add_argument(
        "--gen-robustness",
        action="store_true",
        help="Also build padding/noise generalization splits.",
    )
    parser.add_argument("--output-dir", type=Path, default=MNISTConfig.output_dir)
    args = parser.parse_args()
    return MNISTConfig(
        padding=args.padding,
        seed=args.seed,
        normalize=not args.no_normalize,
        gen_robustness=args.gen_robustness,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    build(parse_args())
