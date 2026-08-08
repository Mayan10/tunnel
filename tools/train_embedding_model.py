#!/usr/bin/env python3
"""Trains audio-embedding-v1 and exports it as a Core ML model.

This is a development tool, not part of the shipped `tunnel` package. It
requires numpy and coremltools:

    pip install ".[ml]"
    python3 tools/train_embedding_model.py

The model is a small feed-forward network trained with a triplet loss on
procedurally generated acoustic archetypes (ambient, acoustic, classical,
pop, dance, rock) that span the tempo, energy, brightness, dynamics, and
spectral-band-balance ranges those genres typically occupy. There is no
real audio in the training set and no network access is used; this is a
local, synthetic bootstrap for a general-purpose "does this sound similar"
embedding, not a model trained on a music catalog.

The 9-dim input layout must match `tunnel/embedding.py`'s `_input_vector`:
    [tempo, energy, brightness, dynamics, duration, band0, band1, band2, band3]

Re-run this script and commit the resulting .mlmodel whenever that input
layout changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "tunnel" / "models" / "audio_embedding_v1.mlmodel"

INPUT_DIMS = 9
HIDDEN1 = 16
HIDDEN2 = 12
EMBEDDING_DIMS = 8

# (tempo, energy, dynamics, brightness) are (mean, stddev) in normalized [0, 1]
# space. band_bias is a rough (low, low-mid, high-mid, high) spectral energy
# split that gets renormalized to sum to 1 after noise is added.
ARCHETYPES: dict[str, dict[str, tuple[float, float] | tuple[float, float, float, float]]] = {
    "ambient": {
        "tempo": (0.15, 0.08),
        "energy": (0.15, 0.06),
        "dynamics": (0.22, 0.08),
        "brightness": (0.22, 0.08),
        "band_bias": (0.55, 0.25, 0.13, 0.07),
    },
    "acoustic": {
        "tempo": (0.28, 0.1),
        "energy": (0.25, 0.07),
        "dynamics": (0.42, 0.08),
        "brightness": (0.35, 0.08),
        "band_bias": (0.40, 0.32, 0.18, 0.10),
    },
    "classical": {
        "tempo": (0.30, 0.14),
        "energy": (0.25, 0.09),
        "dynamics": (0.55, 0.12),
        "brightness": (0.32, 0.09),
        "band_bias": (0.35, 0.30, 0.22, 0.13),
    },
    "pop": {
        "tempo": (0.52, 0.08),
        "energy": (0.58, 0.08),
        "dynamics": (0.38, 0.08),
        "brightness": (0.58, 0.08),
        "band_bias": (0.25, 0.30, 0.28, 0.17),
    },
    "dance": {
        "tempo": (0.75, 0.08),
        "energy": (0.80, 0.07),
        "dynamics": (0.28, 0.08),
        "brightness": (0.70, 0.08),
        "band_bias": (0.30, 0.22, 0.25, 0.23),
    },
    "rock": {
        "tempo": (0.62, 0.1),
        "energy": (0.75, 0.08),
        "dynamics": (0.50, 0.09),
        "brightness": (0.65, 0.08),
        "band_bias": (0.22, 0.25, 0.30, 0.23),
    },
}


def sample_archetype(rng: np.random.Generator, name: str) -> np.ndarray:
    spec = ARCHETYPES[name]
    tempo = float(np.clip(rng.normal(*spec["tempo"]), 0.0, 1.0))
    energy = float(np.clip(rng.normal(*spec["energy"]), 0.0, 1.0))
    dynamics = float(np.clip(rng.normal(*spec["dynamics"]), 0.0, 1.0))
    brightness = float(np.clip(rng.normal(*spec["brightness"]), 0.0, 1.0))
    duration = float(rng.uniform(0.05, 0.95))
    bands = np.array(spec["band_bias"], dtype=np.float64) + rng.normal(0, 0.04, size=4)
    bands = np.clip(bands, 0.01, None)
    bands = bands / bands.sum()
    return np.array([tempo, energy, brightness, dynamics, duration, *bands], dtype=np.float32)


def build_dataset(rng: np.random.Generator, per_class: int = 500) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = list(ARCHETYPES)
    features = []
    labels = []
    for label, name in enumerate(names):
        for _ in range(per_class):
            features.append(sample_archetype(rng, name))
            labels.append(label)
    return np.stack(features), np.array(labels), names


class MLP:
    """A 3-layer feed-forward network with hand-rolled forward/backward passes."""

    def __init__(self, rng: np.random.Generator) -> None:
        def init(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

        self.W1 = init(INPUT_DIMS, HIDDEN1)
        self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
        self.W2 = init(HIDDEN1, HIDDEN2)
        self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
        self.W3 = init(HIDDEN2, EMBEDDING_DIMS)
        self.b3 = np.zeros(EMBEDDING_DIMS, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.W3 + self.b3
        return z3, (x, z1, a1, z2, a2)

    def backward(self, grad_z3: np.ndarray, cache: tuple) -> dict[str, np.ndarray]:
        x, z1, a1, z2, a2 = cache
        grad_w3 = a2.T @ grad_z3
        grad_b3 = grad_z3.sum(axis=0)
        grad_a2 = grad_z3 @ self.W3.T
        grad_z2 = grad_a2 * (z2 > 0)
        grad_w2 = a1.T @ grad_z2
        grad_b2 = grad_z2.sum(axis=0)
        grad_a1 = grad_z2 @ self.W2.T
        grad_z1 = grad_a1 * (z1 > 0)
        grad_w1 = x.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)
        return {"W1": grad_w1, "b1": grad_b1, "W2": grad_w2, "b2": grad_b2, "W3": grad_w3, "b3": grad_b3}

    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def set_param(self, name: str, value: np.ndarray) -> None:
        setattr(self, name, value)


def l2_normalize(z: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(z, axis=1, keepdims=True) + eps
    return z / norm, norm


def _l2_normalize_backward(grad_u: np.ndarray, u: np.ndarray, norm: np.ndarray) -> np.ndarray:
    dot = np.sum(grad_u * u, axis=1, keepdims=True)
    return (grad_u - dot * u) / norm


def triplet_loss_and_grad(
    z_anchor: np.ndarray,
    z_positive: np.ndarray,
    z_negative: np.ndarray,
    margin: float = 0.3,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    u_a, n_a = l2_normalize(z_anchor)
    u_p, n_p = l2_normalize(z_positive)
    u_n, n_n = l2_normalize(z_negative)

    diff_pos = u_a - u_p
    diff_neg = u_a - u_n
    dist_pos = np.sum(diff_pos**2, axis=1)
    dist_neg = np.sum(diff_neg**2, axis=1)
    raw = margin + dist_pos - dist_neg
    active = (raw > 0).astype(np.float32)
    loss = float(np.mean(np.maximum(raw, 0)))
    batch = z_anchor.shape[0]

    grad_u_a = (active[:, None] * 2 * (diff_pos - diff_neg)) / batch
    grad_u_p = (active[:, None] * -2 * diff_pos) / batch
    grad_u_n = (active[:, None] * 2 * diff_neg) / batch

    grad_z_a = _l2_normalize_backward(grad_u_a, u_a, n_a)
    grad_z_p = _l2_normalize_backward(grad_u_p, u_p, n_p)
    grad_z_n = _l2_normalize_backward(grad_u_n, u_n, n_n)
    return loss, grad_z_a, grad_z_p, grad_z_n


def sample_triplets(
    rng: np.random.Generator, features: np.ndarray, labels: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_labels = list(np.unique(labels))
    by_label = {label: np.where(labels == label)[0] for label in unique_labels}
    anchor_idx = rng.integers(0, len(features), size=batch_size)
    positive_idx = np.empty(batch_size, dtype=int)
    negative_idx = np.empty(batch_size, dtype=int)
    for i, anchor in enumerate(anchor_idx):
        label = labels[anchor]
        positive_idx[i] = rng.choice(by_label[label])
        other_label = rng.choice([candidate for candidate in unique_labels if candidate != label])
        negative_idx[i] = rng.choice(by_label[other_label])
    return features[anchor_idx], features[positive_idx], features[negative_idx]


def train(
    rng: np.random.Generator,
    features: np.ndarray,
    labels: np.ndarray,
    steps: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 0.05,
) -> MLP:
    mlp = MLP(rng)
    momentum = {name: np.zeros_like(value) for name, value in mlp.params().items()}
    beta = 0.9

    for step in range(steps):
        x_a, x_p, x_n = sample_triplets(rng, features, labels, batch_size)
        z_a, cache_a = mlp.forward(x_a)
        z_p, cache_p = mlp.forward(x_p)
        z_n, cache_n = mlp.forward(x_n)
        loss, grad_a, grad_p, grad_n = triplet_loss_and_grad(z_a, z_p, z_n)
        grads_a = mlp.backward(grad_a, cache_a)
        grads_p = mlp.backward(grad_p, cache_p)
        grads_n = mlp.backward(grad_n, cache_n)

        for name, value in mlp.params().items():
            grad = grads_a[name] + grads_p[name] + grads_n[name]
            momentum[name] = beta * momentum[name] + (1 - beta) * grad
            mlp.set_param(name, value - learning_rate * momentum[name])

        if step % 300 == 0 or step == steps - 1:
            print(f"  step {step:5d}  triplet loss {loss:.4f}")

    return mlp


def evaluate(rng: np.random.Generator, mlp: MLP, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    z, _ = mlp.forward(features)
    embeddings, _ = l2_normalize(z)
    by_label = {label: np.where(labels == label)[0] for label in np.unique(labels)}
    unique_labels = list(by_label)

    intra_distances = []
    inter_distances = []
    for _ in range(3000):
        anchor = rng.integers(0, len(features))
        label = labels[anchor]
        positive = rng.choice(by_label[label])
        other_label = rng.choice([candidate for candidate in unique_labels if candidate != label])
        negative = rng.choice(by_label[other_label])
        intra_distances.append(np.sum((embeddings[anchor] - embeddings[positive]) ** 2))
        inter_distances.append(np.sum((embeddings[anchor] - embeddings[negative]) ** 2))

    return float(np.mean(intra_distances)), float(np.mean(inter_distances))


def export(mlp: MLP, path: Path) -> None:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb

    w1, b1 = mlp.W1, mlp.b1
    w2, b2 = mlp.W2, mlp.b2
    w3, b3 = mlp.W3, mlp.b3

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, INPUT_DIMS))])
    def prog(audio_features):
        z1 = mb.linear(x=audio_features, weight=w1.T, bias=b1, name="dense1")
        a1 = mb.relu(x=z1, name="relu1")
        z2 = mb.linear(x=a1, weight=w2.T, bias=b2, name="dense2")
        a2 = mb.relu(x=z2, name="relu2")
        return mb.linear(x=a2, weight=w3.T, bias=b3, name="embedding")

    mlmodel = ct.convert(
        prog,
        source="milinternal",
        convert_to="neuralnetwork",
        inputs=[ct.TensorType(name="audio_features", shape=(1, INPUT_DIMS))],
        outputs=[ct.TensorType(name="embedding")],
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.short_description = (
        "Tunnel audio-embedding-v1: a local neural embedding of a track's tempo, "
        "energy, brightness, dynamics, and spectral balance, used to compare tracks "
        "for playlist ordering. Runs entirely on-device via Core ML."
    )
    mlmodel.input_description["audio_features"] = (
        "9-dim vector: [tempo, energy, brightness, dynamics, duration, band0, band1, band2, band3], "
        "each normalized to roughly [0, 1]."
    )
    mlmodel.output_description["embedding"] = "8-dim raw embedding. L2-normalize before comparing vectors."
    path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--per-class", type=int, default=500, help="Synthetic examples per archetype.")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    features, labels, names = build_dataset(rng, per_class=args.per_class)
    print(f"Training on {len(features)} synthetic examples across archetypes: {', '.join(names)}")

    mlp = train(rng, features, labels, steps=args.steps)

    intra, inter = evaluate(rng, mlp, features, labels)
    ratio = inter / max(intra, 1e-6)
    print(f"Mean squared distance, same archetype:      {intra:.4f}")
    print(f"Mean squared distance, different archetype:  {inter:.4f}")
    print(f"Separation ratio (higher is better):          {ratio:.2f}x")

    export(mlp, args.out)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
