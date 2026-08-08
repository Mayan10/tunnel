#!/usr/bin/env python3
"""Generates every plot in docs/ from real, reproducible local runs.

Development tool, not part of the shipped `tunnel` package. Requires the dev
extras (`pip install -e ".[dev]"`, which includes matplotlib). Nothing here is
fabricated or hand-tuned for looks: the loss curve comes from actually
training the model, the embedding plots from actually running Core ML, and
the ordering plots from actually running Tunnel's orderer, reusing the exact
same scenario-loading functions as `scripts/benchmark_ordering.py` so the
numbers in the README and the plots here never drift apart.

Run with: python3 tools/generate_report_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_ordering import evaluate, load_audio_scenario, load_metadata_scenario  # noqa: E402
from train_embedding_model import build_dataset, l2_normalize, train  # noqa: E402

from tunnel.model import LocalFlowModel  # noqa: E402
from tunnel.types import Track  # noqa: E402

DPI = 220

# A restrained, colorblind-safe palette matching macOS system colors.
INK = "#1d1d1f"
SUBTLE = "#8e8e93"
GRID = "#e5e5e7"
ACCENT = "#0071e3"
ARCHETYPE_COLORS = ["#0071e3", "#34c759", "#ff9500", "#af52de", "#ff3b30", "#5e5ce6"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(DOCS / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote docs/{name}")


def plot_training_loss(loss_history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = np.arange(len(loss_history))
    ax.plot(steps, loss_history, color=ACCENT, linewidth=1.0, alpha=0.8)

    window = 50
    if len(loss_history) >= window:
        smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1 :], smoothed, color=INK, linewidth=2.2, label=f"{window}-step moving average")
        ax.legend(frameon=False)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Triplet loss")
    ax.set_title("audio-embedding-v1 training convergence", loc="left")
    ax.set_xlim(0, len(loss_history))
    ax.set_ylim(bottom=0)
    save(fig, "training_loss.png")


def pca(points: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    explained = eigenvalues[order] / eigenvalues.sum()
    components = eigenvectors[:, order[:n_components]]
    return centered @ components, explained[:n_components]


def plot_embedding_space(embeddings: np.ndarray, labels: np.ndarray, names: list[str]) -> None:
    projected, explained = pca(embeddings)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    rng = np.random.default_rng(0)
    for label, name in enumerate(names):
        mask = labels == label
        points = projected[mask]
        if len(points) > 250:
            points = points[rng.choice(len(points), 250, replace=False)]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=14,
            alpha=0.65,
            color=ARCHETYPE_COLORS[label % len(ARCHETYPE_COLORS)],
            label=name,
            linewidths=0,
        )

    ax.set_xlabel(f"PCA component 1 ({explained[0]:.0%} of variance)")
    ax.set_ylabel(f"PCA component 2 ({explained[1]:.0%} of variance)")
    ax.set_title("audio-embedding-v1 embedding space, by archetype", loc="left")
    ax.legend(frameon=False, loc="best", markerscale=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "embedding_space.png")


def plot_embedding_distance_heatmap(embeddings: np.ndarray, labels: np.ndarray, names: list[str]) -> None:
    centroids = np.stack([embeddings[labels == i].mean(axis=0) for i in range(len(names))])
    centroids, _ = l2_normalize(centroids)
    distance = 1.0 - centroids @ centroids.T
    distance[np.abs(distance) < 1e-5] = 0.0

    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(distance, cmap="Blues", vmin=0, vmax=distance.max())
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            value = distance[i, j]
            color = "white" if value > distance.max() * 0.6 else INK
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_title("Cosine distance between archetype centroids", loc="left")
    fig.colorbar(image, ax=ax, shrink=0.82, label="cosine distance")
    save(fig, "embedding_distances.png")


def plot_tempo_smoothing(shuffled: list[Track], ordered: list[Track], audio_features: dict) -> None:
    source_bpm = [audio_features[t.id].bpm for t in shuffled]
    ordered_bpm = [audio_features[t.id].bpm for t in ordered]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(source_bpm) + 1), source_bpm, "o--", color=SUBTLE, label="source order (shuffled)")
    ax.plot(range(1, len(ordered_bpm) + 1), ordered_bpm, "o-", color=ACCENT, label="Tunnel (audio-embedding-v1)")
    ax.set_xlabel("Track position")
    ax.set_ylabel("Tempo (BPM, measured from audio)")
    ax.set_title("Tempo curve before and after ordering", loc="left")
    ax.legend(frameon=False)
    save(fig, "tempo_smoothing.png")


def plot_energy_curve(ordered: list[Track], audio_features: dict) -> None:
    model = LocalFlowModel(audio_features=audio_features)
    total = len(ordered)
    positions = list(range(total))
    target = [model.target_energy(position, total) for position in positions]
    achieved = [model.features_for(track).energy for track in ordered]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(positions, target, "--", color=SUBTLE, linewidth=1.6, label="designed target energy curve")
    ax.plot(positions, achieved, "o-", color=ACCENT, linewidth=2, markersize=5, label="achieved (Tunnel order)")
    ax.set_xlabel("Track position")
    ax.set_ylabel("Energy (0-1)")
    ax.set_title("Designed energy arc vs. achieved order", loc="left")
    ax.legend(frameon=False)
    save(fig, "energy_curve.png")


def plot_benchmark_comparison(
    metadata_scores: dict[str, dict[str, float]],
    audio_scores: dict[str, dict[str, float]],
) -> None:
    metrics = [
        ("tempo_jump", "Mean tempo jump (BPM)"),
        ("energy_jump", "Mean energy jump"),
        ("genre_distance", "Mean genre distance"),
    ]
    scenarios = ["Metadata only", "Audio-aware"]
    scenario_scores = [metadata_scores, audio_scores]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(scenarios))
    width = 0.32

    for ax, (key, label) in zip(axes, metrics, strict=True):
        source_values = [scores["source"][key] for scores in scenario_scores]
        tunnel_values = [scores["tunnel"][key] for scores in scenario_scores]
        ax.bar(x - width / 2, source_values, width, label="source order", color=SUBTLE)
        ax.bar(x + width / 2, tunnel_values, width, label="Tunnel", color=ACCENT)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_ylim(bottom=0)

    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Tunnel vs. source order, both benchmark scenarios", x=0.01, ha="left", fontweight="bold")
    save(fig, "benchmark_comparison.png")


def main() -> int:
    DOCS.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)
    features, labels, names = build_dataset(rng, per_class=500)
    mlp, loss_history = train(rng, features, labels, steps=3000)
    z, _ = mlp.forward(features)
    embeddings, _ = l2_normalize(z)
    embeddings = np.asarray(embeddings)

    plot_training_loss(loss_history)
    plot_embedding_space(embeddings, labels, names)
    plot_embedding_distance_heatmap(embeddings, labels, names)

    metadata_shuffled, metadata_ordered = load_metadata_scenario()
    audio_shuffled, audio_ordered, audio_features = load_audio_scenario()

    plot_tempo_smoothing(audio_shuffled, audio_ordered, audio_features)
    plot_energy_curve(audio_ordered, audio_features)
    plot_benchmark_comparison(
        {"source": evaluate(metadata_shuffled, {}), "tunnel": evaluate(metadata_ordered, {})},
        {
            "source": evaluate(audio_shuffled, audio_features),
            "tunnel": evaluate(audio_ordered, audio_features),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
