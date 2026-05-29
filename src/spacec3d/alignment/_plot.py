"""Verification plots for serial-section registration."""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_point_rainbow(
    aligned_xys: list[np.ndarray], ids: list[str], out_path: Path
) -> None:
    """All aligned sections overlaid in one axis, jet hue per section index."""
    n = len(ids)
    cmap = cm.get_cmap("jet", n)
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, (ali, idd) in enumerate(zip(aligned_xys, ids)):
        ax.scatter(ali[:, 0], ali[:, 1], s=0.2, alpha=0.25,
                   c=[cmap(i)], label=idd, linewidths=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"All {n} sections aligned to {ids[0]} (jet by section index)")
    handles = [
        mlines.Line2D([], [], color=cmap(i), marker="s", linestyle="None",
                      markersize=10, label=lab)
        for i, lab in enumerate(ids)
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_point_adjacent_pairs(
    aligned_xys: list[np.ndarray], ids: list[str], out_path: Path
) -> None:
    """One panel per adjacent pair: section i (red) over section i+1 (green)."""
    n_pairs = len(ids) - 1
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]
    for i in range(n_pairs):
        a, b = aligned_xys[i], aligned_xys[i + 1]
        axes[i].scatter(a[:, 0], a[:, 1], s=0.2, alpha=0.35, c="red", linewidths=0)
        axes[i].scatter(b[:, 0], b[:, 1], s=0.2, alpha=0.35, c="green", linewidths=0)
        axes[i].set_aspect("equal")
        axes[i].invert_yaxis()
        axes[i].set_title(f"{ids[i]} (R)  vs  {ids[i + 1]} (G)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_image_rainbow(stack: np.ndarray, ids: list[str], out_path: Path) -> None:
    """All sections additively blended with a jet hue per section index."""
    n, h, w = stack.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    cmap = cm.get_cmap("jet", n)
    for i in range(n):
        norm = stack[i].astype(np.float32) / 255.0
        color = np.array(cmap(i)[:3], dtype=np.float32)
        rgb += norm[..., None] * color[None, None, :]
    rgb = np.clip(rgb, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(f"All {n} sections aligned to {ids[0]} (jet by section index)")
    handles = [
        mlines.Line2D([], [], color=cmap(i), marker="s", linestyle="None",
                      markersize=10, label=lab)
        for i, lab in enumerate(ids)
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_adjacent_pairs(
    stack: np.ndarray,
    ids: list[str],
    dices: list[tuple[float, float]],
    out_path: Path,
) -> None:
    """One panel per adjacent pair: section i (R) over section i+1 (G)."""
    n_pairs = stack.shape[0] - 1
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]
    for i in range(n_pairs):
        r = stack[i].astype(np.float32) / 255.0
        g = stack[i + 1].astype(np.float32) / 255.0
        rgb = np.zeros((*r.shape, 3), dtype=np.float32)
        rgb[..., 0] = r
        rgb[..., 1] = g
        axes[i].imshow(np.clip(rgb, 0, 1))
        axes[i].axis("off")
        d_before, d_after = dices[i]
        axes[i].set_title(
            f"{ids[i]} (R)  vs  {ids[i + 1]} (G)\n"
            f"Dice before={d_before:.3f}  after={d_after:.3f}"
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
