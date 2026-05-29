"""Rasterize aligned point clouds into a (Z, H, W) voxel volume."""

from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd


def points_to_volume(
    points_csv: str | PathLike[str],
    bin_size: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Rasterize aligned point clouds into a (Z, H, W) uint32 count volume.

    points_csv: path to a CSV with columns `section, x, y` (output of align_points).
    bin_size: data-unit size of a single (y, x) voxel. Larger = coarser/denser
        volume.
    bounds: (xmin, ymin, xmax, ymax) in data units. If None, computed as the
        bounding box across all sections so every point fits.

    Returns a (Z, H, W) uint32 array of per-voxel point counts, with Z ordered
    by first appearance of each section in the CSV.
    """
    df = pd.read_csv(Path(points_csv))
    for col in ("section", "x", "y"):
        if col not in df.columns:
            raise ValueError(f"missing column {col!r} in {points_csv}")

    if bounds is None:
        xmin, ymin = float(df["x"].min()), float(df["y"].min())
        xmax, ymax = float(df["x"].max()), float(df["y"].max())
    else:
        xmin, ymin, xmax, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"degenerate bounds: ({xmin}, {ymin}, {xmax}, {ymax})")

    w = max(1, int(np.ceil((xmax - xmin) / bin_size)))
    h = max(1, int(np.ceil((ymax - ymin) / bin_size)))
    x_edges = xmin + np.arange(w + 1) * bin_size
    y_edges = ymin + np.arange(h + 1) * bin_size

    sections = list(dict.fromkeys(df["section"].tolist()))
    planes = []
    for sec in sections:
        sub = df[df["section"] == sec]
        plane, _, _ = np.histogram2d(
            sub["y"].to_numpy(), sub["x"].to_numpy(),
            bins=(y_edges, x_edges),
        )
        planes.append(plane.astype(np.uint32))
    return np.stack(planes, axis=0)
