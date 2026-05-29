"""Point-based serial-section registration using space-map."""

import shutil
import tempfile
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from space_map import Slice
from space_map.flow import AutoFlowMultiCenter4, FlowImport

from ._plot import plot_point_adjacent_pairs, plot_point_rainbow


def align_points(
    xys: list[np.ndarray],
    ids: list[str],
    output_dir: str | PathLike[str],
    work_dir: str | PathLike[str] | None = None,
    use_ldm: bool = True,
) -> None:
    """
    Align serial-section point clouds (affine + optional LDDMM).

    xys: list of (N_i, 2) arrays of (x, y) cell coordinates, one per section.
    ids: section identifiers, same length as xys.
    output_dir: written outputs go here.
        - aligned_points.csv                 columns: section, x, y
        - rainbow.png                        all aligned sections overlaid, jet hue per section
        - adjacent_pairs.png                 per-pair R/G scatter of adjacent sections
        - transforms/affine_H_<j>_to_<i>.npy 3x3 affine for each ordered pair (i != j)
    work_dir: scratch dir for space-map's project files. Created + deleted if None.
    use_ldm: run LDDMM non-rigid refinement after affine.

    Note: LDDMM transform is not recoverable.
    """

    if len(xys) != len(ids):
        raise ValueError(f"xys and ids length mismatch: {len(xys)} vs {len(ids)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleanup = work_dir is None
    work_dir = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="spacemap_"))

    try:
        aligned, slices, affine_key = _run_spacemap(xys, ids, work_dir, use_ldm=use_ldm)

        rows = []
        for idd, ali in zip(ids, aligned):
            for x, y in ali:
                rows.append((idd, float(x), float(y)))
        pd.DataFrame(rows, columns=["section", "x", "y"]).to_csv(
            output_dir / "aligned_points.csv", index=False
        )

        _save_affine_transforms(slices, affine_key, output_dir / "transforms")

        plot_point_rainbow(aligned, ids, output_dir / "rainbow.png")
        plot_point_adjacent_pairs(aligned, ids, output_dir / "adjacent_pairs.png")
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


def _run_spacemap(xys, ids, work_dir: Path, use_ldm: bool):
    flow = FlowImport(str(work_dir))
    slices = flow.init_xys(xys, ids)

    mgr = AutoFlowMultiCenter4(slices, initJKey=Slice.rawKey)
    mgr.affine(useKey="DF", show=False)

    if use_ldm:
        mgr.ldm_pair(fromKey=Slice.align1Key, toKey=Slice.align2Key, show=False)
        out_key = Slice.align2Key
    else:
        out_key = Slice.align1Key

    aligned = [s.ps(out_key) for s in slices]
    return aligned, slices, mgr.affineKey


def _save_affine_transforms(slices, affine_key: str, transforms_dir: Path) -> None:
    """Pull each persisted pairwise affine H out of the spacemap project and
    write it as a .npy under transforms_dir. Skips pairs the chain didn't compute."""
    transforms_dir.mkdir(parents=True, exist_ok=True)
    for s_from in slices:
        for s_to in slices:
            if s_from.index == s_to.index:
                continue
            try:
                H = s_from.data.loadH(s_to.index, affine_key)
            except Exception:
                H = None
            if H is None:
                continue
            np.save(transforms_dir / f"affine_H_{s_from.index}_to_{s_to.index}.npy", H)
