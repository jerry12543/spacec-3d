"""Image-based serial-section registration using space-map's AutoAffineImgKey."""

from os import PathLike
from pathlib import Path

import numpy as np
import tifffile as tf
from space_map.affine_block import AutoAffineImgKey

from ._plot import plot_adjacent_pairs, plot_image_rainbow
from ._preprocess import pad_to


def align_image_pair(
    fixed: np.ndarray,
    moving: np.ndarray,
    method: str = "auto",
    use_ldm: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Register `moving` to `fixed` with space-map. Returns (warped, H).

    fixed, moving: 2D preprocessed images.
    method: passed to AutoAffineImgKey. "auto" runs LoFTR + RANSAC affine.
    use_ldm: also run LDDMM non-rigid refinement on top of affine. Note that this method is still buggy in the latest space-map version
    """
    mgr = AutoAffineImgKey(imgI=fixed, imgJ=moving, method=method)
    if use_ldm:
        mgr.useLDM = True
    H = mgr.run()
    warped = mgr.imgJ

    warped = pad_to(warped[: fixed.shape[0], : fixed.shape[1]], *fixed.shape)
    return warped, H


def align_image_stack(
    images: list[np.ndarray],
    ids: list[str],
    output_dir: str | PathLike[str],
    method: str = "auto",
    use_ldm: bool = False,
) -> None:
    """
    Sequential pairwise registration of a stack into the first frame.

    Each images[i+1] is registered to the already-aligned images[i] (in
    images[0]'s frame), so transforms compose along the chain.

    images: list of 2D preprocessed images, all the same shape.
    ids: section identifiers, same length as images.
    output_dir: written outputs go here.
        - aligned_stack.tif         multi-page aligned stack
        - rainbow.png               all sections one image, hue per section
        - adjacent_pairs.png        per-pair R/G overlay with Dice before/after
        - H_<ids[i+1]>_to_<ids[i]>.npy   per-pair affine
    method, use_ldm: forwarded to align_image_pair.
    """
    if len(images) != len(ids):
        raise ValueError(f"images and ids length mismatch: {len(images)} vs {len(ids)}")
    if len(images) < 2:
        raise ValueError("need at least 2 sections")
    shapes = {im.shape for im in images}
    if len(shapes) > 1:
        raise ValueError(f"all images must share a shape; got {shapes}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned = [images[0]]
    pair_dices: list[tuple[float, float]] = []
    for i in range(len(images) - 1):
        fixed = aligned[i]
        moving = images[i + 1]
        d_before = _dice(fixed, moving)
        warped, H = align_image_pair(fixed, moving, method=method, use_ldm=use_ldm)
        d_after = _dice(fixed, warped)
        pair_dices.append((d_before, d_after))
        aligned.append(warped)
        np.save(output_dir / f"H_{ids[i + 1]}_to_{ids[i]}.npy", H) # save the affine transform

    stack = np.stack(aligned, axis=0)
    tf.imwrite(output_dir / "aligned_stack.tif", stack)
    plot_image_rainbow(stack, ids, output_dir / "rainbow.png")
    plot_adjacent_pairs(stack, ids, pair_dices, output_dir / "adjacent_pairs.png")


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0
    b_bin = b > 0
    s = a_bin.sum() + b_bin.sum()
    if s == 0:
        return 0.0
    return float(2.0 * np.logical_and(a_bin, b_bin).sum() / s)
