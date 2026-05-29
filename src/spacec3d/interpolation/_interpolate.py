"""Z-axis interpolation of an aligned 2D slice stack into a dense 3D volume. Note that this is a temporary dropin."""

from os import PathLike
from pathlib import Path

import numpy as np
import tifffile as tf
from scipy.ndimage import gaussian_filter, zoom


def interpolate_z(
    volume: np.ndarray,
    output_path: str | PathLike[str],
    z_factor: float = 50.0,
    sigma_z: float = 3.0,
    background_threshold: float = 0.08,
) -> None:
    """
    Cubic-interpolate along Z, smooth slice seams, threshold background, write TIFF.

    volume: (Z, H, W) array of aligned slices in a common frame.
    output_path: multi-page uint16 TIFF written here (ImageJ-compatible).
    z_factor: interpolated Z voxels per original slice gap.
    sigma_z: Gaussian sigma along Z only, smooths brightness jumps at slice boundaries.
    background_threshold: after normalization, voxels below this go to 0.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected (Z, H, W) volume, got shape {volume.shape}")

    vol = volume.astype(np.float32)
    vol = _normalize(vol)
    vol = zoom(vol, (z_factor, 1.0, 1.0), order=3)
    vol = gaussian_filter(vol, sigma=(sigma_z, 0.0, 0.0))
    vol = _normalize(vol)
    vol[vol < background_threshold] = 0.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tf.imwrite(output_path, (vol * 65535).astype(np.uint16), imagej=True)


def _normalize(vol: np.ndarray) -> np.ndarray:
    vol = vol - vol.min()
    m = vol.max()
    if m > 0:
        vol = vol / m
    return vol
