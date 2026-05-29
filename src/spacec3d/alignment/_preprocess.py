"""Image preprocessing helpers for serial-section registration."""

from os import PathLike
from pathlib import Path

import numpy as np
import tifffile as tf
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects


def load_channel(section_dir: str | PathLike[str], channel: str) -> np.ndarray:
    """Load a single named channel from a CODEX section dir.

    Expects `<section_dir>/image.tif` (multi-page TIFF, one page per channel) and
    `<section_dir>/channelnames.txt` (one channel name per line). Returns a 2D
    array — never reads the whole TIFF.
    """
    section_dir = Path(section_dir)
    names = [
        c.strip()
        for c in (section_dir / "channelnames.txt").read_text().splitlines()
        if c.strip()
    ]
    if channel not in names:
        raise ValueError(
            f"channel {channel!r} not in {section_dir.name}; first few: {names[:5]}"
        )
    ch_idx = names.index(channel)
    with tf.TiffFile(section_dir / "image.tif") as tif:
        return tif.pages[ch_idx].asarray()


def downsample(img: np.ndarray, k: int) -> np.ndarray:
    """Stride-based downsample by factor k. k=1 is a no-op."""
    return img[::k, ::k] if k > 1 else img


def normalize(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    """Percentile clip + rescale to uint8 [0, 255]. Percentiles ignore zeros for p_lo."""
    arr = img.astype(np.float32)
    nz = arr[arr > 0]
    if nz.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo = np.percentile(nz, p_lo)
    hi = np.percentile(arr, p_hi)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Zero-pad to (h, w) on the bottom-right."""
    out = np.zeros((h, w), dtype=img.dtype)
    out[: img.shape[0], : img.shape[1]] = img
    return out


def pad_all(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """Zero-pad every image in `imgs` to the per-axis maximum so they share a frame."""
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    return [pad_to(im, h, w) for im in imgs]


def tissue_mask(
    img: np.ndarray,
    close_size: int = 25,
    dilate_size: int = 5,
    blur_sigma: float = 8.0,
    min_object_px: int = 5000,
) -> np.ndarray:
    """Blurred-Otsu tissue mask robust to FOV-tile gaps and varying brightness.

    Steps: gaussian blur -> Otsu on nonzero pixels -> binary closing -> fill
    holes -> remove tiny specks -> final dilation. Returns a bool mask the same
    shape as `img`.
    """
    if img.max() == 0:
        return np.zeros_like(img, dtype=bool)
    blurred = gaussian_filter(img.astype(np.float32), sigma=blur_sigma)
    nz = blurred[blurred > 0]
    if nz.size == 0:
        return np.zeros_like(img, dtype=bool)
    mask = blurred > threshold_otsu(nz)
    if close_size > 0:
        mask = binary_closing(mask, structure=np.ones((close_size, close_size), dtype=bool))
    mask = binary_fill_holes(mask)
    if min_object_px > 0:
        mask = remove_small_objects(mask, min_size=min_object_px)
    if dilate_size > 0:
        mask = binary_dilation(mask, structure=np.ones((dilate_size, dilate_size), dtype=bool))
    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out everything outside `mask`, preserving intensities inside."""
    return np.where(mask, img, 0).astype(img.dtype)
