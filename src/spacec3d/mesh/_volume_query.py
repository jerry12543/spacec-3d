"""Volume measurement on the label voxel volume that backs the meshes.

The meshes are surfaces; the volume math runs on the label volume `create_mesh`
saved alongside them (`query_volume.npz`), so segment IDs match what is displayed.
"""

import numpy as np


def segment_volumes(
    labels: np.ndarray,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[int, float]:
    """Enclosed volume of every segment, keyed by segment id.

    labels: (z, y, x) integer label volume; 0 is background and is excluded.
    voxel_size: (z, y, x) physical size of one voxel. Volumes are in those units
        cubed (raw voxel count if voxel_size is all ones).
    """
    counts = np.bincount(labels.ravel())
    voxel_vol = float(np.prod(voxel_size))
    return {
        int(seg_id): float(count) * voxel_vol
        for seg_id, count in enumerate(counts)
        if seg_id != 0 and count != 0
    }


def box_volume(
    labels: np.ndarray,
    voxel_size: tuple[float, float, float],
    p0,
    p1,
    segment: int | None = None,
) -> tuple[float, int, int]:
    """Foreground volume inside an axis-aligned box, plus voxel counts.

    p0, p1: opposite corners as (z, y, x) voxel indices (any order; floats ok).
    segment: if given, count only that segment's voxels; otherwise all foreground.
    Returns (volume, foreground_count, box_capacity), where box_capacity is the total
    number of voxels the (clamped) box spans -- so foreground_count <= box_capacity and
    their ratio is the fill fraction. The box is clamped to the array bounds.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    shape = np.asarray(labels.shape)

    lo = np.clip(np.floor(np.minimum(p0, p1)).astype(int), 0, shape)
    hi = np.clip(np.ceil(np.maximum(p0, p1)).astype(int), 0, shape)

    sub = labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if segment is not None:
        count = int(np.count_nonzero(sub == segment))
    else:
        count = int(np.count_nonzero(sub))

    voxel_vol = float(np.prod(voxel_size))
    return float(count) * voxel_vol, count, int(sub.size)
