"""Functions and utilities to create a mesh from a volume array."""

import math
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import spatialdata as sd
from skimage import measure
from spatialdata.models import Labels3DModel
from spatialdata.transformations import Scale
from tissue_map_tools.igneous_converters import (
    from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes,
)


def create_mesh(
    volume: np.ndarray,
    output_dir: str | PathLike[str],
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    binary: bool = True,
    spatial_tiles: tuple[int, int, int] = (1, 1, 1),
    mode: Literal["intensity", "labels"] = "intensity",
    threshold: float = 0.05,
) -> None:
    """
    voxel_size: (z,y,x) image scaling factor
    binary: whether to create a binary or connected-component mesh
    spatial_tiles: how many tiles in (z,y,x) directions
    mode: "labels" when each volume voxel in the same region has the same value
    threshold: only for mode "intensity" -- anything with intensity above median intensity * threshold will have value 1 and anything else will have value 0
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "labels":
        labels = volume.astype(np.uint32)
    else:  # intensity-based
        labels = make_labels(volume, binary, threshold)

    nz, ny, nx = spatial_tiles
    labels = add_tiling(labels, nz, ny, nx)

    labels = ndarray_to_labels(labels, voxel_size)

    from_spatialdata_raster_to_sharded_precomputed_raster_and_meshes(
        raster=labels,
        precomputed_path=str(output_dir),
        multiscale=True,
        sharded_raster=True,
        sharded_mesh=True,
    )


def make_labels(
    volume: np.ndarray,
    binary: bool,
    threshold: float,
) -> np.ndarray:
    """Convert intensity to labels."""

    nonzero = volume[volume > 0]
    threshold = threshold * float(np.median(nonzero))

    labels = volume > threshold  # binary mask

    if not binary:
        labels = measure.label(labels)  # connected-component labeling

    return labels.astype(np.uint32)


def round_to_power_of_10(n: int):
    return 10 ** math.ceil(math.log10(n))


def add_tiling(volume: np.ndarray, nz: int, ny: int, nx: int) -> np.ndarray:
    """Add a tiling scheme to labels."""

    z_size, y_size, x_size = volume.shape

    multiplier = round_to_power_of_10(np.max(volume))

    tile_x = np.arange(x_size, dtype=np.uint32) * nx // x_size
    tile_y = np.arange(y_size, dtype=np.uint32) * ny // y_size
    tile_z = np.arange(z_size, dtype=np.uint32) * nz // z_size

    tile_map = (
        multiplier
        * (
            tile_z[:, None, None] * (ny * nx)
            + tile_y[None, :, None] * nx
            + tile_x[None, None, :]
            + 1
        )
    ).astype(np.uint32)

    foreground = volume != 0
    labels = np.where(foreground, tile_map, 0).astype(np.uint32)
    labels += volume

    return labels


def ndarray_to_labels(
    labels: np.ndarray, voxel_size: tuple[float, float, float]
) -> sd.models.Labels3DModel:
    "Convert ndarray labels to Labels3dModel, compatible with spatialData."

    z_scale, y_scale, x_scale = voxel_size
    scale_transform = Scale([z_scale, y_scale, x_scale], axes=("z", "y", "x"))
    element = Labels3DModel.parse(
        labels,
        dims=("z", "y", "x"),
        transformations={"global": scale_transform},
    )
    return element
