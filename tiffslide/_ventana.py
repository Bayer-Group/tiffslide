from __future__ import annotations

from typing import Mapping
from typing import TypedDict

import numpy as np


# dict types for joint info
TileJointInfo = TypedDict("TileJointInfo", {
    "@FlagJoined": str,
    "@Confidence": str,
    "@Direction": str,
    "@Tile1": str,
    "@Tile2": str,
    "@OverlapX": str,
    "@OverlapY": str,
})
ImageInfo = TypedDict("ImageInfo", {
    "@AOIScanned": str,
    "@Width": str,
    "@Height": str,
    "@NumRows": str,
    "@NumCols": str,
    "@Pos-X": str,
    "@Pos-Y": str,
    "@MaxJTPFileSize": str,
    "TileJointInfo": "list[TileJointInfo]",
})


def _ventana_calculate_absolute_offsets(ji: ImageInfo) -> Mapping[int, tuple[int, int]]:
    """takes a ventana BIF slide stitching info and returns absolute tile coordinates"""

    num_joints = len(ji["TileJointInfo"])
    num_tiles = int(ji["@NumRows"]) * int(ji["@NumCols"])
    tile_width = int(ji["@Width"])
    tile_height = int(ji["@Height"])

    A_select = np.zeros((num_joints + 1, num_tiles), dtype=int)
    Y_diff = np.zeros((num_joints + 1, ), dtype=complex)

    # set one tile's coordinates to 0, 0
    A_select[0, num_tiles // 2] = 1
    Y_diff[0] = complex(0, 0)

    for iy, j in enumerate(ji["TileJointInfo"], start=1):
        # make indices start at 0
        i0 = int(j["@Tile1"]) - 1
        i1 = int(j["@Tile2"]) - 1

        # select tiles for comparison
        A_select[iy, i0] = -1  # todo: t0 - t1 or t1 - t0 ???
        A_select[iy, i1] = 1
        overlap_x = int(j["@OverlapX"])
        overlap_y = int(j["@OverlapY"])

        # compute the complex differences
        direction = j["@Direction"]
        # todo: I need to double check the signs on all of these
        if direction == "UP":
            dx = overlap_x
            dy = overlap_y - tile_height
        elif direction == "RIGHT":
            dx = overlap_x - tile_width
            dy = overlap_y
        else:
            # fixme: I think i've seen "LEFT" and "DOWN" on image.sc
            #   would be super easy to add here...
            raise ValueError(direction)
        Y_diff[iy] = complex(dx, dy)

    # todo: check if solution makes sense here...
    c_coords, _, _, _ = np.linalg.lstsq(A_select, Y_diff, rcond=None)
    coords_xy = np.vstack((
        c_coords.real.round().astype(int),
        c_coords.imag.round().astype(int),
    )).T
    return coords_xy
