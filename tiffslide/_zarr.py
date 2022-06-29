"""
provides helpers for handling and compositing arrays and zarr-like groups
"""
from __future__ import annotations

import itertools
import math
from typing import Any
from typing import Iterator
from typing import TYPE_CHECKING

import numpy as np
import zarr

from tiffslide._types import Size3D
from tiffslide._types import Point3D
from tiffslide._types import Slice3D
from tiffslide._types import SeriesCompositionInfo

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from tifffile import TiffFile

__all__ = [
    "composite",
    "CompositedArray",
    "CompositedGroup",
]


# --- composition classes ---------------------------------------------

class CompositedArray:
    """composite zarr.Arrays with offsets

    Combine into a 3D array object with array slice indexing
    """
    def __init__(
        self,
        shape: Size3D,
        located_arrays: dict[Point3D, zarr.Array],
        *,
        fill_value: Any | None = None
    ) -> None:
        """instantiate a CompositedArray

        Parameters
        ----------
        shape:
            3D shape of the composited array
        located_arrays:
            map from offsets to arrays
        fill_value:
            overwrite the fill value of the composition
        """
        self.shape = tuple(shape)
        assert len(self.shape) == 3, "restricting to 3 dims for now"
        self._located_arrays = located_arrays

        self.dtype, self.fill_value = verify_located_arrays(shape, located_arrays)
        if fill_value is not None:
            self.fill_value = fill_value

    def __getitem__(self, item: Slice3D) -> NDArray[np.uint]:
        if not (
            isinstance(item, tuple)
            and len(item) == 3
            and all(isinstance(s, slice) for s in item)
        ):
            raise ValueError("support is limited to 3D slice indexing")

        if not all(s.step == 1 or s.step is None for s in item):
            raise ValueError("support is limited to slices with step 1")

        # todo: getting these overlaps should be done via an RTree...
        overlaps = {}
        for offset, arr in self._located_arrays.items():

            overlap = get_overlap(item, self.shape, offset, arr.shape)
            if overlap is None:
                continue

            # todo: could take a shortcut here...
            # target, source = overlap
            # if target == item:  # < this check is incorrect [:, :] != [0:10, 0:10]
            #     return arr[source]

            overlaps[offset] = overlap

        s0, s1, s2 = item
        r0 = range(self.shape[0])[s0]
        r1 = range(self.shape[1])[s1]
        r2 = range(self.shape[2])[s2]
        shape = (r0.stop - r0.start, r1.stop - r1.start, r2.stop - r2.start)
        # todo: optimize! In the most common case this should be filled entirely
        out = np.full(shape, fill_value=self.fill_value, dtype=self.dtype)

        for offset, (target, source) in overlaps.items():
            arr = self._located_arrays[offset]
            out[target] = arr[source]

        return out


class CompositedGroup:
    """minimal required interface for tiffslide's read_region

    use to group CompositedArrays into a zarr.hierarchy.Group like object
    """
    def __init__(self, arrays: list[CompositedArray]):
        self._groups = {
            str(lvl): array
            for lvl, array in enumerate(arrays)
        }

    def __getitem__(self, item: str) -> CompositedArray:
        return self._groups[item]

    def __len__(self) -> int:
        return len(self._groups)

    def __iter__(self) -> Iterator[str]:
        return iter(self._groups)


def composite(tf: TiffFile, info: SeriesCompositionInfo) -> CompositedGroup | CompositedArray:
    """composite series or arrays into a virtual array"""
    composited_arrays: list[CompositedArray] = []

    shape = info["shape"]
    located_series = info["located_series"]
    for lvl in itertools.count():
        located_arrays: dict[Point3D, zarr.Array] = {}

        area_0 = None
        for offset, series_idx in located_series.items():
            series = tf.series[series_idx]
            try:
                zstore = series.aszarr(lvl)
            except IndexError:
                if len(located_arrays) == 0:
                    break
                raise RuntimeError("series seem to have different levels")
            else:
                zarray = zarr.open(zstore, mode="r")

            # implicitly assuming axes="YXS" ...
            area_i = zarray.shape[0] * zarray.shape[1]
            if area_0 is None:
                area_0 = area_i
            ds = math.sqrt(area_0 / area_i)

            lvl_offset = (math.ceil(offset[0] / ds), math.ceil(offset[1] / ds), offset[2])
            located_arrays[lvl_offset] = zarray

        composited_arrays.append(CompositedArray(shape, located_arrays))

    if len(composited_arrays) == 1:
        return composited_arrays[0]
    else:
        return CompositedGroup(composited_arrays)


# --- helper functions ------------------------------------------------

def get_overlap(
    selection: Slice3D,
    cshape: Size3D,
    offset: Point3D,
    ashape: Size3D,
) -> tuple[Slice3D, Slice3D] | None:
    """return correct slices for overlapping arrays

    CompositedArray
    +------------------------+
    | x------------x         |
    | | Slice o----|---o     |
    | |       |    |   |     |
    | x------------x   |     |
    |         | Array  |     |
    |         o--------o     |
    +------------------------+

    Notes
    -----
    if selection is a slice out of the CompositedArray with shape=cshape,
    this function returns either None, if the slice and the Array share
    no overlap, or two slices (target and source), so that:

    `new_arr[target] = array[source]`

    can be used to assign the overlap correctly to the output array.

    """
    s0, s1, s2 = selection
    d0, d1, d2 = cshape
    o0, o1, o2 = offset
    a0, a1, a2 = ashape

    # dim 0
    x0, x1, _ = s0.indices(d0)
    X0 = o0
    X1 = o0 + a0
    if X1 < x0 or x1 <= X0:
        return None
    # dim 1
    y0, y1, _ = s1.indices(d1)
    Y0 = o1
    Y1 = o1 + a1
    if Y1 < y0 or y1 <= Y0:
        return None
    # dim 2
    z0, z1, _ = s2.indices(d2)
    Z0 = o2
    Z1 = o2 + a2
    if Z1 < z0 or z1 <= Z0:
        return None

    ix0 = max(x0, X0)
    iy0 = max(y0, Y0)
    iz0 = max(z0, Z0)
    ix1 = min(x1, X1)
    iy1 = min(y1, Y1)
    iz1 = min(z1, Z1)

    target = (
        slice(ix0 - x0, ix1 - x0, 1),
        slice(iy0 - y0, iy1 - y0, 1),
        slice(iz0 - z0, iz1 - z0, 1),
    )
    source = (
        slice(ix0 - X0, ix1 - X0, 1),
        slice(iy0 - Y0, iy1 - Y0, 1),
        slice(iz0 - Z0, iz1 - Z0, 1),
    )
    return target, source


def verify_located_arrays(
    shape: Size3D,
    located_arrays: dict[Point3D, zarr.Array]
) -> tuple[str, Any]:
    """verify located arrays

    ensures that dtypes and fill_values agree and that
    arrays are within the composited boundaries.
    """
    dt_fv = set()
    for loc, arr in located_arrays.items():
        dt_fv.add((arr.dtype, arr.fill_value))

        x0, x1, x2 = loc
        d0, d1, d2 = arr.shape
        out_of_bounds = (
            (x0 + d0 < 0 or x0 >= shape[0])
            or (x1 + d1 < 0 or x1 >= shape[1])
            or (x2 + d2 < 0 or x2 >= shape[2])
        )
        if out_of_bounds:
            raise ValueError(f"array at location {loc!r} is out of bounds")

    if not len(dt_fv) == 1:
        raise ValueError(f"arrays don't share the same dtype and fill_value, got: {dt_fv!r}")
    (dtype, fill_value), = dt_fv
    return dtype, fill_value
