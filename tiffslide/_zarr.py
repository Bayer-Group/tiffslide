"""
provides helpers for handling and compositing arrays and zarr-like groups
"""

from __future__ import annotations

import asyncio
import builtins
import json
from collections.abc import AsyncIterator
from collections.abc import Iterable
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import zarr
from fsspec.implementations.reference import ReferenceFileSystem
from tifffile import TiffFile
from tifffile.zarr import ZarrTiffStore
from zarr.abc.store import Store
from zarr.core.buffer import BufferPrototype
from zarr.core.buffer import default_buffer_prototype
from zarr.core.buffer.cpu import Buffer
from zarr.core.sync import sync as _sync
from zarr.storage import FsspecStore
from zarr.storage import MemoryStore

from tiffslide._compat import NotTiffFile
from tiffslide._types import Point3D
from tiffslide._types import SeriesCompositionInfo
from tiffslide._types import Size3D
from tiffslide._types import Slice3D

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray
    from zarr.abc.store import ByteRequest


__all__ = [
    "get_zarr_store",
    "get_zarr_depth_and_dtype",
    "get_zarr_selection",
]


# --- zarr storage classes --------------------------------------------


def _store_exists_sync(store: Store, key: str) -> bool:
    """synchronous wrapper for store.exists()"""
    return bool(_sync(store.exists(key)))


def _store_get_sync(
    store: Store,
    key: str,
    prototype: BufferPrototype | None = None,
    byte_range: ByteRequest | None = None,
) -> Buffer | None:
    """synchronous wrapper for store.get()"""
    if prototype is None:
        prototype = default_buffer_prototype()
    return _sync(store.get(key, prototype, byte_range=byte_range))


def _store_list_sync(store: Store) -> list[str]:
    """synchronous wrapper for store.list()"""

    async def _collect() -> list[str]:
        return [key async for key in store.list()]

    return list(_sync(_collect()))


def _store_list_prefix_sync(store: Store, prefix: str) -> list[str]:
    """synchronous wrapper for store.list_prefix()"""

    async def _collect() -> list[str]:
        return [key async for key in store.list_prefix(prefix)]

    return list(_sync(_collect()))


class _CompositedStore(Store):
    """prefix-routing zarr v3 Store that mounts child stores under prefixes"""

    _stores: dict[str, Store]
    _base: MemoryStore

    def __init__(
        self,
        prefixed_stores: Mapping[str, Store],
        *,
        zattrs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(read_only=True)

        self._base = MemoryStore()
        # create a zarr v2 group in the memory store with optional attributes
        _zgroup = json.dumps({"zarr_format": 2}).encode()
        _sync(self._base.set(".zgroup", Buffer.from_bytes(_zgroup)))
        if zattrs:
            _zattrs = json.dumps({"tiffslide.series-composition": zattrs}).encode()
            _sync(self._base.set(".zattrs", Buffer.from_bytes(_zattrs)))

        self._stores = {}
        for prefix, store in prefixed_stores.items():
            assert not prefix.endswith("/")
            self._stores[prefix] = store

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        prefix, _, subkey = key.partition("/")
        if subkey:
            try:
                store = self._stores[prefix]
            except KeyError:
                pass
            else:
                return await store.get(subkey, prototype, byte_range=byte_range)
        return await self._base.get(key, prototype, byte_range=byte_range)

    async def exists(self, key: str) -> bool:
        prefix, _, subkey = key.partition("/")
        if subkey:
            try:
                store = self._stores[prefix]
            except KeyError:
                pass
            else:
                return bool(await store.exists(subkey))
        # also return True for bare prefix (it's a "directory")
        if not subkey and prefix in self._stores:
            return True
        return bool(await self._base.exists(key))

    async def set(self, key: str, value: Buffer) -> None:
        raise ReadOnlyError()

    async def delete(self, key: str) -> None:
        raise ReadOnlyError()

    async def list(self) -> AsyncIterator[str]:
        for prefix, store in self._stores.items():
            async for key in store.list():
                yield f"{prefix}/{key}"
        async for key in self._base.list():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        if prefix:
            store_prefix = prefix.rstrip("/")
            if store_prefix in self._stores:
                store = self._stores[store_prefix]
                async for key in store.list():
                    yield f"{store_prefix}/{key}"
                return
            # check if prefix is a sub-path within a child store
            top, _, rest = store_prefix.partition("/")
            if top in self._stores:
                store = self._stores[top]
                async for key in store.list_dir(rest):
                    yield f"{top}/{key}"
                return
        # fall back to base store
        async for key in self._base.list_dir(prefix):
            yield key
        # also list child store prefixes as "directories"
        if not prefix:
            for p in self._stores:
                yield p

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_partial_writes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return True

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> builtins.list[Buffer | None]:
        return [
            await self.get(key, prototype, byte_range=byte_range)
            for key, byte_range in key_ranges
        ]

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, _CompositedStore)
            and self._stores == value._stores
            and self._base == value._base
        )


class ReadOnlyError(Exception):
    """raised when write operations are attempted on a read-only store"""


def _get_series_zarr(
    obj: TiffFile | ReferenceFileSystem,
    series_idx: int,
    *,
    num_decode_threads: int | None = None,
) -> Store:
    """return a zarr store from the object"""
    if isinstance(obj, (TiffFile, NotTiffFile)):
        zstore: Store = obj.series[series_idx].aszarr(maxworkers=num_decode_threads)  # type: ignore
    elif isinstance(obj, ReferenceFileSystem):
        zstore = FsspecStore(fs=obj, path=f"s{series_idx}", read_only=True)
    else:
        raise NotImplementedError(f"{type(obj).__name__} unsupported")
    return zstore


def _is_single_array_store(store: Store) -> bool:
    """check if the store represents a single zarr array (not a group)"""
    return _store_exists_sync(store, ".zarray")


def get_zarr_store(
    properties: Mapping[str, Any],
    tf: TiffFile | ReferenceFileSystem | None,
    *,
    num_decode_threads: int | None = None,
) -> Store:
    """return a zarr store

    Parameters
    ----------
    properties:
        the TiffSlide().properties mapping
    tf:
        the corresponding TiffFile instance
    num_decode_threads:
        number of threads used for decoding (default num_cpu / 2)

    Returns
    -------
    store:
        a zarr v3 Store of the tiff
    """
    if tf is None:
        raise NotImplementedError("support in future versions")

    # the tiff might contain multiple series that require composition
    composition: SeriesCompositionInfo | None = properties.get(
        "tiffslide.series-composition"
    )
    store: Store
    if composition:
        prefixed_stores: dict[str, Store] = {}
        for series_idx in composition["located_series"].keys():
            _store = _get_series_zarr(
                tf, series_idx, num_decode_threads=num_decode_threads
            )
            # encapsulate store as group if tifffile returns a zarr array
            if _is_single_array_store(_store):
                _store = _CompositedStore({"0": _store})
            prefixed_stores[str(series_idx)] = _store

        store = _CompositedStore(prefixed_stores, zattrs=composition)

    else:
        series_idx = properties.get("tiffslide.series-index", 0)
        _store = _get_series_zarr(tf, series_idx, num_decode_threads=num_decode_threads)

        # encapsulate store as group if tifffile returns a zarr array
        if _is_single_array_store(_store):
            store = _CompositedStore({"0": _store})
        else:
            store = _store

    return store


def get_zarr_selection(
    grp: zarr.Group,
    level: int,
    selection: Slice3D,
) -> NDArray[np.int_]:
    """retrieve the selection of the zarr Group"""
    composition: SeriesCompositionInfo = grp.attrs.get("tiffslide.series-composition")

    if composition is None:
        # no composition required, simply retrieve the array
        return grp[str(level)][selection]

    else:
        # we need to composite the array
        # todo: getting these overlaps should be done via an RTree...
        overlaps = {}
        dtype = None
        fill_value = None
        level_shape = composition["level_shapes"][level]
        located_series = composition["located_series"]

        for series_idx, level_offsets in located_series.items():
            arr = grp[f"{series_idx}/{level}"]
            offset = level_offsets[level]

            if dtype is None:
                dtype = arr.dtype
            if fill_value is None:
                fill_value = arr.fill_value

            overlap = get_overlap(selection, level_shape, offset, arr.shape)
            if overlap is None:
                continue

            # todo: could take a shortcut here...
            # target, source = overlap
            # if target == item:  # < this check is incorrect [:, :] != [0:10, 0:10]
            #     return arr[source]

            overlaps[series_idx] = overlap

        s0, s1, s2 = selection
        r0 = range(level_shape[0])[s0]
        r1 = range(level_shape[1])[s1]
        r2 = range(level_shape[2])[s2 if s2 is not ... else slice(None)]
        shape = (r0.stop - r0.start, r1.stop - r1.start, r2.stop - r2.start)
        # todo: optimize! In the most common case this should be filled entirely
        out = np.full(shape, fill_value=fill_value, dtype=dtype)

        for series_idx, (target, source) in overlaps.items():
            arr = grp[f"{series_idx}/{level}"]
            out[target] = arr[source]

    return out


def _unwrap_to_zarrtiffstore(store: Store) -> ZarrTiffStore:
    """navigate the store wrapper chain to find the underlying ZarrTiffStore"""
    if isinstance(store, ZarrTiffStore):
        return store

    if isinstance(store, _CompositedStore) and {"0"} == set(store._stores):
        inner = store._stores["0"]
        if isinstance(inner, ZarrTiffStore):
            return inner

    raise NotImplementedError(f"store type: {type(store).__name__!r}")


def get_zarr_chunk_sizes(
    grp: zarr.Group,
    *,
    level: int = 0,
    sum_axis: int | None = None,
) -> NDArray[np.int64]:
    """return an array of the raw chunk byte sizes

    EXPERIMENTAL --- do not rely on this

    """
    store = grp.store
    tiff_store = _unwrap_to_zarrtiffstore(store)

    # determine the level prefix and read metadata via zarr v2 .zarray keys
    if tiff_store.is_multiscales:
        levelstr = f"{level}/"
        zarray_key = f"{level}/.zarray"
    else:
        if level != 0:
            raise ValueError(f"no matching level: {level}")
        levelstr = ""
        zarray_key = ".zarray"

    buf = _store_get_sync(tiff_store, zarray_key)
    if buf is None:
        raise ValueError(f"no matching level: {level}")

    meta = json.loads(buf.to_bytes().decode())
    shape = meta["shape"]
    chunks = meta["chunks"]

    assert len(shape) == len(chunks)
    if len(shape) not in (2, 3):
        raise NotImplementedError("chunk dimensions not in (2, 3)")

    chunked = tuple(i // j + (1 if i % j else 0) for i, j in zip(shape, chunks))

    # fixme:
    #  relies on private functionality of ZarrTiffStore, might break at any time
    try:
        # noinspection PyProtectedMember
        parse_key = tiff_store._parse_key
    except AttributeError:
        raise RuntimeError("probably not supported with your tifffile version")

    chunk_sizes: NDArray[np.int64] = np.full(chunked, dtype=np.int64, fill_value=-1)

    for indices in np.ndindex(*chunked):
        chunkindex = ".".join(str(index) for index in indices)
        key = levelstr + chunkindex
        keyframe, page, _, offset, bytecount = parse_key(key)
        if page and offset is None:
            bytecount = keyframe.nbytes
        if offset and bytecount:
            chunk_sizes[indices] = bytecount

    if chunk_sizes.ndim not in (2, 3):
        raise NotImplementedError("chunk dimensions not in (2, 3)")

    if sum_axis is None:
        return chunk_sizes  # type: ignore
    else:
        return chunk_sizes.sum(axis=sum_axis)  # type: ignore


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
    if s2 is Ellipsis:
        z0, z1 = 0, 1
    else:
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
    shape: Size3D, located_arrays: dict[Point3D, zarr.Array]
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
        raise ValueError(
            f"arrays don't share the same dtype and fill_value, got: {dt_fv!r}"
        )
    ((dtype, fill_value),) = dt_fv
    return dtype, fill_value


def get_zarr_depth_and_dtype(grp: zarr.Group, axes: str) -> tuple[int, DTypeLike]:
    """return the image depth from the zarr group"""
    if "tiffslide.series-composition" in grp.attrs:
        srs = next(iter(grp.attrs["series-composition"]["located_series"]))
        key = f"{srs}/0"
    else:
        key = "0"  # -> level

    zarray: zarr.Array = grp[key]

    if axes == "YXS":
        depth = zarray.shape[2]
    elif axes == "CYX":
        depth = zarray.shape[0]
    elif axes == "YX":
        depth = 1
    else:
        raise NotImplementedError(f"axes={axes!r}")

    return depth, zarray.dtype
