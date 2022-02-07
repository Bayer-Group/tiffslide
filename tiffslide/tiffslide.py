from __future__ import annotations

import math
import os.path
import re
import sys
from fractions import Fraction
from types import TracebackType
from typing import TYPE_CHECKING
from typing import Any
from typing import AnyStr
from typing import Iterator
from typing import Mapping
from typing import overload
from warnings import warn

if sys.version_info[:2] >= (3, 8):
    from functools import cached_property
    from importlib.metadata import version
    from typing import Literal
else:
    from backports.cached_property import cached_property
    from importlib_metadata import version
    from typing_extensions import Literal

import tifffile
import zarr
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from PIL import Image
from tifffile import TiffFile
from tifffile import TiffFileError as TiffFileError
from tifffile import TiffPageSeries
from tifffile.tifffile import svs_description_metadata

from tiffslide._types import OpenFileLike
from tiffslide._types import PathOrFileOrBufferLike
from tiffslide._types import TiffFileIO

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


__all__ = [
    "PROPERTY_NAME_COMMENT",
    "PROPERTY_NAME_VENDOR",
    "PROPERTY_NAME_QUICKHASH1",
    "PROPERTY_NAME_BACKGROUND_COLOR",
    "PROPERTY_NAME_OBJECTIVE_POWER",
    "PROPERTY_NAME_MPP_X",
    "PROPERTY_NAME_MPP_Y",
    "PROPERTY_NAME_BOUNDS_X",
    "PROPERTY_NAME_BOUNDS_Y",
    "PROPERTY_NAME_BOUNDS_WIDTH",
    "PROPERTY_NAME_BOUNDS_HEIGHT",
    "TiffSlide",
    "TiffFileError",
]

# all relevant tifffile version numbers work with this.
_TIFFFILE_VERSION = tuple(
    int(x) if x.isdigit() else x for x in version("tifffile").split(".")
)

# === Constants to support drop-in ===
PROPERTY_NAME_COMMENT = "tiffslide.comment"
PROPERTY_NAME_VENDOR = "tiffslide.vendor"
PROPERTY_NAME_QUICKHASH1 = "tiffslide.quickhash-1"
PROPERTY_NAME_BACKGROUND_COLOR = "tiffslide.background-color"
PROPERTY_NAME_OBJECTIVE_POWER = "tiffslide.objective-power"
PROPERTY_NAME_MPP_X = "tiffslide.mpp-x"
PROPERTY_NAME_MPP_Y = "tiffslide.mpp-y"
PROPERTY_NAME_BOUNDS_X = "tiffslide.bounds-x"
PROPERTY_NAME_BOUNDS_Y = "tiffslide.bounds-y"
PROPERTY_NAME_BOUNDS_WIDTH = "tiffslide.bounds-width"
PROPERTY_NAME_BOUNDS_HEIGHT = "tiffslide.bounds-height"


class TiffSlide:
    """
    tifffile backed whole slide image container emulating openslide.OpenSlide
    """

    def __init__(
        self,
        filename: PathOrFileOrBufferLike[AnyStr],
        *,
        tifffile_options: dict[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        """TiffSlide constructor

        Parameters
        ----------
        filename:
            a local filename or a fsspec urlpath or a file object
        tifffile_options:
            a dictionary with keyword arguments passed to the TiffFile constructor
        storage_options:
            a dictionary with keyword arguments passed to fsspec
        """
        # tifffile instance, can raise TiffFileError
        self.ts_tifffile = _prepare_tifffile(
            filename, storage_options=storage_options, tifffile_options=tifffile_options
        )

    def __enter__(self) -> TiffSlide:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        ts_zarr_grp = self.__dict__.pop("ts_zarr_grp", None)
        if ts_zarr_grp is not None:
            try:
                self.ts_zarr_grp.close()
            except AttributeError:
                pass  # Arrays don't need to be closed
        self.ts_tifffile.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.ts_tifffile.filename!r})"

    @classmethod
    def detect_format(
        cls,
        filename: PathOrFileOrBufferLike[AnyStr],
        *,
        tifffile_options: dict[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> str | None:
        """return the detected format as a str or None if unknown/unimplemented"""
        _vendor_compat_map = dict(
            svs="aperio",
            # add more when needed
        )
        tf = _prepare_tifffile(
            filename, tifffile_options=tifffile_options, storage_options=storage_options
        )
        with tf as t:
            for prop, vendor in _vendor_compat_map.items():
                if getattr(t, f"is_{prop}"):
                    return vendor
        return None

    @cached_property
    def dimensions(self) -> tuple[int, int]:
        """return the width and height of level 0"""
        series0 = self.ts_tifffile.series[0]
        assert series0.ndim == 3, "loosen restrictions in future versions"
        h, w, _ = series0.shape
        return w, h

    @cached_property
    def level_count(self) -> int:
        """return the number of levels"""
        return len(self.ts_tifffile.series[0].levels)

    @cached_property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        """return the dimensions of levels as a list"""
        return tuple(lvl.shape[1::-1] for lvl in self.ts_tifffile.series[0].levels)

    @cached_property
    def level_downsamples(self) -> tuple[float, ...]:
        """return the downsampling factors of levels as a list"""
        w0, h0 = self.dimensions
        return tuple(math.sqrt((w0 * h0) / (w * h)) for w, h in self.level_dimensions)

    @cached_property
    def properties(self) -> dict[str, Any]:
        """image properties / metadata as a dict"""
        aperio_desc = self.ts_tifffile.pages[0].description

        if _TIFFFILE_VERSION >= (2021, 6, 14):
            # tifffile 2021.6.14 fixed the svs parsing.
            _aperio_desc = aperio_desc
            _aperio_recovered_header = None  # no need to recover

        else:
            # this emulates the new description parsing for older versions
            _aperio_desc = re.sub(r";Aperio [^;|]*(?=[|])", "", aperio_desc, 1)
            _aperio_recovered_header = aperio_desc.split("|", 1)[0]

        try:
            aperio_meta = svs_description_metadata(_aperio_desc)
        except ValueError as err:
            if "invalid Aperio image description" in str(err):
                warn(f"{err} - {self!r}")
                aperio_meta = {}
            else:
                raise
            vendor = "generic-tiff"  # todo: need to handle more supported formats in the future
        else:
            # Normalize the aperio metadata
            aperio_meta.pop("", None)
            aperio_meta.pop("Aperio Image Library", None)
            if aperio_meta and "Header" not in aperio_meta:
                aperio_meta["Header"] = _aperio_recovered_header
            vendor = "aperio"

        md = {
            PROPERTY_NAME_COMMENT: aperio_desc,
            PROPERTY_NAME_VENDOR: vendor,
            PROPERTY_NAME_QUICKHASH1: None,
            PROPERTY_NAME_BACKGROUND_COLOR: None,
            PROPERTY_NAME_OBJECTIVE_POWER: aperio_meta.get("AppMag", None),
            PROPERTY_NAME_MPP_X: aperio_meta.get("MPP", None),
            PROPERTY_NAME_MPP_Y: aperio_meta.get("MPP", None),
            PROPERTY_NAME_BOUNDS_X: None,
            PROPERTY_NAME_BOUNDS_Y: None,
            PROPERTY_NAME_BOUNDS_WIDTH: None,
            PROPERTY_NAME_BOUNDS_HEIGHT: None,
        }
        md.update({f"aperio.{k}": v for k, v in sorted(aperio_meta.items())})

        _ds_dimensions = zip(self.level_downsamples, self.level_dimensions)
        for lvl, (ds, (width, height)) in enumerate(_ds_dimensions):
            page = self.ts_tifffile.series[0].levels[lvl].pages[0]
            md[f"tiffslide.level[{lvl}].downsample"] = ds
            md[f"tiffslide.level[{lvl}].height"] = height
            md[f"tiffslide.level[{lvl}].width"] = width
            md[f"tiffslide.level[{lvl}].tile-height"] = page.tilelength
            md[f"tiffslide.level[{lvl}].tile-width"] = page.tilewidth

        md["tiff.ImageDescription"] = aperio_desc

        if md[PROPERTY_NAME_MPP_X] is None or md[PROPERTY_NAME_MPP_Y] is None:
            # recover mpp from tiff tags
            try:
                resolution_unit = (
                    self.ts_tifffile.pages[0].tags["ResolutionUnit"].value
                )
                x_resolution = Fraction(
                    *self.ts_tifffile.pages[0].tags["XResolution"].value
                )
                y_resolution = Fraction(
                    *self.ts_tifffile.pages[0].tags["YResolution"].value
                )
            except KeyError:
                pass
            else:
                md["tiff.ResolutionUnit"] = resolution_unit.name
                md["tiff.XResolution"] = float(x_resolution)
                md["tiff.YResolution"] = float(y_resolution)

                RESUNIT = tifffile.TIFF.RESUNIT
                scale = {
                    RESUNIT.INCH: 25400.0,
                    RESUNIT.CENTIMETER: 10000.0,
                    RESUNIT.MILLIMETER: 1000.0,
                    RESUNIT.MICROMETER: 1.0,
                    RESUNIT.NONE: None,
                }.get(resolution_unit, None)
                if scale is not None:
                    try:
                        mpp_x = scale / x_resolution
                        mpp_y = scale / y_resolution
                    except ArithmeticError:
                        pass
                    else:
                        md[PROPERTY_NAME_MPP_X] = mpp_x
                        md[PROPERTY_NAME_MPP_Y] = mpp_y
        return md

    @cached_property
    def associated_images(self) -> _LazyAssociatedImagesDict:
        """return associated images as a mapping of names to PIL images"""
        return _LazyAssociatedImagesDict(self.ts_tifffile)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """return the best level for a given downsampling factor"""
        if downsample <= 1.0:
            return 0
        for lvl, ds in enumerate(self.level_downsamples):
            if ds >= downsample:
                return lvl - 1
        return self.level_count - 1

    @cached_property
    def ts_zarr_grp(self) -> zarr.core.Array | zarr.hierarchy.Group:
        """return the tiff image as a zarr array or group

        NOTE: this is extra functionality and not part of the drop-in behaviour
        """
        store = self.ts_tifffile.series[0].aszarr()
        return zarr.open(store, mode="r")

    def _read_region_as_array(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> npt.NDArray[np.int_]:
        """return the requested region as a numpy array

        Parameters
        ----------
        location :
            pixel location (x, y) in level 0 of the image
        level :
            target level used to read the image
        size :
            size (width, height) of the requested region
        """
        warn(
            "use: Tiffslide.read_region(loc, lvl, size, as_array=True)",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.read_region(location, level, size, as_array=True)

    @overload
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        ...

    @overload
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        as_array: Literal[False] = ...,
    ) -> Image.Image:
        ...

    @overload
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        as_array: Literal[True] = ...,
    ) -> npt.NDArray[np.int_]:
        ...

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        as_array: bool = False,
    ) -> Image.Image | npt.NDArray[np.int_]:
        """return the requested region as a PIL.Image

        Parameters
        ----------
        location :
            pixel location (x, y) in level 0 of the image
        level :
            target level used to read the image
        size :
            size (width, height) of the requested region
        as_array :
            if True, return the region as numpy array
        """
        base_x, base_y = location
        base_w, base_h = self.dimensions
        level_w, level_h = self.level_dimensions[level]
        rx0 = (base_x * level_w) // base_w
        ry0 = (base_y * level_h) // base_h
        _rw, _rh = size
        rx1 = rx0 + _rw
        ry1 = ry0 + _rh
        arr: npt.NDArray[np.int_]
        if isinstance(self.ts_zarr_grp, zarr.core.Array):
            arr = self.ts_zarr_grp[ry0:ry1, rx0:rx1]
        else:
            arr = self.ts_zarr_grp[str(level)][ry0:ry1, rx0:rx1]

        if as_array:
            return arr
        else:
            return Image.fromarray(arr)

    def get_thumbnail(
        self, size: tuple[int, int], *, use_embedded: bool = False
    ) -> Image.Image:
        """return the thumbnail of the slide as a PIL.Image with a maximum size

        Parameters
        ----------
        size:
            width,height tuple defining maximum size of the thumbnail in each direction.
            the thumbnail itself keeps the image aspect ratio
        use_embedded:
            if True uses the embedded thumbnail in the image (if available and smaller
            than the highest level) to generate the thumbnail image

        """
        if (
            use_embedded
            and "thumbnail" in self.associated_images
            and size <= self.associated_images.series_map["thumbnail"].shape[1::-1]
        ):
            thumb_byte_size = self.associated_images.series_map["thumbnail"].size
        else:
            thumb_byte_size = -1

        slide_w, slide_h = self.dimensions
        thumb_w, thumb_h = size
        downsample = max(slide_w / thumb_w, slide_h / thumb_h)
        level = self.get_best_level_for_downsample(downsample)

        level_byte_size = self.ts_tifffile.series[0].levels[level].size

        if 0 < thumb_byte_size < level_byte_size:
            # read the embedded thumbnail if it uses fewer bytes
            img = self.associated_images["thumbnail"]
        else:
            # read the best suited level
            _level_dimensions = self.level_dimensions[level]
            img = self.read_region((0, 0), level, _level_dimensions)

        # now composite the thumbnail
        thumb = Image.new(
            mode="RGB",
            size=img.size,
            color=f"#{self.properties[PROPERTY_NAME_BACKGROUND_COLOR] or 'ffffff'}",
        )
        thumb.paste(img, box=None, mask=None)
        thumb.thumbnail(size, Image.ANTIALIAS)
        return thumb


class _LazyAssociatedImagesDict(Mapping[str, Image.Image]):
    """lazily load associated images"""

    def __init__(self, tifffile: TiffFile):
        series = tifffile.series[1:]
        self.series_map: dict[str, TiffPageSeries] = {s.name.lower(): s for s in series}
        self._m: dict[str, Image.Image] = {}

    def __repr__(self) -> str:
        args = ", ".join(
            f"{name!r}: <lazy-loaded PIL.Image.Image size={s.shape[1]}x{s.shape[0]} ...>"
            for name, s in self.series_map.items()
        )
        # pretend to be a normal dictionary
        return f"{{{args}}}"

    def __getitem__(self, k: str) -> Image.Image:
        if k in self._m:
            return self._m[k]
        else:
            s = self.series_map[k]
            self._m[k] = img = Image.fromarray(s.asarray())
            return img

    def __len__(self) -> int:
        return len(self.series_map)

    def __iter__(self) -> Iterator[str]:
        yield from self.series_map


def _prepare_tifffile(
    fb: PathOrFileOrBufferLike[AnyStr],
    *,
    tifffile_options: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> TiffFile:
    """prepare a TiffFile instance

    Allows providing fsspec urlpaths as well as fsspec OpenFile instances directly.

    Parameters
    ----------
    fb:
        an urlpath like string, a fsspec OpenFile like instance or a buffer like instance
    tifffile_options:
        keyword arguments passed to tifffile.TiffFile
    storage_options:
        keyword arguments passed to fsspec AbstractFileSystem.open()
    """
    tf_kw: dict[str, Any] = tifffile_options or {}
    st_kw: dict[str, Any] = storage_options or {}

    def _warn_unused_storage_options(kw: Any) -> None:
        if kw:
            warn(
                "storage_options ignored when providing file or buffer like object",
                stacklevel=3,
            )

    if isinstance(fb, TiffFileIO):
        # provided an IO stream like instance
        _warn_unused_storage_options(st_kw)

        return TiffFile(fb, **tf_kw)

    elif isinstance(fb, OpenFileLike):
        # provided a fsspec compatible OpenFile instance
        _warn_unused_storage_options(st_kw)

        fs, path = fb.fs, fb.path

        # set name for tifffile.FileHandle
        if "name" not in tf_kw:
            if hasattr(fb, "full_name"):
                name = os.path.basename(fb.full_name)  # type: ignore
            else:
                name = os.path.basename(path)
            tf_kw["name"] = name

        return TiffFile(fs.open(path), **tf_kw)

    elif isinstance(fb, (str, os.PathLike)):
        # provided a string like url
        urlpath = os.fspath(fb)
        fs, path = url_to_fs(urlpath, **st_kw)
        if isinstance(fs, LocalFileSystem):
            return TiffFile(path, **tf_kw)
        else:
            # set name for tifffile.FileHandle
            if "name" not in tf_kw:
                tf_kw["name"] = os.path.basename(path)

            return TiffFile(fs.open(path), **tf_kw)

    else:
        # let's try anyways ...
        _warn_unused_storage_options(st_kw)

        return TiffFile(fb, **tf_kw)
