from __future__ import annotations

import io
import math
import os.path
import sys
from collections import defaultdict
from collections.abc import Iterator
from collections.abc import Mapping
from fractions import Fraction
from functools import cached_property
from itertools import count
from types import TracebackType
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import TypeVar
from typing import overload
from warnings import warn
from xml.etree import ElementTree

import numpy as np
import tifffile
import zarr
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.reference import ReferenceFileSystem
from PIL import Image
from PIL import ImageCms
from tifffile import TiffFile
from tifffile import TiffFileError as TiffFileError
from tifffile import TiffPage
from tifffile import TiffPageSeries
from tifffile.tifffile import svs_description_metadata

from tiffslide._compat import NotTiffFile
from tiffslide._types import OpenFileLike
from tiffslide._types import PathOrFileOrBufferLike
from tiffslide._types import SeriesCompositionInfo
from tiffslide._types import Slice3D
from tiffslide._types import TiffFileIO
from tiffslide._zarr import get_zarr_depth_and_dtype
from tiffslide._zarr import get_zarr_selection
from tiffslide._zarr import get_zarr_store

if TYPE_CHECKING:
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
    "NotTiffSlide",
]

# all relevant tifffile version numbers work with this.
_TIFFFILE_VERSION = tuple(
    int(x) if x.isdigit() else x for x in tifffile.__version__.split(".")
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
        filename: PathOrFileOrBufferLike,
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
        self._tifffile = _prepare_tifffile(
            filename,
            storage_options=storage_options,
            tifffile_options=tifffile_options,
            _cls=TiffFile,
        )

    @property
    def ts_tifffile(self) -> TiffFile:
        """get the underlying tifffile instance"""
        # backwards compatibility
        if isinstance(self._tifffile, ReferenceFileSystem):
            raise RuntimeError(
                "instance is backed by kerchunk: no ts_tifffile available"
            )
        elif not isinstance(self._tifffile, (TiffFile, NotTiffFile)):
            raise NotImplementedError(
                f"instance backed by {type(self._tifffile).__name__}"
            )
        return self._tifffile

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
        try:
            grp = self.__dict__.pop("zarr_group")
        except KeyError:
            pass
        else:
            try:
                grp.close()
            except AttributeError:
                pass
            del grp
        self.ts_tifffile.close()

    def __repr__(self) -> str:
        fn = _get_filename(self._tifffile)
        r = repr(fn) if fn else "<unknown filename ...>"
        return f"{type(self).__name__}({r})"

    @classmethod
    def detect_format(
        cls,
        filename: PathOrFileOrBufferLike,
        *,
        tifffile_options: dict[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> str | None:
        """return the detected format as a str or None if unknown/unimplemented"""
        try:
            tf = _prepare_tifffile(
                filename,
                tifffile_options=tifffile_options,
                storage_options=storage_options,
                _cls=TiffFile,
            )
        except TiffFileError:
            return None
        with tf as t:
            return _PropertyParser.detect_format(t)

    @cached_property
    def dimensions(self) -> tuple[int, int]:
        """return the width and height of level 0"""
        prop = self.properties
        return (
            prop["tiffslide.level[0].width"],
            prop["tiffslide.level[0].height"],
        )

    @cached_property
    def level_count(self) -> int:
        """return the number of levels"""
        prop = self.properties
        lvl = 1
        while f"tiffslide.level[{lvl}].width" in prop:
            lvl += 1
        return lvl

    @cached_property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        """return the dimensions of levels as a list"""
        prop = self.properties
        lvl_dims = [self.dimensions]
        for lvl in count(1):
            try:
                lvl_dim = (
                    prop[f"tiffslide.level[{lvl}].width"],
                    prop[f"tiffslide.level[{lvl}].height"],
                )
            except KeyError:
                break
            else:
                lvl_dims.append(lvl_dim)
        return tuple(lvl_dims)

    @cached_property
    def level_downsamples(self) -> tuple[float, ...]:
        """return the downsampling factors of levels as a list"""
        w0, h0 = self.dimensions
        return tuple(((w0 / w) + (h0 / h)) / 2.0 for w, h in self.level_dimensions)

    @cached_property
    def properties(self) -> dict[str, Any]:
        """image properties / metadata as a dict"""
        return _PropertyParser(self._tifffile).parse()

    @cached_property
    def associated_images(self) -> _LazyAssociatedImagesDict:
        """return associated images as a mapping of names to PIL images"""
        idx = self.properties["tiffslide.series-index"]
        series = self.ts_tifffile.series[idx + 1 :]
        return _LazyAssociatedImagesDict(series)

    @cached_property
    def color_profile(self) -> ImageCms.ImageCmsProfile | None:
        """return the color profile of the image if present"""
        if self._profile is None:
            return None
        return ImageCms.getOpenProfile(io.BytesIO(self._profile))

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """return the best level for a given downsampling factor"""
        if downsample <= 1.0:
            return 0
        for lvl, ds in enumerate(self.level_downsamples):
            if ds > downsample:
                return lvl - 1
        return self.level_count - 1

    @cached_property
    def zarr_group(self) -> zarr.hierarchy.Group:
        """return the tiff image as a zarr-like group

        NOTE: this is extra functionality and not part of the drop-in behaviour
        """
        try:
            _num_decode = os.environ["TIFFSLIDE_NUM_DECODE_THREADS"]
        except KeyError:
            num_decode_threads = None  # half of num CPU
        else:
            if _num_decode:
                num_decode_threads = int(_num_decode)
            else:
                num_decode_threads = None
        store = get_zarr_store(
            self.properties, self._tifffile, num_decode_threads=num_decode_threads
        )
        return zarr.open_group(store, mode="r")

    @property
    def ts_zarr_grp(self) -> zarr.hierarchy.Group:
        """use .zarr_group instead"""
        # backwards compatibility
        return self.zarr_group

    @overload
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image: ...

    @overload
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        as_array: Literal[False] = ...,
        padding: bool = ...,
    ) -> Image.Image: ...

    @overload
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        as_array: Literal[True] = ...,
        padding: bool = ...,
    ) -> npt.NDArray[np.int_]: ...

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        as_array: bool = False,
        padding: bool = True,
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
        padding :
            if True, will ensure that the size of the returned image is deterministic.
        """
        base_x, base_y = map(int, location)
        _rw, _rh = map(int, size)
        axes = self.properties["tiffslide.series-axes"]

        try:
            if level < 0:
                raise IndexError
            level_w, level_h = self.level_dimensions[level]
        except IndexError:
            if not padding:
                raise IndexError(f"level={level} out of range")
            warn(
                f"level={level} is out-of-bounds, but padding is requested",
                stacklevel=2,
            )

            depth, dtype = get_zarr_depth_and_dtype(self.zarr_group, axes)
            return np.zeros((_rh, _rw, depth), dtype=dtype)

        rx0, ry0 = self._read_region_loc_transform((base_x, base_y), level)
        rx1 = rx0 + _rw
        ry1 = ry0 + _rh

        in_bound = 0 <= rx0 and rx1 <= level_w and 0 <= ry0 and ry1 <= level_h
        requires_padding = padding and not in_bound

        if requires_padding:
            # compute padding
            pad_x0 = _clip(-rx0, 0, _rw)
            pad_x1 = _clip(rx1 - level_w, 0, _rw)
            pad_y0 = _clip(-ry0, 0, _rh)
            pad_y1 = _clip(ry1 - level_h, 0, _rh)

        if not in_bound:
            # crop coord to valid zone
            rx0 = _clip(rx0, 0, level_w)
            rx1 = _clip(rx1, 0, level_w)
            ry0 = _clip(ry0, 0, level_h)
            ry1 = _clip(ry1, 0, level_h)

        selection: Slice3D
        if axes == "YXS":
            selection = slice(ry0, ry1), slice(rx0, rx1), slice(None)
        elif axes == "CYX":
            selection = slice(None), slice(ry0, ry1), slice(rx0, rx1)
        elif axes == "YX":
            selection = slice(ry0, ry1), slice(rx0, rx1), ...
        else:
            raise NotImplementedError(f"axes={axes!r}")

        arr: npt.NDArray[np.int_] = get_zarr_selection(
            self.zarr_group,
            selection=selection,
            level=level,
        )

        if axes == "CYX":
            arr = arr.transpose((1, 2, 0))
        elif axes == "YX":
            arr = arr[..., np.newaxis]

        if requires_padding:
            if arr.shape[0] == 0 or arr.shape[1] == 0:
                warn(
                    f"location={location!r}, level={level}, size={size!r} is out-of-bounds, but padding is requested",
                    stacklevel=2,
                )

            # noinspection PyUnboundLocalVariable
            arr = np.pad(
                arr,
                ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        if as_array:
            return arr
        elif axes == "YX":
            image = Image.fromarray(arr[..., 0])
        else:
            image = Image.fromarray(arr)
        if self._profile is not None:
            image.info["icc_profile"] = self._profile
        return image

    def _read_region_loc_transform(
        self, location: tuple[int, int], level: int
    ) -> tuple[int, int]:
        """return the location at the provided level

        Notes
        -----
        Overwrite in subclasses in case you want to change the default
        interpretation of the `loc` argument in `read_region()`.

        """
        base_x, base_y = location
        level_ds = self.level_downsamples[level]
        rx0 = int(base_x / level_ds)
        ry0 = int(base_y / level_ds)
        return rx0, ry0

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

        idx = self.properties["tiffslide.series-index"]
        level_byte_size = self.ts_tifffile.series[idx].levels[level].size

        if 0 < thumb_byte_size < level_byte_size:
            # read the embedded thumbnail if it uses fewer bytes
            img = self.associated_images["thumbnail"]
        else:
            # read the best suited level
            _level_dimensions = self.level_dimensions[level]
            img = self.read_region((0, 0), level, _level_dimensions)

        # now composite the thumbnail
        thumb = Image.new(
            mode=img.mode,
            size=img.size,
            color=f"#{self.properties[PROPERTY_NAME_BACKGROUND_COLOR] or 'ffffff'}",
        )
        thumb.paste(img, box=None, mask=None)
        try:
            thumb.thumbnail(size, Image.Resampling.LANCZOS)
        except ValueError:
            # see: https://github.com/python-pillow/Pillow/blob/95cff6e959/src/libImaging/Resample.c#L559-L588
            thumb.thumbnail(size, Image.Resampling.NEAREST)
        if self._profile is not None:
            thumb.info["icc_profile"] = self._profile
        return thumb

    @cached_property
    def _profile(self) -> bytes | None:
        """return the color profile of the image if present"""
        parser = _IccParser(self._tifffile)
        return parser.parse()


class NotTiffSlide(TiffSlide):
    # noinspection PyMissingConstructor
    def __init__(
        self,
        filename: PathOrFileOrBufferLike,
        *,
        tifffile_options: dict[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        # tifffile instance, can raise TiffFileError
        self._tifffile = _prepare_tifffile(  # type: ignore[assignment]
            filename,
            storage_options=storage_options,
            tifffile_options=tifffile_options,
            _cls=NotTiffFile,
        )

    @classmethod
    def detect_format(
        cls,
        filename: PathOrFileOrBufferLike,
        *,
        tifffile_options: dict[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> str | None:
        """return the detected format as a str or None if unknown/unimplemented"""
        try:
            tf = _prepare_tifffile(
                filename,
                tifffile_options=tifffile_options,
                storage_options=storage_options,
                _cls=NotTiffFile,
            )
        except ValueError:
            return None
        with tf as t:
            # noinspection PyProtectedMember
            return t.pages[0]._codec


class _LazyAssociatedImagesDict(Mapping[str, Image.Image]):
    """lazily load associated images"""

    def __init__(self, series: list[TiffPageSeries]):
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


TF = TypeVar("TF", TiffFile, NotTiffFile)


def _prepare_tifffile(
    fb: PathOrFileOrBufferLike,
    *,
    tifffile_options: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
    _cls: type[TF] = TiffFile,  # type: ignore[assignment]
) -> TF:
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

    _monkey_patch()

    if isinstance(fb, TiffFileIO):
        # provided an IO stream like instance
        _warn_unused_storage_options(st_kw)

        return _cls(fb, **tf_kw)  # type: ignore[arg-type]

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

        return _cls(fs.open(path), **tf_kw)

    elif isinstance(fb, (str, os.PathLike)):
        # provided a string like url
        urlpath = os.fspath(fb)
        fs, path = url_to_fs(urlpath, **st_kw)
        if isinstance(fs, LocalFileSystem):
            return _cls(path, **tf_kw)
        else:
            # set name for tifffile.FileHandle
            if "name" not in tf_kw:
                tf_kw["name"] = os.path.basename(path)

            return _cls(fs.open(path), **tf_kw)

    else:
        # let's try anyways ...
        _warn_unused_storage_options(st_kw)

        return _cls(fb, **tf_kw)


# --- property / metadata related functionality -------------------------------


class _PropertyParser:
    """parse tiffslide properties for different slide types"""

    vendor_map = dict(
        svs="aperio",
        scn="leica",
        bif="ventana",
        ndpi="hamamatsu",
        philips="philips_tiff",
        # add more when needed
    )

    def __init__(self, tf: TiffFile) -> None:
        self._tf = tf

    @staticmethod
    def new_metadata() -> dict[str, Any]:
        return dict.fromkeys(
            [
                PROPERTY_NAME_COMMENT,
                PROPERTY_NAME_VENDOR,
                PROPERTY_NAME_QUICKHASH1,
                PROPERTY_NAME_BACKGROUND_COLOR,
                PROPERTY_NAME_OBJECTIVE_POWER,
                PROPERTY_NAME_MPP_X,
                PROPERTY_NAME_MPP_Y,
                PROPERTY_NAME_BOUNDS_X,
                PROPERTY_NAME_BOUNDS_Y,
                PROPERTY_NAME_BOUNDS_WIDTH,
                PROPERTY_NAME_BOUNDS_HEIGHT,
            ]
        )

    @classmethod
    def detect_format(cls, tf: TiffFile) -> str:
        for prop, vendor in cls.vendor_map.items():
            if getattr(tf, f"is_{prop}"):
                return vendor
        return "generic-tiff"

    @classmethod
    def collect_level_info(cls, series: TiffPageSeries) -> dict[str, Any]:
        # calculate level info
        md: dict[str, Any] = {}
        if series.ndim not in (2, 3):
            raise NotImplementedError(
                "currently no support for series.ndim not in (2, 3)"
            )

        axes = md["tiffslide.series-axes"] = series.axes

        if axes == "YXS":
            h0, w0, _ = map(int, series.shape)
            level_dimensions = ((lvl.shape[1], lvl.shape[0]) for lvl in series.levels)
        elif axes == "CYX":
            _, h0, w0 = map(int, series.shape)
            level_dimensions = ((lvl.shape[2], lvl.shape[1]) for lvl in series.levels)
        elif axes == "YX":
            h0, w0 = map(int, series.shape)
            level_dimensions = ((lvl.shape[1], lvl.shape[0]) for lvl in series.levels)
        else:
            raise NotImplementedError(f"series with axes={axes!r} not supported yet")

        for lvl, (width, height) in enumerate(level_dimensions):
            downsample = ((w0 / width) + (h0 / height)) / 2.0
            page: TiffPage = series.levels[lvl][0]  # type: ignore[assignment]
            md[f"tiffslide.level[{lvl}].downsample"] = downsample
            md[f"tiffslide.level[{lvl}].height"] = int(height)
            md[f"tiffslide.level[{lvl}].width"] = int(width)
            md[f"tiffslide.level[{lvl}].tile-height"] = page.tilelength
            md[f"tiffslide.level[{lvl}].tile-width"] = page.tilewidth
        return md

    @classmethod
    def recover_mpp(cls, series: TiffPageSeries) -> dict[str, Any]:
        """recover mpp from tiff tags"""
        page0: TiffPage = series[0]  # type: ignore[assignment]
        md: dict[str, Any] = {}

        try:
            resolution_unit = page0.tags["ResolutionUnit"].value
            x_resolution = Fraction(*page0.tags["XResolution"].value)
            y_resolution = Fraction(*page0.tags["YResolution"].value)
        except KeyError:
            pass
        else:
            md["tiff.ResolutionUnit"] = resolution_unit.name
            md["tiff.XResolution"] = float(x_resolution)
            md["tiff.YResolution"] = float(y_resolution)

            RESUNIT = tifffile.RESUNIT
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

    def parse(self) -> dict[str, Any]:
        fmt = self.detect_format(self._tf)
        fmt = fmt.replace("-", "_")  # generic-tiff
        return getattr(self, f"parse_{fmt}")()  # type: ignore

    def parse_aperio(self) -> dict[str, Any]:
        """parse Aperio SVS properties"""
        md = self.new_metadata()

        # parse metadata from description
        page: TiffPage = self._tf.pages[0]  # type: ignore[assignment]
        desc = _check_page_description_encoding(page)
        md.update(_parse_metadata_aperio(desc))
        md["tiff.ImageDescription"] = desc

        # get series 0
        series0 = self._tf.series[0]
        md["tiffslide.series-index"] = 0  # svs standard

        # in case mpp wasn't available recover from tags
        if not _has_mpp(md):
            md.update(self.recover_mpp(series0))

        # collect level info
        md.update(self.collect_level_info(series0))

        return md

    def parse_leica(self) -> dict[str, Any]:
        """parse leica SCN properties"""
        md = self.new_metadata()

        # parse metadata from scn xml
        desc = self._tf.scn_metadata
        md["tiff.ImageDescription"] = desc

        # get all leica info
        md.update(_parse_metadata_leica(desc or ""))

        # fill tile-width / tile-height
        idx = md["tiffslide.series-index"]
        for lvl, page in enumerate(self._tf.series[idx]):
            md[f"tiffslide.level[{lvl}].tile-width"] = page.tilewidth
            md[f"tiffslide.level[{lvl}].tile-height"] = page.tilelength

        return md

    def parse_ventana(self) -> dict[str, Any]:
        warn(
            "no special ventana-format metadata parsing implemented yet!",
            stacklevel=2,
        )
        return self.parse_generic_tiff()

    def parse_hamamatsu(self) -> dict[str, Any]:
        warn(
            "hamamatsu-format metadata parsing only partially implemented!",
            stacklevel=2,
        )
        md = self.parse_generic_tiff()

        # collect hamamatsu tags
        tags = self._tf.series[0][0].tags  # type: ignore[union-attr]
        tag_map = {
            "65421": "hamamatsu.SourceLens",
            "65422": "hamamatsu.XOffsetFromSlideCentre",
            "65423": "hamamatsu.YOffsetFromSlideCentre",
            "Model": "hamamatsu.Model",
        }
        for tf_t, ts_t in tag_map.items():
            tag = tags.get(tf_t)
            if tag:
                md[ts_t] = tag.value

        md[PROPERTY_NAME_VENDOR] = "hamamatsu"
        if "hamamatsu.SourceLens" in md:
            md[PROPERTY_NAME_OBJECTIVE_POWER] = md["hamamatsu.SourceLens"]

        return md

    def parse_generic_tiff(self) -> dict[str, Any]:
        # todo: need to handle more supported formats in the future
        md = self.new_metadata()

        # store the description
        page: TiffPage = self._tf.pages[0]  # type: ignore[assignment]
        desc = _check_page_description_encoding(page)
        md["tiff.ImageDescription"] = desc

        md["tiffslide.series-index"] = 0  # use series 0
        series0 = self._tf.series[0]

        # in case mpp wasn't available recover from tags
        if not _has_mpp(md):
            md.update(self.recover_mpp(series0))

        # collect level info
        md.update(self.collect_level_info(series0))
        return md

    def parse_philips_tiff(self) -> dict[str, Any]:
        """parse Philips tiff properties"""
        md = self.parse_generic_tiff()
        if self._tf.philips_metadata is None:
            return md
        philips_metadata = ElementTree.fromstring(self._tf.philips_metadata)

        def get_first_attribute_with_name(
            root: ElementTree.Element, name: str
        ) -> str | None:
            return next(root.iterfind(f".//Attribute[@Name='{name}']")).text

        md[PROPERTY_NAME_VENDOR] = get_first_attribute_with_name(
            philips_metadata, "DICOM_MANUFACTURER"
        )
        pixel_spacing_attribute = get_first_attribute_with_name(
            philips_metadata, "DICOM_PIXEL_SPACING"
        )
        if pixel_spacing_attribute is not None:
            pixel_spacings = [
                float(element.strip('"')) * 1000
                for element in pixel_spacing_attribute.split(" ")
            ]
            mpp_y, mpp_x = pixel_spacings[0], pixel_spacings[1]
            md[PROPERTY_NAME_MPP_X] = mpp_x
            md[PROPERTY_NAME_MPP_Y] = mpp_y
        return md


def _parse_metadata_aperio(desc: str) -> dict[str, Any]:
    """Aperio SVS metadata"""
    aperio_meta = svs_description_metadata(desc)
    assert "Header" in aperio_meta

    md = {
        PROPERTY_NAME_COMMENT: desc,
        PROPERTY_NAME_VENDOR: "aperio",
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
    return md


def _parse_metadata_leica(image_description: str) -> dict[str, Any]:
    """return the leica SCN properties"""
    # todo: clean up. this is pretty convoluted
    md: dict[str, Any] = {PROPERTY_NAME_COMMENT: image_description}

    dct = _xml_to_dict(image_description)
    collection = dct["scn"]["collection"]

    slide_x_nm = int(collection["@sizeX"])
    slide_y_nm = int(collection["@sizeY"])

    first_non_macro_idx: int | None = None
    lvl_resolutions = defaultdict(list)
    series_offsets_nm = {}
    levels_start_x_nm: int | None = None
    levels_start_y_nm: int | None = None
    levels_end_x_nm: int | None = None
    levels_end_y_nm: int | None = None

    for idx, image in enumerate(collection["image"]):
        image_x_nm = int(image["view"]["@sizeX"])
        image_y_nm = int(image["view"]["@sizeY"])
        offset_x_nm = int(image["view"]["@offsetX"])
        offset_y_nm = int(image["view"]["@offsetY"])

        is_macro_image = (
            offset_x_nm == 0
            and offset_y_nm == 0
            and image_x_nm == slide_x_nm
            and image_y_nm == slide_y_nm
        )
        if is_macro_image:
            continue
        if levels_start_x_nm is None or levels_start_x_nm > offset_x_nm:
            levels_start_x_nm = offset_x_nm
        if levels_start_y_nm is None or levels_start_y_nm > offset_y_nm:
            levels_start_y_nm = offset_y_nm
        level_end_x_nm = offset_x_nm + image_x_nm
        level_end_y_nm = offset_y_nm + image_y_nm
        if levels_end_x_nm is None or levels_end_x_nm < level_end_x_nm:
            levels_end_x_nm = level_end_x_nm
        if levels_end_y_nm is None or levels_end_y_nm < level_end_y_nm:
            levels_end_y_nm = level_end_y_nm

        if first_non_macro_idx is None:
            first_non_macro_idx = idx

            _scan = image["scanSettings"]
            obj_pow = float(_scan["objectiveSettings"]["objective"])
            aperture = float(_scan["illuminationSettings"]["numericalAperture"])
            isource = str(_scan["illuminationSettings"]["illuminationSource"])

            md.update(
                {
                    PROPERTY_NAME_VENDOR: "leica",
                    PROPERTY_NAME_OBJECTIVE_POWER: obj_pow,
                    "leica.aperture": aperture,
                    "leica.creation-date": str(image["creationDate"]),
                    "leica.device-model": str(image["device"]["@model"]),
                    "leica.device-version": str(image["device"]["@version"]),
                    "leica.illumination-source": isource,
                }
            )

        for lvl, info in enumerate(image["pixels"]["dimension"]):
            resolution = image_x_nm / int(info["@sizeX"])
            # image_y_nm / int(info["@sizeY"])  <-- openslide just uses X
            lvl_resolutions[lvl].append(resolution)

        series_offsets_nm[idx] = offset_y_nm, offset_x_nm

    if not lvl_resolutions:
        raise ValueError("no non-macro images in file")

    assert levels_start_x_nm is not None
    assert levels_start_y_nm is not None
    assert levels_end_x_nm is not None
    assert levels_end_y_nm is not None

    resolutions0 = np.array(lvl_resolutions[0])
    # allow some threshold
    _r_avg = resolutions0.mean()
    if np.any(np.abs((resolutions0 - _r_avg) / _r_avg) > 0.02):
        raise ValueError(
            f"non-macro images vary too much in resolution: {lvl_resolutions[0]!r}"
        )

    nm_per_px = min(lvl_resolutions[0])
    mpp = nm_per_px / 1000.0
    md[PROPERTY_NAME_MPP_X] = mpp
    md[PROPERTY_NAME_MPP_Y] = mpp
    md[PROPERTY_NAME_BOUNDS_X] = int(levels_start_x_nm / nm_per_px)
    md[PROPERTY_NAME_BOUNDS_Y] = int(levels_start_y_nm / nm_per_px)
    md[PROPERTY_NAME_BOUNDS_WIDTH] = int(
        (levels_end_x_nm - levels_start_x_nm) / nm_per_px
    )
    md[PROPERTY_NAME_BOUNDS_HEIGHT] = int(
        (levels_end_y_nm - levels_start_y_nm) / nm_per_px
    )
    slide_x_px = math.ceil(slide_x_nm / nm_per_px)
    slide_y_px = math.ceil(slide_y_nm / nm_per_px)

    level_shapes = []
    located_series = defaultdict(list)
    for lvl, resolutions in sorted(lvl_resolutions.items()):
        lvl_nm_per_px = min(resolutions)

        for srs, offset_nm in series_offsets_nm.items():
            # implicitly assuming axes="YXS" ... (might be wrong?)
            offset_px = (
                int(offset_nm[0] / lvl_nm_per_px),
                int(offset_nm[1] / lvl_nm_per_px),
                0,
            )
            located_series[srs].append(offset_px)
        assert len(set(map(len, located_series.values()))) == 1

        lvl_size_x = math.ceil(slide_x_nm / lvl_nm_per_px)
        lvl_size_y = math.ceil(slide_y_nm / lvl_nm_per_px)
        md[f"tiffslide.level[{lvl}].height"] = lvl_size_y
        md[f"tiffslide.level[{lvl}].width"] = lvl_size_x
        md[f"tiffslide.level[{lvl}].downsample"] = (
            (slide_x_px / lvl_size_x) + (slide_y_px / lvl_size_y)
        ) / 2.0
        level_shapes.append((lvl_size_y, lvl_size_x, 3))

    md["tiffslide.series-index"] = first_non_macro_idx
    md["tiffslide.series-axes"] = "YXS"  # todo: verify
    md["tiffslide.series-composition"] = SeriesCompositionInfo(
        level_shapes=level_shapes,
        located_series=located_series,
    )

    return md


class _IccParser:
    """parse ICC profile from tiff tags"""

    def __init__(self, tf: TiffFile) -> None:
        self._tf = tf

    def parse(self) -> bytes | None:
        """return the ICC profile if present"""
        page = self._tf.pages[0]
        if isinstance(page, TiffPage) and "InterColorProfile" in page.tags:
            icc_profile = page.tags["InterColorProfile"].value
            if isinstance(icc_profile, bytes):
                return icc_profile
        return None


# --- helper functions --------------------------------------------------------


def _xml_to_dict(xml: str) -> dict[str, Any]:
    """helper function to convert xml string to a dictionary"""
    x = ElementTree.fromstring(xml)  # nosec B314

    def _to_dict(e):  # type: ignore
        tag = e.tag[e.tag.find("}") + 1 :]
        d = {f"@{k}": v for k, v in e.attrib.items()}
        for c in e:
            key, val = _to_dict(c).popitem()
            if key not in d:
                d[key] = val
            elif not isinstance(d[key], list):
                d[key] = [d[key], val]
            else:
                d[key].append(val)
        if e.text and e.text.strip():
            if d:
                d["#text"] = e.text
            else:
                d = e.text
        return {tag: d}

    return _to_dict(x)  # type: ignore


def _has_mpp(md: dict[str, Any]) -> bool:
    """check if metadata has MPP defined"""
    return md[PROPERTY_NAME_MPP_X] is not None and md[PROPERTY_NAME_MPP_Y] is not None


def _clip(x: int, min_: int, max_: int) -> int:
    """clip a value to a range"""
    return min(max(x, min_), max_)


def _get_filename(obj: Any) -> str:
    """return a filename or a placeholder"""
    if isinstance(obj, ReferenceFileSystem):
        for name in obj.templates.values():
            return name or ""
        else:
            return ""
    else:
        try:
            return obj.filename or ""
        except AttributeError:
            return ""


def _check_page_description_encoding(page: TiffPage) -> str:
    """try to return the description tag of a tiffpage"""
    value = page.description
    if isinstance(value, str) and value:
        return value

    # value was empty, try to recover
    value = page.tags.valueof(270, default="")

    if isinstance(value, str):
        return value

    elif isinstance(value, bytes) and value:
        raise TiffFileError(
            "ImageDescription tag has incompatible encoding.\n"
            "# Try setting the environment variable TIFFSLIDE_MONKEY_PATCH_NON_ASCII_DESCRIPTIONS=1\n"
            "# Or better fix your original file by running:\n"
            "$ python -m tiffslide.repair description-tag-encoding <WSI_FILENAME>",
        )

    else:
        raise ValueError(f"unsupported description type: {type(value)!r}")


# === monkey patching =================================================


def _env2bool(value: str) -> bool:
    """convert an environment variable to bool"""
    if not isinstance(value, str):
        raise TypeError(f"value must be of type str, got: {type(value).__name__}")
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    elif value in {"no", "false", "f", "n", "0", ""}:
        return False
    else:
        raise ValueError(f"could not parse value: {value!r}")


def _monkey_patch() -> None:
    """applying monkey tiffslide patches if requested"""
    if _env2bool(os.getenv("TIFFSLIDE_MONKEY_PATCH_NON_ASCII_DESCRIPTIONS", "")):
        from tiffslide.repair import monkey_patch_description_tag_encoding

        warn(
            "TIFFSLIDE_MONKEY_PATCH_NON_ASCII_DESCRIPTIONS is active!",
            RuntimeWarning,
            stacklevel=4,
        )
        monkey_patch_description_tag_encoding()
