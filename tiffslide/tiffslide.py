from __future__ import annotations

import math
import os.path
import re
import sys
from fractions import Fraction
from itertools import count
from types import TracebackType
from typing import TYPE_CHECKING
from typing import Any
from typing import AnyStr
from typing import Iterator
from typing import Mapping
from typing import TypeVar
from typing import overload
from warnings import warn
from xml.etree import ElementTree

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
import numpy as np
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from PIL import Image
from tifffile import TiffFile
from tifffile import TiffFileError as TiffFileError
from tifffile import TiffPageSeries
from tifffile.tifffile import svs_description_metadata

from tiffslide._compat import NotTiffFile
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
    "NotTiffSlide",
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
            filename,
            storage_options=storage_options,
            tifffile_options=tifffile_options,
            _cls=TiffFile,
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
            return _detect_format(t)

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
        return tuple(math.sqrt((w0 * h0) / (w * h)) for w, h in self.level_dimensions)

    @cached_property
    def properties(self) -> dict[str, Any]:
        """image properties / metadata as a dict"""
        md: dict[str, Any] = {
            PROPERTY_NAME_COMMENT: None,
            PROPERTY_NAME_VENDOR: None,
            PROPERTY_NAME_QUICKHASH1: None,
            PROPERTY_NAME_BACKGROUND_COLOR: None,
            PROPERTY_NAME_OBJECTIVE_POWER: None,
            PROPERTY_NAME_MPP_X: None,
            PROPERTY_NAME_MPP_Y: None,
            PROPERTY_NAME_BOUNDS_X: None,
            PROPERTY_NAME_BOUNDS_Y: None,
            PROPERTY_NAME_BOUNDS_WIDTH: None,
            PROPERTY_NAME_BOUNDS_HEIGHT: None,
        }
        tf = self.ts_tifffile

        if tf.is_svs:
            desc = tf.pages[0].description
            series_idx = 0
            _md = _parse_metadata_aperio(desc)

        elif tf.is_scn:
            desc = tf.scn_metadata
            series_idx = _auto_select_series_scn(desc)
            _md = _parse_metadata_scn(desc, series_idx)

        else:
            # todo: need to handle more supported formats in the future
            if tf.is_bif or tf.is_ndpi:
                vendor = _detect_format(tf)
                warn(f"no special {vendor!r}-format metadata parsing implemented yet!")
            desc = tf.pages[0].description
            series_idx = 0
            _md = {
                PROPERTY_NAME_COMMENT: desc,
                PROPERTY_NAME_VENDOR: "generic-tiff",
            }

        md.update(_md)
        md["tiff.ImageDescription"] = desc
        md["tiffslide.series-index"] = series_idx

        # calculate level info
        series0 = tf.series[series_idx]
        assert series0.ndim == 3, "loosen restrictions in future versions"
        axes0 = md["tiffslide.series-axes"] = series0.axes

        if axes0 == "YXS":
            h0, w0, _ = map(int, series0.shape)
            level_dimensions = ((lvl.shape[1], lvl.shape[0]) for lvl in series0.levels)
        elif axes0 == "CYX":
            _, h0, w0 = map(int, series0.shape)
            level_dimensions = ((lvl.shape[2], lvl.shape[1]) for lvl in series0.levels)
        else:
            raise NotImplementedError(f"series with axes={axes0!r} not supported yet")

        for lvl, (width, height) in enumerate(level_dimensions):
            downsample = math.sqrt((w0 * h0) / (width * height))
            page = series0.levels[lvl][0]
            md[f"tiffslide.level[{lvl}].downsample"] = downsample
            md[f"tiffslide.level[{lvl}].height"] = int(height)
            md[f"tiffslide.level[{lvl}].width"] = int(width)
            md[f"tiffslide.level[{lvl}].tile-height"] = page.tilelength
            md[f"tiffslide.level[{lvl}].tile-width"] = page.tilewidth

        if md[PROPERTY_NAME_MPP_X] is None or md[PROPERTY_NAME_MPP_Y] is None:
            # recover mpp from tiff tags
            page0 = series0[0]
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
        idx = self.properties["tiffslide.series-index"]
        series = self.ts_tifffile.series[idx + 1 :]
        return _LazyAssociatedImagesDict(series)

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
        idx = self.properties["tiffslide.series-index"]
        store = self.ts_tifffile.series[idx].aszarr()
        return zarr.open(store, mode="r")

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
        padding: bool = ...,
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
        padding: bool = ...,
    ) -> npt.NDArray[np.int_]:
        ...

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
        base_w, base_h = self.dimensions
        _rw, _rh = map(int, size)
        axes = self.properties["tiffslide.series-axes"]

        try:
            if level < 0:
                raise IndexError
            level_w, level_h = self.level_dimensions[level]
        except IndexError:
            if not padding:
                raise
            warn(
                f"level={level} is out-of-bounds, but padding is requested",
                stacklevel=2,
            )

            zarray: zarr.core.Array
            if isinstance(self.ts_zarr_grp, zarr.core.Array):
                zarray = self.ts_zarr_grp
            else:
                zarray = self.ts_zarr_grp["0"]

            if axes == "YXS":
                depth = zarray.shape[2]
            elif axes == "CYX":
                depth = zarray.shape[0]
            else:
                raise NotImplementedError(f"axes={axes!r}")

            return np.zeros((_rh, _rw, depth), dtype=zarray.dtype)

        rx0 = (base_x * level_w) // base_w
        ry0 = (base_y * level_h) // base_h
        rx1 = rx0 + _rw
        ry1 = ry0 + _rh

        in_bound = 0 <= rx0 and rx1 <= base_w and 0 <= ry0 and ry1 <= base_h
        if not in_bound:
            # crop coord to valid zone
            rx0 = _clip(rx0, 0, level_w)
            rx1 = _clip(rx1, 0, level_w)
            ry0 = _clip(ry0, 0, level_h)
            ry1 = _clip(ry1, 0, level_h)

        requires_padding = padding and not in_bound
        if requires_padding:
            # compute padding
            pad_x0 = _clip(-rx0, 0, _rw)
            pad_x1 = _clip(rx1 - level_w, 0, _rw)
            pad_y0 = _clip(-ry0, 0, _rh)
            pad_y1 = _clip(ry1 - level_h, 0, _rh)

        if axes == "YXS":
            selection = slice(ry0, ry1), slice(rx0, rx1), slice(None)
        elif axes == "CYX":
            selection = slice(None), slice(ry0, ry1), slice(rx0, rx1)
        else:
            raise NotImplementedError(f"axes={axes!r}")

        arr: npt.NDArray[np.int_]
        if isinstance(self.ts_zarr_grp, zarr.core.Array):
            arr = self.ts_zarr_grp[selection]
        else:
            arr = self.ts_zarr_grp[str(level)][selection]

        if axes == "CYX":
            arr = arr.transpose((1, 2, 0))

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
            mode="RGB",
            size=img.size,
            color=f"#{self.properties[PROPERTY_NAME_BACKGROUND_COLOR] or 'ffffff'}",
        )
        thumb.paste(img, box=None, mask=None)
        thumb.thumbnail(size, Image.ANTIALIAS)
        return thumb


class NotTiffSlide(TiffSlide):
    # noinspection PyMissingConstructor
    def __init__(
        self,
        filename: PathOrFileOrBufferLike[AnyStr],
        *,
        tifffile_options: dict[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        # tifffile instance, can raise TiffFileError
        self.ts_tifffile = _prepare_tifffile(
            filename,
            storage_options=storage_options,
            tifffile_options=tifffile_options,
            _cls=NotTiffFile,
        )

    @classmethod
    def detect_format(
        cls,
        filename: PathOrFileOrBufferLike[AnyStr],
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


def _detect_format(tf: TiffFile) -> str:
    _vendor_compat_map = dict(
        svs="aperio",
        scn="leica",
        bif="ventana",
        ndpi="hamamatsu",
        # add more when needed
    )
    for prop, vendor in _vendor_compat_map.items():
        if getattr(tf, f"is_{prop}"):
            return vendor
    return "generic-tiff"


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
    fb: PathOrFileOrBufferLike[AnyStr],
    *,
    tifffile_options: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
    _cls: type[TF] = TiffFile,
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

    if isinstance(fb, TiffFileIO):
        # provided an IO stream like instance
        _warn_unused_storage_options(st_kw)

        return _cls(fb, **tf_kw)

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


def _parse_metadata_aperio(desc: str) -> dict[str, Any]:
    """Aperio SVS metadata"""
    if _TIFFFILE_VERSION >= (2021, 6, 14):
        # tifffile 2021.6.14 fixed the svs parsing.
        _aperio_desc = desc
        _aperio_recovered_header = None  # no need to recover

    else:
        # this emulates the new description parsing for older versions
        _aperio_desc = re.sub(r";Aperio [^;|]*(?=[|])", "", desc, 1)
        _aperio_recovered_header = desc.split("|", 1)[0]

    try:
        aperio_meta = svs_description_metadata(_aperio_desc)
    except ValueError:
        raise
    else:
        # Normalize the aperio metadata
        aperio_meta.pop("", None)
        aperio_meta.pop("Aperio Image Library", None)
        if aperio_meta and "Header" not in aperio_meta:
            aperio_meta["Header"] = _aperio_recovered_header
        vendor = "aperio"

    md = {
        PROPERTY_NAME_COMMENT: desc,
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
    return md


def _auto_select_series_scn(desc: str) -> int:
    """selects the first non-macro series of an SCN file"""
    tree = _xml_to_dict(desc)

    marco_sizeX = int(tree["scn"]["collection"]["@sizeX"])
    marco_sizeY = int(tree["scn"]["collection"]["@sizeY"])

    for idx, image in enumerate(tree["scn"]["collection"]["image"]):
        offsetX = int(image["view"]["@offsetX"])
        offsetY = int(image["view"]["@offsetY"])
        sizeX = int(image["view"]["@sizeX"])
        sizeY = int(image["view"]["@sizeY"])

        is_macro_image = (
            offsetX == 0
            and offsetY == 0
            and sizeX == marco_sizeX
            and sizeY == marco_sizeY
        )
        if not is_macro_image:
            break
    else:
        raise ValueError("SCN: no main image found")

    return idx


def _parse_metadata_scn(desc: str, series_idx: int) -> dict[str, Any]:
    """Leica metadata"""
    tree = _xml_to_dict(desc)

    image = tree["scn"]["collection"]["image"][series_idx]

    # use auto recovery
    # mpp_x = float(image['pixels']['@sizeX']) / float(image['view']['@sizeX']) / 1000
    # mpp_y = float(image['pixels']['@sizeY']) / float(image['view']['@sizeY']) / 1000
    mpp_x = None
    mpp_y = None

    obj_pow = float(image["scanSettings"]["objectiveSettings"]["objective"])

    md = {
        PROPERTY_NAME_COMMENT: desc,
        PROPERTY_NAME_VENDOR: "leica",
        PROPERTY_NAME_QUICKHASH1: None,
        PROPERTY_NAME_BACKGROUND_COLOR: None,
        PROPERTY_NAME_OBJECTIVE_POWER: obj_pow,
        PROPERTY_NAME_MPP_X: mpp_x,
        PROPERTY_NAME_MPP_Y: mpp_y,
        PROPERTY_NAME_BOUNDS_X: None,
        PROPERTY_NAME_BOUNDS_Y: None,
        PROPERTY_NAME_BOUNDS_WIDTH: None,
        PROPERTY_NAME_BOUNDS_HEIGHT: None,
        "leica.aperture": float(
            image["scanSettings"]["illuminationSettings"]["numericalAperture"]
        ),
        "leica.creation-date": str(image["creationDate"]),
        "leica.device-model": str(image["device"]["@model"]),
        "leica.device-version": str(image["device"]["@version"]),
        "leica.illumination-source": str(
            image["scanSettings"]["illuminationSettings"]["illuminationSource"]
        ),
    }

    return md


def _xml_to_dict(xml: str) -> dict[str, Any]:
    """helper function to convert xml string to a dictionary"""
    x = ElementTree.fromstring(xml)

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


def _label_series_axes(axes: str) -> tuple[str, ...]:
    """helper to make series shapes more understandable"""
    return tuple(tifffile.TIFF.AXES_LABELS[c] for c in axes)


def _clip(x, min_, max_):
    """clip a value to a range"""
    return min(max(x, min_), max_)
