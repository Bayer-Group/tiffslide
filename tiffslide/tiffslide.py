from __future__ import annotations

import math
import re
import sys
from types import TracebackType
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Type
from typing import Union
from warnings import warn

if sys.version_info[:2] >= (3, 8):
    from functools import cached_property
    from importlib.metadata import version
else:
    from backports.cached_property import cached_property
    from importlib_metadata import version

import zarr
from PIL import Image
from tifffile import TiffFile
from tifffile import TiffFileError as TiffFileError
from tifffile import TiffPageSeries
from tifffile.tifffile import svs_description_metadata

from tiffslide._types import PathOrFileLike

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


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

    def __init__(self, filename: PathOrFileLike):
        self.ts_tifffile: TiffFile = TiffFile(filename)  # may raise TiffFileError
        self._zarr_grp: Optional[Union[zarr.core.Array, zarr.hierarchy.Group]] = None
        self._metadata: Optional[Dict[str, Any]] = None

    def __enter__(self) -> TiffSlide:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._zarr_grp:
            try:
                self._zarr_grp.close()
            except AttributeError:
                pass  # Arrays dont need to be closed
            self._zarr_grp = None
        self.ts_tifffile.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.ts_tifffile.filename!r})"

    @classmethod
    def detect_format(cls, filename: PathOrFileLike) -> Optional[str]:
        """return the detected format as a str or None if unknown/unimplemented"""
        _vendor_compat_map = dict(
            svs="aperio",
            # add more when needed
        )
        with TiffFile(filename) as t:
            for prop, vendor in _vendor_compat_map.items():
                if getattr(t, f"is_{prop}"):
                    return vendor
        return None

    @property
    def dimensions(self) -> Tuple[int, int]:
        """return the width and height of level 0"""
        series0 = self.ts_tifffile.series[0]
        assert series0.ndim == 3, "loosen restrictions in future versions"
        h, w, _ = series0.shape
        return w, h

    @property
    def level_count(self) -> int:
        """return the number of levels"""
        return len(self.ts_tifffile.series[0].levels)

    @property
    def level_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        """return the dimensions of levels as a list"""
        return tuple(lvl.shape[1::-1] for lvl in self.ts_tifffile.series[0].levels)

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
        """return the downsampling factors of levels as a list"""
        w0, h0 = self.dimensions
        return tuple(math.sqrt((w0 * h0) / (w * h)) for w, h in self.level_dimensions)

    @cached_property
    def properties(self) -> Dict[str, Any]:
        """image properties / metadata as a dict"""
        if self._metadata is None:
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
            else:
                # Normalize the aperio metadata
                aperio_meta.pop("", None)
                aperio_meta.pop("Aperio Image Library", None)
                if aperio_meta and "Header" not in aperio_meta:
                    aperio_meta["Header"] = _aperio_recovered_header

            md = {
                PROPERTY_NAME_COMMENT: aperio_desc,
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

            _ds_dimensions = zip(self.level_downsamples, self.level_dimensions)
            for lvl, (ds, (width, height)) in enumerate(_ds_dimensions):
                page = self.ts_tifffile.series[0].levels[lvl].pages[0]
                md[f"tiffslide.level[{lvl}].downsample"] = ds
                md[f"tiffslide.level[{lvl}].height"] = height
                md[f"tiffslide.level[{lvl}].width"] = width
                md[f"tiffslide.level[{lvl}].tile-height"] = page.tilelength
                md[f"tiffslide.level[{lvl}].tile-width"] = page.tilewidth

            md["tiff.ImageDescription"] = aperio_desc
            self._metadata = md
        return self._metadata

    @cached_property
    def associated_images(self) -> Mapping[str, Image.Image]:
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

    @property
    def ts_zarr_grp(self) -> Union[zarr.core.Array, zarr.hierarchy.Group]:
        """return the tiff image as a zarr array or group

        NOTE: this is extra functionality and not part of the drop-in behaviour
        """
        if self._zarr_grp is None:
            store = self.ts_tifffile.series[0].aszarr()
            self._zarr_grp = zarr.open(store, mode="r")
        return self._zarr_grp

    def _read_region_as_array(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> ArrayLike:
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
        base_x, base_y = location
        base_w, base_h = self.dimensions
        level_w, level_h = self.level_dimensions[level]
        rx0 = (base_x * level_w) // base_w
        ry0 = (base_y * level_h) // base_h
        _rw, _rh = size
        rx1 = rx0 + _rw
        ry1 = ry0 + _rh
        arr: ArrayLike
        if isinstance(self.ts_zarr_grp, zarr.core.Array):
            arr = self.ts_zarr_grp[ry0:ry1, rx0:rx1]
        else:
            arr = self.ts_zarr_grp[str(level)][ry0:ry1, rx0:rx1]
        return arr

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> Image.Image:
        """return the requested region as a PIL.Image

        Parameters
        ----------
        location :
            pixel location (x, y) in level 0 of the image
        level :
            target level used to read the image
        size :
            size (width, height) of the requested region
        """
        arr = self._read_region_as_array(location, level, size)
        return Image.fromarray(arr)

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """return the thumbnail of the slide as a PIL.Image with a maximum size"""
        slide_w, slide_h = self.dimensions
        thumb_w, thumb_h = size
        downsample = max(slide_w / thumb_w, slide_h / thumb_h)
        level = self.get_best_level_for_downsample(downsample)

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
        self.series_map: Dict[str, TiffPageSeries] = {s.name.lower(): s for s in series}
        self._m: Dict[str, Image.Image] = {}

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
