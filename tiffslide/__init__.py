"""tiffslide

a somewhat drop-in replacement for openslide-python using tifffile and zarr

"""
import math
import os
from warnings import warn
from collections.abc import Mapping
from functools import cached_property
from os import PathLike
from typing import Dict
from typing import Iterator

import zarr
from PIL import Image
from tifffile import TiffFile
from tifffile import TiffFileError as TiffFileError
from tifffile import TiffPageSeries
from tifffile.tifffile import svs_description_metadata

try:
    from tiffslide._version import version as __version__
except ImportError:
    __version__ = "not-installed"


__all__ = ["TiffSlide", "TiffFileError"]


def __getattr__(name):
    """support some drop-in behavior"""
    # alias the most important bits
    if name in {"OpenSlideUnsupportedFormatError", "OpenSlideError"}:
        warn(f'compatibility: aliasing tiffslide.TiffFileError to {name!r}')
        return TiffFileError
    elif name in {"OpenSlide", "ImageSlide"}:
        warn(f'compatibility: aliasing tiffslide.TiffSlide to {name!r}')
        return TiffSlide
    # warn if internals are imported that we dont support
    if name in {"AbstractSlide", "__library_version__"}:
        warn(f'{name!r} is not provided by tiffslide')
    raise AttributeError(name)


PROPERTY_NAME_COMMENT = u'tiffslide.comment'
PROPERTY_NAME_VENDOR = u'tiffslide.vendor'
PROPERTY_NAME_QUICKHASH1 = u'tiffslide.quickhash-1'
PROPERTY_NAME_BACKGROUND_COLOR = u'tiffslide.background-color'
PROPERTY_NAME_OBJECTIVE_POWER = u'tiffslide.objective-power'
PROPERTY_NAME_MPP_X = u'tiffslide.mpp-x'
PROPERTY_NAME_MPP_Y = u'tiffslide.mpp-y'
PROPERTY_NAME_BOUNDS_X = u'tiffslide.bounds-x'
PROPERTY_NAME_BOUNDS_Y = u'tiffslide.bounds-y'
PROPERTY_NAME_BOUNDS_WIDTH = u'tiffslide.bounds-width'
PROPERTY_NAME_BOUNDS_HEIGHT = u'tiffslide.bounds-height'


def open_slide(filename):
    """drop-in helper function"""
    return TiffSlide(filename)


class TiffSlide:
    """
    tifffile backed whole slide image container emulating openslide.OpenSlide
    """

    def __init__(self, filename: PathLike):
        self.ts_filename = os.fspath(filename)
        self.ts_tifffile = TiffFile(self.ts_filename)  # may raise TiffFileError
        self._zarr_grp = None
        self._metadata = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._zarr_grp:
            self._zarr_grp.close()
            self._zarr_grp = None
        self.ts_tifffile.close()

    def __repr__(self):
        return f"{type(self).__name__}({self.ts_filename!r})"

    @classmethod
    def detect_format(cls, filename):
        _vendor_compat_map = dict(
            svs='aperio',
            # add more when needed
        )
        with TiffFile(filename) as t:
            for attr in dir(t):
                if not attr.startswith("is_"):
                    continue
                if getattr(t, attr):
                    vendor = _vendor_compat_map.get(attr[3:])
                    if vendor is not None:
                        return vendor
        return None

    @property
    def dimensions(self):
        assert self.ts_tifffile.series[0].ndim == 3, "loosen restrictions in future versions"
        return self.ts_tifffile.series[0].shape[1::-1]

    @property
    def level_count(self):
        return len(self.ts_tifffile.series[0].levels)

    @property
    def level_dimensions(self):
        return tuple(
            lvl.shape[1::-1]
            for lvl in self.ts_tifffile.series[0].levels
        )

    @property
    def level_downsamples(self):
        w0, h0 = self.dimensions
        return tuple(
            math.sqrt((w0*h0) / (w*h))
            for w, h in self.level_dimensions
        )

    @cached_property
    def properties(self):
        """image properties"""
        if self._metadata is None:
            aperio_desc = self.ts_tifffile.pages[0].description
            aperio_meta = svs_description_metadata(aperio_desc)
            aperio_meta.pop("")
            aperio_meta.pop("Aperio Image Library")

            md = {
                PROPERTY_NAME_COMMENT: aperio_desc,
                PROPERTY_NAME_VENDOR: "aperio",
                PROPERTY_NAME_QUICKHASH1: None,
                PROPERTY_NAME_BACKGROUND_COLOR: None,
                PROPERTY_NAME_OBJECTIVE_POWER: aperio_meta["AppMag"],
                PROPERTY_NAME_MPP_X: aperio_meta["MPP"],
                PROPERTY_NAME_MPP_Y: aperio_meta["MPP"],
                PROPERTY_NAME_BOUNDS_X: None,
                PROPERTY_NAME_BOUNDS_Y: None,
                PROPERTY_NAME_BOUNDS_WIDTH: None,
                PROPERTY_NAME_BOUNDS_HEIGHT: None,
            }
            md.update({f"aperio.{k}": v for k, v in sorted(aperio_meta.items())})
            for lvl, (ds, (width, height)) in enumerate(zip(
                    self.level_downsamples, self.level_dimensions,
            )):
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
    def associated_images(self):
        return _LazyAssociatedImagesDict(self.ts_tifffile)

    def get_best_level_for_downsample(self, downsample):
        if downsample <= 1.0:
            return 0
        for lvl, ds in enumerate(self.level_downsamples):
            if ds >= downsample:
                return lvl - 1
        return self.level_count - 1

    @property
    def ts_zarr_grp(self):
        if self._zarr_grp is None:
            store = self.ts_tifffile.series[0].aszarr()
            self._zarr_grp = zarr.open(store, mode='r')
        return self._zarr_grp

    def read_region(self, location, level, size):
        base_x, base_y = location
        base_w, base_h = self.dimensions
        level_w, level_h = self.level_dimensions[level]
        level_rw, level_rh = size
        level_rx = (base_x * level_w) // base_w
        level_ry = (base_y * level_h) // base_h
        arr = self.ts_zarr_grp[level][level_ry:level_ry + level_rh, level_rx:level_rx + level_rw]
        return Image.fromarray(arr)

    def get_thumbnail(self, size):
        slide_w, slide_h = self.dimensions
        thumb_w, thumb_h = size
        downsample = max(slide_w / thumb_w, slide_h / thumb_h)
        level = self.get_best_level_for_downsample(downsample)
        tile = self.read_region((0, 0), level, self.level_dimensions[level])
        # Apply on solid background
        bg_color = '#' + self.properties.get(PROPERTY_NAME_BACKGROUND_COLOR,
                                             'ffffff')
        thumb = Image.new('RGB', tile.size, bg_color)
        thumb.paste(tile, None, tile)
        thumb.thumbnail(size, Image.ANTIALIAS)
        return thumb


# === internal utility classes ========================================

class _LazyAssociatedImagesDict(Mapping):
    """lazily load associated images"""

    def __init__(self, tifffile):
        series = tifffile.series[1:]
        self._k: Dict[str, TiffPageSeries] = {s.name.lower(): s for s in series}
        self._m = {}

    def __repr__(self):
        args = ", ".join(
            f"{name!r}: <lazy-loaded PIL.Image.Image size={s.shape[1]}x{s.shape[0]} ...>"
            for name, s in self._k.items()
        )
        return f"{{{args}}}"

    def __getitem__(self, k: str) -> Image.Image:
        if k in self._m:
            return self._m[k]
        else:
            s = self._k[k]
            self._m[k] = img = Image.fromarray(s.asarray())
            return img

    def __len__(self) -> int:
        return len(self._k)

    def __iter__(self) -> Iterator[str]:
        yield from self._k
