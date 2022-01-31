"""tiffslide

a somewhat drop-in replacement for openslide-python using tifffile and zarr

"""
from typing import AnyStr
from warnings import warn

from tiffslide._types import PathOrFileOrBufferLike
from tiffslide.tiffslide import PROPERTY_NAME_BACKGROUND_COLOR
from tiffslide.tiffslide import PROPERTY_NAME_BOUNDS_HEIGHT
from tiffslide.tiffslide import PROPERTY_NAME_BOUNDS_WIDTH
from tiffslide.tiffslide import PROPERTY_NAME_BOUNDS_X
from tiffslide.tiffslide import PROPERTY_NAME_BOUNDS_Y
from tiffslide.tiffslide import PROPERTY_NAME_COMMENT
from tiffslide.tiffslide import PROPERTY_NAME_MPP_X
from tiffslide.tiffslide import PROPERTY_NAME_MPP_Y
from tiffslide.tiffslide import PROPERTY_NAME_OBJECTIVE_POWER
from tiffslide.tiffslide import PROPERTY_NAME_QUICKHASH1
from tiffslide.tiffslide import PROPERTY_NAME_VENDOR
from tiffslide.tiffslide import TiffFileError
from tiffslide.tiffslide import TiffSlide

try:
    from tiffslide._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "not-installed"

__all__ = ["TiffSlide", "TiffFileError"]


def __getattr__(name):  # type: ignore
    """support some drop-in behavior"""
    # alias the most important bits
    if name in {"OpenSlideUnsupportedFormatError", "OpenSlideError"}:
        warn(
            f"compatibility: aliasing tiffslide.TiffFileError to {name!r}", stacklevel=2
        )
        return TiffFileError
    elif name in {"OpenSlide", "ImageSlide"}:
        warn(f"compatibility: aliasing tiffslide.TiffSlide to {name!r}", stacklevel=2)
        return TiffSlide
    # warn if internals are imported that we dont support
    if name in {"AbstractSlide", "__library_version__"}:
        warn(f"{name!r} is not provided by tiffslide", stacklevel=2)
    raise AttributeError(name)


def open_slide(filename: PathOrFileOrBufferLike[AnyStr]) -> TiffSlide:
    """drop-in helper function"""
    return TiffSlide(filename)
