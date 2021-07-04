"""tiffslide

a somewhat drop-in replacement for openslide-python using tifffile and zarr

"""
from warnings import warn

from tiffslide._types import PathOrFileLike
from tiffslide.tiffslide import TiffSlide
from tiffslide.tiffslide import TiffFileError
from tiffslide.tiffslide import (
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
)


try:
    from tiffslide._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "not-installed"

__all__ = ["TiffSlide", "TiffFileError"]


def __getattr__(name):  # type: ignore
    """support some drop-in behavior"""
    # alias the most important bits
    if name in {"OpenSlideUnsupportedFormatError", "OpenSlideError"}:
        warn(f"compatibility: aliasing tiffslide.TiffFileError to {name!r}")
        return TiffFileError
    elif name in {"OpenSlide", "ImageSlide"}:
        warn(f"compatibility: aliasing tiffslide.TiffSlide to {name!r}")
        return TiffSlide
    # warn if internals are imported that we dont support
    if name in {"AbstractSlide", "__library_version__"}:
        warn(f"{name!r} is not provided by tiffslide")
    raise AttributeError(name)


def open_slide(filename: PathOrFileLike) -> TiffSlide:
    """drop-in helper function"""
    return TiffSlide(filename)
