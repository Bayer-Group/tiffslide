from __future__ import annotations

import os
import sys
import warnings
from typing import IO
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import TypedDict
from typing import Union
from typing import runtime_checkable

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from types import TracebackType

    from fsspec import AbstractFileSystem

    if sys.version_info >= (3, 10):
        from types import EllipsisType
    else:
        EllipsisType: TypeAlias = Any


__all__ = [
    "PathLikeStr",
    "PathOrFileOrBufferLike",
    "OpenFileLike",
    "TiffFileIO",
    "Point3D",
    "Size3D",
    "Slice3D",
    "SeriesCompositionInfo",
]

if TYPE_CHECKING:
    from typing import AnyStr

    PathLikeAnyStr: TypeAlias = os.PathLike[AnyStr]


def __getattr__(name: str) -> Any:
    if name == "PathLikeAnyStr":
        warnings.warn(
            "use PathLikeStr instead of PathLikeAnyStr",
            DeprecationWarning,
            stacklevel=2,
        )
        return "os.PathLike[AnyStr]"
    else:
        raise AttributeError(name)


PathLikeStr: TypeAlias = "os.PathLike[str]"


@runtime_checkable
class OpenFileLike(Protocol):
    """minimal fsspec open file type"""

    fs: AbstractFileSystem
    path: str

    def __enter__(self) -> IO[bytes]:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        ...


@runtime_checkable
class TiffFileIO(Protocol):
    """minimal stream io for use in tifffile.FileHandle"""

    def seek(self, offset: int, whence: int = 0) -> int:
        ...

    def tell(self) -> int:
        ...

    def read(self, n: int = -1) -> bytes:
        ...

    def readinto(self, __buffer: Any) -> int | None:
        ...


PathOrFileOrBufferLike = Union[str, PathLikeStr, OpenFileLike, TiffFileIO]


Point3D: TypeAlias = "tuple[int, int, int]"
Size3D: TypeAlias = "tuple[int, int, int]"
Slice2D: TypeAlias = "tuple[slice, slice]"
_Slice2D_YX0: TypeAlias = "tuple[slice, slice, EllipsisType]"
Slice3D: TypeAlias = "tuple[slice, slice, slice] | _Slice2D_YX0"


class SeriesCompositionInfo(TypedDict):
    """composition information for combining tifffile series

    Notes
    -----
    level_shapes:
        A list of 3D sizes for each composited level
    located_series:
        A dict mapping the TiffFile series index to a list of
        3D offsets for each level for compositing

    """

    level_shapes: list[Size3D]
    located_series: dict[int, list[Point3D]]
