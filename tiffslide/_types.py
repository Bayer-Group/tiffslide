from __future__ import annotations

import os
import sys
from typing import IO
from typing import TYPE_CHECKING
from typing import Any
from typing import AnyStr
from typing import Union

if sys.version_info >= (3, 8):
    from typing import Protocol
    from typing import TypedDict
    from typing import runtime_checkable

else:
    from typing_extensions import Protocol
    from typing_extensions import TypedDict
    from typing_extensions import runtime_checkable

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
    "PathOrFileOrBufferLike",
    "OpenFileLike",
    "TiffFileIO",
    "Point3D",
    "Size3D",
    "Slice3D",
    "SeriesCompositionInfo",
]

if sys.version_info >= (3, 9):
    PathLikeAnyStr: TypeAlias = os.PathLike[AnyStr]
else:
    PathLikeAnyStr: TypeAlias = os.PathLike


@runtime_checkable
class OpenFileLike(Protocol):
    """minimal fsspec open file type"""

    fs: AbstractFileSystem
    path: str

    def __enter__(self) -> IO[AnyStr]:
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

    def read(self, n: int = -1) -> AnyStr:
        ...

    def readinto(self, __buffer: Any) -> int | None:
        ...


PathOrFileOrBufferLike = Union[AnyStr, PathLikeAnyStr, OpenFileLike, TiffFileIO]


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
