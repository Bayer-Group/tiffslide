from __future__ import annotations

import os
import sys
from types import TracebackType
from typing import IO
from typing import Any
from typing import AnyStr
from typing import Optional
from typing import Type
from typing import Union

if sys.version_info >= (3, 8):
    from typing import Protocol
    from typing import TypedDict
    from typing import runtime_checkable
else:
    from typing_extensions import Protocol
    from typing_extensions import TypedDict
    from typing_extensions import runtime_checkable

from fsspec import AbstractFileSystem

__all__ = [
    "PathOrFileOrBufferLike",
    "OpenFileLike",
    "TiffFileIO",
    "Size3D",
    "Point3D",
    "Slice3D",
    "SeriesCompositionInfo",
]

if sys.version_info >= (3, 9):
    PathLikeAnyStr = os.PathLike[AnyStr]
else:
    PathLikeAnyStr = os.PathLike


@runtime_checkable
class OpenFileLike(Protocol):
    """minimal fsspec open file type"""

    fs: AbstractFileSystem
    path: str

    def __enter__(self) -> IO[AnyStr]:
        ...

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
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

    def readinto(self, __buffer: Any) -> Optional[int]:
        ...


PathOrFileOrBufferLike = Union[AnyStr, PathLikeAnyStr, OpenFileLike, TiffFileIO]


Slice3D = "tuple[slice, slice, slice]"
Point3D = "tuple[int, int, int]"
Size3D = "tuple[int, int, int]"


class SeriesCompositionInfo(TypedDict):
    """composition information for combining tifffile series"""
    shape: Size3D
    located_series: dict[int, Point3D]
