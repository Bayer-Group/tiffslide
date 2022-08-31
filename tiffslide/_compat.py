"""tiffslide._compat

compatibility layer to support loading non-tiff images

"""
from __future__ import annotations

import json
import os.path
import sys
from pathlib import PurePath
from types import MappingProxyType
from types import TracebackType
from typing import TYPE_CHECKING
from typing import Any
from typing import Mapping
from typing import Sequence

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
import zarr
from imagecodecs import __version__ as _imagecodecs_version
from imagecodecs import imread

try:
    from tiffslide._version import version as _tiffslide_version
except ImportError:  # pragma: no cover
    _tiffslide_version = "not-installed"

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "NotTiffFile",
]


class NotTiffFile:
    def __init__(
        self,
        arg: Any,
        mode: Literal["rb"] | None = None,
        name: str | None = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        if mode is not None and mode != "rb":
            raise ValueError("mode must be 'rb'")
        if name is None:
            if isinstance(arg, str):
                name = os.path.basename(arg)
            elif isinstance(arg, PurePath):
                name = arg.name
            elif hasattr(arg, "fullname"):
                name = arg.fullname
        self.filename = name

        array, codec = imread(arg, return_codec=True)

        self.pages = [NotTiffPage(array, codec=codec.__name__)]
        self.series = [NotTiffPageSeries(self.pages)]

    def close(self) -> None:
        pass

    def __enter__(self) -> NotTiffFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def __getattr__(self, item: str) -> Any:
        if item.startswith("is_"):
            return False
        raise AttributeError(item)


class NotTiffPageSeries:
    axes = "YXS"  # todo: check if it's always this

    def __init__(self, pages: NotTiffPages) -> None:
        self.pages = pages

    def __getitem__(self, item: int) -> NotTiffPage:
        return self.pages[item]

    def __len__(self) -> int:
        return len(self.pages)

    @property
    def ndim(self) -> int:
        return self.pages[0].ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.pages[0].shape

    @property
    def levels(self) -> list[NotTiffPageSeries]:
        return [self]

    @property
    def size(self) -> int:
        return self.pages[0].size

    def aszarr(self, **_: Any) -> zarr.storage.MemoryStore:
        return self.pages[0].aszarr()

    def asarray(self) -> NDArray[np.uint8]:
        return self.pages[0].asarray()


NotTiffPages = Sequence["NotTiffPage"]


class NotTiffPage:
    tags: Mapping[Any, Any] = MappingProxyType({})
    tilelength = 0
    tilewidth = 0

    def __init__(self, array: NDArray[np.uint8], codec: str):
        self._array = array
        self._codec = codec

    @property
    def description(self) -> str:
        data = json.dumps(
            {
                "tiffslide.__version__": _tiffslide_version,
                "imagecodecs.__version__": _imagecodecs_version,
                "codec": self._codec,
            },
            separators=(",", ":"),
        )
        return f"tiffslide={data}"

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def size(self) -> int:
        return self._array.size

    def aszarr(self) -> zarr.storage.MemoryStore:
        return zarr.creation.array(self._array).store

    def asarray(self) -> NDArray[np.uint8]:
        return self._array
