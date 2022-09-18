"""python3.7 compatibility code

This will be removed as soon as we drop python 3.7

"""
from __future__ import annotations

import re
import sys
from threading import RLock
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterator
from typing import Mapping
from typing import TypeVar

import zarr
from tifffile import __version__ as tifffile_version

__all__ = [
    "REQUIRES_STORE_FIX",
    "py37_fix_store",
    "cached_property",
]


def _requires_store_fix(ver_zarr: str, ver_tifffile: str) -> bool:
    _v_regex = re.compile(r"^([0-9]+)[.]([0-9]+)[.]([0-9]+)(a|b|rc)?([0-9]+)?$")
    _v_map = {"a": 0, "b": 1, "rc": 2, None: 99}

    mz = _v_regex.match(ver_zarr)
    mt = _v_regex.match(ver_tifffile)
    if mz is None or mt is None:
        return True

    _ver_zarr: list[int] = [
        int(x) if x and x.isdigit() else _v_map[x] for x in mz.groups()
    ]
    _ver_tifffile: list[int] = [
        int(x) if x and x.isdigit() else _v_map[x] for x in mt.groups()
    ]

    _new_zarr = _ver_zarr >= [2, 11, 0, 99, 99]
    _old_tifffile = _ver_tifffile < [2022, 3, 29, 99, 99]
    return _new_zarr and _old_tifffile


REQUIRES_STORE_FIX = _requires_store_fix(zarr.__version__, tifffile_version)


# --- zarr - tifffile compatibility patches ---------------------------
#
# note: we can drop this once we drop python 3.7


class _IncompatibleStoreShim(Mapping[str, Any]):
    """
    A compatibility shim, for python=3.7
    with zarr>=2.11.0 with tifffile<2022.3.29
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        self._mutable_mapping = mapping

    def __getitem__(self, key: str) -> Any:
        if key.endswith((".zarray", ".zgroup")) and key not in self._mutable_mapping:
            raise KeyError(key)
        try:
            return self._mutable_mapping[key]
        except ValueError:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._mutable_mapping)

    def __len__(self) -> int:
        return len(self._mutable_mapping)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._mutable_mapping, item)


def py37_fix_store(zstore: Mapping[str, Any]) -> Mapping[str, Any]:
    """python 3.7 compatibility fix for tifffile and zarr"""
    if REQUIRES_STORE_FIX:
        return _IncompatibleStoreShim(zstore)
    else:
        return zstore


if sys.version_info < (3, 8):
    # --- vendored cached_property from CPython with added type information ---

    _T = TypeVar("_T")
    _NOT_FOUND = object()

    class cached_property(Generic[_T]):
        def __init__(self, func: Callable[..., _T]) -> None:
            self.func = func
            self.attrname = None
            self.__doc__ = func.__doc__
            self.lock = RLock()

        def __set_name__(self, owner, name):  # type: ignore
            if self.attrname is None:
                self.attrname = name
            elif name != self.attrname:
                raise TypeError(
                    "Cannot assign the same cached_property to two different names "
                    f"({self.attrname!r} and {name!r})."
                )

        def __get__(self, instance: Any, owner: type[Any] | None = None) -> _T:
            if instance is None:
                return self  # type: ignore
            if self.attrname is None:
                raise TypeError(
                    "Cannot use cached_property instance without calling __set_name__ on it."
                )
            try:
                cache = instance.__dict__
            except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
                msg = (
                    f"No '__dict__' attribute on {type(instance).__name__!r} "
                    f"instance to cache {self.attrname!r} property."
                )
                raise TypeError(msg) from None
            val = cache.get(self.attrname, _NOT_FOUND)
            if val is _NOT_FOUND:
                with self.lock:
                    # check if another thread filled cache while we awaited lock
                    val = cache.get(self.attrname, _NOT_FOUND)
                    if val is _NOT_FOUND:
                        val = self.func(instance)
                        try:
                            cache[self.attrname] = val
                        except TypeError:
                            msg = (
                                f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                                f"does not support item assignment for caching {self.attrname!r} property."
                            )
                            raise TypeError(msg) from None
            return val

        # __class_getitem__ = classmethod(GenericAlias)

else:
    from functools import cached_property
