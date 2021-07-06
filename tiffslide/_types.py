# noinspection PyUnresolvedReferences
from os import PathLike
from typing import IO
from typing import Union

__all__ = [
    "PathOrFileLike",
]

# todo: check if this covers all relevant use cases
PathOrFileLike = Union[str, bytes, 'PathLike[str]', IO[str], IO[bytes]]
