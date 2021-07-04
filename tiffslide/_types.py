import os
from typing import IO
from typing import Union

# todo: check if this covers all relevant use cases
PathOrFileLike = Union[str, bytes, os.PathLike, IO[str], IO[bytes]]
