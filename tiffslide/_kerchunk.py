from __future__ import annotations

import os
from io import StringIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiffslide.tiffslide import TiffSlide


def extract_kerchunk(slide: TiffSlide):
    """take a tiffslide instance and extract a kerchunk representation"""
    ...

def to_fsspec(self, url=None, key=None, series=None, level=None, chunkmode=None, version=None):
    ts = self.ts_tifffile
    if ts.filehandle.name is None:
        # noinspection PyProtectedMember
        _fh = ts.filehandle._fh
        if hasattr(_fh, 'path'):
            # fixme: this should be upstreamed
            ts.filehandle._name = os.path.basename(_fh.path)

    if url is None:
        if self._urlpath is None:
            raise NotImplementedError("needs a urlpath")
        else:
            url = self._urlpath

    with StringIO() as f:
        with self.ts_tifffile.aszarr(
            key=key, series=series, level=level, chunkmode=chunkmode
        ) as store:
            store.write_fsspec(f, url, version=version)
        return f.getvalue()

