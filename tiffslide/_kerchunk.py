from __future__ import annotations

import json
from io import StringIO
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiffslide.tiffslide import TiffSlide


KERCHUNK_SPEC_VERSION = 1

def extract_kerchunk(
    slide: TiffSlide,
    *,
    urlpath: str,
    group_name: str | None = None,
    level: int | None = None,
) -> dict[str, Any]:
    """take a tiffslide instance and extract a kerchunk representation"""
    if slide.ts_tifffile.filename is None:
        raise ValueError("can't kerchunk a slide that's not backed by a named file")

    with StringIO() as f:
        series_idx = slide.properties["tiffslide.series-index"]
        series = slide.ts_tifffile.series[series_idx]
        with series.aszarr(level=level) as store:
            store.write_fsspec(f, urlpath, groupname=group_name, version=KERCHUNK_SPEC_VERSION)
        kerchunk_dct = json.loads(f.getvalue())

    return kerchunk_dct
