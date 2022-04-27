from __future__ import annotations

import json
from io import StringIO
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiffslide.tiffslide import TiffSlide


__all__ = [
    "get_kerchunk_specification",
]


KERCHUNK_SPEC_VERSION = 1

def get_kerchunk_specification(
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


if __name__ == "__main__":
    import argparse
    import sys

    from tiffslide.tiffslide import TiffSlide

    if sys.version_info >= (3, 8):
        from pprint import pp
    else:
        from pprint import pprint as pp

    parser = argparse.ArgumentParser("tiffslide._kerchunk")
    parser.add_argument("urlpath", help="fsspec compatible urlpath to image")
    parser.add_argument("--storage-options", type=json.loads, help="json encoded storage options", default=None)
    args = parser.parse_args()

    slide = TiffSlide(args.urlpath, storage_options=args.storage_options)
    kc = get_kerchunk_specification(slide, urlpath=args.urlpath)

    pp(kc, compact=True, indent=2)
