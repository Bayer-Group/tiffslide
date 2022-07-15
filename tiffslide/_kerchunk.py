from __future__ import annotations

import json
import sys
from io import StringIO
from typing import Any
from typing import TYPE_CHECKING

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

if TYPE_CHECKING:
    from tiffslide.tiffslide import TiffSlide


__all__ = [
    "get_kerchunk_specification",
]


TIFFSLIDE_SPEC_VERSION = 1
KERCHUNK_SPEC_VERSION = 1


class KerchunkSpec(TypedDict):
    """simple kerchunk version 1 spec"""
    version: int
    templates: dict[str, str]
    gen: list[Any]
    refs: dict[str, Any]


def get_kerchunk_specification(
    slide: TiffSlide,
    *,
    urlpath: str,
    templatename: str | None = None,
) -> KerchunkSpec:
    """take a tiffslide instance and extract a kerchunk representation"""
    if slide.ts_tifffile.filename is None:
        raise ValueError("can't kerchunk a slide that's not backed by a named file")

    kw = {}
    if templatename is not None:
        kw["templatename"] = templatename

    combined = KerchunkSpec(
        version=KERCHUNK_SPEC_VERSION,
        templates={},
        gen=[],
        refs={},
    )
    for idx, series in enumerate(slide.ts_tifffile.series):
        with StringIO() as f, series.aszarr() as store:
            try:
                store.write_fsspec(
                    f,
                    urlpath,
                    groupname=f"s{idx}",
                    version=KERCHUNK_SPEC_VERSION,
                    **kw,
                )
            except ValueError as err:
                if "incomplete chunks" in str(err):
                    continue
                raise
            kc = json.loads(f.getvalue())

        # combine individual series
        combined["templates"].update(kc["templates"])
        combined["gen"].extend(kc["gen"])

        combined_refs = combined["refs"]
        refs = kc["refs"]

        # insert .zgroup
        _zgroup = refs.pop(".zgroup")
        if ".zgroup" in combined_refs:
            assert combined_refs[".zgroup"] == _zgroup
        combined_refs[".zgroup"] = _zgroup

        # insert .zattrs
        if ".zattrs" not in combined_refs:
            combined_refs[".zattrs"] = json.dumps({
                "tiffslide.spec_version": TIFFSLIDE_SPEC_VERSION,
                "tiffslide.properties": slide.properties,
            })
        assert ".zattrs" not in refs

        # insert refs
        combined_refs.update(refs)

    return combined


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

    _slide = TiffSlide(args.urlpath, storage_options=args.storage_options)
    k = get_kerchunk_specification(_slide, urlpath=args.urlpath)

    pp(k, compact=True, indent=2)
