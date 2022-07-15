"""tiffslide._kerchunk

Test implementation of serialization and deserialization code to and from kerchunk
Don't rely on this API until it's public.

"""
from __future__ import annotations

import json
import os
import sys
from collections import ChainMap
from io import StringIO
from typing import TYPE_CHECKING
from typing import Any

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import fsspec
from imagecodecs.numcodecs import register_codecs

from tiffslide.tiffslide import TiffSlide

if TYPE_CHECKING:
    from fsspec.implementations.reference import ReferenceFileSystem


__all__ = [
    "to_kerchunk",
    "from_kerchunk",
]


TIFFSLIDE_SPEC_VERSION = 1
KERCHUNK_SPEC_VERSION = 1


class KerchunkSpec(TypedDict):
    """simple kerchunk version 1 spec"""

    version: int
    templates: dict[str, str]
    gen: list[Any]
    refs: dict[str, Any]


def to_kerchunk(
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

    if not isinstance(urlpath, str):
        urlpath = os.fspath(urlpath)
    if urlpath.endswith(slide.ts_tifffile.filename):
        urlpath = urlpath[: -len(slide.ts_tifffile.filename)]

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
            combined_refs[".zattrs"] = json.dumps(
                {
                    "tiffslide.spec_version": TIFFSLIDE_SPEC_VERSION,
                    "tiffslide.properties": slide.properties,
                }
            )
        assert ".zattrs" not in refs

        # insert refs
        combined_refs.update(refs)

    return combined


def from_kerchunk(
    kc: KerchunkSpec,
    *,
    urlpath: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> TiffSlide:
    """deserialize a TiffSlide from a kerchunk specification"""
    if urlpath:
        # replace urlpath in kerchunk spec
        templates = kc.get("templates", {}).copy()
        if len(set(templates.values())) != 1:
            raise NotImplementedError(
                "urlpath replacement support only for length 1 templates"
            )
        templates = {key: urlpath for key in templates}
        kc = ChainMap({"templates": templates}, kc)  # type: ignore

    if storage_options is None:
        storage_options = {}

    fs: ReferenceFileSystem = fsspec.filesystem(
        "reference",
        fo=kc,
        **storage_options,
    )
    zattrs = json.loads(fs.cat_file(".zattrs"))

    if "tiffslide.spec_version" not in zattrs or "tiffslide.properties" not in zattrs:
        raise ValueError("")

    if zattrs["tiffslide.spec_version"] != 1:
        raise NotImplementedError("TiffSlide spec version unsupported")
    properties = zattrs["tiffslide.properties"]

    # register codecs now...
    register_codecs(verbose=False)

    # prepare the TiffSlide instance
    inst = object.__new__(TiffSlide)
    inst.__dict__["properties"] = properties
    inst._tifffile = fs  # fixme: ...
    return inst


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("tiffslide._kerchunk")
    parser.add_argument("urlpath", help="fsspec compatible urlpath to image")
    parser.add_argument(
        "--storage-options",
        type=json.loads,
        help="json encoded storage options",
        default=None,
    )
    args = parser.parse_args()

    # open slide, serialize and reopen
    _slide = TiffSlide(args.urlpath, storage_options=args.storage_options)
    k = to_kerchunk(_slide, urlpath=args.urlpath)
    # kc_slide = from_kerchunk(k)
    print(json.dumps(k, separators=(",", ":")))
