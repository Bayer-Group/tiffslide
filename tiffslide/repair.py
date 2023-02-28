"""helpers for broken wsi"""
from __future__ import annotations

import functools
import inspect
import os
import os.path
import warnings
from typing import Any

import tifffile


def fix_description_tag_encoding(
    filename: str | os.PathLike[str],
    *,
    errors: str = "ignore",
    no_dry_run: bool = False,
) -> int:
    """overwrites broken description tags"""
    filename = os.path.expanduser(filename)
    tf = tifffile.TiffFile(filename, mode="r+")

    applied_fixes = False
    for page in tf.pages:
        try:
            description_tag = page.tags["ImageDescription"]
        except KeyError:
            continue
        value = description_tag.value
        if isinstance(value, bytes) and value:
            if not no_dry_run:
                print("# would fix tag", description_tag)
                print("- ", repr(value)[2:-1])
                print("+ ", repr(value.decode(errors=errors).encode())[2:-1])
            else:
                print("# fixing tag", description_tag)
                new_value = value.decode("ascii", errors=errors)
                if len(new_value.encode()) > len(value):
                    raise RuntimeError("fixed string is longer than original?")
                description_tag.overwrite(new_value)
            applied_fixes = True
    if not applied_fixes:
        print("# found no problems with description tag encoding")
    elif applied_fixes and not no_dry_run:
        print("\n# run the command again with `--no-dry-run` to apply fixes")
        print("# PLEASE BACKUP THE ORIGINAL FILE BEFORE!")
    return 0


def monkey_patch_description_tag_encoding() -> None:
    from tifffile.tifffile import TiffTags

    original_valueof_method = TiffTags.valueof
    if getattr(original_valueof_method, "__monkey_patched__", False):
        return

    sig = inspect.signature(original_valueof_method)
    if "key" not in sig.parameters:
        raise AssertionError("monkey patch won't work on this tifffile version")

    @functools.wraps(original_valueof_method)
    def patched_valueof(*args: Any, **kwargs: Any) -> Any:
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        value = original_valueof_method(*ba.args, **ba.kwargs)
        if ba.arguments["key"] == 270 and isinstance(value, bytes):
            value = value.decode(errors="ignore")
        return value

    patched_valueof.__monkey_patched__ = True  # type: ignore

    warnings.warn(
        "Monkey patching `tifffile.tifffile.TiffTags` to recover non-ascii description tags.\n"
        "IT IS NOT RECOMMENDED TO USE THIS WORKAROUND to load files that contain non-ascii\n"
        "characters in their image description tags. To fix the WSI File, please run:\n"
        "$ python -m tiffslide.repair fix-description-tag-encoding <wsi-file>\n",
        RuntimeWarning,
        stacklevel=2,
    )

    # monkey patch the TiffTags class
    TiffTags.valueof = patched_valueof


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collection of helper scripts for broken wsi files"
    )
    subparsers = parser.add_subparsers(required=True)

    parser0 = subparsers.add_parser("fix-description-tag-encoding")
    parser0.add_argument("filename", help="local filename for wsi")
    parser0.add_argument("--no-dry-run", action="store_true", help="modify wsi")
    parser0.set_defaults(func=fix_description_tag_encoding)

    args = parser.parse_args()
    kw = vars(args).copy()
    del kw["func"]

    raise SystemExit(args.func(**kw))
