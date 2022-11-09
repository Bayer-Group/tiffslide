import importlib
import os
import platform
import random
import subprocess
import sys
import warnings
from getpass import getpass
from itertools import product

import numpy as np
import pytest

OPENSLIDE_TESTDATA_DIR = os.getenv("OPENSLIDE_TESTDATA_DIR", None)
FILES = {
    "svs": "Aperio/CMU-2.svs",
    "generic": "Generic-TIFF/CMU-1.tiff",
    "hamamatsu": "Hamamatsu/OS-3.ndpi",
    "leica": "Leica/Leica-2.scn",
    "ventana": "Ventana/OS-2.bif",
}
if OPENSLIDE_TESTDATA_DIR is None:
    pytestmark = pytest.mark.skip

else:
    for key, fn in FILES.items():
        FILES[key] = os.path.join(OPENSLIDE_TESTDATA_DIR, fn)


MODULES = ["tiffslide", "openslide"]


@pytest.fixture(scope="module")
def root_pw(request):
    capture = request.config.pluginmanager.getplugin("capturemanager")
    capture.suspend_global_capture(in_=True)
    if not sys.stdin.isatty() and sys.stdout.isatty():
        warnings.warn("DISK CACHE INVALIDATION NEEDS ROOT, but running non-interactive")
        root_pw = None
    else:
        root_pw = getpass(
            "\n\nPLEASE ENTER ROOT PW (empty disables disk cache invalidation): "
        )
    capture.resume_global_capture()
    yield root_pw


def try_to_clear_disk_cache(root_pass):
    env = os.environ.copy()
    if root_pw:  # type: ignore
        env["SUDO_PASSWORD"] = root_pass

    system = platform.system()
    if system == "Linux":
        # from: https://github.com/sdvillal/jagged/blob/6fc83aa11e/jagged/benchmarks/utils.py#L95-L118
        max_size = "1000G"
        drop_level = 3

        if 0 != os.system(f'vmtouch -e -f -q -m {max_size} "{OPENSLIDE_TESTDATA_DIR}"'):
            subprocess.run(
                r"""printf '%%s\n' "$SUDO_PASSWORD" """
                r"""| sudo -p "" -S -- sh -c 'sync && echo %d > /proc/sys/vm/drop_caches'"""
                % drop_level,
                env=env,
                check=True,
                shell=True,
            )

    elif system == "Darwin":
        subprocess.run(
            r"""printf '%s\n' "$SUDO_PASSWORD" | sudo -p "" -S -- sh -c 'sync && purge'""",
            env=env,
            check=True,
            shell=True,
        )

    elif system == "Windows":
        warnings.warn("not implemented for windows yet")


@pytest.fixture(
    params=[pytest.param((ft, m), id=f"{m}-{ft}") for ft, m in product(FILES, MODULES)]
)
def slide_with_tile_size(request, root_pw):
    """yield a slide together with the internal tile size"""
    file_type, module_name = request.param
    file_name = FILES[file_type]

    # get the corresponding module
    mod = importlib.import_module(module_name)
    open_slide = getattr(mod, "open_slide")

    try_to_clear_disk_cache(root_pw)

    slide = open_slide(file_name)

    # attach tile_size
    try:
        t_size = (
            int(slide.properties[f"{module_name}.level[0].tile-width"]),
            int(slide.properties[f"{module_name}.level[0].tile-height"]),
        )
    except KeyError:
        warnings.warn("recovering tile_size via tiffslide")
        ts_cls = getattr(importlib.import_module("tiffslide"), "TiffSlide")
        p = ts_cls(file_name).properties
        t_size = (
            int(p[f"tiffslide.level[0].tile-width"]),
            int(p[f"tiffslide.level[0].tile-height"]),
        )

    try:
        yield slide, t_size
    finally:
        slide.close()


def get_locations(slide, order, tsize):
    def _locations(s, ts, c_order):
        w, h = s.dimensions
        tw, th = ts

        out = []
        if c_order:
            for y in range(0, h, th):
                for x in range(0, w, tw):
                    out.append((x, y))
        else:
            for x in range(0, w, tw):
                for y in range(0, h, th):
                    out.append((x, y))
        return out

    if order == "seqrow":
        locations = _locations(slide, tsize, c_order=True)
    elif order == "seqcol":
        locations = _locations(slide, tsize, c_order=False)
    elif order == "random":
        locations = _locations(slide, tsize, c_order=True)
        random.seed(42)
        random.shuffle(locations)
    else:
        raise ValueError(order)
    return locations


@pytest.mark.parametrize("order", ["seqrow", "seqcol", "random"])
def test_read_tiles_as_pil(order, slide_with_tile_size, benchmark):
    slide, tsize = slide_with_tile_size
    locations = get_locations(slide, order, tsize)
    lociter = iter(locations[len(locations) // 2 :])

    def setup():
        return (next(lociter), 0, tsize), {}

    read_tile = slide.read_region

    benchmark.pedantic(read_tile, setup=setup, rounds=64, iterations=1, warmup_rounds=1)


@pytest.mark.parametrize("order", ["seqrow", "seqcol", "random"])
def test_read_tiles_as_numpy(order, slide_with_tile_size, benchmark):
    slide, tsize = slide_with_tile_size
    locations = get_locations(slide, order, tsize)
    lociter = iter(locations[len(locations) // 2 :])

    def setup():
        return (next(lociter), 0, tsize), {}

    _read_tile = slide.read_region
    if slide.__class__.__name__ == "OpenSlide":

        def read_tile_np(loc, level, size):
            return np.array(_read_tile(loc, level, size))

    elif slide.__class__.__name__ == "TiffSlide":

        def read_tile_np(loc, level, size):
            return _read_tile(loc, level, size, as_array=True)

    else:
        raise NotImplementedError("unknown class")

    benchmark.pedantic(
        read_tile_np, setup=setup, rounds=64, iterations=1, warmup_rounds=1
    )
