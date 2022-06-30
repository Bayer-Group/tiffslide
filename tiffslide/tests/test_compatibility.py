import itertools
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

OPENSLIDE_TESTDATA_DIR = os.getenv("OPENSLIDE_TESTDATA_DIR", None)
_FILES = {
    "svs": [
        "Aperio/CMU-1-JP2K-33005.svs",
        "Aperio/CMU-1-Small-Region.svs",
        "Aperio/CMU-1.svs",
        "Aperio/CMU-2.svs",
        "Aperio/CMU-3.svs",
        "Aperio/JP2K-33003-1.svs",
        "Aperio/JP2K-33003-2.svs",
    ],
    "generic": [
        "Generic-TIFF/CMU-1.tiff",
    ],
    "hamamatsu": [
        "Hamamatsu/CMU-1.ndpi",
        "Hamamatsu/CMU-2.ndpi",
        "Hamamatsu/CMU-3.ndpi",
        "Hamamatsu/OS-1.ndpi",
        "Hamamatsu/OS-2.ndpi",
        "Hamamatsu/OS-3.ndpi",
    ],
    "leica": [
        "Leica/Leica-1.scn",
        "Leica/Leica-2.scn",
        # todo: these two can't be opened by OpenSlide
        # "Leica/Leica-3.scn",
        # "Leica/Leica-Fluorescence-1.scn",
    ],
    "ventana": [
        "Ventana/OS-1.bif",
        "Ventana/OS-2.bif",
    ],
}
FILES = {}
if OPENSLIDE_TESTDATA_DIR is None:
    pytestmark = pytest.mark.skip

else:
    pytestmark = pytest.mark.compat  # type: ignore

    for key, fns in _FILES.items():
        for fn in fns:
            k = f"{key}-{fn.split('/')[1]}"
            FILES[k] = os.path.join(OPENSLIDE_TESTDATA_DIR, fn)


MODULES = ["tiffslide", "openslide"]


def matches(fn, vendor=None, filename=None, ext=None):
    if vendor is not None:
        return any(Path(x).name == Path(fn).name for x in _FILES[vendor])
    elif filename is not None:
        return fn == file_name
    elif ext is not None:
        return Path(fn).suffix == f".{ext}"
    else:
        raise ValueError("all none")


@pytest.fixture(params=list(FILES))
def file_name(request):
    yield FILES[request.param]


@pytest.fixture()
def ts_slide(file_name):
    from tiffslide import TiffSlide

    yield TiffSlide(file_name)


@pytest.fixture()
def os_slide(file_name):
    from openslide import OpenSlide

    yield OpenSlide(file_name)


def test_dimensions(ts_slide, os_slide):
    assert ts_slide.dimensions == os_slide.dimensions


def test_level_count(ts_slide, os_slide, file_name):
    if matches(file_name, vendor="hamamatsu"):
        pytest.xfail("'ndpi' no computed levels")

    assert ts_slide.level_count == os_slide.level_count


def test_level_dimensions(ts_slide, os_slide, file_name):
    if matches(file_name, vendor="hamamatsu"):
        pytest.xfail("'ndpi' no computed levels")

    assert ts_slide.level_dimensions == os_slide.level_dimensions


def test_level_downsamples(ts_slide, os_slide, file_name):
    if matches(file_name, vendor="hamamatsu"):
        pytest.xfail("'ndpi' no computed levels")

    np.testing.assert_allclose(
        ts_slide.level_downsamples,
        os_slide.level_downsamples,
        rtol=1e-5,
    )


def test_strict_subset_level_dimensions(ts_slide, os_slide):
    # test that all available tiffslide levels are in os_slide
    assert set(ts_slide.level_dimensions).issubset(os_slide.level_dimensions)


def test_strict_subset_level_downsamples(ts_slide, os_slide):
    # test that all available tiffslide levels are in os_slide
    for ds in ts_slide.level_downsamples:
        assert any(
            math.isclose(ds, x, rel_tol=1e-5) for x in os_slide.level_downsamples
        )


def test_read_region_equality_level_min(ts_slide, os_slide, file_name):
    exact = True
    if "JP2K-33003" in file_name:
        warnings.warn(
            f"JP2K file {file_name} is tested to be almost equal (not exactly equal)!",
            stacklevel=2,
        )
        exact = False

    width, height = ts_slide.dimensions

    ws = range(0, width, width // 5)
    hs = range(0, height, height // 5)
    for loc in itertools.product(ws[:-1], hs[:-1]):
        ts_img = ts_slide.read_region(loc, 0, (128, 128))
        os_img = os_slide.read_region(loc, 0, (128, 128))

        ts_arr = np.array(ts_img)
        os_arr = np.array(os_img)
        # np.testing.assert_equal(os_arr[:, :, 3], 255)
        os_arr = os_arr[:, :, :3]

        if exact:
            np.testing.assert_equal(ts_arr, os_arr)
        else:
            np.testing.assert_allclose(ts_arr, os_arr, atol=1, rtol=0)


def test_read_region_equality_level_common_max(ts_slide, os_slide, file_name):
    exact = True
    if "JP2K-33003" in file_name:
        warnings.warn(
            f"JP2K file {file_name} is tested to be almost equal (not exactly equal)!",
            stacklevel=2,
        )
        exact = False

    ts_lvl = ts_slide.level_count - 1
    os_lvl = os_slide.level_dimensions.index(ts_slide.level_dimensions[ts_lvl])
    size = ts_slide.level_dimensions[ts_lvl]

    if size[0] >= 5000 or size[1] >= 5000:
        pytest.skip("smallest common level is too big...")

    ts_img = ts_slide.read_region((0, 0), ts_lvl, size)
    os_img = os_slide.read_region((0, 0), os_lvl, size)

    ts_arr = np.array(ts_img)
    os_arr = np.array(os_img)
    os_arr = os_arr[:, :, :3]

    max_difference = np.max(np.abs(ts_arr.astype(int) - os_arr.astype(int)))
    assert max_difference <= (0 if exact else 1)
