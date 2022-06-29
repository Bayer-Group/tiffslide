import os

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
    pytestmark = pytest.mark.compat

    for key, fns in _FILES.items():
        for fn in fns:
            k = f"{key}-{fn.split('/')[1]}"
            FILES[k] = os.path.join(OPENSLIDE_TESTDATA_DIR, fn)


MODULES = ["tiffslide", "openslide"]


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


def test_level_count(ts_slide, os_slide):
    assert ts_slide.level_count == os_slide.level_count


def test_level_dimensions(ts_slide, os_slide):
    assert ts_slide.level_dimensions == os_slide.level_dimensions


def test_level_downsamples(ts_slide, os_slide):
    np.testing.assert_allclose(
        ts_slide.level_downsamples,
        os_slide.level_downsamples,
        rtol=1e-5,
    )
