import os

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
    pytestmark = pytest.mark.compat

    for key, fn in FILES.items():
        FILES[key] = os.path.join(OPENSLIDE_TESTDATA_DIR, fn)


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
    np.testing.assert_allclose(ts_slide.level_downsamples, os_slide.level_downsamples)
