import os
import platform

import pytest

from tiffslide import TiffSlide
from tiffslide._kerchunk import from_kerchunk
from tiffslide._kerchunk import to_kerchunk


@pytest.fixture
def slide(wsi_file):
    yield TiffSlide(wsi_file)


def test_to_kerchunk(slide, wsi_file):
    kc = to_kerchunk(slide, urlpath=wsi_file)
    assert kc["version"] == 1
    assert kc["gen"] == []
    if platform.system() != "Windows":
        assert os.fspath(wsi_file) in set(kc["templates"].values())
    assert kc["refs"]


def test_from_kerchunk(slide, wsi_file):
    kc = to_kerchunk(slide, urlpath=wsi_file)

    ts = from_kerchunk(kc)
    assert ts.properties
    assert ts.read_region((0, 0), 0, (100, 100), as_array=True).size > 0


def test_kerchunked_repr(slide, wsi_file):
    kc = to_kerchunk(slide, urlpath=wsi_file)

    ts = from_kerchunk(kc)
    assert repr(ts)
