import os

import pytest
import importlib

import tiffslide
from tiffslide import TiffSlide
from tiffslide import TiffFileError


def test_image_detect_format(svs_small):
    assert TiffSlide.detect_format(svs_small) == "aperio"


def test_image_open_incorrect_path():
    with pytest.raises(FileNotFoundError):
        TiffSlide('i-do-not-exist')


def test_image_open_unsupported_image(tmp_path):
    f = tmp_path.joinpath("test_file.svs")
    f.write_text("123")
    with pytest.raises(TiffFileError):
        TiffSlide(f)


def test_image_open(svs_small):
    TiffSlide(svs_small)


def test_image_context_manager(svs_small):
    with TiffSlide(svs_small) as t:
        # trigger full context
        _ = t.ts_zarr_grp


def test_image_repr(svs_small):
    assert os.path.basename(os.fspath(svs_small)) in repr(TiffSlide(svs_small))


def test_image_dimensions(svs_small):
    assert TiffSlide(svs_small).dimensions == (2220, 2967)


def test_image_level_count(svs_small):
    assert TiffSlide(svs_small).level_count == 1


def test_image_level_dimensions(svs_small):
    assert TiffSlide(svs_small).level_dimensions == ((2220, 2967), )


def test_image_level_downsamples(svs_small):
    assert TiffSlide(svs_small).level_downsamples == (1.0, )


def test_image_properties(svs_small, svs_small_props):
    assert TiffSlide(svs_small).properties == svs_small_props


def test_image_get_best_level_for_downsample(svs_small):
    # single layer image...
    assert TiffSlide(svs_small).get_best_level_for_downsample(1.0) == 0
    assert TiffSlide(svs_small).get_best_level_for_downsample(2.0) == 0
    assert TiffSlide(svs_small).get_best_level_for_downsample(4.0) == 0


def test_image_read_region(svs_small):
    assert TiffSlide(svs_small).read_region((0, 0), 0, (2220, 2967)).size == (2220, 2967)


def test_image_get_thumbnail(svs_small):
    assert TiffSlide(svs_small).get_thumbnail((200, 200)).size == (150, 200)


def test_image_associated_images(svs_small):
    assoc = TiffSlide(svs_small).associated_images
    assert repr(assoc)
    assert len(assoc) == 3
    assert 'thumbnail' in assoc
    assert 'label' in assoc
    assert 'macro' in assoc
    for key in assoc:
        img = assoc[key]
        assert img.size


# === test aliases and fallbacks ========================================

def test_compat_open_slide(svs_small):
    assert isinstance(tiffslide.open_slide(svs_small), TiffSlide)


@pytest.mark.parametrize(
    "exc_name", [
        "OpenSlideUnsupportedFormatError",
        "OpenSlideError",
    ]
)
def test_compat_alias_exception(exc_name):
    with pytest.warns(UserWarning):
        cls = getattr(importlib.import_module('tiffslide'), exc_name)
    assert cls is TiffFileError


@pytest.mark.parametrize(
    "cls_name", [
        "OpenSlide",
        "ImageSlide",
    ]
)
def test_compat_alias_tiffslide(cls_name):
    with pytest.warns(UserWarning):
        cls = getattr(importlib.import_module('tiffslide'), cls_name)
    assert cls is TiffSlide


def test_compat_unsupported_abstractslide():
    with pytest.warns(UserWarning):
        with pytest.raises(ImportError):
            from tiffslide import AbstractSlide

