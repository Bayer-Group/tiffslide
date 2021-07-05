import os

import fsspec
import pytest
import importlib

import tiffslide
from tiffslide import TiffSlide
from tiffslide import TiffFileError


@pytest.fixture
def slide(svs_small):
    yield TiffSlide(svs_small)


def test_image_detect_format(svs_small):
    assert TiffSlide.detect_format(svs_small) == "aperio"


def test_image_open_incorrect_path():
    with pytest.raises(FileNotFoundError):
        TiffSlide("i-do-not-exist")


def test_image_open_unsupported_image(tmp_path):
    f = tmp_path.joinpath("test_file.svs")
    f.write_text("123")
    with pytest.raises(TiffFileError):
        TiffSlide(f)


def test_image_open(svs_small):
    TiffSlide(svs_small)


def test_image_context_manager(slide):
    with slide as t:
        # trigger full context
        _ = t.ts_zarr_grp


def test_image_repr(svs_small, slide):
    assert os.path.basename(os.fspath(svs_small)) in repr(slide)


def test_image_dimensions(slide):
    assert slide.dimensions == (2220, 2967)


def test_image_level_count(slide):
    assert slide.level_count == 1


def test_image_level_dimensions(slide):
    assert slide.level_dimensions == ((2220, 2967),)


def test_image_level_downsamples(slide):
    assert slide.level_downsamples == (1.0,)


def test_image_properties(slide, svs_small_props):
    assert slide.properties == svs_small_props


def test_image_get_best_level_for_downsample(slide):
    # single layer image...
    assert slide.get_best_level_for_downsample(1.0) == 0
    assert slide.get_best_level_for_downsample(2.0) == 0
    assert slide.get_best_level_for_downsample(4.0) == 0


def test_image_read_region(slide):
    assert slide.read_region((0, 0), 0, (2220, 2967)).size == (
        2220,
        2967
    )


def test_image_get_thumbnail(slide):
    assert slide.get_thumbnail((200, 200)).size == (150, 200)


def test_image_associated_images(slide):
    assoc = slide.associated_images
    assert repr(assoc)
    assert len(assoc) == 3
    assert "thumbnail" in assoc
    assert "label" in assoc
    assert "macro" in assoc
    for key in assoc:
        img = assoc[key]
        assert img.size


def test_tiffslide_from_fsspec(svs_small_urlpath):
    with fsspec.open(svs_small_urlpath) as f:
        slide = TiffSlide(f)
        _ = slide.get_thumbnail((200, 200))


# === test aliases and fallbacks ========================================
def test_compat_open_slide(svs_small):
    assert isinstance(tiffslide.open_slide(svs_small), TiffSlide)


@pytest.mark.parametrize(
    "exc_name",
    [
        "OpenSlideUnsupportedFormatError",
        "OpenSlideError",
    ],
)
def test_compat_alias_exception(exc_name):
    with pytest.warns(UserWarning):
        cls = getattr(importlib.import_module("tiffslide"), exc_name)
    assert cls is TiffFileError


@pytest.mark.parametrize(
    "cls_name",
    [
        "OpenSlide",
        "ImageSlide",
    ],
)
def test_compat_alias_tiffslide(cls_name):
    with pytest.warns(UserWarning):
        cls = getattr(importlib.import_module("tiffslide"), cls_name)
    assert cls is TiffSlide


def test_compat_unsupported_abstractslide():
    with pytest.warns(UserWarning):
        with pytest.raises(ImportError):
            from tiffslide import AbstractSlide
            _ = AbstractSlide
