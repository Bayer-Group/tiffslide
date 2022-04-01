import importlib
import os
import subprocess
import sys
from textwrap import dedent

import fsspec
import numpy as np
import pytest

import tiffslide
from tiffslide import TiffFileError
from tiffslide import TiffSlide


@pytest.fixture
def slide(wsi_file):
    yield TiffSlide(wsi_file)


def test_image_detect_format(wsi_file):
    fmt = TiffSlide.detect_format(wsi_file)
    assert fmt is not None
    assert isinstance(fmt, str)


def test_image_open_incorrect_path():
    with pytest.raises(FileNotFoundError):
        TiffSlide("i-do-not-exist")


def test_image_open_unsupported_image(tmp_path):
    f = tmp_path.joinpath("test_file.svs")
    f.write_text("123")
    with pytest.raises(TiffFileError):
        TiffSlide(f)


def test_image_open(wsi_file):
    TiffSlide(wsi_file)


def test_image_context_manager(slide):
    with slide as t:
        # trigger full context
        _ = t.ts_zarr_grp


def test_image_repr(wsi_file, slide):
    assert os.path.basename(os.fspath(wsi_file)) in repr(slide)


def test_image_dimensions(slide):
    assert isinstance(slide.dimensions, tuple)
    assert len(slide.dimensions) == 2
    assert slide.dimensions[0] > 0 and slide.dimensions[1] > 0


def test_image_level_count(slide):
    assert slide.level_count >= 1


def test_image_level_dimensions(slide):
    assert isinstance(slide.level_dimensions, tuple)
    assert len(slide.level_dimensions) >= 1
    assert slide.dimensions == slide.level_dimensions[0]
    x0, y0 = slide.dimensions
    for x1, y1 in slide.level_dimensions[1:]:
        assert x1 < x0 and y1 < y0
        x0, y0 = x1, y1


def test_image_level_downsamples(slide):
    assert isinstance(slide.level_downsamples, tuple)
    assert len(slide.level_downsamples) >= 1
    for ds in slide.level_downsamples:
        assert isinstance(ds, float)
        assert ds >= 1.0


def test_image_properties(slide, svs_small_props):
    if slide.ts_tifffile.filename == "CMU-1-Small-Region.svs":
        assert slide.properties == svs_small_props
    else:
        pytest.skip("needs examples")


def test_image_get_best_level_for_downsample(slide):
    assert slide.get_best_level_for_downsample(1.0) == 0
    lvl = 0
    for ds in np.arange(0, 10, 0.5):
        lvl_new = slide.get_best_level_for_downsample(ds)
        assert lvl <= lvl_new
        lvl = lvl_new


def test_image_read_region(slide):
    assert slide.read_region((0, 0), 0, (2220, 2967)).size == (2220, 2967)


def test_image_read_region_as_array(slide):
    assert slide.read_region((0, 0), 0, (2220, 2967), as_array=True).shape[:2] == (
        2967,
        2220,
    )


@pytest.mark.parametrize("use_embedded", [True, False])
def test_image_get_thumbnail(slide, use_embedded):
    thumb = slide.get_thumbnail((200, 200), use_embedded=use_embedded)
    assert max(thumb.size) == 200


def test_image_associated_images(slide):
    assoc = slide.associated_images
    assert repr(assoc)
    assert len(assoc) >= 0
    if slide.ts_tifffile.filename.endswith(".svs"):
        assert "thumbnail" in assoc
        assert "label" in assoc
        assert "macro" in assoc
        for key in assoc:
            img = assoc[key]
            assert img.size


def test_tiffslide_from_fsspec_buffer(wsi_file_urlpath):
    with fsspec.open(wsi_file_urlpath) as f:
        slide = TiffSlide(f)
        _ = slide.get_thumbnail((200, 200))


def test_tiffslide_from_fsspec_openfile(wsi_file_urlpath):
    of = fsspec.open(wsi_file_urlpath)
    slide = TiffSlide(of)
    _ = slide.get_thumbnail((200, 200))


def test_tiffslide_from_fsspec_urlpath(wsi_file_urlpath):
    slide = TiffSlide(wsi_file_urlpath)
    _ = slide.get_thumbnail((200, 200))


def test_tiffslide_reject_unsupported_file():
    with pytest.raises(ValueError):
        TiffSlide(dict())  # type: ignore


# === test aliases and fallbacks ========================================


def test_compat_open_slide(wsi_file):
    assert isinstance(tiffslide.open_slide(wsi_file), TiffSlide)


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
    assert issubclass(cls, TiffSlide)


def test_compat_unsupported_abstractslide():
    with pytest.warns(UserWarning):
        with pytest.raises(ImportError):
            from tiffslide import AbstractSlide

            _ = AbstractSlide


def test_thread_safety(wsi_file, tmp_path):
    """reproduce threading issue reported in:

    https://github.com/bayer-science-for-a-better-life/tiffslide/issues/14

    """
    bug_py = tmp_path.joinpath("threading_bug.py")
    bug_py.write_text(
        dedent(
            f"""\
            import tiffslide
            from multiprocessing.pool import ThreadPool

            def read_region(slide):
                _ = slide.read_region((0, 0), 0, (10, 10))

            # number of threads
            num_threads = 8

            ts = tiffslide.TiffSlide({os.fspath(wsi_file)!r})

            with ThreadPool(num_threads) as pool:
                pool.starmap(read_region, [(ts, ) for _ in range(num_threads*2)])
            """
        )
    )
    for _ in range(10):
        out = subprocess.run([sys.executable, bug_py], capture_output=True)
        assert out.returncode == 0, out.stderr.decode()


def test_non_tiff_fallback(jpg_file):
    # noinspection PyProtectedMember
    from tiffslide import open_slide

    ts = open_slide(jpg_file)
    assert ts.properties
    assert ts.get_thumbnail((10, 10))
    assert ts.associated_images == {}
    assert ts.read_region((0, 0), 0, (10, 10))


def test_image_read_region_out_of_area(slide):
    assert slide.read_region((-100, -100), 0, (50, 50), as_array=True, padding=True).shape[:2] == (50, 50)
    assert slide.read_region((-100, -100), 100, (50, 50), as_array=True, padding=True).shape[:2] == (50, 50)
    assert slide.read_region((-50, -50), 0, (100, 100), as_array=True, padding=True).shape[:2] == (100, 100)
    with pytest.raises(AssertionError):
        slide.read_region((-50, -50), 0, (50, 50), as_array=True, padding=False)
    assert slide.read_region((-50, -50), 0, (100, 100), as_array=True, padding=False).shape[:2] == (50, 50)
