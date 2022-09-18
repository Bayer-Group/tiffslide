from __future__ import annotations

import enum
import hashlib
import os
import pathlib
import shutil
import sys
import urllib.request
from itertools import cycle
from itertools import groupby
from itertools import islice
from operator import attrgetter

import fsspec
import numpy as np
import pytest
import tifffile
from imagecodecs import imwrite

# openslide aperio test images
IMAGES_BASE_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/"

try:
    APERIO_JP2000_RGB = tifffile.COMPRESSION.APERIO_JP2000_RGB
except AttributeError:
    # python3.7
    APERIO_JP2000_RGB = tifffile.TIFF.COMPRESSION.APERIO_JP2000_RGB


def md5(fn):
    m = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()


class TestImageType(enum.Enum):
    DOWNLOAD_SMALLEST_CMU = enum.auto()
    GENERATE_PYRAMIDAL_IMG = enum.auto()
    GENERATE_PYRAMIDAL_1CH_16B_SVS = enum.auto()


def _wsi_files():
    """gather images"""

    local_test_images = os.environ.get("TIFFSLIDE_TEST_IMAGES", None)
    if local_test_images:
        _paths = []
        for img in pathlib.Path(local_test_images).glob("**/*"):
            # todo:
            #  - .zip files need to be tested
            #  - .zvi files need to be handled
            if img.suffix in {".svs", ".ndpi", ".scn", ".bif", ".tiff"}:
                _paths.append(img.absolute())

        def roundrobin(iterables):
            """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
            num_active = len(iterables)
            nexts = cycle(iter(it).__next__ for it in iterables)
            while num_active:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    # Remove the iterator we just exhausted from the cycle.
                    num_active -= 1
                    nexts = cycle(islice(nexts, num_active))

        get_suffix = attrgetter("suffix")
        paths = [
            pytest.param(img, id=img.name)
            for img in roundrobin(
                [
                    sorted(list(group), key=lambda x: int(x.stat().st_size))
                    for key, group in groupby(
                        sorted(_paths, key=get_suffix), key=get_suffix
                    )
                ]
            )
        ]

    else:
        paths = [
            pytest.param(
                TestImageType.DOWNLOAD_SMALLEST_CMU, id="CMU-1-Small-Region.svs"
            ),
            pytest.param(
                TestImageType.GENERATE_PYRAMIDAL_IMG, id="generated-pyramidal"
            ),
            pytest.param(
                TestImageType.GENERATE_PYRAMIDAL_1CH_16B_SVS,
                id="generated-pyramidal-1ch-16b-svs",
            ),
        ]
    return paths


def _write_test_tiff(
    pth: os.PathLike[str],
    size: tuple[int, int],
    tile_size: int = 128,
    mpp: float = 0.5,
) -> None:
    def _num_pyramids(ldim: int, tsize: int) -> int:
        assert ldim > 0 and tsize > 0
        n = 0
        while ldim > tsize:
            ldim //= 2
            n += 1
        return n

    data = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)

    with tifffile.TiffWriter(pth, bigtiff=True, ome=True) as tif:
        im_height, im_width, _ = data.shape
        options0 = {}
        metadata = {}
        if mpp:
            metadata = {
                "PhysicalSizeX": mpp,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": mpp,
                "PhysicalSizeYUnit": "µm",
            }
            options0["resolution"] = (1.0 / mpp, 1.0 / mpp, "MICROMETER")
        options = dict(
            tile=(tile_size, tile_size),
            photometric="rgb",
            compression="jpeg",
            metadata=metadata,
        )
        num_pyramids = _num_pyramids(max(im_height, im_width), tile_size)
        tif.write(
            data,
            subifds=num_pyramids,
            **options0,
            **options,
        )
        lvl_data = data
        for _ in range(num_pyramids):
            lvl_data = lvl_data[::2, ::2, :]
            tif.write(
                lvl_data,
                subfiletype=1,
                **options,
            )


def _write_test_svs_with_axes_YX_dtype_uint16(pth):
    """write a special tiff in svs format with single channel levels and dtype uint16

    author: One-Sixth https://github.com/One-sixth
    reported: https://github.com/bayer-science-for-a-better-life/tiffslide/issues/46
    """

    # fake data
    def gen_im(size_hw):
        while True:
            im = np.full(size_hw, 255, np.uint16)
            yield im

    thumbnail_im = np.zeros([762, 762], dtype=np.uint16)
    label_im = np.zeros([762, 762], dtype=np.uint16)
    macro_im = np.zeros([762, 762], dtype=np.uint16)

    # fake descriptions
    svs_desc = "Aperio Image Library Fake\nABC |AppMag = {mag}|Filename = {filename}|MPP = {mpp}"
    label_desc = "Aperio Image Library Fake\nlabel {W}x{H}"
    macro_desc = "Aperio Image Library Fake\nmacro {W}x{H}"

    tile_hw = (512, 512)
    # multi resolution
    multi_hw = [(10240, 10240), (5120, 5120), (2560, 2560)]
    mpp = 0.25
    mag = 40
    filename = "ASD"
    if sys.version_info >= (3, 8):
        resolution_kw = {
            "resolution": (10000 / mpp, 10000 / mpp),
            "resolutionunit": "CENTIMETER",
        }
    else:
        resolution_kw = {"resolution": (10000 / mpp, 10000 / mpp, "CENTIMETER")}

    # write to svs format
    with tifffile.TiffWriter(pth, bigtiff=True) as tif:
        kwargs = {
            "subifds": 0,
            "photometric": "MINISBLACK",
            "compression": APERIO_JP2000_RGB,
            "dtype": np.uint16,
            "metadata": None,
        }

        # write level 0
        tif.write(
            data=gen_im(tile_hw),
            shape=(*multi_hw[0], 1),
            tile=tile_hw[::-1],
            description=svs_desc.format(mag=mag, filename=filename, mpp=mpp),
            **resolution_kw,
            **kwargs,
        )
        # write thumbnail image
        tif.write(data=thumbnail_im, description="", **kwargs)

        # write level 1 to N
        for hw in multi_hw[1:]:
            tif.write(
                data=gen_im(tile_hw),
                shape=(*hw, 1),
                tile=tile_hw[::-1],
                description="",
                **resolution_kw,
                **kwargs,
            )

        # write label
        tif.write(
            data=label_im,
            subfiletype=1,
            description=label_desc.format(W=label_im.shape[1], H=label_im.shape[0]),
            **kwargs,
        )
        # write marco
        tif.write(
            data=macro_im,
            subfiletype=9,
            description=macro_desc.format(W=macro_im.shape[1], H=macro_im.shape[0]),
            **kwargs,
        )


def _write_test_jpg(
    pth: os.PathLike[str],
    size: tuple[int, int],
):
    arr = np.random.randint(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
    imwrite(pth, arr)


@pytest.fixture(scope="session")
def jpg_file(tmp_path_factory):
    jpg_pth = tmp_path_factory.mktemp("test_images").joinpath("test.jpg")
    _write_test_jpg(jpg_pth, (1024, 1024))
    yield jpg_pth


@pytest.fixture(scope="session", params=_wsi_files())
def wsi_file(request, tmp_path_factory):
    """download the smallest aperio test image svs or use local"""
    if request.param == TestImageType.DOWNLOAD_SMALLEST_CMU:
        small_image = "CMU-1-Small-Region.svs"
        small_image_md5 = "1ad6e35c9d17e4d85fb7e3143b328efe"
        data_dir = pathlib.Path(__file__).parent / "data"

        data_dir.mkdir(parents=True, exist_ok=True)
        img_fn = data_dir / small_image

        if not img_fn.is_file():
            # download svs from openslide test images
            url = IMAGES_BASE_URL + small_image
            with urllib.request.urlopen(url) as response, open(
                img_fn, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)

        if md5(img_fn) != small_image_md5:  # pragma: no cover
            shutil.rmtree(img_fn)
            pytest.fail("incorrect md5")

    elif request.param == TestImageType.GENERATE_PYRAMIDAL_IMG:
        img_fn = tmp_path_factory.mktemp("_generated_test_tiffs").joinpath(
            "_small_pyramid.tiff"
        )
        _write_test_tiff(img_fn, (4096, 4096))

    elif request.param == TestImageType.GENERATE_PYRAMIDAL_1CH_16B_SVS:
        img_fn = tmp_path_factory.mktemp("_generated_test_tiffs").joinpath(
            "_small_pyramid_2.svs"
        )
        _write_test_svs_with_axes_YX_dtype_uint16(img_fn)

    else:
        img_fn = request.param

    yield img_fn.absolute()


@pytest.fixture
def wsi_file_urlpath(wsi_file):
    if wsi_file.stat().st_size > 100 * 1024 * 1024:
        pytest.skip("reduce ram usage of tests")
    urlpath = f"memory://{wsi_file.name}"
    fs: fsspec.AbstractFileSystem = fsspec.get_filesystem_class("memory")()
    of = fsspec.open(urlpath, mode="wb")
    with of as f:
        f.write(wsi_file.read_bytes())
    try:
        yield urlpath
    finally:
        fs.rm(wsi_file.name)


@pytest.fixture(scope="session")
def svs_small_props():
    yield {
        'aperio.AppMag': 20,
        'aperio.Date': '12/29/09',
        'aperio.Filename': 'CMU-1',
        'aperio.Filtered': 5,
        'aperio.Focus Offset': 0.0,
        'aperio.Header': 'Aperio Image Library v11.2.1 \r\n'
                         '46000x32914 [42673,5576 2220x2967] (240x240) JPEG/RGB Q=30'
                         ';Aperio Image Library v10.0.51\r\n'
                         '46920x33014 [0,100 46000x32914] (256x256) JPEG/RGB Q=30',
        'aperio.ImageID': 1004486,
        'aperio.Left': 25.691574,
        'aperio.LineAreaXOffset': 0.019265,
        'aperio.LineAreaYOffset': -0.000313,
        'aperio.LineCameraSkew': -0.000424,
        'aperio.MPP': 0.499,
        'aperio.OriginalHeight': 32914,
        'aperio.OriginalWidth': 46000,
        'aperio.Originalheight': 33014,
        'aperio.Parmset': 'USM Filter',
        'aperio.ScanScope ID': 'CPAPERIOCS',
        'aperio.StripeWidth': 2040,
        'aperio.Time': '09:59:15',
        'aperio.Top': 23.449873,
        'aperio.User': 'b414003d-95c6-48b0-9369-8010ed517ba7',
        'tiff.ImageDescription': 'Aperio Image Library v11.2.1 \r\n'
                                 '46000x32914 [42673,5576 2220x2967] (240x240) '
                                 'JPEG/RGB Q=30;Aperio Image Library v10.0.51\r\n'
                                 '46920x33014 [0,100 46000x32914] (256x256) JPEG/RGB '
                                 'Q=30|AppMag = 20|StripeWidth = '
                                 '2040|ScanScope ID = CPAPERIOCS|Filename = '
                                 'CMU-1|Date = 12/29/09|Time = 09:59:15|User = '
                                 'b414003d-95c6-48b0-9369-8010ed517ba7|Parmset = USM '
                                 'Filter|MPP = 0.4990|Left = 25.691574|Top = '
                                 '23.449873|LineCameraSkew = '
                                 '-0.000424|LineAreaXOffset = '
                                 '0.019265|LineAreaYOffset = -0.000313|Focus Offset = '
                                 '0.000000|ImageID = 1004486|OriginalWidth = '
                                 '46920|Originalheight = 33014|Filtered = '
                                 '5|OriginalWidth = 46000|OriginalHeight = 32914',
        'tiffslide.series-index': 0,
        'tiffslide.series-axes': 'YXS',
        'tiffslide.background-color': None,
        'tiffslide.bounds-height': None,
        'tiffslide.bounds-width': None,
        'tiffslide.bounds-x': None,
        'tiffslide.bounds-y': None,
        'tiffslide.comment': 'Aperio Image Library v11.2.1 \r\n'
                             '46000x32914 [42673,5576 2220x2967] (240x240) JPEG/RGB '
                             'Q=30;Aperio Image Library v10.0.51\r\n'
                             '46920x33014 [0,100 46000x32914] (256x256) JPEG/RGB '
                             'Q=30|AppMag = 20|StripeWidth = 2040|ScanScope ID = '
                             'CPAPERIOCS|Filename = CMU-1|Date = 12/29/09|Time = '
                             '09:59:15|User = '
                             'b414003d-95c6-48b0-9369-8010ed517ba7|Parmset = USM '
                             'Filter|MPP = 0.4990|Left = 25.691574|Top = '
                             '23.449873|LineCameraSkew = -0.000424|LineAreaXOffset = '
                             '0.019265|LineAreaYOffset = -0.000313|Focus Offset = '
                             '0.000000|ImageID = 1004486|OriginalWidth = '
                             '46920|Originalheight = 33014|Filtered = 5|OriginalWidth '
                             '= 46000|OriginalHeight = 32914',
        'tiffslide.level[0].downsample': 1.0,
        'tiffslide.level[0].height': 2967,
        'tiffslide.level[0].tile-height': 240,
        'tiffslide.level[0].tile-width': 240,
        'tiffslide.level[0].width': 2220,
        'tiffslide.mpp-x': 0.499,
        'tiffslide.mpp-y': 0.499,
        'tiffslide.objective-power': 20,
        'tiffslide.quickhash-1': None,
        'tiffslide.vendor': 'aperio',
    }  # fmt: skip
