import hashlib
import os
import pathlib
import shutil
import urllib.request
from itertools import cycle
from itertools import groupby
from itertools import islice
from operator import attrgetter

import fsspec
import pytest

# openslide aperio test images
IMAGES_BASE_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/"


def md5(fn):
    m = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()


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
        paths = [pytest.param(None, id="CMU-1-Small-Region.svs")]
    return paths


@pytest.fixture(scope="session", params=_wsi_files())
def wsi_file(request):
    """download the smallest aperio test image svs or use local"""
    if request.param is None:
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
    else:
        img_fn = request.param

    yield img_fn.absolute()


@pytest.fixture
def wsi_file_urlpath(wsi_file):
    if wsi_file.stat().st_size > 100 * 1024 * 1024:
        pytest.skip(msg="reduce ram usage of tests")
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
