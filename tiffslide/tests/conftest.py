import hashlib
import pathlib
import shutil
import urllib.request

import pytest

# openslide aperio test images
IMAGES_BASE_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/"


def md5(fn):
    m = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()


@pytest.fixture(scope='session')
def svs_small():
    """download the smallest aperio test image svs"""
    small_image = "CMU-1-Small-Region.svs"
    small_image_md5 = "1ad6e35c9d17e4d85fb7e3143b328efe"
    data_dir = pathlib.Path(__file__).parent / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    img_fn = data_dir / small_image

    if not img_fn.is_file():
        # download svs from openslide test images
        url = IMAGES_BASE_URL + small_image
        with urllib.request.urlopen(url) as response, open(img_fn, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    if md5(img_fn) != small_image_md5:  # pragma: no cover
        shutil.rmtree(img_fn)
        pytest.fail("incorrect md5")
    else:
        yield img_fn.absolute()


@pytest.fixture(scope='session')
def svs_small_props():
    yield {
        'tiff.ImageDescription': (
            'Aperio Image Library v11.2.1 \r\n'
            '46000x32914 [42673,5576 2220x2967] (240x240) '
            'JPEG/RGB Q=30;Aperio Image Library v10.0.51\r\n'
            '46920x33014 [0,100 46000x32914] (256x256) JPEG/RGB '
            'Q=30|AppMag = 20|StripeWidth = 2040|ScanScope ID = '
            'CPAPERIOCS|Filename = CMU-1|Date = 12/29/09|Time = '
            '09:59:15|User = '
            'b414003d-95c6-48b0-9369-8010ed517ba7|Parmset = USM '
            'Filter|MPP = 0.4990|Left = 25.691574|Top = '
            '23.449873|LineCameraSkew = '
            '-0.000424|LineAreaXOffset = '
            '0.019265|LineAreaYOffset = -0.000313|Focus Offset = '
            '0.000000|ImageID = 1004486|OriginalWidth = '
            '46920|Originalheight = 33014|Filtered = '
            '5|OriginalWidth = 46000|OriginalHeight = 32914'
        ),
        'tiffslide.background-color': None,
        'tiffslide.bounds-height': None,
        'tiffslide.bounds-width': None,
        'tiffslide.bounds-x': None,
        'tiffslide.bounds-y': None,
        'tiffslide.comment': (
            'Aperio Image Library v11.2.1 \r\n'
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
            '= 46000|OriginalHeight = 32914'
        ),
        'tiffslide.level[0].downsample': 1.0,
        'tiffslide.level[0].height': 2967,
        'tiffslide.level[0].tile-height': 240,
        'tiffslide.level[0].tile-width': 240,
        'tiffslide.level[0].width': 2220,
        'tiffslide.mpp-x': None,
        'tiffslide.mpp-y': None,
        'tiffslide.objective-power': None,
        'tiffslide.quickhash-1': None,
        'tiffslide.vendor': 'aperio',
    }
