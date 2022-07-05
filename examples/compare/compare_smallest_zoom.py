"""small helper script to compare the smallest zoom between openslide and tiffslide"""

from pathlib import Path
from pprint import pprint

import numpy as np
from openslide import OpenSlide
from PIL import Image

from tiffslide import TiffSlide


def write_lvl(fn):

    ts_slide = TiffSlide(fn)
    os_slide = OpenSlide(fn)

    print("ts ds:", ts_slide.level_downsamples)
    print("os ds:", os_slide.level_downsamples)

    c = ts_slide.properties.get("tiffslide.series-composition")
    if c:
        pprint(c)

    ts_lvl = ts_slide.level_count - 1
    os_lvl = os_slide.level_dimensions.index(ts_slide.level_dimensions[ts_lvl])
    size = ts_slide.level_dimensions[ts_lvl]

    ts_img = ts_slide.read_region((0, 0), ts_lvl, size)
    os_img = os_slide.read_region((0, 0), os_lvl, size)

    _p = Path(fn)
    ts_fn = f"{_p.stem}-{_p.suffix}_lvl{ts_lvl}_ts.png"
    os_fn = f"{_p.stem}-{_p.suffix}_lvl{ts_lvl}_os.png"
    err_fn = f"{_p.stem}-{_p.suffix}_lvl{ts_lvl}_err.png"

    ts_arr = np.array(ts_img)
    os_arr = np.array(os_img)
    os_arr = os_arr[:, :, :3]

    arr = np.abs(ts_arr.astype(int) - os_arr.astype(int))
    out = np.zeros_like(ts_arr, dtype=int)
    out = np.array(
        (out + (arr[:, :, :] * np.array([32, 32, 32]).reshape((1, 1, 3)))),
        dtype=np.uint8,
    )
    print("max difference:", np.max(arr))

    Image.fromarray(out).save(err_fn)

    ts_img.save(ts_fn)
    Image.fromarray(os_arr).save(os_fn)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("provide single filename")
        raise SystemExit(1)

    print(f"storing {sys.argv[1]}")
    write_lvl(sys.argv[1])
    print("done")
