"""small helper script to compare tiles between openslide and tiffslide"""

import itertools
from pathlib import Path

import numpy as np
from openslide import OpenSlide
from PIL import Image

from tiffslide import TiffSlide


def write_tiles(fn):

    ts_slide = TiffSlide(fn)
    os_slide = OpenSlide(fn)

    width, height = ts_slide.dimensions

    ws = range(0, width, width // 5)
    hs = range(0, height, height // 5)
    for loc in itertools.product(ws[:-1], hs[:-1]):
        ts_img = ts_slide.read_region(loc, 0, (512, 512))
        os_img = os_slide.read_region(loc, 0, (512, 512))

        _p = Path(fn)

        ts_fn = f"{_p.stem}-{_p.suffix}_x{loc[0]}y{loc[1]}_ts.png"
        os_fn = f"{_p.stem}-{_p.suffix}_x{loc[0]}y{loc[1]}_os.png"
        err_fn = f"{_p.stem}-{_p.suffix}_x{loc[0]}y{loc[1]}_zzz_err.png"

        ts_arr = np.array(ts_img)
        os_arr = np.array(os_img)
        # assert np.all(os_arr[:, :, 3] == 255)
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

    print(f"tiling {sys.argv[1]}")
    write_tiles(sys.argv[1])
    print("done")
