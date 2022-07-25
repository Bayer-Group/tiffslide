"""tiffslide.deepzoom

tiffslide.deepzoom is not intended to be used as a drop-in replacement
for openslide's DeepZoomGenerator. It's aiming to provide the basis for a
minimal overhead tile server for viewing tiled whole slide images in the
browser without having to re-tile the existing layers in the slide.

"""
import math
from io import BytesIO
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Sequence
from typing import Tuple
from typing import Union
from warnings import warn
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import SubElement

import fsspec
from PIL import Image
from PIL import ImageFile
from tifffile import TIFF
from tifffile import TiffFile
from tifffile import TiffPage

# improve robustness when encountering corrupted tiles
ImageFile.LOAD_TRUNCATED_IMAGES = True

# prevent pillow>=9.1.0 deprecation warning
try:
    _ANTIALIAS = Image.Resampling.LANCZOS
except AttributeError:
    _ANTIALIAS = Image.ANTIALIAS


def __getattr__(name):  # type: ignore
    if name == "DeepZoomGenerator":
        warn(
            "tiffslide.deepzoom does not provide a drop-in replacement for openslide's deepzoom generator"
        )
    raise AttributeError(name)


class MinimalComputeAperioDZGenerator:
    """Minimal-compute tiffslide Deep Zoom tile generator

    Note:
      - this is providing tiles of existing layers in the svs directly from disk
      - it would be a zero-compute DZG if we'd create a openseadragon TileSource
        that supports arbitrary (not 2x) spaced tile layers

    """

    def __init__(
        self, urlpath: Union[str, fsspec.core.OpenFile], **kwargs: Any
    ) -> None:
        self._kwargs = kwargs
        if isinstance(urlpath, fsspec.core.OpenFile):
            self._openfile = urlpath
        else:
            self._openfile = fsspec.open(urlpath, **kwargs)

        with self._openfile as f, TiffFile(f) as tiff:
            baseline = tiff.series[0]
            assert baseline.is_pyramidal  # aperio svs

            # store tile size and image level sizes
            assert baseline.keyframe.tilewidth == baseline.keyframe.tilelength
            self._tile_size = baseline.keyframe.tilewidth
            self._im_levels = tuple(lvl.shape[1::-1] for lvl in baseline.levels)

            # generate the levels for deep zoom
            dz_levels = (dz_lvl,) = [self._im_levels[0]]
            while dz_lvl[0] > 1 or dz_lvl[1] > 1:
                dz_lvl = tuple(max(1, int(math.ceil(z / 2))) for z in dz_lvl)
                dz_levels.append(dz_lvl)
            self._dz_levels = tuple(reversed(dz_levels))

            self._mapped_levels = {}
            for im_idx, im_lvl in enumerate(self._im_levels):
                for dz_idx, dz_lvl in enumerate(self._dz_levels):
                    if (
                        abs(im_lvl[0] - dz_lvl[0]) <= 1
                        and abs(im_lvl[1] - dz_lvl[1]) <= 1
                    ):
                        self._mapped_levels[dz_idx] = im_idx

            self._page_info = {}
            for lvl_idx, page_series in enumerate(baseline.levels):
                # extract the current page from page_series
                page: TiffPage
                (page,) = page_series.pages

                # more assumptions to ensure programmer sanity
                assert page.compression in {
                    TIFF.COMPRESSION.JPEG,
                    TIFF.COMPRESSION.APERIO_JP2000_YCBC,
                    TIFF.COMPRESSION.JPEG_2000_LOSSY,
                    TIFF.COMPRESSION.APERIO_JP2000_RGB,
                }
                assert page.is_tiled
                assert page.planarconfig == TIFF.PLANARCONFIG.CONTIG

                # calculate indices
                st_length, st_width = page.tilelength, page.tilewidth
                im_length, im_width = page.shaped[2:4]
                idx_width = (im_width + st_width - 1) // st_width
                idx_length = (im_length + st_length - 1) // st_length

                self._page_info[lvl_idx] = {
                    "jpeg_tables": page.jpegtables,
                    "idx_wh": (idx_width, idx_length),
                    "tile_wh": (st_width, st_length),
                    "image_wh": (im_width, im_length),
                    "offsets": page.dataoffsets,
                    "bytecounts": page.databytecounts,
                    "requires_rgb_color_fix": page.photometric == TIFF.PHOTOMETRIC.RGB,
                }

    @property
    def level_size(self) -> Dict[int, Tuple[int, int]]:
        """return a dict mapping deep zoom level to tile index size"""
        return {
            idx: (
                int(math.ceil(lvl[0] / self._tile_size)),
                int(math.ceil(lvl[1] / self._tile_size)),
            )
            for idx, lvl in enumerate(self._dz_levels)
        }

    def get_dzi(self) -> str:
        """return the dzi XML metadata"""
        # noinspection HttpUrlsUsage
        image = Element(
            "Image",
            TileSize=str(self._tile_size),
            Overlap="0",  # tiles in AperioSVS files have 0 overlap
            Format="jpeg",  # tiles are stored in jpeg format
            xmlns="http://schemas.microsoft.com/deepzoom/2008",
        )
        width, height = self._im_levels[0]
        SubElement(image, "Size", Width=str(width), Height=str(height))
        tree = ElementTree(element=image)

        with BytesIO() as buffer:
            tree.write(buffer, encoding="UTF-8")
            return buffer.getvalue().decode("UTF-8")

    def _read_svs_tile(self, svs_level: int, x: int, y: int) -> bytes:
        """return a single tile from an svs as a jpeg into a buffer"""
        info = self._page_info[svs_level]
        jpeg_tables = info["jpeg_tables"]
        (idx_width, idx_length) = info["idx_wh"]
        (st_width, st_length) = info["tile_wh"]
        (im_width, im_length) = info["image_wh"]
        dataoffsets = info["offsets"]
        databytecounts = info["bytecounts"]
        requires_rgb_color_fix = info["requires_rgb_color_fix"]

        if not ((0 <= x < idx_width) and (0 <= y < idx_length)):
            raise IndexError(
                f"({x}, {y}) level={svs_level} out of bounds: max ({idx_width}, {idx_length})"
            )

        # index for reading tile
        tile_index = y * idx_width + x

        with self._openfile as f:
            f.seek(dataoffsets[tile_index])
            data = f.read(databytecounts[tile_index])

        if jpeg_tables is not None:
            with BytesIO() as buffer:
                buffer.write(jpeg_tables[:-2])
                if requires_rgb_color_fix:
                    # to directly provide the stored tiles from disk, we need to fix that svs tiles
                    # use a jpeg colorspace that doesn't show up correctly in the browser if the default
                    # jpeg headers are used
                    # See https://stackoverflow.com/questions/8747904/extract-jpeg-from-tiff-file/9658206#9658206
                    buffer.write(
                        b"\xFF\xEE\x00\x0E\x41\x64\x6F\x62\x65\x00\x64\x80\x00\x00\x00\x00"
                    )  # colorspace fix
                buffer.write(data[2:])
                tile_data = buffer.getvalue()
        else:
            tile_data = data

        # the outer edges need to be cropped to be interpreted correctly by default openseadragon
        out_width = ((im_width - 1) % st_width) + 1 if x == idx_width - 1 else st_width
        out_length = (
            ((im_length - 1) % st_length) + 1 if y == idx_length - 1 else st_length
        )
        if out_width < st_width or out_length < st_length:  # cut output if needed
            with BytesIO(tile_data) as buffer:
                im = Image.open(buffer).crop((0, 0, out_width, out_length))
            with BytesIO() as buffer:
                im.save(buffer, format="JPEG")
                return buffer.getvalue()

        return tile_data

    def get_tile(self, level: int, x: int, y: int) -> bytes:
        """return the jpeg tile as bytes"""

        if level in self._mapped_levels:
            # FAST PATH:
            # -> there's a direct mapping to a svs tiled level
            # note: if we optimize the frontend, this will be the only path that's hit!
            svs_lvl = self._mapped_levels[level]
            return self._read_svs_tile(svs_level=svs_lvl, x=x, y=y)

        elif 8 <= level <= max(self._mapped_levels):
            # SLOW PATH:
            # -> the frontend requests a dzi layer that's not available in the svs
            # we need to compute new tiles from lower levels
            dst = Image.new("RGB", (2 * self._tile_size, 2 * self._tile_size))

            out_width = out_height = 0
            for ix, iy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                nx, ny = 2 * x + ix, 2 * y + iy

                try:
                    # the recursive call is only really a problem for the small (very low magnification)
                    # deep zoom tile levels, the bigger ones (high magnification) hit after one recursion
                    # fixme: we could optimize this by doing oneshot crop resizing from the svs thumbnail layer
                    data = self.get_tile(level + 1, nx, ny)
                except IndexError:
                    continue

                with BytesIO(data) as buffer:
                    im = Image.open(buffer)
                    if ix == 0:
                        out_height += im.height
                    if iy == 0:
                        out_width += im.width
                    dst.paste(im, (ix * self._tile_size, iy * self._tile_size))

            if out_width == 0 or out_height == 0:
                raise IndexError(
                    f"tile index ({x}, {y}) at INTERMEDIATE level={level} out of bounds"
                )

            elif (out_width, out_height) != dst.size:
                dst = dst.crop((0, 0, out_width, out_height))
                thumb_size = (
                    max(1, math.ceil(out_width / 2)),
                    max(1, math.ceil(out_height / 2)),
                )

            else:
                thumb_size = (self._tile_size, self._tile_size)

            dst.thumbnail(thumb_size, _ANTIALIAS)
            with BytesIO() as buffer:
                dst.save(buffer, format="JPEG")
                return buffer.getvalue()

        else:
            raise IndexError(f"requested level {level} invalid")


def _test_tifffile_timing() -> None:
    # build some load stats for a slide
    import random
    import sys
    import time
    from contextlib import contextmanager
    from operator import itemgetter

    if len(sys.argv) != 2:
        print("please provide a svs file as a cli argument")
        sys.exit(1)
    else:
        svs_fn = sys.argv[1]

    num_samples = 100

    @contextmanager
    def timer(label: str, samples: int = 1) -> Iterator[None]:
        t0 = time.time()
        yield
        avg = (time.time() - t0) / samples
        print(f"{label} took {avg} seconds")

    with timer("create dz"):
        dz = MinimalComputeAperioDZGenerator(svs_fn)
    # noinspection PyProtectedMember
    print("mapped levels:", dz._mapped_levels)

    for test_lvl, lvl_size in sorted(
        dz.level_size.items(), key=itemgetter(0), reverse=True
    ):
        if test_lvl < 8:
            print("skipping small levels")
            break

        test_idx: Sequence[int] = range(lvl_size[0] * lvl_size[1])
        if len(test_idx) > num_samples:
            test_idx = random.sample(test_idx, num_samples)

        with timer(f"accessing level {test_lvl}", samples=len(test_idx)):
            for flat_idx in test_idx:
                addr = flat_idx % lvl_size[0], flat_idx // lvl_size[0]
                dz.get_tile(test_lvl, addr[0], addr[1])

    print("done.")


if __name__ == "__main__":
    _test_tifffile_timing()
