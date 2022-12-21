"""Mirax support via zarr

__author__: Andreas Poehlmann
"""
from __future__ import annotations

import colorsys
import json
import os
import struct
from collections import defaultdict
from configparser import ConfigParser
from functools import cached_property
from io import BytesIO
from itertools import cycle
from pathlib import PurePath
from typing import TYPE_CHECKING
from typing import Any
from typing import BinaryIO
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import TextIO

import numpy as np
from fsspec.core import url_to_fs
from imagecodecs import imread
from imagecodecs.numcodecs import register_codecs

from tiffslide._types import OpenFileLike
from tiffslide._types import SeriesCompositionInfo
from tiffslide._types import TiffFileIO

if TYPE_CHECKING:
    from numpy.typing import NDArray


register_codecs()


def read_int(fp: BinaryIO) -> int:
    return struct.unpack("<i", fp.read(4))[0]


def read_ints(fp: BinaryIO, num: int) -> tuple[int, ...]:
    return struct.unpack(f"<{'i' * num}", fp.read(4 * num))


def to_rgb(num: int) -> tuple[int, int, int]:
    r = num % 256
    g = (num >> 8) % 256
    b = (num >> 16) % 256
    return r, g, b


class Record(NamedTuple):
    offset: int
    length: int
    file_number: int


class Tile(NamedTuple):
    tile_index: int
    record: Record


class SlideData:
    """MIRAX slide data"""

    def __init__(self, fp: TextIO) -> None:
        self.c = c = ConfigParser(default_section="")
        c.read_file(fp)

        self.slide_version: str = c["GENERAL"]["SLIDE_VERSION"]
        self.slide_uuid: str = c["GENERAL"]["SLIDE_ID"]
        self.index_fn: str = c["HIERARCHICAL"]["INDEXFILE"]

        self.image_number_x = num_ix = c["GENERAL"].getint("IMAGENUMBER_X")
        self.image_number_y = num_iy = c["GENERAL"].getint("IMAGENUMBER_Y")
        self.image_divisions = div_i = c["GENERAL"].getint(
            "CameraImageDivisionsPerSide"
        )
        self.num_positions = (num_ix / div_i) * (num_iy / div_i)

        zi = self._get_zoom_hierarchy_index()
        self.zoom_levels = c["HIERARCHICAL"].getint(f"HIER_{zi}_COUNT")
        zl = c["HIERARCHICAL"].get(f"HIER_{zi}_VAL_0_SECTION")
        self.fill_value = to_rgb(c[zl].getint("IMAGE_FILL_COLOR_BGR", 0))
        self.tile_size = (
            c[zl].getint("DIGITIZER_WIDTH"),
            c[zl].getint("DIGITIZER_HEIGHT"),
        )
        self.tile_overlap = (
            c[zl].getint("OVERLAP_X"),
            c[zl].getint("OVERLAP_Y"),
        )
        self.image_size = self._get_base_dimensions()

    def flatten(self, prefix: str = "") -> dict[str, Any]:
        """flattened representation of the slide data configuration"""
        return {
            f"{prefix}{skey.upper()}.{key.upper()}": value
            for skey, section in self.c.items()
            for key, value in section.items()
        }

    @cached_property
    def file_map(self) -> dict[int, str]:
        """map file numbers from Index.dat to corresponding file names"""
        df = self.c["DATAFILE"]
        return {f_idx: df[f"FILE_{f_idx}"] for f_idx in range(int(df["FILE_COUNT"]))}

    def _get_zoom_hierarchy_index(self) -> int:
        h = self.c["HIERARCHICAL"]
        for i in range(h.getint("HIER_COUNT")):
            if h.get(f"HIER_{i}_NAME") == "Slide zoom level":
                return i
        else:
            raise ValueError("no 'Slide zoom level' hierarchy")

    @property
    def position_record_index(self) -> int | None:
        n_total = 0
        h = self.c["HIERARCHICAL"]
        for n in range(h.getint("NONHIER_COUNT")):
            if h.get(f"NONHIER_{n}_NAME") == "VIMSLIDE_POSITION_BUFFER":
                return n_total
            n_total += h.getint(f"NONHIER_{n}_COUNT")
        return None

    def _get_base_dimensions(self) -> tuple[int, int]:
        div_i = self.image_divisions

        def _compute(num_i, t_size, t_over):
            d = 0
            for i in range(num_i):
                not_last_division = i % div_i != div_i - 1
                last_image = i == num_i - 1
                if not_last_division or last_image:
                    d += t_size
                else:
                    d += t_size - t_over
            return d

        return (
            _compute(self.image_number_x, self.tile_size[0], self.tile_overlap[0]),
            _compute(self.image_number_y, self.tile_size[1], self.tile_overlap[1]),
        )


class IndexData:
    """MIRAX index data:

    provides access to tile and record information
    """

    def __init__(self, fh: BinaryIO, *, version: str, uuid: str):
        self.fh = fh = BytesIO(fh.read())

        sv = fh.read(len(version)).decode()
        if sv != version:
            raise ValueError(
                "Index.dat does not have expected version:" f" {sv!r} != {version!r}"
            )
        si = fh.read(len(uuid)).decode()
        if uuid != uuid:
            raise ValueError(
                "Index.dat does not have a matching slide identifier:"
                f" {si!r} != {uuid!r}"
            )

        self._ptr_hierarchy = fh.tell()
        self._ptr_non_hierarchy = self._ptr_hierarchy + 4
        self.offset_hierarchy, self.offset_non_hierarchy = read_ints(self.fh, 2)

    def get_non_hierarchy_record(self, record_index: int) -> Record:
        """get file, offset and length info of a non-hierarchical record"""
        if record_index < 0:
            raise ValueError(f"record_index: {record_index} < 0")
        self.fh.seek(self.offset_non_hierarchy + 4 * record_index)
        record_ptr = read_int(self.fh)
        self.fh.seek(record_ptr)
        page_start, page_ptr = read_ints(self.fh, 2)
        assert page_start == 0, f"page_start should be 0, got: {page_start}"
        self.fh.seek(page_ptr)
        (page_size, z0, z1, z2, offset, length, file_number) = read_ints(self.fh, 7)
        assert page_size == 1, f"page_size should be 1, got: {page_size}"
        assert z1 == z2 == 0
        return Record(offset, length, file_number)

    def get_hierarchy_zoom_map(self, zoom_levels: int) -> dict[int, list[Tile]]:
        """get a mapping from zoom level to stored tiles"""
        out = {}
        for zoom_level_index in range(zoom_levels):
            self.fh.seek(self.offset_hierarchy + (4 * zoom_level_index))
            zoom_level_offset = read_int(self.fh)
            self.fh.seek(zoom_level_offset)
            page_start, page_offset = read_ints(self.fh, 2)
            assert page_start == 0
            self.fh.seek(page_offset)

            out[zoom_level_index] = tiles = []
            while True:
                page_len, page_next = read_ints(self.fh, 2)
                for _ in range(page_len):
                    tile_index, offset, length, file_number = read_ints(self.fh, 4)
                    tile = Tile(tile_index, Record(offset, length, file_number))
                    tiles.append(tile)
                if page_next == 0:
                    break
        return out


class Mirax:
    """MIRAX whole slide image"""

    def __init__(
        self,
        arg,
        *,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        st_kw = storage_options or {}

        if isinstance(arg, TiffFileIO):
            raise ValueError(
                "Multi-file MIRAX can't read from single io buffer. "
                "Please provide OpenFileLike or PathLike instead."
            )
        elif isinstance(arg, OpenFileLike):
            self.fs, self.path = arg.fs, arg.path
        elif isinstance(arg, (str, os.PathLike)):
            # provided a string like url
            urlpath = os.fspath(arg)
            self.fs, self.path = url_to_fs(urlpath, **st_kw)
        else:
            raise ValueError(f"incompatible location: {arg!r}")

        if not self.path.endswith(".mrxs"):
            raise ValueError(f"requires '.mrxs' suffix, got: {self.path!r}")

        pth = PurePath(self.path)
        ddir = pth.parent / pth.stem
        self.data_dir = str(ddir)
        if not self.fs.isdir(self.data_dir):
            raise NotADirectoryError(self.data_dir)

        # --- gather metadata, levels, tile indices and tile positions

        with self.fs.open(str(ddir / "Slidedat.ini"), mode="rt") as f:
            self.slide_data = sd = SlideData(f)

        with self.fs.open(str(ddir / sd.index_fn), mode="rb") as f:
            self.index_data = IndexData(f, version=sd.slide_version, uuid=sd.slide_uuid)

        self.tile_data = self._get_tile_data()
        self.tile_positions = self._get_tile_positions()

    def _read_record(self, record: Record) -> bytes:
        """read a record from mirax data"""
        fn = self.slide_data.file_map[record.file_number]
        with self.fs.open(os.path.join(self.data_dir, fn)) as f:
            f.seek(record.offset)
            return f.read(record.length)

    def _get_tile_data(self) -> dict[int, list[Tile]]:
        """get the tile data for all zoom levels"""
        sd = self.slide_data
        ii = self.index_data
        return ii.get_hierarchy_zoom_map(zoom_levels=sd.zoom_levels)

    def _get_tile_positions(self) -> NDArray[int]:
        """get the tile positions for stored tiles

        note: there seem to be 3 different ways of how these can be stored,
              this (wip) implementation currently supports only one

        shape: (N, 2)
        """
        sd = self.slide_data
        ii = self.index_data

        # test if position info is stored directly
        record_index = sd.position_record_index
        if record_index is not None:
            level_0_factor = 1  # fixme: this should be calculated from sd...
            positions = []

            record = ii.get_non_hierarchy_record(record_index)
            raw_positions = self._read_record(record)
            if not len(raw_positions) == sd.num_positions * 9:
                raise RuntimeError("position buffer has incorrect size")

            for i in range(0, record.length, 9):
                zz, x, y = struct.unpack("<bii", raw_positions[i : i + 9])
                assert zz & 0xFE == 0
                positions.append((x * level_0_factor, y * level_0_factor))

            return np.array(positions, dtype=int)
        raise RuntimeError("todo: implement other position stores...")

    def get_tile_coordinates(self, idx: int, level: int) -> tuple[int, int]:
        """get coordinates of tile #idx at level"""
        if level < 0:
            raise ValueError(level)

        idiv = self.slide_data.image_divisions
        iw = self.slide_data.image_number_x
        # ih = self.slide_data.image_number_y

        tile = self.tile_data[level][idx]
        tx = tile.tile_index % iw
        ty = tile.tile_index // iw

        cx, dx = divmod(tx, idiv)
        cy, dy = divmod(ty, idiv)

        c_idx = cy * (iw // idiv) + cx
        ix0, iy0 = self.tile_positions[c_idx]
        ix = ix0 + dx * self.slide_data.tile_size[0]
        iy = iy0 + dy * self.slide_data.tile_size[1]
        return ix, iy

    def get_tile_raw(self, idx: int, level: int) -> bytes:
        """get raw tile #idx at level"""
        if level < 0:
            raise ValueError(level)
        tile = self.tile_data[level][idx]
        return self._read_record(tile.record)

    def get_tile(self, idx: int, level: int, **_kw) -> NDArray[np.uint8]:
        """get tile #idx at level"""
        raw = self.get_tile_raw(idx, level)
        return imread(raw, **_kw)

    # noinspection DuplicatedCode
    def _debug_image(self):
        import PIL.Image

        w, h = self.slide_data.image_size
        w //= 16
        h //= 16
        arr = np.full((h, w, 3), 0, dtype=np.uint8)
        print(arr.shape)

        idiv = self.slide_data.image_divisions
        iw = self.slide_data.image_number_x
        # ih = self.slide_data.image_number_y
        # tw, th = self.slide_data.tile_size

        _colors = [
            np.round(np.array(colorsys.hsv_to_rgb(h, 1.0, 1.0)) * 255)
            .astype(np.uint8)
            .T[np.newaxis, np.newaxis, :]
            for h in np.linspace(0, 1.0, 7, endpoint=False)
        ]
        print(_colors)
        color = iter(cycle(_colors))

        for t in self.tile_data[0]:
            tx = t.tile_index % iw
            ty = t.tile_index // iw
            cx, dx = divmod(tx, idiv)
            cy, dy = divmod(ty, idiv)
            c_idx = cy * (iw // idiv) + cx

            x0, y0 = self.tile_positions[c_idx]
            # assert x0 % 4 == 0
            # assert y0 % 4 == 0
            x0 = x0 // 16
            y0 = y0 // 16
            x1 = x0 + (self.slide_data.tile_size[0] // 4)
            y1 = y0 + (self.slide_data.tile_size[1] // 4)
            c = next(color)
            print((y0, y1), (x0, x1), c)
            arr[y0:y1, x0:x1, :] = c

        PIL.Image.fromarray(arr).save("output.png")

    def build_properties(self):
        # fixme: minimal version
        return {
            "tiffslide.series-axes": "YXS",
            "tiffslide.level[0].width": self.slide_data.image_size[0],
            "tiffslide.level[0].height": self.slide_data.image_size[1],
        }

    def build_reference(self):
        """write a kerchunk reference loadable via zarr"""
        idiv = self.slide_data.image_divisions
        iw = self.slide_data.image_number_x
        # ih = self.slide_data.image_number_y
        image_size = self.slide_data.image_size
        tw, th = self.slide_data.tile_size

        # query the codec, ... todo improve
        _, codec = self.get_tile(0, 0, return_codec=True)
        print("codec", codec.__name__, codec)

        # tiled image groups
        kc = {
            "version": 1,
            "templates": {},
            "refs": {},
        }
        templates = kc["templates"]
        refs = kc["refs"]
        _common_zgroup = json.dumps({"zarr_format": 2})

        def _common_zarray(max_idiv_x, max_idiv_y):
            return json.dumps(
                {
                    "zarr_format": 2,
                    "shape": [th * max_idiv_y, tw * max_idiv_x, 3],
                    "chunks": [th, tw, 3],
                    "dtype": "u1",
                    "compressor": {
                        "id": "imagecodecs_jpeg"
                    },  # fixme: set dependent on file info
                    "fill_value": min(self.slide_data.fill_value),
                    "order": "C",
                    "filters": None,
                }
            )

        refs[".zgroup"] = _common_zgroup
        # print("tile_data.keys():", list(self.tile_data))
        level_series = defaultdict(set)
        for level, tiles in self.tile_data.items():

            sx, sy = image_size
            sx //= 2**level
            sy //= 2**level

            dxdy_max = defaultdict(lambda: [0, 0])
            for tile in tiles:
                # get indices
                tx = tile.tile_index % iw
                ty = tile.tile_index // iw
                cx, dx = divmod(tx, idiv)
                cy, dy = divmod(ty, idiv)
                c_idx = cy * (iw // idiv) + cx

                level_series[level].add(c_idx)

                # build key
                key = f"s{c_idx}/{level}/{dy}.{dx}.0"
                dxdy = dxdy_max[c_idx]
                dxdy[0] = max(dxdy[0], dx)
                dxdy[1] = max(dxdy[1], dx)

                # add chunks
                rec = tile.record
                templatename = f"u{rec.file_number}"
                templates[
                    templatename
                ] = f"file://{self.data_dir}/{self.slide_data.file_map[rec.file_number]}"

                refs[key] = ["{{%s}}" % templatename, rec.offset, rec.length]

            for c_idx, dxdy in dxdy_max.items():
                # === set arrays
                refs.setdefault(f"s{c_idx}/.zgroup", _common_zgroup)
                refs[f"s{c_idx}/{level}/.zgroup"] = _common_zgroup
                refs.setdefault(
                    f"s{c_idx}/{level}/.zarray",
                    _common_zarray(dxdy[0] + 1, dxdy[1] + 1),
                )

        series_indices = set.union(*level_series.values())
        level_shapes = []
        located_series = defaultdict(list)
        for level, c_indices in level_series.items():
            sx, sy = self.slide_data.image_size
            sx //= 2**level
            sy //= 2**level
            level_shapes.append((int(sy), int(sx), 3))

            for c_idx in series_indices - c_indices:
                located_series[f"s{c_idx}"].append(None)
            for c_idx in c_indices:
                x0, y0 = self.tile_positions[c_idx]
                x0 //= 2**level
                y0 //= 2**level
                located_series[f"s{c_idx}"].append((int(y0), int(x0), 0))

        series_composition_info: SeriesCompositionInfo = dict(
            level_shapes=level_shapes,
            located_series=located_series,
        )

        refs[".zattrs"] = json.dumps(
            {
                # "tiffslide.series": sorted(level_series),
                "tiffslide.series-composition": series_composition_info,
                "tiffslide.properties": self.build_properties(),
                "tiffslide.spec_version": 1,
            }
        )

        kc["refs"] = {
            k: refs[k]
            for k in sorted(
                refs.keys(),
                key=lambda x: f".{x}" if x.endswith((".zgroup", ".zarray")) else x,
            )
        }
        kc["templates"] = dict(sorted(templates.items()))

        return kc


def _debug_get_zarr(kc_dct):
    from fsspec.implementations.reference import ReferenceFileSystem
    from zarr import open_group

    fs = ReferenceFileSystem(kc_dct)
    return open_group(fs.get_mapper(""), mode="r")


if __name__ == "__main__":
    import sys
    from itertools import islice
    from pathlib import Path

    from tiffslide._kerchunk import from_kerchunk

    try:
        pth = Path(sys.argv[1])
    except IndexError:
        print(
            "usage: python -m tiffslide._mirax <file.mrxs>",
            file=sys.stderr,
        )
        raise SystemExit(1)

    mm = Mirax(pth)
    _kc = mm.build_reference()
    print('"templates": {')
    for t, x in _kc["templates"].items():
        print(" ", f'"{t}": {x!r},')
    print("},")
    print('"refs": {')
    _N = len(_kc["refs"])
    print("  ...")
    for _k, r in islice(_kc["refs"].items(), _N // 2, _N // 2 + 10):
        print(" ", f'"{_k}": {r!r},')
    print("  ...")
    print("}")

    with open(f"{pth.stem}.kerchunk.json", "w") as f:
        json.dump(_kc, f)

    grp = _debug_get_zarr(_kc)
    # todo:
    #  > [ ] support None in series_composition
    #  > [ ] gather minimal properties and add to zarr
    #  > [ ] load via from_kerchunk

    slide = from_kerchunk(_kc)
    slide.read_region((55000, 150000), 0, (8192, 8192)).save("test_cmu_big.jpeg")
