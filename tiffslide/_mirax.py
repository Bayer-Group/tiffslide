from __future__ import annotations

import configparser
import pathlib
import struct
from collections import defaultdict
from pprint import pprint
from typing import BinaryIO
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import TYPE_CHECKING
from typing import TextIO
from typing import Tuple

import numpy as np
from imagecodecs import imread
from imagecodecs import imwrite

if TYPE_CHECKING:
    from numpy.typing import NDArray


def read_int(fp) -> int:
    return struct.unpack("<i", fp.read(4))[0]


def read_ints(fp, num: int):
    return struct.unpack(f"<{'i' * num}", fp.read(4 * num))


def to_rgb(num: int):
    r = num % 256
    g = (num >> 8) % 256
    b = (num >> 16) % 256
    return r, g, b


class _RecordData(NamedTuple):
    file_number: int
    chunk_offset: int
    chunk_length: int


class _TileData(NamedTuple):
    tile_index: int
    file_number: int
    chunk_offset: int
    chunk_length: int


class _SlideData:
    def __init__(self, fp: TextIO):
        self.sd = sd = configparser.ConfigParser()
        sd.read_file(fp)

        g = sd["GENERAL"]

        self.camera_image_divisions_per_side = div_i = g.getint("CameraImageDivisionsPerSide")
        self.slide_version = g.get("SLIDE_VERSION")
        self.slide_id = g.get("SLIDE_ID")
        self.image_number_x = num_ix = g.getint("IMAGENUMBER_X")
        self.image_number_y = num_iy = g.getint("IMAGENUMBER_Y")

        h = sd["HIERARCHICAL"]
        self.index_filename = h.get("INDEXFILE")
        self.num_hierarchies = h.getint("HIER_COUNT")
        self.fill_value = to_rgb(0)
        for h_idx in range(self.num_hierarchies):
            if h.get(f"HIER_{h_idx}_NAME") == "Slide zoom level":
                level_key = h.get(f"HIER_{h_idx}_VAL_0_SECTION")
                l = sd[level_key]
                self.fill_value = to_rgb(l.getint("IMAGE_FILL_COLOR_BGR", 0))
                break
        else:
            raise ValueError("could not find 'slide zoom level' hierarchy")
        self.zoom_levels = h.getint(f"HIER_{h_idx}_COUNT")

        self.num_nonhierarchies = h.getint("NONHIER_COUNT")
        self.position_buffer_record_index = None
        _n_total = 0
        for n_idx in range(self.num_nonhierarchies):
            if h.get(f"NONHIER_{n_idx}_NAME") == "VIMSLIDE_POSITION_BUFFER":
                self.position_buffer_record_index = _n_total
                break
            _n_total += h.getint(f"NONHIER_{n_idx}_COUNT")

        df = sd["DATAFILE"]
        self.file_map = {
            f_idx: df.get(f"FILE_{f_idx}")
            for f_idx in range(df.getint("FILE_COUNT"))
        }

        self.num_positions = (num_ix / div_i) * (num_iy / div_i)
        self.position_buffer_size = 9 * self.num_positions


class _Index:
    def __init__(
        self,
        fp: BinaryIO,
        sd: _SlideData,
    ):
        self.sd = sd

        v = fp.read(len(sd.slide_version)).decode()
        if v != self.sd.slide_version:
            raise ValueError(
                "Index.dat does not have expected version:"
                f" {v!r} != {sd.slide_version!r}"
            )
        uuid = fp.read(len(sd.slide_id)).decode()
        if uuid != sd.slide_id:
            raise ValueError(
                "Index.dat does not have a matching slide identifier:"
                f" {uuid!r} != {sd.slide_id!r}"
            )

        hier_pos = fp.tell()
        nonhier_pos = hier_pos + 4
        hier_offset, nonhier_offset = read_ints(fp, 2)

        self.tile_data = self._get_mirax_hierarchy(
            fp,
            hier_offset,
            sd.zoom_levels,
        )

        self.position_record = None
        self.position_filename = None
        if sd.position_buffer_record_index is not None:
            _pr = self.position_record = self._get_mirax_nonhierarchy_record(
                fp, nonhier_pos, sd.position_buffer_record_index
            )
            self.position_filename = sd.file_map[_pr.file_number]

    @staticmethod
    def _get_mirax_nonhierarchy_record(
        fp: BinaryIO,
        nonhier_offset: int,
        record_index: int,
    ) -> _RecordData:
        fp.seek(nonhier_offset)
        ptr = read_int(fp)
        assert record_index >= 0
        fp.seek(ptr + 4 * record_index)
        record_ptr = read_int(fp)
        fp.seek(record_ptr)
        page_start, page_ptr = read_ints(fp, 2)
        assert page_start == 0, page_start
        fp.seek(page_ptr)
        (page_size, z0, z1, z2, offset, length, file_number) = read_ints(fp, 7)
        assert page_size == 1
        assert z1 == z2 == 0
        return _RecordData(file_number, offset, length)

    @staticmethod
    def _get_mirax_hierarchy(
        fp: BinaryIO,
        hier_offset: int,
        zoom_levels: int,
    ) -> Dict[int, List[_TileData]]:
        out = {}
        for zoom_level_index in range(zoom_levels):
            fp.seek(hier_offset + (4 * zoom_level_index))
            zoom_level_offset = read_int(fp)
            fp.seek(zoom_level_offset)
            page_start, page_offset = read_ints(fp, 2)
            assert page_start == 0
            fp.seek(page_offset)

            out[zoom_level_index] = tiles = []
            while True:
                page_len, page_next = read_ints(fp, 2)
                for _ in range(page_len):
                    _index, offset, length, file_number = read_ints(fp, 4)
                    tiles.append(_TileData(_index, file_number, offset, length))
                if page_next == 0:
                    break
        return out


class _PositionRecord:
    def __init__(
        self,
        fp: BinaryIO,
        offset: int,
        length: int,
        level_0_factor: int = 1,
    ) -> None:  # :
        fp.seek(offset)
        out = []
        assert length % 9 == 0
        for _ in range(length // 9):
            zz, x, y = struct.unpack("<bii", fp.read(9))
            assert zz & 0xfe == 0
            out.append((x * level_0_factor, y * level_0_factor))
        self.positions: List[Tuple[int, int]] = out


class Mirax:
    def __init__(self, pth: str) -> None:
        pth = pathlib.Path(pth)
        assert pth.suffix == ".mrxs"
        parent_dir = pth.parent
        self.data_dir = dd = parent_dir / pth.stem
        assert dd.is_dir()

        with dd.joinpath("Slidedat.ini").open() as f:
            self.data = _SlideData(f)
        with dd.joinpath(self.data.index_filename).open("rb") as f:
            self.index = _Index(f, self.data)

        self.positions = None
        if self.index.position_filename:
            rec = self.index.position_record
            with dd.joinpath(self.index.position_filename).open("rb") as f:
                self.positions = _PositionRecord(f, rec.chunk_offset, rec.chunk_length)

    def get_coords(
        self,
        idx,
        level=0
    ) -> Tuple[int, int]:
        assert level == 0,  "fixme"  # todo
        tiles = self.index.tile_data[level]
        trec = tiles[idx]
        x = trec.tile_index % self.data.image_number_x
        y = trec.tile_index // self.data.image_number_x
        cx = x // self.data.camera_image_divisions_per_side
        dx = x % self.data.camera_image_divisions_per_side
        cy = y // self.data.camera_image_divisions_per_side
        dy = y % self.data.camera_image_divisions_per_side
        cp = cy * (self.data.image_number_x // self.data.camera_image_divisions_per_side) + cx
        pos = self.positions.positions[cp]
        ix, iy = pos
        y0 = iy + (dy * 256)  # fixme
        x0 = ix + (dx * 340)  # fixme
        return x0, y0

    def get_tile(
        self,
        idx,
        level=0
    ) -> NDArray[np.uint8]:
        assert level == 0,  "fixme"  # todo
        tiles = self.index.tile_data[level]
        trec = tiles[idx]
        fn = self.data.file_map[trec.file_number]
        with self.data_dir.joinpath(fn).open("rb") as f:
            f.seek(trec.chunk_offset)
            raw = f.read(trec.chunk_length)
        return imread(raw)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    FILE = "CMU-1.mrxs"
    m = Mirax(FILE)

    print('positions', len(m.positions.positions or []))
    print('tiles', len(m.index.tile_data[0]))
    print('images xy', m.data.image_number_y * m.data.image_number_x // m.data.camera_image_divisions_per_side ** 2)
    print('non-zero', sum(1 for p in m.positions.positions if p[0] != 0 != p[1]))

    total = 0
    ys = set()
    xs = set()
    for idx in tqdm(range(len(m.index.tile_data[0]))):
        x0, y0 = m.get_coords(idx)
        x1 = x0 + 340
        y1 = y0 + 256
        xs.add(x0)
        ys.add(y0)
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs) + 340
    max_y = max(ys) + 256
    print("X", min_x, max_x)
    print("Y", min_y, max_y)

    sx = (max_x - min_x) // 100
    sy = (max_y - min_y) // 100
    coll = np.zeros((sy, sx, 3), dtype=np.uint8)
    coll[:, :, :] = m.data.fill_value
    print("fill", m.data.fill_value)

    for idx in tqdm(range(len(m.index.tile_data[0]))):
        x0, y0 = m.get_coords(idx)
        x0 = x0 - min_x
        y0 = y0 - min_y
        x1 = x0 + 340
        y1 = y0 + 256

        total += 1
        arr = m.get_tile(idx)
        x0 //= 100
        y0 //= 100
        x1 = x0 + 4
        y1 = y0 + 3
        if x1 >= sx or y1 >= sy:
            continue
        coll[y0:y1, x0:x1, :] = arr[::100, ::100, :]
    print("added", total, "tiles")
    imwrite("coll.png", coll)
