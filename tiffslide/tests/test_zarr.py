from tifffile import ZarrStore

from tiffslide import TiffSlide

# noinspection PyProtectedMember
from tiffslide._zarr import get_zarr_chunk_sizes


def test_get_chunk_sizes(wsi_file):
    chunk_sizes = get_zarr_chunk_sizes(TiffSlide(wsi_file).zarr_group)
    assert chunk_sizes.ndim in (2, 3)
    assert chunk_sizes.sum() > 0


def test_decode_only_once(wsi_file, monkeypatch):
    ts = TiffSlide(wsi_file)

    called_keys = []
    _ZarrStore___getitem__ = ZarrStore.__getitem__

    def _cnt_getitem(self, key):
        nonlocal called_keys
        called_keys.append(key)
        return _ZarrStore___getitem__(self, key)

    monkeypatch.setattr(ZarrStore, "__getitem__", _cnt_getitem)

    _ = ts.read_region((0, 0), 0, (1, 1), as_array=True)
    assert len(called_keys) == len(set(called_keys))
