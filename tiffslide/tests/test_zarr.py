from tifffile.zarr import ZarrTiffStore

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
    _ZarrTiffStore_get = ZarrTiffStore.get

    async def _cnt_get(self, key, prototype, byte_range=None):
        nonlocal called_keys
        called_keys.append(key)
        return await _ZarrTiffStore_get(self, key, prototype, byte_range=byte_range)

    monkeypatch.setattr(ZarrTiffStore, "get", _cnt_get)

    _ = ts.read_region((0, 0), 0, (1, 1), as_array=True)
    # filter out metadata keys — only check chunk data keys
    data_keys = [k for k in called_keys if not k.startswith(".z")]
    assert len(data_keys) == len(set(data_keys))
