from tiffslide import TiffSlide

# noinspection PyProtectedMember
from tiffslide._zarr import get_zarr_chunk_sizes


def test_get_chunk_sizes(wsi_file):
    chunk_sizes = get_zarr_chunk_sizes(TiffSlide(wsi_file).zarr_group)
    assert chunk_sizes.ndim in (2, 3)
    assert chunk_sizes.sum() > 0
