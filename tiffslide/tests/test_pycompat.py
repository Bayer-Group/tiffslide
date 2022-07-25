import pytest

from tiffslide._pycompat import _requires_store_fix


@pytest.mark.parametrize(
    "vz,vt,fix",
    [
        ("2.11.0", "2022.3.29rc7", True),
        ("2.10.1a3", "2022.3.29", False),
        ("2.12.0", "2022.4.0", False),
        ("2.13.0rc1", "2022.12.0", False),
        ("2.11.3b1", "2022.2.11", True),
    ],
)
def test_requires_fix(vz, vt, fix):
    assert _requires_store_fix(vz, vt) == fix
