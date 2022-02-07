# tiffslide: a drop-in replacement for openslide-python

[![PyPI Version](https://img.shields.io/pypi/v/tiffslide)](https://pypi.org/project/tiffslide/)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/tiffslide?label=conda)](https://anaconda.org/conda-forge/tiffslide)
[![tiffslide ci](https://github.com/bayer-science-for-a-better-life/tiffslide/actions/workflows/run_pytests.yaml/badge.svg)](https://github.com/bayer-science-for-a-better-life/tiffslide/actions/workflows/run_pytests.yaml)
[![GitHub issues](https://img.shields.io/github/issues/bayer-science-for-a-better-life/tiffslide)](https://github.com/bayer-science-for-a-better-life/tiffslide/issues)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/tiffslide?label=pypi)](https://pypi.org/project/tiffslide/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tiffslide)](https://github.com/bayer-science-for-a-better-life/tiffslide)

Welcome to `tiffslide` :wave:, a [tifffile](https://github.com/cgohlke/tifffile/) based
drop-in replacement for [openslide-python](https://github.com/openslide/openslide-python).

`tiffslide`'s goal is to provide an easy way to migrate existing code from an
openslide dependency to the excellently maintained tifffile module.

We strive to make your lives as easy as possible: If using `tiffslide` is
unintuitive, slow, or if it's drop-in behavior differs from what you expect,
it's a bug in `tiffslide`. Feel free to report any issues or feature requests in
the issue tracker!

Development [happens on github](https://github.com/bayer-science-for-a-better-life/tiffslide) :octocat:


## Notes

Currently, we are targeting support for Aperio SVS but contributions to expand
to a larger variety of file formats that tifffile supports are very welcome :heart:
<br>
If there are any questions open an issue, and we'll do our best to help!


## Documentation

### Installation

tiffslide's stable releases can be installed via `pip`:
```bash
pip install tiffslide
```
Or via `conda`:
```bash
conda install -c conda-forge tiffslide
```

### Usage

tiffslide's behavior aims to be identical to openslide-python where it makes sense.
If you rely heavily on the internals of openslide, this is not the package you are looking for.
In case we add more features, we will add documentation here.

#### as a drop-in replacement

```python
# directly
from tiffslide import TiffSlide
slide = TiffSlide('path/to/my/file.svs')

# or via its drop-in behavior
import tiffslide as openslide
slide = openslide.OpenSlide('path/to/my/file.svs')
```

#### access files in the cloud

A nice side effect of using tiffslide is that your code will also work with
[filesystem_spec](https://github.com/fsspec/filesystem_spec), which enables you
to access your whole slide images from various supported filesystems:

```python
import fsspec
from tiffslide import TiffSlide

# read from any io buffer
with fsspec.open("s3://my-bucket/file.svs") as f:
    slide = TiffSlide(f)
    thumb = slide.get_thumbnail((200, 200))

# read from fsspec urlpaths directly, using your AWS_PROFILE 'aws'
slide = TiffSlide("s3://my-bucket/file.svs", storage_options={'profile': 'aws'})
thumb = slide.get_thumbnail((200, 200))

# read via fsspec from google cloud and use fsspec's caching mechanism to cache locally
slide = TiffSlide("simplecache::gcs://my-bucket/file.svs", storage_options={'project': 'my-project'})
region = slide.read_region((300, 400), 0, (512, 512))
```

#### read numpy arrays instead of PIL images

Very often you'd actually want your region returned as a numpy array instead
getting a PIL Image and then having to convert to numpy:

```python
import numpy as np
from tiffslide import TiffSlide

slide = TiffSlide("myfile.svs")
arr = slide.read_region((100, 200), 0, (256, 256), as_array=True)
assert isinstance(arr, np.ndarray)
```


## Development Installation

If you want to help improve tiffslide, you can setup your development environment
in two different ways:

With conda:

1. Clone tiffslide `git clone https://github.com/bayer-science-for-a-better-life/tiffslide.git`
2. `cd tiffslide`
3. `conda env create -f dev_environment.yml`
4. Activate the environment `conda activate tiffslide`

Without conda:

1. Clone tiffslide `git clone https://github.com/bayer-science-for-a-better-life/tiffslide.git`
2. `cd tiffslide`
3. `python -m venv venv && source venv/bin/activate && python -m pip install -U pip`
4. `pip install -e .[dev]`

Note that in these environments `tiffslide` is already installed in development
mode, so go ahead and hack.


## Contributing Guidelines

- Please follow [pep-8 conventions](https://www.python.org/dev/peps/pep-0008/) but:
  - We allow 120 character long lines (try anyway to keep them short)
- Please use [numpy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
- When contributing code, please try to use Pull Requests.
- tests go hand in hand with modules on ```tests``` packages at the same level. We use ```pytest```.

You can setup your IDE to help you adhering to these guidelines.
<br>
_([Santi](https://github.com/sdvillal) is happy to help you setting up pycharm in 5 minutes)_


## Acknowledgements

Build with love by Andreas Poehlmann and Santi Villalba from the _Machine Learning Research_ group at Bayer.

`tiffslide`: copyright 2020-2022 Bayer AG, licensed under [BSD](https://github.com/bayer-science-for-a-better-life/tiffslide/blob/master/LICENSE)
