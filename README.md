# tiffslide: a drop-in replacement for openslide-python

[![PyPI Version](https://img.shields.io/pypi/v/tiffslide)](https://pypi.org/project/tiffslide/)
[![Read the Docs](https://img.shields.io/readthedocs/tiffslide)](https://tiffslide.readthedocs.io)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/bayer-science-for-a-better-life/tiffslide/tiffslide%20ci?label=tests)](https://github.com/bayer-science-for-a-better-life/tiffslide/actions)
[![Codecov](https://img.shields.io/codecov/c/github/bayer-science-for-a-better-life/tiffslide)](https://codecov.io/gh/bayer-science-for-a-better-life/tiffslide)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tiffslide)](https://github.com/bayer-science-for-a-better-life/tiffslide)
[![GitHub issues](https://img.shields.io/github/issues/bayer-science-for-a-better-life/tiffslide)](https://github.com/bayer-science-for-a-better-life/tiffslide/issues)

Welcome to `tiffslide` :wave:, a [tifffile](https://github.com/cgohlke/tifffile/)
drop-in replacement for [openslide-python](https://github.com/openslide/openslide-python).

`tiffslide`'s goal is to provide an easy way to migrate existing code from an
openslide dependency to the excellently maintained tifffile module.

We strive to make your lives as easy as possible: If using `tiffslide` is
unintuitive to use, slow, or if it's drop-in behavior differs from what you expect,
it's a bug in `tiffslide`. Feel free to report any issues or feature requests in
the issue tracker!

Development [happens on github](https://github.com/bayer-science-for-a-better-life/tiffslide) :octocat:

## Documentation

tiffslide's behavior aims to be identical to openslide-python where it makes sense.
If you rely heavily on the internals of openslide, this is not the package you are looking for.
In case we add more features, we will add documentation here.

```python
import tiffslide as openslide
```

## Development Installation

1. Install conda and git
2. Clone tiffslide `git clone https://github.com/bayer-science-for-a-better-life/tiffslide.git`
3. Run `conda env create -f environment.yaml`
4. Activate the environment `conda activate tiffslide`

Note that in this environment `tiffslide` is already installed in development mode,
so go ahead and hack.


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

`tiffslide`: copyright 2020 Bayer AG, licensed under [BSD](https://github.com/bayer-science-for-a-better-life/tiffslide/blob/master/LICENSE)
