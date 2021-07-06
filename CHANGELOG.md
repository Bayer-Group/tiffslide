# tiffslide changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ...
...

## [0.1.0] - 2021-07-06
### Fixed
- allow passing file objects
- fix svs metadata handling (fixed upstream in tifffile)
- fix single-level svs handling

### Added
- tests via pytest
- type annotations and mypy type checking
- `TiffSlide.read_region_array` method for reading `np.ndarray`

### Changed
- removed TiffSlide.ts_filename attribute

## [0.0.1] - 2021-03-18
### Added
- initial release of tiffslide and 

[Unreleased]: 
https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/tree/v0.1.0
[0.0.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/tree/v0.0.1
