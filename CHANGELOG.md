# tiffslide changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ...

## [0.2.1] - 2021-09-28
### Fixed
- fix [XY]Resolution are rational numbers

## [0.2.0] - 2021-08-29
### Added
- add `use_embedded` kwarg to `TiffSlide.get_thumbnail`
- add `as_array` kwarg to `TiffSlide.get_region`

### Change
- deprecate internal `_read_region_as_array`

## [0.1.1] - 2021-08-25
### Fixed
- fixed typing with newer versions of numpy
- fixed missing mpp for generic tiffs that provide `[XY]Resolution` and `ResolutionUnit`
- allow providing OpenFile objects to `tiffslide.deepzoom`

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
https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/tree/v0.0.1
