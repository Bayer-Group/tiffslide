# tiffslide changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ...
### Changes
- tiffslide: refactor .properties to simplify adding new formats
- tiffslide.deepzoom: allow JPEG2000 LOSSY

## [1.0.1] - 2022-02-24
### Fixed
- tiffslide.deepzoom: support ycbc and rgb svs with and without jpeg tables

## [1.0.0] - 2022-01-31
### Added
- docs: add stable version installation instructions

### Changed
- tiffslide: remove deprecated private method (breaking)
- tiffslide: remove obsolete internal state after using `cached_property`
- docs: update and improve the README

### Fixed
- tiffslide: complete typing overload for `read_region`
- tiffslide: change order of checks in _prepare_tifffile

## [0.3.0] - 2022-01-31
### Added
- support opening fsspec urlpaths and openfiles directly

### Fixed
- fix multithreaded access of uninitialized TiffSlide instance
- `tiffslide.deepzoom` only use rgb colorfix if needed

## [0.2.2] - 2021-11-17
### Fixed
- fix black bar artifacts on computed intermediate levels in tiffslide.deepzoom
- warnings regarding aliasing now report at the correct stack level

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
- removed `TiffSlide.ts_filename` attribute

## [0.0.1] - 2021-03-18
### Added
- initial release of tiffslide and

[Unreleased]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/tree/v0.0.1
