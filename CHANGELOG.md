# tiffslide changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ...

## [2.0.0] - 2023-02-09
## Changed
- `tiffslide`: the downsamples are now calculated identical to openslide (breaking)

## Fixed
- `tiffslide`: fix repr when backed by ReferenceFileSystem
- `tiffslide`: prevent deprecation warning with non-rgb images
- prevent incompatibility issues with fsspec on py3.7
- support newer pillow

## [1.10.1] - 2022-11-09
## Fixed
- `tiffslide`: support `.getitems()` fetching via ReferenceFileSystem

## [1.10.0] - 2022-10-07
## Added
- `tiffslide`: parse additional hamamatsu specific tags

## Fixed
- `tiffslide`: fix bug in `TiffSlide.get_best_level_for_downsample`

## [1.9.0] - 2022-09-18
## Added
- `tiffslide`: support single-channel 16bit svs/tiff files

## [1.8.1] - 2022-09-13
## Fixed
- `tiffslide._zarr`: unwrap the compatibility shim correctly on py37

## [1.8.0] - 2022-08-31
## Added
- `tiffslide`: allow limiting worker threads via env `TIFFSLIDE_NUM_DECODE_THREADS`

## [1.7.0] - 2022-08-23
## Added
- `tiffslide._zarr`: added experimental support for getting chunk sizes

## [1.6.0] - 2022-08-15
## Changed
- `tiffslide`: remove old svs metadata parsing compatibility patch

## [1.5.0] - 2022-07-25
## Fixed
- `tiffslide`: prevent deprecation warning with `ANTIALIAS` for `pillow>=9.1.0`

## Changed
- `tiffslide`: don't require `backports.cached_property` on python3.7 anymore

## [1.4.0] - 2022-07-15
## Added
- `tiffslide._kerchunk`: added experimental support for kerchunk serialization

## [1.3.0] - 2022-06-30
## Fixed
- compatibility: fixed scn compositing (#36)

## Added
- `TiffSlide().zarr_group` property to replace `.ts_zarr_grp` in a future version

## [1.2.1] - 2022-06-16
## Fixed
- correct padding on nonzero levels (#38)

## [1.2.0] - 2022-04-03
## Added
- compatibility: add fallback support for non-tiff images via `tiffslide.open_slide` (#19)
- make region padding configurable

## Fixed
- compatibility: support padding regions if out-of-bounds region requested (#27)
- prevent numpy scalar overflows when using np.int32 coords in `read_region` (#29)

## [1.1.1] - 2022-03-31
## Changes
- change conda environment back to have conda devenv features in a compatibility way

## Fixed
- fixes py37 zarr/tifffile version compatibility issue for installs via pypi

## [1.1.0] - 2022-03-22
### Added
- support Leica SCN format
- support using local images for tests via TIFFSLIDE_TEST_IMAGES env var

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

[Unreleased]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.10.1...v2.0.0
[1.10.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.10.0...v1.10.1
[1.10.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.9.0...v1.10.0
[1.9.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.8.1...v1.9.0
[1.8.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bayer-science-for-a-better-life/tiffslide/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/bayer-science-for-a-better-life/tiffslide/tree/v0.0.1
