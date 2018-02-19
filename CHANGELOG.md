# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## 3.2 - 2018-02-?
### Added
- The whole build system of the project migrated from makefiles to
  cmake. In particualr this includes `libsleef`, `libsleefgnuabi`,
  `libdft` and all the tests.
- Benchmarks that compare `libsleef` vs `SVML` on X86 Linux are
  available in the project tree. @shibatch please add the path here
  once https://github.com/shibatch/sleef/pull/168 is in.
- Extensive upstream testing via Travis CI and Appveyor, on the
  following systems:
  * OS: Windows / Linux/ OSX.
  * Compilers: gcc / clang / MSVC.
  * Targets: X86 (SSE/AVX/AVX2/AVX512F), AArch64 (Advanced SIMD), ARM
    (NEON). Emulators like QEMU or SDE can be used to run the tests.
- Added the following new vector functions (with relative testing):
  * `log2`
  * ... @shibatch , any other function I am missing?
- New compatibility tests have been added to check that
  `libsleefgnuabi` exports the GNUABI symbols correctly.
- The library can be compiled to an LLVM bitcode object. @xoofx,
  do you have anything more to say here?
- Added masked interface to the library to support AVX512F masked
  vectorization.

### Changed
- Use native instructions if available for `sqrt`.
- Fixed fmax and fmin behavior on AArch64:
  https://github.com/shibatch/sleef/pull/140
- Speed improvements for `asin`, `acos`, `fmod` and `log`.
- Removed `libm` dependency.

### Removed
- Makefile build system
