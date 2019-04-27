# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Next release
## 3.4 - 2019-04-28
### Added
- Faster and low precision functions are added.
  https://github.com/shibatch/sleef/pull/229
- Functions that return consistent results across platforms are
  added
  https://github.com/shibatch/sleef/pull/216
  https://github.com/shibatch/sleef/pull/224
- Quad precision math library(libsleefquad) is added
  https://github.com/shibatch/sleef/pull/235
  https://github.com/shibatch/sleef/pull/237
  https://github.com/shibatch/sleef/pull/240
- AArch64 Vector Procedure Call Standard (AAVPCS) support.
### Changed
- Many functions are now faster
- Testers are now faster
## 3.3.1 - 2018-08-20
### Added
- FreeBSD support is added
### Changed
- i386 build problem is fixed
- Trigonometric functions now evaluate correctly with full FP
  domain.
  https://github.com/shibatch/sleef/pull/210
## 3.3 - 2018-07-06
### Added
- SVE target support is added to libsleef.
  https://github.com/shibatch/sleef/pull/180
- SVE target support is added to DFT. With this patch, DFT operations
  can be carried out using 256, 512, 1024 and 2048-bit wide vectors
  according to runtime availability of vector registers and operators.
  https://github.com/shibatch/sleef/pull/182
- 3.5-ULP versions of sinh, cosh, tanh, sinhf, coshf, tanhf, and the
  corresponding testing functionalities are added.
  https://github.com/shibatch/sleef/pull/192
- Power VSX target support is added to libsleef.
  https://github.com/shibatch/sleef/pull/195
- Payne-Hanek like argument reduction is added to libsleef.
  https://github.com/shibatch/sleef/pull/197
## 3.2 - 2018-02-26
### Added
- The whole build system of the project migrated from makefiles to
  cmake. In particualr this includes `libsleef`, `libsleefgnuabi`,
  `libdft` and all the tests.
- Benchmarks that compare `libsleef` vs `SVML` on X86 Linux are
  available in the project tree under src/libm-benchmarks directory.
- Extensive upstream testing via Travis CI and Appveyor, on the
  following systems:
  * OS: Windows / Linux / OSX.
  * Compilers: gcc / clang / MSVC.
  * Targets: X86 (SSE/AVX/AVX2/AVX512F), AArch64 (Advanced SIMD), ARM
    (NEON). Emulators like QEMU or SDE can be used to run the tests.
- Added the following new vector functions (with relative testing):
  * `log2`
- New compatibility tests have been added to check that
  `libsleefgnuabi` exports the GNUABI symbols correctly.
- The library can be compiled to an LLVM bitcode object.
- Added masked interface to the library to support AVX512F masked
  vectorization.

### Changed
- Use native instructions if available for `sqrt`.
- Fixed fmax and fmin behavior on AArch64:
  https://github.com/shibatch/sleef/pull/140
- Speed improvements for `asin`, `acos`, `fmod` and `log`. Computation
  speed of other functions are also improved by general optimization.
  https://github.com/shibatch/sleef/pull/97
- Removed `libm` dependency.

### Removed
- Makefile build system
