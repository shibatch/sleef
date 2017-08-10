# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v[3.0.1] - 2017-07-19
### Added:
- AArch64 support
- Support for 32-bit x86, Cygwin, etc.
- Dispatcher for x86 functions
- Implementation for the remaining C99 math functions: lgamma, tgamma, erf, erfc, fabs, copysign, fmax, fmin, fdim, trunc, floor, ceil, round, rint, modf, ldexp, nextafter, frexp, hypot, and fmod
### Changed:
- Improved reduction of trigonometric functions
- Improved tester
### Removed:
### Fixed:

## v[3.0.0] - 2017-02-07
### Added:
- New API is defined
- Support for Linux, Windows and MacOSX
- Support for compilers: GCC, Clang, Intel Compiler, Microsoft Visual C++
- DFT (Discrete Fourier Transform) functions
- sincospi functions
- Files needed for creating a debian package
### Changed:
- gencoef now supports single, extended and quad precision in addition to double precision
- The library can be compiled as DLLs
### Removed:
### Fixed:

## v[3.11.0] - 2017-81-30
### Added:
### Changed:
- Relicensed to Boost Software License Version 1.0
### Removed:
### Fixed:

## v[2.1.1] - 2016-12-11
### Added:
- Specification of each functions regarding to the domain and accuracy
- A coefficient generation tool
- New testing tools
### Changed:
- The valid range of argument is extended for trig functions
### Removed:
- Support for Java language (because no one seems using this)
### Fixed:
- Following functions returned incorrect values when the argument is very large or small : exp, pow, asinh, acosh
- SIMD xsin and xcos returned values more than 1 when FMA is enabled
- Pure C cbrt returned incorrect values when the argument is negative
- tan_u1 returned values with more than 1 ulp of error on rare occasions
