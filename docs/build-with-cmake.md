# Introduction

[CMake](http://www.cmake.org/) is an open-source and cross-platform building
tool for software packages that provides easy managing of multiple build systems
at a time. It works by allowing the developer to specify build parameters and
rules in a simple text file that CMake then processes to generate project files
for the actual native build tools (e.g. UNIX Makefiles, Microsoft Visual Studio,
Apple XCode, etc). That means you can easily maintain multiple separate builds
for one project and manage cross-platform hardware and software complexity.

If you are not already familiar with CMake, please refer to the [official
documentation](https://cmake.org/documentation/) or the
[Basic Introductions](https://cmake.org/Wiki/CMake#Basic_Introductions) in the
wiki (recommended).

Before using CMake you will need to install/build the binaries on your system.
Most systems have CMake already installed or provided by the standard package
manager. If that is not the case for you, please
[download](https://cmake.org/download/) and install now.

**For building SLEEF, CMake 3.18 is the minimum required.**

# Quick start

1. Make sure CMake is available. The following command line should display a
version number greater than or equal to 3.18.

```
cmake --version
```

2. Download the tar from [sourceforge][forge_url],
[GitHub releases][release_url], or checkout out the source code from the
[GitHub repository][repo_url].

```
git clone https://github.com/shibatch/sleef
```

3. Make separate directories to create out-of-source build and install. SLEEF
does not allow for in-tree builds.

```
cd sleef
mkdir build
mkdir install
```

4. Run CMake to configure your project and generate the system to build it.

```
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_INSTALL_PREFIX=install/ \
      -S . -B build/
```

By default, CMake will detect your system platform and configure the build
using the default parameters. You can control and modify these parameters by
setting variables when running CMake. See the list of
[options and variables](#build-customization) for customizing your build.

In the above example:
- `-DCMAKE_BUILD_TYPE=RelWithDebInfo` configures an optimised `libsleef`
shared library build with basic debug info.
- `-DCMAKE_INSTALL_PREFIX=install/` sets SLEEF to be installed in `install/`.

6. Run make to build the project

```
cmake --build build -j --clean-first
```

> NOTE: On **Windows**, you need to use a specific generator like this:
> `cmake -G"Visual Studio 14 2015 Win64" ..` specifying the Visual Studio version
> and targeting specifically `Win64` (to support compilation of AVX/AVX2)
> Check `cmake -G` to get a full list of supported Visual Studio project generators.
> This generator will create a proper solution `SLEEF.sln` under the build
> directory.
> You can still use `cmake --build build` to build without opening Visual Studio.

7. Run ctest suite (CMake 3.20+)

```
ctest --test-dir build -j
```

or for older CMake versions

```
cd build/ && ctest -j
```

8. Install at path provided earlier or at new path `<prefix>`

```
cmake --install build/ [--prefix <prefix>]
```

Refer to our web page for [detailed build instructions][build_info_url].

# Build customization

Variables dictate how the build is generated; options are defined and undefined,
respectively, on the CMake command line like this:

```
cmake -DVARIABLE=<value> <cmake-build-dir>
cmake -UVARIABLE <cmake-build-dir>
```

Build configurations allow a project to be built in different ways for debug,
optimized, or any other special set of flags.


## CMake Variables

- `CMAKE_BUILD_TYPE`: By default, CMake supports the following configuration:
  * `Debug`: basic debug flags turned on
  * `Release`: basic optimizations turned on
  * `MinSizeRel`: builds the smallest (but not fastest) object code
  * `RelWithDebInfo`: builds optimized code with debug information as well

- `CMAKE_INSTALL_PREFIX`: The prefix the use when running `make install`.
			  Defaults to /usr/local on GNU/Linux and MacOS.
			  Defaults to C:/Program Files on Windows.

- `CMAKE_C_FLAGS_RELEASE` : The optimization options used by the compiler.

## SLEEF Variables

### Library

- `SLEEF_SHOW_CONFIG` : Show relevant CMake variables upon configuring a build
- `SLEEF_SHOW_ERROR_LOG` : Show the content of CMakeError.log

- `BUILD_SHARED_LIBS` : Static libs are built if set to FALSE
- `SLEEF_BUILD_SHARED_LIBS` : If set, override the value of `BUILD_SHARED_LIBS` when configuring SLEEF.
- `SLEEF_BUILD_GNUABI_LIBS` : Avoid building libraries with GNU ABI if set to FALSE
- `SLEEF_BUILD_INLINE_HEADERS` : Generate header files for inlining whole SLEEF functions

- `SLEEF_DISABLE_OPENMP` : Disable support for OpenMP
- `SLEEF_ENFORCE_OPENMP` : Build fails if OpenMP is not supported by the compiler

- `SLEEF_ENABLE_LTO` : Enable support for LTO with gcc, or thinLTO with llvm
- `SLEEF_LLVM_AR_COMMAND` : Specify LLVM AR command when you build the library with thinLTO support with clang.
- `SLEEF_ENABLE_LLVM_BITCODE` : Generate LLVM bitcode

### Tests

- `SLEEF_BUILD_TESTS` : Avoid building testing tools if set to FALSE
- `SLEEF_ENFORCE_TESTER3` : Build fails if tester3 cannot be built

### Quad and DFT

- `SLEEF_BUILD_QUAD` : Build quad-precision libraries if set to TRUE
- `SLEEF_BUILD_DFT` : Build DFT libraries if set to TRUE
- `SLEEFDFT_MAXBUTWIDTH` : This variable specifies the maximum length of combined butterfly block used in the DFT. Setting this value to 7 makes DFT faster but compilation takes more time and the library size will be larger.
- `SLEEF_DISABLE_FFTW` : Disable FFTW-based testing of the DFT library.

### Vector extensions and instructions

- `SLEEF_ENABLE_ALTDIV` : Enable alternative division method (aarch64 only)
- `SLEEF_ENABLE_ALTSQRT` : Enable alternative sqrt method (aarch64 only)
- `SLEEF_DISABLE_LONG_DOUBLE` : Disable support for long double data type
- `SLEEF_ENFORCE_LONG_DOUBLE` : Build fails if long double data type is not supported by the compiler
- `SLEEF_DISABLE_FLOAT128` : Disable support for float128 data type
- `SLEEF_ENFORCE_FLOAT128` : Build fails if float128 data type is not supported by the compiler
- `SLEEF_DISABLE_SSE2` : Disable support for x86 SSE2
- `SLEEF_ENFORCE_SSE2` : Build fails if SSE2 is not supported by the compiler
- `SLEEF_DISABLE_SSE4` : Disable support for x86 SSE4
- `SLEEF_ENFORCE_SSE4` : Build fails if SSE4 is not supported by the compiler
- `SLEEF_DISABLE_AVX` : Disable support for x86 AVX
- `SLEEF_ENFORCE_AVX` : Build fails if AVX is not supported by the compiler
- `SLEEF_DISABLE_FMA4` : Disable support for x86 FMA4
- `SLEEF_ENFORCE_FMA4` : Build fails if FMA4 is not supported by the compiler
- `SLEEF_DISABLE_AVX2` : Disable support for x86 AVX2
- `SLEEF_ENFORCE_AVX2` : Build fails if AVX2 is not supported by the compiler
- `SLEEF_DISABLE_AVX512F` : Disable support for x86 AVX512F
- `SLEEF_ENFORCE_AVX512F` : Build fails if AVX512F is not supported by the compiler
- `SLEEF_DISABLE_SVE` : Disable support for AArch64 SVE
- `SLEEF_ENFORCE_SVE` : Build fails if SVE is not supported by the compiler
- `SLEEF_DISABLE_VSX` : Disable support for PowerPC VSX
- `SLEEF_ENFORCE_VSX` : Build fails if VSX is not supported by the compiler
- `SLEEF_DISABLE_VSX3` : Disable support for PowerPC VSX-3
- `SLEEF_ENFORCE_VSX3` : Build fails if VSX-3 is not supported by the compiler
- `SLEEF_DISABLE_VXE` : Disable support for System/390 VXE
- `SLEEF_ENFORCE_VXE` : Build fails if System/390 VXE is not supported by the compiler
- `SLEEF_DISABLE_VXE2` : Disable support for System/390 VXE2
- `SLEEF_ENFORCE_VXE2` : Build fails if System/390 VXE2 is not supported by the compiler

<!-- Repository links -->

[build_info_url]: https://sleef.org/compile.xhtml
[repo_url]: https://github.com/shibatch/sleef
[release_url]: https://github.com/shibatch/sleef/releases
[forge_url]: https://sourceforge.net/projects/sleef
