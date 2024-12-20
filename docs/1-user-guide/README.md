---
layout: default
title: User Guide
nav_order: 2
has_children: true
permalink: /1-user-guide/
---

<h1>User Guide</h1>

Guidelines on how to compile and install the library.

<h2>Table of contents</h2>

* [Preliminaries](#preliminaries)
* [Quick start](#quickstart)
* [Compiling and installing the library on Linux](#linux)
* [Compiling the library with Microsoft Visual C++](#MSVC)
* [Compiling and running "Hello SLEEF"](#hello)
* [Importing SLEEF into your project](#import)
* [Cross compilation for Linux](#cross_linux)
* [Cross compilation for iOS and Android](#cross)

<h2 id="preliminaries">Preliminaries</h2>

In order to build SLEEF, you need [CMake](http://www.cmake.org/), which is an
open-source and cross-platform building tool.

CMake works by allowing the developer to specify build parameters and rules in
a simple text file that cmake then processes to generate project files for the
actual native build tools (e.g. UNIX Makefiles, Microsoft Visual Studio, Apple
XCode, etc). If you are not already familiar with cmake, please refer to the
[official documentation](https://cmake.org/documentation/) or the [basic
introductions in the
wiki](https://gitlab.kitware.com/cmake/community/-/wikis/home).

<h2 id="quickstart">Quick start</h2>

You will find quick start instructions in the sources or via GitHub in the
[README.md](https://github.com/shibatch/sleef/blob/master/README.md#how-to-build-sleef)
file.

A more detailed description of CMake usage is provided in the [Build with
CMake](build-with-cmake) page, along with [a list of CMake
variables](build-with-cmake#sleef-variables) relevant to users.

<h2 id="linux">Compiling and installing the library on Linux</h2>

In order to build the library, you may want to install OpenMP (optional).

In order to test the library, you need to install:

* [the GNU MPFR Library](http://www.mpfr.org/) - libmpfr,
* [Libssl](https://wiki.openssl.org/index.php/Libssl_API) - openssl (recommended),
and you may want to install
* [FFTW](http://www.fftw.org/) - libfftw3 (recommended).

Availability of these libraries are checked upon execution of cmake.

Please run the following from the root directory of SLEEF:

```sh
sudo apt-get install libmpfr-dev libssl-dev libfftw3-dev
cmake -S . -B build/ ..
cmake --build build/ --clean-first -j
ctest --test-dir build/ -j
sudo cmake --install build/ --prefix /path/to/install/dir
```

In order to uninstall the libraries and headers, run the following command.

```sh
sudo xargs rm -v < build/install_manifest.txt
```

<h3 id="lto">Building the library with LTO support</h3>

You can build the library with [link time opimization(LTO)](../3-extra#lto)
support with the following commands. Note that you can only build static
libraries with LTO support. You also have to use the same compiler with the
same version to build the library and other source codes.

```sh
CC=gcc cmake -DBUILD_SHARED_LIBS=FALSE -DSLEEF_ENABLE_LTO=TRUE ..
```

In order to build the library with thinLTO support with clang, you need to
specify LLVM AR command that exactly corresponds to the clang compiler.

```sh
CC=clang-9 cmake -DBUILD_SHARED_LIBS=FALSE -DSLEEF_ENABLE_LTO=TRUE -DSLEEF_LLVM_AR_COMMAND=llvm-ar-9 ..
```

<h3 id="inline">Building the header files for inlining the whole SLEEF functions</h3>

[Header files for inlining the whole SLEEF functions](../3-extra#inline) can be
built with the following commands. With these header files, it may be easier to
inline the whole SLEEF functions than using LTO. You have to specify
`-ffp-contract=off` compiler option when compiling a source code that includes
one of these header files.

```sh
cmake -DSLEEF_BUILD_INLINE_HEADERS=TRUE ..
```

<h2 id="MSVC">Compiling the library with Microsoft Visual C++</h2>

You need Visual Studio 2019. Open developer command prompt for VS2019 and
change directory to sleef root. When configuring a build with CMake, you need to
use a specific generator: `cmake -G"Visual Studio 16 2019" ..` This generator
will create a proper solution `SLEEF.sln` under the build directory. You can
still use `cmake --build .` to build the library without opening Visual Studio.

Below is an example of commands for building SLEEF with Visual Studio.

```sh
mkdir build
cd build
cmake -G"Visual Studio 15 2017 Win64" ..    &:: If you are using VS2017
cmake -G"Visual Studio 16 2019" ..          &:: If you are using VS2019
cmake --build . --config Release -- /maxcpucount:1
```

<h3 id="cow">Compiling the library with Clang on Windows</h3>

You need Visual Studio 2019. Install ninja via VS2019 installer.  Download and
install clang on Windows from
[llvm.org](https://releases.llvm.org/download.html). Below is an example of
commands for building SLEEF with Clang on Windows.

```sh
"c:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -GNinja -DCMAKE_C_COMPILER:PATH="C:\Program Files\LLVM\bin\clang.exe" ..
cmake --build build --clean-first
```

<h2 id="hello">Compiling and running "Hello SLEEF"</h2>

Now, let's try compiling the [source code shown in Fig.
2.1](../src/hellox86.c).

```c
#include <stdio.h>
#include <x86intrin.h>
#include <sleef.h>
int main(int argc, char **argv) {
  double a[] = {2, 10};
  double b[] = {3, 20};
  __m128d va, vb, vc;
  va = _mm_loadu_pd(a);
  vb = _mm_loadu_pd(b);
  vc = Sleef_powd2_u10(va, vb);
  double c[2];
  _mm_storeu_pd(c, vc);
  printf("pow(%g, %g) = %g\n", a[0], b[0], c[0]);
  printf("pow(%g, %g) = %g\n", a[1], b[1], c[1]);
}
```
<p style="text-align:center;">
  Fig. 2.1: <a href="../src/hellox86.c">Source code for testing</a>
</p>

```sh
gcc hellox86.c -o hellox86 -lsleef
./hellox86
  pow(2, 3) = 8
  pow(10, 20) = 1e+20
```
<p style="text-align:center;">
  Fig. 2.2: Commands for compiling and executing hellox86.c
</p>

You may need to set `LD_LIBRARY_PATH` environment variable appropriately. If
you are trying to execute the program on Mac OSX or Windows, try copying the
DLLs to the current directory.

<h2 id="import">Importing SLEEF into your project</h2>

Below is an example [CMakeLists.txt](../src/CMakeLists.txt) for compiling the
above hellox86.c. CMake will automatically download SLEEF from GitHub
repository, and thus there is no need to include SLEEF in your software
package. If you prefer importing SLEEF as a submodule in git, you can use
`SOURCE_DIR` option instead of `GIT_REPOSITORY` option for
`ExternalProject_Add`.

```cmake
cmake_minimum_required(VERSION 3.5.1)
include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(libsleef
  GIT_REPOSITORY https://github.com/shibatch/sleef
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib
)

include_directories(${CMAKE_BINARY_DIR}/contrib/include)
link_directories(${CMAKE_BINARY_DIR}/contrib/lib)

add_executable(hellox86 hellox86.c)
add_dependencies(hellox86 libsleef)
target_link_libraries(hellox86 sleef)
```
<p style="text-align:center;">
  Fig. 2.3: <a href="../src/CMakeLists.txt">Example CMakeLists.txt</a>
</p>


<h2 id="cross_linux">Cross compilation for Linux</h2>

Two methods are used for cross-compiling SLEEF. Both rely on existing toolchain
files provided in the `toolchains/` directory.

Here are examples of cross-compiling SLEEF for the AArch64 on a platform with 
x86_64 and Linux OS:

<h3 id="method1">Method 1</h3>

Please run the following from the root directory of SLEEF:

1. First, compile the native SLEEF.
```bash
cmake -S . -B build-native

cmake --build build-native -j --clean-first
```

2. Cross-compile the target platform's SLEEF.
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=./toolchains/aarch64-gcc.cmake -DNATIVE_BUILD_DIR=$(pwd)/build-native/ -S . -B build

cmake --build build -j --clean-first
```

<h3 id="method2">Method 2</h3>

If running via an emulator like QEMU, there is no need to compile the native SLEEF.

Please run the following from the root directory of SLEEF:

1. Install qemu on Ubuntu.
```bash
sudo apt install -y qemu-user-static binfmt-support
```

2. Set the environment variable.
```bash
export SLEEF_TARGET_EXEC_USE_QEMU=ON
```

3. Set the dynamic linker/loader path.
```bash
# for AArch64
export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu/
```

4. Cross-compile the target platform's SLEEF.
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=./toolchains/aarch64-gcc.cmake -S . -B build

cmake --build build -j --clean-first
```


<h2 id="cross">Cross compilation for iOS and Android</h2>

SLEEF has preliminary support for iOS and Android. Here, "preliminary" means
that the library is only tested to be built, but not tested to work correctly
on the real devices. In order to cross compile the library, you need a cmake
tool chain file for the corresponding OS. The tool chain file for iOS can be
downloaded from
[https://github.com/leetal/ios-cmake](https://github.com/leetal/ios-cmake).
The tool chain file for Android is included in the SDK. You first need to build
the library for the host computer, then for the target OS. Below is an example
sequence of commands for cross compiling the library for iOS.

```sh
# Build natively first
cmake -S . -B build-native
cmake --build build-native -j --clean-first
# Then cross-compile for iOS
cmake  -S . -B build-cross -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake -DNATIVE_BUILD_DIR=$(pwd)/build-native -DSLEEF_DISABLE_MPFR=TRUE -DSLEEF_DISABLE_SSL=TRUE
```

Below is an example sequence of commands for cross compiling the library for
Android.

```sh
# Build natively first
cmake -S . -B build-native
cmake --build build-native -j --clean-first
# Then cross-compile for Android
cmake  -S . -B build-cross -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r21d/build/cmake/android.toolchain.cmake -DNATIVE_BUILD_DIR=$(pwd)/build-native -DANDROID_ABI=arm64-v8a
```

