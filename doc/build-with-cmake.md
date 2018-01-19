# Introduction

[Cmake](http://www.cmake.org/) is an open-source and cross-platform building
tool for software packages that provides easy managing of multiple build systems
at a time. It works by allowing the developer to specify build parameters and
rules in a simple text file that cmake then processes to generate project files
for the actual native build tools (e.g. UNIX Makefiles, Microsoft Visual Studio,
Apple XCode, etc). That means you can easily maintain multiple separate builds
for one project and manage cross-platform hardware and software complexity.

If you are not already familiar with cmake, please refer to the [official
documentation](https://cmake.org/documentation/) or the
[Basic Introductions](https://cmake.org/Wiki/CMake#Basic_Introductions) in the
wiki (recommended).

Before using CMake you will need to install/build the binaries on your system.
Most systems have cmake already installed or provided by the standard package
manager. If that is not the case for you, please
[download](https://cmake.org/download/) and install now.
For building SLEEF, version 3.4.3 is the minimum required.

# Quick start

1. Make sure cmake is available on the command-line.
```
$ cmake --version
(should display a version number greater than or equal to 3.4.3)
```

2. Download the tar from the
[software repository](http://shibatch.sourceforge.net/)
or checkout out the source code from the
[github repository](https://github.com/shibatch/sleef):
```
$ git clone https://github.com/shibatch/sleef
```

3. Make a separate directory to create an out-of-source build. SLEEF does not
allow for in-tree builds.
```
$ cd sleef-project
$ mkdir my-sleef-build && cd my-sleef-build
```

4. Run cmake to configure your project and generate the system to build it:
```
$ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_INSTALL_PREFIX=../my-sleef-install \
	..
```
This flag configures an optimised `libsleef` shared library build with basic
debug info.
By default, cmake will autodetect your system platform and configure the build
using the default parameters. You can control and modify these parameters by
setting variables when running cmake. See the list of
[options and variables](#build-customization) for customizing your build.

> NOTE: On **Windows**, you need to use a specific generator like this:
> `cmake -G"Visual Studio 14 2015 Win64" ..` specifying the Visual Studio version
> and targeting specifically `Win64` (to support compilation of AVX/AVX2)
> Check `cmake -G` to get a full list of supported Visual Studio project generators.
> This generator will create a proper solution `SLEEF.sln` under the build
> directory. 
> You can still use `cmake --build .` to build without opening Visual Studio.

5. Now that you have the build files created by cmake, proceed from the top
of the build directory:
```
$ make sleef
```

6. Install the library under ../my-sleef/install by running:
```
$ make install
```

7. You can execute the tests by running:
```
$ make test
```

# Build customization

Variables dictate how the build is generated; options are defined and undefined,
respectively, on the cmake command line like this:
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

## SLEEF Variables
