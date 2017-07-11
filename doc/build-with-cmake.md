Sleef is moving from the existing configure/make system to using
[Cmake](http://www.cmake.org/).

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

Before using CMake you will need to install or build the binaries on your system.
Most systems have cmake already installed or provided by the standard package
manager. If that is not the case for you, please [download](https://cmake.org/download/)
and install now. For building SLEEF, version 3.4.3 is the minimum required.

# Quick start

1. Make sure cmake is available on the command-line.
```
$ cmake --version
(should display a version number greater than or equal to 3.4.3)
```

2. Download the tar from the [software repository](http://shibatch.sourceforge.net/)
or checkout out the source code from the [github repository](https://github.com/shibatch/sleef):
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
$ cmake ..
```
By default, cmake will autodetect your system platform and configure the build
using the default parameters. You can control and modify these parameters by
setting variables when running cmake. See the list of [options and variables](#build-customization)
for customizing your build.

5. Now that you have the build files created by cmake, proceed from the top
of the build directory:
```
$ make sleef
```
Running this command will create the shared library `libsleef`. On UNIX you
should see the following files in the `my-sleef-build/lib` directory.
```
libsleef.3.0.so
libsleef.3.dylib -> libsleef.3.0.so
libsleef.dylib -> libsleef.3.so
```

`
TODO: Add make install instructions.
`

# Build customization
`
TODO: Populate lists.
`

## CMake Variables
## SLEEF Variables
