---
layout: default
title: User Guide
nav_order: 2
has_children: true
permalink: /1-user-guide/
---

<h1>Compiling and installing the library</h1>

<h2>Table of contents</h2>

* [Preliminaries](#preliminaries)
* [Quick start](#quickstart)
* [Compiling and installing the library on Linu](#linux)
* [Compiling the library with Microsoft Visual C++](#MSVC)
* [Compiling and running "Hello SLEEF"](#hello)
* [Importing SLEEF into your project](#import)
* [Cross compilation for iOS and Android](#cross)

<h2 id="preliminaries">Preliminaries</h2>

In order to build SLEEF, you need <a class="underlined"
href="http://www.cmake.org/">CMake</a>, which is an open-source and
cross-platform building tool. In order to test the library, it is
better to have <a class="underlined" href="http://www.mpfr.org/">the
GNU MPFR Library</a>, <a class="underlined"
href="https://wiki.openssl.org/index.php/Libssl_API">Libssl</a> and
<a class="underlined" href="http://www.fftw.org/">FFTW</a>.

CMake works by allowing the developer to specify build parameters and
rules in a simple text file that cmake then processes to generate
project files for the actual native build tools (e.g. UNIX Makefiles,
Microsoft Visual Studio, Apple XCode, etc). If you are not already
familiar with cmake, please refer to the
<a class="underlined" href="https://cmake.org/documentation/">official documentation</a>
or
the <a class="underlined" href="https://gitlab.kitware.com/cmake/community/-/wikis/home">basic
introductions in the wiki</a>.

<h2 id="quickstart">Quick start</h2>

You will find quick start instructions in the sources or via GitHub in the
[README.md](https://github.com/shibatch/sleef/blob/master/README.md#how-to-build-sleef)
file.

A more detailed description of CMake usage is provided in the
[Build with CMake](build-with-cmake) page, along with
[a list of CMake variables](build-with-cmake#sleef-variables)
relevant to users.

<h2 id="linux">Compiling and installing the library on Linux</h2>

In order to build the library, you may want to install OpenMP (optional).
In order to test the library, you need to install libmpfr, libssl and
libfftw3 (optional). Availability of these libraries are checked upon
execution of cmake.

Please run the following from the root directory of SLEEF:

```sh
sudo apt-get install libmpfr-dev libssl-dev libfftw3-dev
cmake -S . -B build/ ..
cmake --build build/ --clean-first -j
ctest --test-dir build/ -j
sudo cmake --install build/ --prefix /path/to/install/dir
```

In order to uninstall the libraries and headers, run the following command.

<pre class="command" style="margin-top: 1em;">$ sudo xargs rm -v < build/install_manifest.txt</pre>

<h3 id="lto">Building the library with LTO support</h3>

You can build the library with <a class="underlined"
href="../3-extra#lto">link time opimization(LTO)</a> support
with the following commands. Note that you can only build static
libraries with LTO support. You also have to use the same compiler
with the same version to build the library and other source codes.

```sh
CC=gcc cmake -DBUILD_SHARED_LIBS=FALSE -DSLEEF_ENABLE_LTO=TRUE ..
```

In order to build the library with thinLTO support with clang, you
need to specify LLVM AR command that exactly corresponds to the
clang compiler.

```sh
CC=clang-9 cmake -DBUILD_SHARED_LIBS=FALSE -DSLEEF_ENABLE_LTO=TRUE -DSLEEF_LLVM_AR_COMMAND=llvm-ar-9 ..
```

<h3 id="inline">Building the header files for inlining the whole SLEEF functions</h3>

<p>
  <a class="underlined" href="additional.xhtml#inline">Header files
  for inlining the whole SLEEF functions</a> can be built with the
  following commands. With these header files, it may be easier to
  inline the whole SLEEF functions than using LTO. You have to specify
  "-ffp-contract=off" compiler option when compiling a source code
  that includes one of these header files.
</p>

<pre class="command">$ cmake -DSLEEF_BUILD_INLINE_HEADERS=TRUE ..</pre>

<h2 id="MSVC">Compiling the library with Microsoft Visual C++</h2>

<p class="noindent">
  You need Visual Studio 2019. Open developer command prompt for
  VS2019 and change directory to sleef-3.X. When configuring a build
  with cmake, you need to use a specific generator: `cmake -G"Visual
  Studio 16 2019" ..` This generator will create a proper solution
  `SLEEF.sln` under the build directory. You can still use `cmake
  --build .` to build the library without opening Visual Studio.
</p>

<p>
  Below is an example of commands for building SLEEF with Visual
  Studio.
</p>

<pre class="command">D:\sleef-3.X> mkdir build
D:\sleef-3.X> cd build
D:\sleef-3.X\build> cmake -G"Visual Studio 15 2017 Win64" ..    &amp;:: If you are using VS2017
D:\sleef-3.X\build> cmake -G"Visual Studio 16 2019" ..          &amp;:: If you are using VS2019
D:\sleef-3.X\build> cmake --build . --config Release -- /maxcpucount:1</pre>


<h3 id="cow">Compiling the library with Clang on Windows</h3>

<p class="noindent">
  You need Visual Studio 2019. Install ninja via VS2019 installer.
  Download and install clang on Windows from <a class="underlined"
  href="https://releases.llvm.org/download.html">llvm.org</a>. Below
  is an example of commands for building SLEEF with Clang on Windows.
</p>

<pre class="command">D:\sleef-3.X> "c:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
D:\sleef-3.X> cmake -S . -B build -GNinja -DCMAKE_C_COMPILER:PATH="C:\Program Files\LLVM\bin\clang.exe" ..
D:\sleef-3.X> cmake --build build --clean-first</pre>


<h2 id="hello">Compiling and running "Hello SLEEF"</h2>

<p class="noindent">
  Now, let's try compiling the <a class="underlined"
  href="hellox86.c">source code shown in Fig. 2.1</a>.
</p>

<pre class="code">
<code>#include &lt;stdio.h&gt;</code>
<code>#include &lt;x86intrin.h&gt;</code>
<code>#include &lt;sleef.h&gt;</code>
<code></code>
<code>int main(int argc, char **argv) {</code>
<code>  double a[] = {2, 10};</code>
<code>  double b[] = {3, 20};</code>
<code></code>
<code>  __m128d va, vb, vc;</code>
<code>  </code>
<code>  va = _mm_loadu_pd(a);</code>
<code>  vb = _mm_loadu_pd(b);</code>
<code></code>
<code>  vc = Sleef_powd2_u10(va, vb);</code>
<code></code>
<code>  double c[2];</code>
<code></code>
<code>  _mm_storeu_pd(c, vc);</code>
<code></code>
<code>  printf("pow(%g, %g) = %g\n", a[0], b[0], c[0]);</code>
<code>  printf("pow(%g, %g) = %g\n", a[1], b[1], c[1]);</code>
<code>}</code>
</pre>
<p style="text-align:center;">
  Fig. 2.1: <a href="hellox86.c">Source code for testing</a>
</p>



<p style="margin-top: 2cm;">
  Fig.2.2 shows typical commands for compiling and executing the hello
  code on Linux computers.
</p>

<pre class="command" style="margin-top: 2.0em; margin-bottom: 0.5cm;">$ gcc hellox86.c -o hellox86 -lsleef
$ ./hellox86
pow(2, 3) = 8
pow(10, 20) = 1e+20
$ &block;</pre>
<p style="text-align:center;">
  Fig. 2.2: Commands for compiling and executing hellox86.c
</p>

<p style="margin-top: 2cm;">
  You may need to set LD_LIBRARY_PATH environment variable
  appropriately. If you are trying to execute the program on Mac OSX
  or Windows, try copying the DLLs to the current directory.
</p>

<h2 id="import">Importing SLEEF into your project</h2>


<p class="noindent">
  Below is an example <a class="underlined"
  href="CMakeLists.txt">CMakeLists.txt</a> for compiling the above
  hellox86.c. CMake will automatically download SLEEF from GitHub
  repository, and thus there is no need to include SLEEF in your
  software package. If you prefer importing SLEEF as a submodule in
  git, you can use SOURCE_DIR option instead of GIT_REPOSITORY option
  for ExternalProject_Add.
</p>

<pre class="code">
<code>cmake_minimum_required(VERSION 3.5.1)</code>
<code>include(ExternalProject)</code>
<code>find_package(Git REQUIRED)</code>
<code></code>
<code>ExternalProject_Add(libsleef</code>
<code>  GIT_REPOSITORY https://github.com/shibatch/sleef</code>
<code>  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib</code>
<code>)</code>
<code></code>
<code>include_directories(${CMAKE_BINARY_DIR}/contrib/include)</code>
<code>link_directories(${CMAKE_BINARY_DIR}/contrib/lib)</code>
<code></code>
<code>add_executable(hellox86 hellox86.c)</code>
<code>add_dependencies(hellox86 libsleef)</code>
<code>target_link_libraries(hellox86 sleef)</code>
</pre>
<p style="text-align:center;">
  Fig. 2.3: <a href="CMakeLists.txt">Example CMakeLists.txt</a>
</p>


<h2 id="cross">Cross compilation for iOS and Android</h2>
