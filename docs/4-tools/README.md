---
layout: default
title: Tools
nav_order: 5
permalink: /4-tools/
---

<h1>Tools</h1>

Other tools included in the package.

<h2>Table of contents</h2>

* [Testers for libm](#testerlibm)
* [Testers for DFT](#testerdft)
* [Tool for generating coefficients](#gencoef)
* [Benchmarking tool](#benchmark)

<h2 id="testerlibm">Libm tester</h2>

SLEEF libm has three kinds of testers, and each kind of testers has its own
role.

The first kind of testers consists of a tester and an IUT (which stands for
Implementation Under Test). The role for this tester is to perform a
perfunctory set of tests to check if the build is correct. In this test, the
functions in the library are tested if the evaluation error is within the
designed limit by comparing the returned values against high-precision
evaluation using [the GNU MPFR Library](http://www.mpfr.org/). The tester and
IUT are built as separate executables, and communicate with each other using a
pipe. Since these two are separate, the IUT can be implemented with an exotic
languages or on an operating system that does not support libraries required
for testing. It is also possible to perform a test over the network.

The second kind of testers are designed to run continuously. It repeats
randomly generating arguments for each function, and comparing the results of
each function to the results calculated with the corresponding function in the
MPFR library. This tester is expected to find bugs if it is run for
sufficiently long time. In these tests, we especially carefully check the error
of the trigonometric functions with arguments close to an integral multiple of
<i class="math">&pi;</i>/2.

The third kind of testers are for testing if bit-identical results are returned
from the functions that are supposed to return such results. The MD5 hash value
of all returned values from each function is calculated and checked if it
matches the precomputed value.

<h2 id="testerdft">DFT tester</h2>

SLEEF DFT has three kinds of testers. The first ones, named `naivetest`, compare
the results computed by SLEEF DFT with those by a naive DFT implementation.
These testers cannot be built with MSVC since complex data types are not
supported. The second testers, named `fftwtest`, compare the results of
computation between SLEEF DFT and FFTW. This test requires FFTW library. The
third testers, named `roundtriptest`, executes a forward transform followed by a
backward transform. Then, it compares the results with the original data.
While this test does not require external library and it runs on all
environment, there could be cases where this test does not find some flaw. The
roundtrip testers are used only if FFTW is not available.

<h2 id="gencoef">Gencoef</h2>

Gencoef is a small tool for generating the coefficients for polynomial
approximation used in the kernels.

In order to change the configurations, please edit `gencoefdp.c`. In the
beginning of the file, specifications of the parameters for generating
coefficients are listed. Please enable one of them by changing #if. Then, run
make to compile the source code. Run the gencoef, and it will show the
generated coefficients in a few minutes. It may take longer time depending on
the settings.

There are two phases of the program. The first phase is the regression for
minimizing the maximum relative error. This problem can be reduced to a linear
programming problem, and the Simplex method is used in this implementation.
This requires multi-precision calculation, and the implementation uses the MPFR
library. In this phase, it uses only a small number of values (specified by the
macro S, usually less than 100) within the input domain of the kernel function
to approximate the function. The function to approximate is given by FRFUNC
function. Specifying higher values for S does not always give better results.

The second phase is to optimize the coefficients so that it gives good accuracy
with double precision calculation. In this phase, it checks 10000 points
(specified by the macro Q) within the specified argument range to see if the
polynomial gives good error bounds. In some cases, the last few terms have to
be calculated in higher precision in order to achieve 1 ULP or better overall
accuracy, and this implementation can take care of that. The L parameter
specifies the number of high precision coefficients.

In some cases, it is desirable to fix the last few coefficients to values like
1 or 0.5. This can be specified if you define FIXCOEF0 macro.

Finding a set of good parameters is not a straightforward process.

<h2 id="benchmark">Benchmarking tool</h2>

This tool uses the [googlebench](https://github.com/google/benchmark) framework to benchmark SLEEF
functions.
It is integrated with SLEEF via CMake.
In order to build this tool automatically when SLEEF is
built, pass the `-DSLEEF_BUILD_BENCH=ON` CMake option when
setting up the build directory:
```sh
cmake -S . -B build -DSLEEF_BUILD_BENCH=ON
```
After building SLEEF:
```sh
cmake --build build -j
```
in `build/bin` folder you will find an executable named
benchsleef128.
Run this executable with `./build/bin/benchsleef128` in
order to obtain microbenchmarks for the functions in the project.
A filter option can also be provided to the executable.
This feature in inherited from googlebench, and takes
a regular expression, and executes only the benchmarks
whose name matches the regular expression.
The set of all the benchmarks available can be obtained
when running the benchmark tool when no filter is set
and corresponds to all the benchmarks listed in
`benchsleef.cpp`.
```sh
# Examples:
# * This will benchmark Sleef_sinf_u10 on all intervals enabled in the tool.
./build/bin/benchsleef128 --benchmark_filter=sinf_u10
# * This will benchmark all single precision sin functions (scalar, vector and sve if available):
./build/bin/benchsleef128 --benchmark_filter=sinf
# * This will benchmark all single precision vector functions:
./build/bin/benchsleef128 --benchmark_filter=vectorf
```
Note: all corresponds to all functions available in SLEEF and enabled in the benchmarks in this context.

This tool also supports multiple output formats, a feature
also inherited from googlebench framework.
The output formats available are `console`(default), `json` and `csv`.
```sh
# Examples:
# * This will print output in the terminal in json format:
./build/bin/benchsleef128 --benchmark_format=json
```

<h3 id="benchmark">Benchmarking on aarch64</h3>
If you're running SLEEF on a machine with SVE support the executable generated will have SVE benchmarks
available for functions specified in `benchsleef.cpp`.
<h3 id="benchmark">Benchmarking on x86</h3>
If you're running SLEEF on an x86 machine, two extra
executables may be built (according to feature detection):

```sh
./build/bin/benchsleef256
./build/bin/benchsleef512
```
These will benchmark 256bit and 512bit vector implementations
for vector functions respectively.
Note these executables can also be used to benchmark scalar
functions.

<h3 id="benchmark">Benchmarking libm</h3>
This tool can also benchmark libm functions listed in 
`benchlibm.cpp`.
To enable this, pass on extra build option `-DSLEEF_BUILD_BENCH_REF=ON`
(on top of `-DSLEEF_BUILD_BENCH` from before).
This will produce a new script `./build/bin/benchlibm128`.
You can interact with this in a similar fashion as described for 
`./build/bin/benchsleef128`.