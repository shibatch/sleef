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

SLEEF has a tool for measuring and plotting execution time of each function in
the library. It consists of an executable for measurements, a makefile for
driving measurement and plotting, and a couple of scripts.

In order to start a measurement, you need to first build the executable for
measurement. CMake builds the executable along with the library. Please refer
to [compiling and installing the library](../1-user-guide) for this.

Then, change directory to `sleef/src/libm-benchmarks/`. You also need to set
the build directory to `BUILDDIR` environment variable. You also need Java
runtime environment.

```sh
export BUILDDIR=$PATH:`pwd`/../../build
```

Type "make measure". After compiling the tools, it will prompt a label for
measurement. After you input a label, measurement begins. After a measurement
finishes, you can repeat measurements under different configurations. If you
want to measure on a different computer, please copy the entire directory on to
that computer and continue measurements. If you have Intel Compiler installed
on your computer, you can type "make measureSVML" to measure the computation
time of SVML functions.

```sh
make measure
./measure.sh benchsleef
       ...
  Enter label of measurement(e.g. My desktop PC) : Skylake
  Measurement in progress. This may take several minutes.
  Sleef_sind2_u10
  Sleef_cosd2_u10
  Sleef_tand2_u10
  Sleef_sincosd2_u10
       ...
  Sleef_atanf8_u10
  Sleef_atan2f8_u10
  Sleef_atanf8_u35
  Sleef_atan2f8_u35

  Now, you can plot the results of measurement by 'make plot'.
  You can do another measurement by 'make measure'.
  You can start over by 'make restart'.
```

Then,

```sh
make plot
  javac ProcessData.java
  java ProcessData *dptrig*.out
  gnuplot script.out
  mv output.png trigdp.png
  java ProcessData *dpnontrig*.out
  gnuplot script.out
  mv output.png nontrigdp.png
  java ProcessData *sptrig*.out
  gnuplot script.out
  mv output.png trigsp.png
  java ProcessData *spnontrig*.out
  gnuplot script.out
  mv output.png nontrigsp.png
```

Then type `make plot` to generate graphs. You need to have JDK and gnuplot
installed on your computer. Four graphs are generated : trigdp.png,
nontrigdp.png, trigsp.png and nontrigsp.png. Please see our [benchmark
results](../5-performance/) for an example of generated graphs by this tool.

