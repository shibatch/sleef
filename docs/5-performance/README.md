---
layout: default
title: Performance
nav_order: 6
permalink: /5-performance/
---

<h1>Performance</h1>

<h2>Vectorized math lib</h2>

These graphs show comparison of the execution time between
[SLEEF](https://github.com/shibatch/sleef)-3.2 compiled with GCC-7.2 and Intel
SVML included in Intel C Compiler 18.0.1.

The execution time of each function is measured by executing each function 10^8
times and taking the average time. Each time a function is executed, a
uniformly distributed random number is set to each element of the argument
vector (each element is set a different value).  The ranges of the random number
for each function are shown below. Argument vectors are generated before the
measurement, and the time to generate random argument vectors is not included
in the execution time.

* Trigonometric functions : [0, 6.28] and [0, 10^6] for double-precision functions. [0, 6.28] and [0, 30000] for single-precision functions.
* Log : [0, 10^300] and [0, 10^38] for double-precision functions and single-precision functions, respectively.
* Exp : [-700, 700] and [-100, 100] for double-precision functions and single-precision functions, respectively.
* Pow : [-30, 30] for both the first and the second arguments.
* Asin : [-1, 1]
* Atan : [-10, 10]
* Atan2 : [-10, 10] for both the first and the second arguments.

The accuracy of SVML functions can be chosen by compiler options, not the
function names. `-fimf-max-error=1.0` option is specified to icc to obtain the
1-ulp-accuracy results, and `-fimf-max-error=5.0` option is used for the
5-ulp-accuracy results.

Those results are measured on a PC with Intel Core i7-6700 CPU @ 3.40GHz with
Turbo Boost turned off. The CPU should be always running at 3.4GHz during the
measurement.

<p style="font-size:1.2em; margin-top:1.0cm;">
  <b>Click graphs to magnify</b>.
</p>

<p style="text-align:center; margin-bottom:2cm;">
  <a class="nothing" href="../img/trigdp.png">
    <img src="../img/trigdp.png" alt="Performance graph for DP trigonometric functions"/>
  </a>
  <br />
  Fig. 6.1: Execution time of double precision trigonometric functions
</p>

<p style="text-align:center; margin-bottom:2cm;">
  <a class="nothing" href="../img/trigsp.png">
    <img src="../img/trigsp.png" alt="Performance graph for SP trigonometric functions"/>
  </a>
  <br />
  Fig. 6.2: Execution time of single precision trigonometric functions
</p>

<p style="text-align:center; margin-bottom:2cm;">
  <a class="nothing" href="../img/nontrigdp.png">
    <img src="../img/nontrigdp.png" alt="Performance graph for other DP functions"/>
  </a>
  <br />
  Fig. 6.3: Execution time of double precision log, exp, pow and inverse trigonometric functions
</p>

<p style="text-align:center; margin-bottom:2cm;">
  <a class="nothing" href="../img/nontrigsp.png">
    <img src="../img/nontrigsp.png" alt="Performance graph for other SP functions"/>
  </a>
  <br />
  Fig. 6.4: Execution time of single precision log, exp, pow and inverse trigonometric functions
</p>


<h2>Discrete Fourier transform</h2>

Below is the result of performance comparison between SleefDFT and FFTW3.

The graphs show the performance of complex transform by both
libraries, with the following settings.

* Compiler : gcc version 14.2.0 (Ubuntu 14.2.0-4ubuntu2~24.04)
* CPU : Ryzen 9 7950X (clock frequency fixed at 4.5GHz)
* SLEEF build option : -DSLEEF_BUILD_DFT=True -DSLEEFDFT_ENABLE_STREAM=True -DSLEEFDFT_MAXBUTWIDTH=7
* FFTW version 3.3.10-1ubuntu3

The vertical axis represents the performance in Mflops calculated in
the way indicated in the FFTW web site. The horizontal axis represents
log2 of the size of transform.

Execution plans were made with SLEEF_MODE_MEASURE mode and
FFTW_MEASURE mode, respectively.

<p style="text-align:center; margin-bottom:2cm;">
  <a class="nothing" href="../img/dftzen4dp.png">
    <img src="../img/dftzen4dp.png" alt="Performance graph for DP DFT"/>
  </a>
  <br />
  Fig. 6.5: Performance of transform in double precision
</p>

<p style="text-align:center; margin-bottom:2cm;">
  <a class="nothing" href="../img/dftzen4sp.png">
    <img src="../img/dftzen4sp.png" alt="Performance graph for SP DFT"/>
  </a>
  <br />
  Fig. 6.6: Performance of transform in single precision
</p>
