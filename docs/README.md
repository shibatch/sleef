---
layout: home
title: Home
nav_order: 1
permalink: /
---

<p style="text-align:center; margin-top:1cm;">
  <a class="nothing" href="img/sleef-logo.svg">
    <img src="img/sleef-logo.svg" alt="logo" width="50%" height="50%" />
  </a>
</p>


<h1>Home</h1>

<h2>Table of contents</h2>

  [Overview](#overview) |
  [Supported environment](#environment) |
  [Credit](#credit) |
  [Partners](#partners) |
  [License](#license) |
  [History](#history) |
  [Related](#related) |
  [Publication](#publication)

<h2 id="overview">Overview</h2>

SLEEF stands for SIMD Library for Evaluating Elementary Functions. It implements
manually [vectorized](https://en.wikipedia.org/wiki/Automatic_vectorization)
versions of all C99 real floating point math functions. It can utilize SIMD
instructions that are available on modern processors. SLEEF is designed to
effciently perform computation with SIMD instructions by reducing the use of
conditional branches and scatter/gather memory access. Our
[benchmarks](5-performance) show that the performance of SLEEF is comparable to
that of the best commercial library.

Unlike closed-source commercial libraries, SLEEF is designed to work with
various architectures, operating systems and compilers. It is distributed under
[the Boost Software License](http://www.boost.org/users/license.html) , which is
a permissive open source license. SLEEF can be easily ported to other
architectures by writing a helper file, which is a thin abstraction layer of
SIMD intrinsics. SLEEF also provides [dispatchers](3-extra#dispatcher) that
automatically choose the best subroutines for the computer on which the library
is executed. In order to further optimize the application code that calls SLEEF
functions, [link time optimization(LTO)](3-extra#lto) can be used to reduce the
overhead of functions calls, and the build system of SLEEF supports usage of
LTO. The library also has a functionality to generate [header
files](3-extra#inline) in which the library functions are all defined as inline
functions. SLEEF can be used for [GPGPU](2-references/libm/cuda) and
[WebAssembly](3-extra#wasm) with these header files.  In addition to the
vectorized functions, SLEEF provides scalar functions. Calls to these scalar
SLEEF functions can be [auto-vectorized](3-extra#vectorizing) by GCC.

The library contains implementations of all C99 real FP math functions in double
precision and single precision. Different accuracy of the results can be chosen
for a subset of the elementary functions; for this subset there are versions
with up to 1 [ULP](3-extra#ulp) error (which is the maximum error, not the
average) and even faster versions with a few ULPs of error. For non-finite
inputs and outputs, the functions return correct results as specified in the C99
standard. All the functions in the library are [thoroughly
tested](4-tools#testerlibm) and confirmed that the evaluation error is within
the designed limit by comparing the returned values against high-precision
evaluation using the GNU MPFR Library.

As of version 3.6, SLEEF also includes a [quad-precision math
library](2-references/quad) . This library includes fully vectorized IEEE 754
quadruple-precision (QP) functions that correspond to the standard C math
functions. It also includes I/O functions for converting between QP numbers and
strings.

SLEEF also includes a library of [discrete Fourier
transform(DFT)](2-references/dft) . These subroutines are fully vectorized,
heavily unrolled, and parallelized in such a way that modern SIMD instructions
and multiple cores can be utilized for efficient computation. It has an API
similar to that of FFTW for easy migration. The subroutines can utilize long
vectors up to 2048 bits. The helper files for abstracting SIMD intrinsics are
shared with SLEEF libm, and thus it is easy to port the DFT subroutines to other
architectures. [Preliminary results of
benchmark](https://github.com/shibatch/sleef/wiki/DFT-Performance) are now
available.

<h2 id="environment">Supported environments</h2>

This library supports the following architectures :

* [x86](2-references/libm/x86) - SSE2, SSE4.1, AVX, AVX2+FMA3, AVX512F
* [AArch64](2-references/libm/aarch64) - Advanced SIMD, SVE
* [AArch32](2-references/libm/aarch32) - NEON
* [PowerPC64](2-references/libm/ppc64) - VSX (POWER8), VSX-3 (POWER9)
* [System/390](2-references/libm/s390x) - VXE (z14), VXE2 (z15)
* RISC-V - RVV1, RVV2
* [CUDA](2-references/libm/cuda)
* [WebAssembly](3-extra#wasm) - SSE2

The supported combinations of the architecture, operating system and
compiler are shown in Table 1.1.

<table style="text-align:center;" align="center">
  <tr align="center">
    <td class="caption">Table 1.1: Environment support matrix</td>
  </tr>
  <tr align="center">
    <td>
      <table class="lt">
        <tr>
          <td class="lt-hl"></td>
          <td class="lt-hl"></td>
          <td class="lt-hl"></td>
          <td class="lt-hl"></td>
          <td class="lt-hl"></td>
        </tr>
	<tr>
	  <td class="lt-br"></td>
	  <td class="lt-blr" align="center">GCC</td>
	  <td class="lt-br" align="center">Clang</td>
	  <td class="lt-br" align="center">Intel Compiler</td>
	  <td class="lt-b" align="center">MSVC</td>
	</tr>
        <tr>
          <td class="lt-hl"></td>
	  <td class="lt-hl"></td>
          <td class="lt-hl"></td>
          <td class="lt-hl"></td>
          <td class="lt-hl"></td>
        </tr>
	<tr>
	  <td class="lt-r" align="left">x86_64, Linux</td>
	  <td class="lt-lr" align="center">Supported</td>
	  <td class="lt-r" align="center">Supported</td>
	  <td class="lt-r" align="center">Supported</td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">AArch64, Linux</td>
	  <td class="lt-lr" align="center">Supported</td>
	  <td class="lt-r" align="center">Supported</td>
	  <td class="lt-r" align="center">N/A</td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">x86_64, macOS</td>
	  <td class="lt-lr" align="center">Supported(*2)</td>
	  <td class="lt-r" align="center">Supported(*2)</td>
	  <td class="lt-r" align="center"></td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">x86_64, Windows</td>
	  <td class="lt-lr" align="center">Supported(Cygwin)(*3)</td>
	  <td class="lt-r" align="center"><a class="underlined" href="1-user-guide#cow">Supported</a></td>
	  <td class="lt-r" align="center"></td>
	  <td class="lt-" align="center"><a class="underlined" href="1-user-guide#MSVC">Supported</a></td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">AArch32, Linux</td>
	  <td class="lt-lr" align="center">Supported(*1)</td>
	  <td class="lt-r" align="center">Supported(*1)</td>
	  <td class="lt-r" align="center">N/A</td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">PowerPC (64 bit), Linux</td>
	  <td class="lt-lr" align="center">Supported</td>
	  <td class="lt-r" align="center">Supported</td>
	  <td class="lt-r" align="center">N/A</td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">System/390 (64 bit), Linux</td>
	  <td class="lt-lr" align="center">Supported</td>
	  <td class="lt-r" align="center">Supported</td>
	  <td class="lt-r" align="center">N/A</td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">x86_64, FreeBSD</td>
	  <td class="lt-lr" align="center"></td>
	  <td class="lt-r" align="center">Supported</td>
	  <td class="lt-r" align="center">N/A</td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">x86 (32 bit), Linux</td>
	  <td class="lt-lr" align="center">Supported</td>
	  <td class="lt-r" align="center">Supported</td>
	  <td class="lt-r" align="center"></td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">AArch64, macOS</td>
	  <td class="lt-lr" align="center">Supported</td>
	  <td class="lt-r" align="center">Supported</td>
	  <td class="lt-r" align="center">N/A</td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">AArch64, Android</td>
	  <td class="lt-lr" align="center"></td>
	  <td class="lt-r" align="center"><a class="underlined" href="1-user-guide#cross">Preliminary</a></td>
	  <td class="lt-r" align="center">N/A</td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-br" align="left">AArch64, iOS</td>
	  <td class="lt-blr" align="center"></td>
	  <td class="lt-br" align="center"><a class="underlined" href="1-user-guide#cross">Preliminary</a></td>
	  <td class="lt-br" align="center">N/A</td>
	  <td class="lt-b" align="center">N/A</td>
	</tr>
	<tr>
	  <td class="lt-r" align="left">RISC-V (64-bit), Linux</td>
	  <td class="lt-lr" align="center">Supported</td>
	  <td class="lt-r" align="center">Supported</td>
	  <td class="lt-r" align="center">N/A</td>
	  <td class="lt-" align="center">N/A</td>
	</tr>
      </table>
    </td>
  </tr>
</table>

The supported compiler versions are as follows.

* GCC : version 5 and later
* Clang : version 6 and later
* Intel Compiler : ICC version 17
* MSVC : Visual Studio 2019

(*1) NEON has only single precision support. The computation results are not in
full accuracy because NEON is not IEEE 754-compliant.

(*2) LTO is not supported.

(*3) AVX functions are not supported for Cygwin, because AVX is not supported by
Cygwin ABI.  SLEEF also builds with MinGW for Windows on x86, but only DFT can
be tested for now.

(*4) Some compiler versions simply do not support certain vector extensions, for
instance SVE is only supported for gcc version 9 onwards. Similarly, the RISC-V
interface in SLEEF is based on version 1.0 of the intrinsics, which is only
supported from llvm version 17 and gcc version 14 onwards. Toolchain files
provide some information on supported compiler versions.

All functions in the library are thread safe unless otherwise noted.

<h2 id="credit">Credit</h2>

* The main developer is [Naoki Shibata](https://github.com/shibatch)
  ([shibatch@users.sourceforge.net](mailto:shibatch@users.sourceforge.net))
  formerly at Nara Institute of Science and Technology.

* [Pierre Blanchard](https://github.com/blapie), at Arm Ltd. is the current
  global maintainer for SLEEF, alongside
  [Joana Cruz](https://github.com/joanaxcruz) and
  [Joe Ramsay](https://github.com/joeramsay).

* [Ludovic Henry](https://github.com/luhenry) at Rivos Inc.  participated in
  adding support for RISC-V vector extensions as well as a whole new framework
  for CI tests based on Github Actions. They provided substantial help to
  improve maintainership of the library. Thanks to the RISC-V community,
  namely [Eric Love](https://github.com/ericlove) (SiFive),
  [@sh1boot](https://github.com/sh1boot), and
  [@GlassOfWhiskey](https://github.com/GlassOfWhiskey) for the help in
  extending SLEEF support to RISC-V vector extensions.

* [Francesco Petrogalli](https://github.com/fpetrogalli) at ARM Ltd.
  contributed the helper for AArch64 (helperadvsimd.h, helpersve.h) and GNUABI
  interface of the library. He also worked on migrating the build system to
  CMake, and reviewed the code, gave precious comments and suggestions.

* [Hal Finkel](https://github.com/hfinkel)  at Argonne Leadership Computing
  Facility is now working on [importing and adapting SLEEF as an LLVM
  runtime](https://reviews.llvm.org/D24951). He also gave precious comments.

* [Diana Bite](https://github.com/diaena) at University of Manchester and
  [Alexandre Mutel](https://github.com/xoofx) at Unity Technologies worked
  on migrating the build system to CMake.

* [Martin Krastev](https://github.com/blu) at Chaos Group contributed faster
  implementation of fmin and fmax functions.

* [Mo Zhou](https://github.com/cdluminate) is managing packages for
  [Debian](https://tracker.debian.org/pkg/sleef) and
  [UbuntuPPA](https://launchpad.net/~lumin0/+archive/ubuntu/sleef).

* [And many more contributors you can find listed on
  GitHub.](https://github.com/shibatch/sleef/graphs/contributors)

<h2 id="partners">Partner institutes and corporations</h2>

|               |                   |
|:-------------:|:------------------|
| [![IBM logo](img/ibm-logo.svg)](https://ibm.com) | As the leading company in a wide range of information technologies, [IBM](https://ibm.com/) participates through David Edelsohn. |
| [![ARM logo](img/arm-logo.svg)](https://www.arm.com) | As the leading IP company in semiconductors design, [ARM](https://www.arm.com/) participates through Pierre Blanchard, Joe Ramsay and Joana Cruz. |
| [![Unity Technologies logo](img/unity-logo.svg)](https://unity3d.com) | As the leading company in developing a video game engine, [Unity Technologies](https://unity3d.com/) participates through Alexandre Mutel. |

<h2 id="license">License</h2>

SLEEF is distributed under [Boost Software License Version
1.0](http://www.boost.org/LICENSE_1_0.txt).

|               |                   |
|:-------------:|:------------------|
| [![open source logo](img/osi-logo.svg)](https://opensource.org) | Boost Software License is [OSI-certified](https://opensource.org/licenses/BSL-1.0). See [this page](http://www.boost.org/users/license.html) for more information about Boost Software License. |

<h2 id="history">History</h2>

See <a href="https://github.com/shibatch/sleef/blob/master/CHANGELOG.md">Changelog</a> for a full history of changes to SLEEF.

<h2 id="related">Related projects and links</h2>

* SLEEF is adopted in [the Burst Compiler in Unity](https://youtu.be/QkM6zEGFhDY?t=2328),
  and also in the [Android version](https://youtu.be/WnJV6J-taIM?t=426).
* SLEEF is now included in [Arm Compiler for HPC](https://developer.arm.com/documentation/101458/1930/Vector-math-routines/Vector-math-routines-in-Arm-C-C---Compiler).
* SLEEF is adopted in [PyTorch](https://github.com/pytorch/pytorch/pull/6725).
* SLEEF is adopted in [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html#appendix-acknowledgements).

* SLEEF is adopted in the following projects.

  * [A pure Julia port of the SLEEF math library](https://github.com/musm/SLEEF.jl)
  * [A Rust FFI bindings to SLEEF](https://docs.rs/crate/sleef-sys/)
  * [MenuetOS](http://www.menuetos.net/sc151.html)
  * [RawTherapee](https://github.com/Beep6581/RawTherapee/tree/dev/rtengine)
  * [John's ECMA-55 Minimal BASIC Compiler](http://buraphakit.sourceforge.net/BASIC.shtml)
  * [Portable Computing Language(pocl)](https://github.com/pocl/pocl/tree/master/lib/kernel/sleef)
  * [The Region Vectorizer(Compiler Design Lab at Saarland University)](https://github.com/cdl-saarland/rv)
  * [Agenium Scale NSIMD](https://agenium-scale.github.io/nsimd)
  * [J Language](https://code.jsoftware.com/wiki/System/ReleaseNotes/J902)
  * [SIMD Everywhere](https://github.com/simd-everywhere/simde)
  * [Minocore](https://github.com/dnbaker/minocore)
  * [OctaSine](https://github.com/greatest-ape/OctaSine)
  * [Simdeez](https://github.com/jackmott/simdeez)

<h2 id="publication">Publication</h2>

* Naoki Shibata and Francesco Petrogalli : <B>SLEEF: A Portable Vectorized Library of C Standard Mathematical Functions,</B> <I>in IEEE Transactions on Parallel and Distributed Systems,</I> [DOI:10.1109/TPDS.2019.2960333](https://doi.org/10.1109/TPDS.2019.2960333) (Dec. 2019). [[PDF](https://arxiv.org/pdf/2001.09258)]
* Francesco Petrogalli and Paul Walker : <B>LLVM and the automatic vectorization of loops invoking math routines: -fsimdmath</B>, <I>2018 IEEE/ACM 5th Workshop on the LLVM Compiler Infrastructure in HPC (LLVM-HPC), pp. 30-38.,</I> <a  href="https://doi.org/10.1109/LLVM-HPC.2018.8639354">DOI:10.1109/LLVM-HPC.2018.8639354</a> (Nov. 2018). [[PDF](https://sc18.supercomputing.org/proceedings/workshops/workshop_files/ws_llvmf106s2-file1.pdf)]

