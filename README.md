[![Build Status](https://travis-ci.org/shibatch/sleef.svg?branch=master)](https://travis-ci.org/shibatch/sleef)

In this library, functions for evaluating some elementary functions
are implemented. The algorithm is intended for efficient evaluation
utilizing SIMD instruction sets like SSE or AVX, but it is also fast
using usual scalar operations.

The package contains a few directories in which implementation in the
corresponding languages are contained. You can run "make test" in
order to test the functions in each directory.

The software is distributed under the Boost Software License, Version
1.0.  See accompanying file LICENSE.txt or copy at
http://www.boost.org/LICENSE_1_0.txt.
Contributions to this project are accepted under the same license.

Copyright Naoki Shibata and contributors 2010 - 2017.


Main download page : http://shibatch.sourceforge.net/

--

Compiling library with Microsoft Visual C++

Below is the instruction for compiling SLEEF with Microsoft Visual
C++.  Only 64bit architecture is supported. Only DLLs are built.


1. Install Visual Studio 2015 or later, along with Cygwin
2. Copy vcvars64.bat to a working directory.
   This file is usually in the following directory.
   C:\Program Files (x86)\MSVCCommunity2015\VC\bin\amd64
3. Add the following line at the end of vcvars64.bat
   if "%SHELL%"=="/bin/bash" c:\cygwin64\bin\bash.exe
4. Execute vcvars64.bat within the Cygwin bash shell.
5. Go to sleef-3.X directory
6. Run "make -f Makefile.vc"

--


History

3.0
* New API is defined
* Functions for DFT are added
* sincospi functions are added
* gencoef now supports single, extended and quad precision in addition to double precision
* Linux, Windows and Mac OS X are supported
* GCC, Clang, Intel Compiler, Microsoft Visual C++ are supported
* The library can be compiled as DLLs
* Files needed for creating a debian package are now included


2.121
* Renamed LICENSE_1_0.txt to LICENSE.txt

2.120
* Relicensed to Boost Software License Version 1.0

2.110
* The valid range of argument is extended for trig functions
* Specification of each functions regarding to the domain and accuracy is added
* A coefficient generation tool is added
* New testing tools are introduced
* Following functions returned incorrect values when the argument is very large or small : exp, pow, asinh, acosh
* SIMD xsin and xcos returned values more than 1 when FMA is enabled
* Pure C cbrt returned incorrect values when the argument is negative
* tan_u1 returned values with more than 1 ulp of error on rare occasions
* Removed support for Java language(because no one seems using this)

2.100 Added support for AVX-512F and Clang Extended Vectors.

2.90 Added ilogbf. All the reported bugs(listed below) are fixed.
* Log function returned incorrect values when the argument is very small.
* Signs of returned values were incorrect when the argument is signed zero.
* Tester incorrectly counted ULP in some cases.
* ilogb function returned incorrect values in some cases.

2.80 Added support for ARM NEON. Added higher accuracy single
precision functions : sinf_u1, cosf_u1, sincosf_u1, tanf_u1, asinf_u1,
acosf_u1, atanf_u1, atan2f_u1, logf_u1, and cbrtf_u1.

2.70 Added higher accuracy functions : sin_u1, cos_u1, sincos_u1,
tan_u1, asin_u1, acos_u1, atan_u1, atan2_u1, log_u1, and
cbrt_u1. These functions evaluate the corresponding function with at
most 1 ulp of error.

2.60 Added the remaining single precision functions : powf, sinhf,
coshf, tanhf, exp2f, exp10f, log10f, log1pf. Added support for FMA4
(for AMD Bulldozer). Added more test cases. Fixed minor bugs (which
degraded accuracy in some rare cases).

2.50 Added support for AVX2. SLEEF now compiles with ICC.

2.40 Fixed incorrect denormal/nonnumber handling in ldexp, ldexpf,
sinf and cosf. Removed support for Go language.

2.31 Added sincosf.

2.30 Added single precision functions : sinf, cosf, tanf, asinf,
acosf, atanf, logf, expf, atan2f and cbrtf.

2.20 Added exp2, exp10, expm1, log10, log1p, and cbrt.

2.10 asin() and acos() are back. Added ilogb() and ldexp(). Added
hyperbolic functions.  Eliminated dependency on frexp, ldexp, fabs,
isnan and isinf.

2.00 All of the algorithm has been updated. Both accuracy and speed
are improved since version 1.10. Denormal number handling is also
improved.

1.10 AVX support is added. Accuracy tester is added.

1.00 Initial release
