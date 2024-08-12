---
layout: default
title: Extra
nav_order: 4
permalink: /3-extra/
---

<h1>Extra</h1>

Additional notes on SLEEF components.

<h2>Table of contents</h2>

* [Frequently asked questions](#faq)
* [Vectorizing calls to scalar functions](#vectorizing)
* [About the GNUABI version of the library](#gnuabi)
* [Using link time optimization](#lto)
* [Using header files of inlinable functions](#inline)
* [Utilizing SLEEF for WebAssembly](#wasm)
* [How the dispatcher works](#dispatcher)
* [About libsleefscalar](#libsleefscalar)
* [ULP, gradual underflow and flush-to-zero mode](#ulp)
* [Explanatory source code for the modified Payne Hanek reduction method](#paynehanek)
* [About the logo](#logo)

<h2 id="faq">Frequently asked questions</h2>

<b>Q1:</b> Is the scalar functions in SLEEF faster than the
corresponding functions in the standard C library?

<b>A1:</b> No. Todays standard C libraries are very well optimized,
and there is small room for further optimization. The reason why
SLEEF is fast is that it computes directly with SIMD registers and
ALUs. This is not simple as it sounds, because conditional branches
have to be eliminated in order to take full advantage of SIMD
computation. If the algorithm requires conditional branches
according to the argument, it must prepare for the cases where the
elements in the input vector contain both values that would make a
branch happen and not happen. This would spoil the advantage of SIMD
computation, because each element in a vector would require a
different code path.

<br/>

<b>Q2:</b> Do the trigonometric functions (e.g. sin) in SLEEF return
correct values for the whole range of inputs?

<b>A2:</b> Yes. SLEEF does implement a [vectorized version of Payne Hanek range
reduction](#paynehanek), and all the trigonometric functions return a correct
value with the specified accuracy.

<br/>

<b>Q3:</b> What can I do to make sleef run faster?

<b>A3:</b> The most important thing is to choose the fastest available vector
extension. SLEEF is optimized for computers with
[FMA](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation#Fused_multiply%E2%80%93add)
instructions, and it runs slow on Ivy Bridge or older CPUs and Atom, that do
not have FMA instructions. If you are not sure, use the dispatcher. [The
dispatcher in SLEEF is not slow](#dispatcher). If you want to further speed up
computation, try using [LTO](#lto). By using LTO, the compiler fuses the code
within the library to the code calling the library functions, and this
sometimes results in considerable performance boost. In this case, you should
not use the dispatcher, and you should use the same compiler with the same
version to build SLEEF and the program against which SLEEF is linked.

<h2 id="vectorizing">Vectorizing calls to scalar functions</h2>

Recent x86_64 gcc can
[auto-vectorize](https://gcc.gnu.org/projects/tree-ssa/vectorization.html)
calls to functions. In order to utilize this functionality, OpenMP SIMD pragmas
can be added to declarations of scalar functions like `Sleef_sin_u10` by
defining `SLEEF_ENABLE_OMP_SIMD` macro before including `sleef.h` on x86_64
computers. With these pragmas, gcc can use its auto-vectorizer to vectorize
calls to these scalar functions. For example, [the following
code](../src/sophomore.c) can be vectorized by gcc-10.

```c
#include <stdio.h>

#define SLEEF_ENABLE_OMP_SIMD
#include "sleef.h"

#define N 65536
#define M (N + 3)

static double func(double x) { return Sleef_pow_u10(x, -x); }

double int_simpson(double a, double b) {
   double h = (b - a) / M;
   double sum_odd = 0.0, sum_even = 0.0;
   for(int i = 1;i <= M-3;i += 2) {
     sum_odd  += func(a + h * i);
     sum_even += func(a + h * (i + 1));
   }
   return h / 3 * (func(a) + 4 * sum_odd + 2 * sum_even + 4 * func(b - h) + func(b));
}

int main() {
  double sum = 0;
  for(int i=1;i<N;i++) sum += Sleef_pow_u10(i, -i);
  printf("%g %g\n", int_simpson(0, 1), sum);
}
```

```sh
gcc-10 -fopenmp -ffast-math -mavx2 -O3 sophomore.c -lsleef -S -o- | grep _ZGV
        call    _ZGVdN4vv_Sleef_pow_u10@PLT
        call    _ZGVdN4vv_Sleef_pow_u10@PLT
        call    _ZGVdN4vv_Sleef_pow_u10@PLT
        call    _ZGVdN4vv_Sleef_pow_u10@PLT
        call    _ZGVdN4vv_Sleef_pow_u10@PLT
        call    _ZGVdN4vv_Sleef_pow_u10@PLT
        call    _ZGVdN4vv_Sleef_pow_u10@PLT
```

<h2 id="gnuabi">About the GNUABI version of the library</h2>

The GNUABI version of the library (`libsleefgnuabi.so`) is built for x86 and
aarch64 architectures. This library provides an API compatible with
[libmvec](https://sourceware.org/glibc/wiki/libmvec) in glibc, and the API
comforms to the [x86 vector
ABI](https://sourceware.org/glibc/wiki/libmvec?action=AttachFile&amp;do=view&amp;target=VectorABI.txt),
[AArch64 vector ABI](https://developer.arm.com/docs/101129/latest) and [Power
Vector ABI](https://github.com/power8-abi-doc/vector-function-abi/). The
auto-vectorizer in x86_64 gcc is capable of vectorizing calls to the standard
math functions and generates calls to
[libmvec](https://sourceware.org/glibc/wiki/libmvec). The GNUABI version of
SLEEF library can be used as a substitute for libmvec.

```c
#include <stdio.h>
#include <math.h>

#define N 65536
#define M (N + 3)

static double func(double x) { return pow(x, -x); }

double int_simpson(double a, double b) {
   double h = (b - a) / M;
   double sum_odd = 0.0, sum_even = 0.0;
   for(int i = 1;i <= M-3;i += 2) {
     sum_odd  += func(a + h * i);
     sum_even += func(a + h * (i + 1));
   }
   return h / 3 * (func(a) + 4 * sum_odd + 2 * sum_even + 4 * func(b - h) + func(b));
}

int main() {
  double sum = 0;
  for(int i=1;i<N;i++) sum += pow(i, -i);
  printf("%g %g\n", int_simpson(0, 1), sum);
}
```

For example, [the above code](../src/sophomore2.c) can be linked against
libsleefgnuabi as shown below. You have to specify `-lsleefgnuabi` compiler
option before `-lm` option.

```sh
gcc-10 -ffast-math -O3 sophomore2.c -lsleefgnuabi -lm -L./lib
ldd a.out
        linux-vdso.so.1 (0x00007ffd0c5ff000)
        libsleefgnuabi.so.3 => not found
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f73f5f98000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f73f5da6000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f73f60fb000)
LD_LIBRARY_PATH=./lib ./a.out
  1.29127 1.29129
```

<h2 id="lto">Using link time optimization</h2>

[Link time optimization
(LTO)](https://en.wikipedia.org/wiki/Interprocedural_optimization) is a
functionality implemented in gcc, clang and other compilers for optimizing
across translation units (or source files.)  This can sometimes dramatically
improve the performance of the code, because it is capable of fusing library
functions into the code calling those functions. The build system in SLEEF
supports LTO and thus it can be built with LTO support by just specifying
`-DSLEEF_ENABLE_LTO=TRUE` cmake option. However, there are a few things to
note in order to get the optimal performance.

1. You should not use the dispatcher with LTO. Dispatchers prevent the functions from being fused with LTO.
2. You have to use the same compiler with the same version to build the library and your code.
3. You cannot build shared libraries with LTO.

<h2 id="inline">Using header files of inlinable functions</h2>

Although LTO is considered to be a smart technique for improving the
performance of the library functions, there are difficulties in using
this functionality in real situations. One of the reasons is that
people still need to use old compilers to build their projects. SLEEF
can generate header files in which the library functions are all
defined as inline functions. This can be compiled with old compilers.
In theory, inline functions should give similar performance to LTO,
but in reality, inline functions are better. In order to generate
those header files, specify `-DSLEEF_BUILD_INLINE_HEADERS=TRUE` cmake
option. Below is an example code utilizing the generated header files
for SSE2 and AVX2. You cannot use a dispatcher with these header
files.

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <x86intrin.h>

#include <sleefinline_sse2.h>
#include <sleefinline_avx2128.h>

int main(int argc, char **argv) {
  __m128d va = _mm_set_pd(2, 10);
  __m128d vb = _mm_set_pd(3, 20);

  __m128d vc = Sleef_powd2_u10sse2(va, vb);

  double c[2];
  _mm_storeu_pd(c, vc);

  printf("%g, %g\n", c[0], c[1]);

  __m128d vd = Sleef_powd2_u10avx2128(va, vb);

  double d[2];
  _mm_storeu_pd(d, vd);

  printf("%g, %g\n", d[0], d[1]);
}
```

```sh
gcc-10 -ffp-contract=off -O3 -march=native helloinline.c -I./include
./a.out
  1e+20, 8
  1e+20, 8
nm -g a.out
  00000000000036a0 R Sleef_rempitabdp
  0000000000003020 R Sleef_rempitabsp
  0000000000003000 R _IO_stdin_used
                   w _ITM_deregisterTMCloneTable
                   w _ITM_registerTMCloneTable
  000000000000d010 D __TMC_END__
  000000000000d010 B __bss_start
                   w __cxa_finalize@@GLIBC_2.2.5
  000000000000d000 D __data_start
  000000000000d008 D __dso_handle
                   w __gmon_start__
  00000000000020a0 T __libc_csu_fini
  0000000000002030 T __libc_csu_init
                   U __libc_start_main@@GLIBC_2.2.5
                   U __printf_chk@@GLIBC_2.3.4
  000000000000d010 D _edata
  000000000000d018 B _end
  00000000000020a8 T _fini
  0000000000001f40 T _start
  000000000000d000 W data_start
  0000000000001060 T main
```

<h2 id="wasm">Utilizing SLEEF for WebAssembly</h2>

Since [Emscripten](https://emscripten.org/) supports SSE2 intrinsics, the SSE2
inlinable function header can be used for
[WebAssembly](https://webassembly.org/).

```c
#include <stdio.h>
#include <emmintrin.h>

#include "sleefinline_sse2.h"

int main(int argc, char **argv) {
  double a[] = {2, 10};
  double b[] = {3, 20};

  __m128d va, vb, vc;

  va = _mm_loadu_pd(a);
  vb = _mm_loadu_pd(b);

  vc = Sleef_powd2_u10sse2(va, vb);

  double c[2];

  _mm_storeu_pd(c, vc);

  printf("pow(%g, %g) = %g\n", a[0], b[0], c[0]);
  printf("pow(%g, %g) = %g\n", a[1], b[1], c[1]);
}
```

```sh
emcc -O3 -msimd128 -msse2 hellowasm.c
../node-v15.7.0-linux-x64/bin/node --experimental-wasm-simd ./a.out.js
  pow(2, 3) = 8
  pow(10, 20) = 1e+20
```

<h2 id="dispatcher">How the dispatchers work</h2>

SLEEF implements versions of functions that are implemented with each vector
extension of the architecture. A dispatcher is a function that dynamically
selects the fastest implementatation for the computer it runs. The dispatchers
in SLEEF are designed to have very low overhead.

Fig. 7.1 shows a simplified code of our dispatcher. There is only one exported
function `mainFunc`. When `mainFunc` is called for the first time,
`dispatcherMain` is called internally, since `funcPtr` is initialized to the
pointer to `dispatcherMain` (line 14). It then detects if the CPU supports SSE
4.1 (line 7), and rewrites `funcPtr` to a pointer to the function that utilizes
SSE 4.1 or SSE 2, depending on the result of CPU feature detection (line 10).
When `mainFunc` is called for the second time, it does not execute the
`dispatcherMain`. It just executes the function pointed by the pointer stored
in `funcPtr` during the execution of `dispatcherMain`.

There are advantages in our dispatcher. The first advantage is that it does not
require any compiler-specific extension. The second advantage is simplicity.
There are only 18 lines of simple code. Since the dispatchers are completely
separated for each function, there is not much room for bugs to get in.

The third advantage is low overhead. You might think that the overhead is one
function call including execution of the prologue and the epilogue. However,
modern compilers are smart enough to eliminate redundant execution of the
prologue, epilogue and return instruction. The actual overhead is just one jmp
instruction, which has very small overhead since it is not conditional. This
overhead is likely hidden by out-of-order execution.

The fourth advantage is thread safety. There is only one variable shared among
threads, which is `funcPtr`. There are only two possible values for this
pointer variable. The first value is the pointer to the `dispatcherMain`, and
the second value is the pointer to either `funcSSE4`, depending on the
availability of extensions. Once `funcPtr` is substituted with the pointer to
`funcSSE4`, it will not be changed in the future. It should be easy to confirm
that the code works in all the cases.

```c
static double (*funcPtr)(double arg);

static double dispatcherMain(double arg) {
    double (*p)(double arg) = funcSSE2;

#if the compiler supports SSE4.1
    if (SSE4.1 is available on the CPU) p = funcSSE4;
#endif

    funcPtr = p;
    return (*funcPtr)(arg);
}

static double (*funcPtr)(double arg) = dispatcherMain;

double mainFunc(double arg) {
    return (*funcPtr)(arg);
}
```
<p style="text-align:center; margin-bottom: 1.0cm;">
  Fig. 7.1: Simplified code of our dispatcher
</p>


<h2 id="libsleefscalar">About libsleefscalar</h2>

The scalar functions like `Sleef_sin_u10` were the functions implemented in
`sleefdp.c` and `sleefsp.c`. These functions are provided to make it easier to
understand how each sleef function works. These functions are now moved to
`libsleefscalar`, because they run slower than the scalar functions implemented
in `sleefsimddp.c` and `sleefsimdsp.c`. As of version 3.6, the scalar functions
whose names do not end with `purec</b> or <b>purecfma` are dispatchers that
choose from the scalar functions whose names end with `purec` or `purecfma`.
For example, `Sleef_sin_u10` in `libsleef` is now a dispatcher that chooses
from `Sleef_sind1_u10purec` and `Sleef_sind1_u10purecfma`.

<h2 id="ulp">ULP, gradual underflow and flush-to-zero mode</h2>

ULP stands for "unit in the last place", which is sometimes used for
representing accuracy of calculation. 1 ULP is the distance between the two
closest floating point number, which depends on the exponent of the FP number.
The accuracy of calculation by reputable math libraries is usually between 0.5
and 1 ULP. Here, the accuracy means the largest error of calculation. SLEEF
math library provides multiple accuracy choices for most of the math functions.
Many functions have 3.5-ULP and 1-ULP versions, and 3.5-ULP versions are faster
than 1-ULP versions. If you care more about execution speed than accuracy, it
is advised to use the 3.5-ULP versions along with -ffast-math or "unsafe math
optimization" options for the compiler.

Note that 3.5 ULPs of error is small enough in many applications. If you do not
manage the error of computation by carefully ordering floating point operations
in your code, you would easily have that amount of error in the computation
results.

In IEEE 754 standard, underflow does not happen abruptly when the exponent
becomes zero. Instead, when a number to be represented is smaller than a
certain value, a denormal number is produced which has less precision. This is
sometimes called gradual underflow. On some processor implementation, a
flush-to-zero mode is used since it is easier to implement by hardware. In
flush-to-zero mode, numbers smaller than the smallest normalized number are
replaced with zero. FP operations are not IEEE-754 conformant if a
flush-to-zero mode is used. A flush-to-zero mode influences the accuracy of
calculation in some cases. The smallest normalized precision number can be
referred with `DBL_MIN` for double precision, and `FLT_MIN` for single
precision. The naming of these macros is a little bit confusing because
`DBL_MIN` is not the smallest double precision number.

You can see known maximum errors in math functions in glibc at [this
page.](http://www.gnu.org/software/libc/manual/html_node/Errors-in-Math-Functions.html)

<h2 id="paynehanek">Explanatory source code for our modified Payne Hanek reduction method</h2>

In order to evaluate a trigonometric function with a large argument, an
argument reduction method is used to find an FP remainder of dividing the
argument `x` by &pi;. We devised a variation of the Payne-Hanek argument
reduction method which is suitable for vector computation. Fig. 7.2 shows [an
explanatory source code](../src/ph.c) for this method.  See [our
paper](http://dx.doi.org/10.1109/TPDS.2019.2960333) for the details.

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpfr.h>

typedef struct { double x, y; } double2;
double2 dd(double d) { double2 r = { d, 0 }; return r; }
int64_t d2i(double d) { union { double f; int64_t i; } tmp = {.f = d }; return tmp.i; }
double i2d(int64_t i) { union { double f; int64_t i; } tmp = {.i = i }; return tmp.f; }
double upper(double d) { return i2d(d2i(d) & 0xfffffffff8000000LL); }
double clearlsb(double d) { return i2d(d2i(d) & 0xfffffffffffffffeLL); }

double2 ddrenormalize(double2 t) {
  double2 s = dd(t.x + t.y);
  s.y = t.x - s.x + t.y;
  return s;
}

double2 ddadd(double2 x, double2 y) {
  double2 r = dd(x.x + y.x);
  double v = r.x - x.x;
  r.y = (x.x - (r.x - v)) + (y.x - v) + (x.y + y.y);
  return r;
}

double2 ddmul(double x, double y) {
  double2 r = dd(x * y);
  r.y = fma(x, y, -r.x);
  return r;
}

double2 ddmul2(double2 x, double2 y) {
  double2 r = ddmul(x.x, y.x);
  r.y += x.x * y.y + x.y * y.x;
  return r;
}

// This function computes remainder(a, PI/2)
double2 modifiedPayneHanek(double a) {
  double table[4];
  int scale = fabs(a) > 1e+200 ? -128 : 0;
  a = ldexp(a, scale);

  // Table genration

  mpfr_set_default_prec(2048);
  mpfr_t pi, m;
  mpfr_inits(pi, m, NULL);
  mpfr_const_pi(pi, GMP_RNDN);

  mpfr_d_div(m, 2, pi, GMP_RNDN);
  mpfr_set_exp(m, mpfr_get_exp(m) + (ilogb(a) - 53 - scale));
  mpfr_frac(m, m, GMP_RNDN);
  mpfr_set_exp(m, mpfr_get_exp(m) - (ilogb(a) - 53));

  for(int i=0;i<4;i++) {
    table[i] = clearlsb(mpfr_get_d(m, GMP_RNDN));
    mpfr_sub_d(m, m, table[i], GMP_RNDN);
  }

  mpfr_clears(pi, m, NULL);

  // Main computation

  double2 x = dd(0);
  for(int i=0;i<4;i++) {
    x = ddadd(x, ddmul(a, table[i]));
    x.x = x.x - round(x.x);
    x = ddrenormalize(x);
  }

  double2 pio2 = { 3.141592653589793*0.5, 1.2246467991473532e-16*0.5 };
  x = ddmul2(x, pio2);
  return fabs(a) < 0.785398163397448279 ? dd(a) : x;
}
```
<p style="text-align:center; margin-bottom: 1.0cm;">
  <a href="../src/ph.c">Fig. 7.2: Explanatory source code for our modified Payne Hanek reduction method</a>
</p>



<h2 id="logo">About the logo</h2>

It is a soup ladle. A sleef means a soup ladle in Dutch.

<br/>

<p style="text-align:center; margin-top:1cm;">
  <a class="nothing" href="../img/sleef-logo.svg">
    <img src="../img/sleef-logo.svg" alt="logo" width="40%" height="40%" />
  </a>
  <br />
  Fig. 7.2: SLEEF logo
</p>

