---
layout: default
title: CUDA
parent: Quadruple Precision
grand_parent: References
permalink: /2-references/quad/cuda
---

<h1>Quadruple precision Math Library reference (CUDA)</h1>

<h2>Table of contents</h2>

* [Tutorial](#tutorial)
* [Conversion functions](#conversion)
* [Comparison and selection functions](#comparison)
* [Math functions](#mathfunctions)

<h2 id="tutorial">Tutorial</h2>

Below is a [test code](../../src/hellocudaquad.cu) for the CUDA functions. CUDA
devices cannot directly compute with the QP FP data type. Thus, you have to use
`Sleef_quadx1` data type to retain a QP FP value in CUDA device codes. This
data type has the same structure as the QP FP data type, and you can directly
access the number by casting the pointer to the QP FP data type supported by
the compiler. Beware of the strict-aliasing rule in this case.

```c

#include <iostream>
#include <quadmath.h>

#include "sleefquadinline_cuda.h"

// Based on the tutorial code at https://developer.nvidia.com/blog/even-easier-introduction-cuda/

__global__ void pow_gpu(int n, Sleef_quadx1 *r, Sleef_quadx1 *x, Sleef_quadx1 *y) {
  int index = threadIdx.x, stride = blockDim.x;

  for (int i = index; i < n; i += stride)
    r[i] = Sleef_powq1_u10cuda(x[i], y[i]);
}

int main(void) {
  int N = 1 << 20;

  Sleef_quadx1 *rd, *xd, *yd;
  cudaMallocManaged(&rd, N*sizeof(Sleef_quadx1));
  cudaMallocManaged(&xd, N*sizeof(Sleef_quadx1));
  cudaMallocManaged(&yd, N*sizeof(Sleef_quadx1));

  __float128 *r = (__float128 *)rd, *x = (__float128 *)xd, *y = (__float128 *)yd;

  for (int i = 0; i < N; i++) {
    r[i] = 0.0;
    x[i] = 1.00001Q;
    y[i] = i;
  }
  pow_gpu<<<1, 256>>> (N, rd, xd, yd);

  cudaDeviceSynchronize();

  double maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabsq(r[i]-powq(x[i], y[i])));
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(yd);
  cudaFree(xd);
  cudaFree(rd);

  return 0;
```
<p style="text-align:center;">
<a class="underlined" href="../../src/hellocudaquad.cu">Source code for testing CUDA functions</a>
</p>

You may want to use both CPU and GPU functions in the same source code. This is
possible, as shown in [the following test code](../../src/hellocudaquad2.cu).
You cannot use the library version of the SLEEF functions in CUDA source codes.
Please include the header files for inlineable functions along with the header
file for CUDA functions. The I/O functions are defined in
`sleefquadinline_purec_scalar.h`.  You cannot use `SLEEF_QUAD_C` or `sleef_q`
in device functions.

```c

// nvcc -O3 hellocudaquad2.cu -I./include --fmad=false -Xcompiler -ffp-contract=off

#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <ctype.h>
#include <assert.h>
#include <emmintrin.h>

#include "sleefquadinline_sse2.h"
#include "sleefquadinline_purec_scalar.h"
#include "sleefquadinline_cuda.h"
#include "sleefinline_sse2.h"

// Based on the tutorial code at https://developer.nvidia.com/blog/even-easier-introduction-cuda/

__global__ void pow_gpu(int n, Sleef_quadx1 *r, Sleef_quadx1 *x, Sleef_quadx1 *y) {
  int index = threadIdx.x, stride = blockDim.x;

  for (int i = index; i < n; i += stride)
    r[i] = Sleef_powq1_u10cuda(x[i], y[i]);
}

int main(void) {
  int N = 1 << 20;

  Sleef_quadx1 *rd, *xd, *yd;
  cudaMallocManaged(&rd, N*sizeof(Sleef_quadx1));
  cudaMallocManaged(&xd, N*sizeof(Sleef_quadx1));
  cudaMallocManaged(&yd, N*sizeof(Sleef_quadx1));

  Sleef_quad *r = (Sleef_quad *)rd, *x = (Sleef_quad *)xd, *y = (Sleef_quad *)yd;

  //

  for (int i = 0; i < N; i++) {
    r[i] = Sleef_cast_from_doubleq1_purec(0);
    x[i] = Sleef_cast_from_doubleq1_purec(1.00001);
    y[i] = Sleef_cast_from_doubleq1_purec(i);
  }

  pow_gpu<<<1, 256>>>(N, rd, xd, yd);

  cudaDeviceSynchronize();

  Sleef_quadx2 maxError = Sleef_splatq2_sse2(Sleef_strtoq("0.0", NULL));

  for (int i = 0; i < N; i += 2) {
    Sleef_quadx2 r2 = Sleef_loadq2_sse2(&r[i]);
    Sleef_quadx2 x2 = Sleef_loadq2_sse2(&x[i]);
    Sleef_quadx2 y2 = Sleef_loadq2_sse2(&y[i]);

    Sleef_quadx2 q = Sleef_fabsq2_sse2(Sleef_subq2_u05sse2(r2, Sleef_powq2_u10sse2(x2, y2)));
    maxError = Sleef_fmaxq2_sse2(maxError, q);
  }

  Sleef_printf("Max error: %Qg\n",
               Sleef_fmaxq1_purec(Sleef_getq2_sse2(maxError, 0), Sleef_getq2_sse2(maxError, 1)));

  //

  cudaFree(yd);
  cudaFree(xd);
  cudaFree(rd);

  return 0;
```
<p style="text-align:center;">
<a class="underlined" href="../../src/hellocudaquad2.cu">Source code for testing CUDA functions with CPU functions</a>
</p>

<h2 id="conversion">Conversion functions</h2>

### Convert QP number to double-precision number

```c
#include <sleefquadinline_cuda.h>

__device__ double Sleef_cast_to_doubleq1_cuda( Sleef_quadx1 a );
```

These functions convert a QP FP value to a double-precision value.

### Convert double-precision number to QP number

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_cast_from_doubleq1_cuda( double a );
```

These functions convert a double-precision value to a QP FP value.

### Convert QP number to 64-bit signed integer

```c
#include <sleefquadinline_cuda.h>

__device__ int64_t Sleef_cast_to_int64q1_cuda( Sleef_quadx1 a );
```

These functions convert a QP FP value to a 64-bit signed integer.

### Convert 64-bit signed integer to QP number

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_cast_from_int64q1_cuda( int64_t a );
```

These functions convert a 64-bit signed integer to a QP FP value.

### Convert QP number to 64-bit unsigned integer

```c
#include <sleefquadinline_cuda.h>

__device__ uint64_t Sleef_cast_to_uint64q1_cuda( Sleef_quadx1 a );
```

These functions convert a QP FP value to a 64-bit signed integer.

### Convert 64-bit unsigned integer to QP number

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_cast_from_uint64q1_cuda( uint64_t a );
```

These functions convert a 64-bit unsigned integer to a QP FP value.

<h2 id="comparison">Comparison and selection functions</h2>

### QP comparison functions

```c
#include <sleefquadinline_cuda.h>

__device__ int32_t Sleef_icmpltq1_cuda(Sleef_quadx1 a, Sleef_quadx1 b);
__device__ int32_t Sleef_icmpleq1_cuda(Sleef_quadx1 a, Sleef_quadx1 b);
__device__ int32_t Sleef_icmpgtq1_cuda(Sleef_quadx1 a, Sleef_quadx1 b);
__device__ int32_t Sleef_icmpgeq1_cuda(Sleef_quadx1 a, Sleef_quadx1 b);
__device__ int32_t Sleef_icmpeqq1_cuda(Sleef_quadx1 a, Sleef_quadx1 b);
__device__ int32_t Sleef_icmpneq1_cuda( Sleef_quadx1 a, Sleef_quadx1 b );
```

These are the vectorized functions of [comparison
functions](../quad#basicComparison).

### QP comparison functions of the second kind

```c
#include <sleefquadinline_cuda.h>

__device__ int32_t Sleef_icmpq1_cuda( Sleef_quadx1 a, Sleef_quadx1 b );
```

These are the vectorized functions
of [Sleef_icmpq1_purec](../quad#sleef_icmpq1_purec).

### Check orderedness

```c
#include <sleefquadinline_cuda.h>

__device__ int32_t Sleef_iunordq1_cuda( Sleef_quadx1 a, Sleef_quadx1 b );
```

These are the vectorized functions
of [Sleef_iunordq1_purec](../quad#sleef_iunordq1_purec).

### Select elements

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_iselectq1_cuda( int32_t c, Sleef_quadx1 a, Sleef_quadx1 b );
```

These are the vectorized functions that operate in the same way as the ternary operator.

<h2 id="mathfunctions">Math functions</h2>

### QP functions for basic arithmetic operations

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_addq1_u05cuda(Sleef_quadx1 a, Sleef_quadx1 b);
__device__ Sleef_quadx1 Sleef_subq1_u05cuda(Sleef_quadx1 a, Sleef_quadx1 b);
__device__ Sleef_quadx1 Sleef_mulq1_u05cuda(Sleef_quadx1 a, Sleef_quadx1 b);
__device__ Sleef_quadx1 Sleef_divq1_u05cuda(Sleef_quadx1 a, Sleef_quadx1 b);
__device__ Sleef_quadx1 Sleef_negq1_cuda( Sleef_quadx1 a );
```

These are the vectorized functions of [the basic arithmetic
operations](../quad#basicArithmetic).

### Square root functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_sqrtq1_u05cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_sqrtq1_u05purec](../quad#sleef_sqrtq1_u05purec).

### Sine functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_sinq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_sinq1_u10purec](../quad#sleef_sinq1_u10purec).

### Cosine functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_cosq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_cosq1_u10purec](../quad#sleef_cosq1_u10purec).

### Tangent functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_tanq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_tanq1_u10purec](../quad#sleef_tanq1_u10purec).

### Arc sine functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_asinq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_asinq1_u10purec](../quad#sleef_asinq1_u10purec).

### Arc cosine functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_acosq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_acosq1_u10purec](../quad#sleef_acosq1_u10purec).

### Arc tangent functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_atanq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_atanq1_u10purec](../quad#sleef_atanq1_u10purec).

### Base-<i>e</i> exponential functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_expq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_expq1_u10purec](../quad#sleef_expq1_u10purec).

### Base-2 exponential functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_exp2q1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_exp2q1_u10purec](../quad#sleef_exp2q1_u10purec).

### Base-10 exponentail

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_exp10q1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_exp10q1_u10purec](../quad#sleef_exp10q1_u10purec).

### Base-<i>e</i> exponential functions minus 1

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_expm1q1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_expm1q1_u10purec](../quad#sleef_expm1q1_u10purec).

### Natural logarithmic functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_logq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_logq1_u10purec](../quad#sleef_logq1_u10purec).

### Base-2 logarithmic functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_log2q1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_log2q1_u10purec](../quad#sleef_log2q1_u10purec).

### Base-10 logarithmic functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_log10q1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_log10q1_u10purec](../quad#sleef_log10q1_u10purec).

### Logarithm of one plus argument

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_log1pq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_log1pq1_u10purec](../quad#sleef_log1pq1_u10purec).

### Power functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_powq1_u10cuda( Sleef_quadx1 x, Sleef_quadx1 y );
```

These are the vectorized functions
of [Sleef_powq1_u10purec](../quad#sleef_powq1_u10purec).

### Hyperbolic sine functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_sinhq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_sinhq1_u10purec](../quad#sleef_sinhq1_u10purec).

### Hyperbolic cosine functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_coshq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_coshq1_u10purec](../quad#sleef_coshq1_u10purec).

### Hyperbolic tangent functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_tanhq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_tanhq1_u10purec](../quad#sleef_tanhq1_u10purec).

### Inverse hyperbolic sine functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_asinhq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_asinhq1_u10purec](../quad#sleef_asinhq1_u10purec).

### Inverse hyperbolic cosine functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_acoshq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_acoshq1_u10purec](../quad#sleef_acoshq1_u10purec).

### Inverse hyperbolic tangent functions

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_atanhq1_u10cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_atanhq1_u10purec](../quad#sleef_atanhq1_u10purec).

### Round to integer towards zero

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_truncq1_cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_truncq1_purec](../quad#sleef_truncq1_purec).

### Round to integer towards minus infinity

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_floorq1_cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_floorq1_purec](../quad#sleef_floorq1_purec).

### Round to integer towards plus infinity

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_ceilq1_cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_ceilq1_purec](../quad#sleef_ceilq1_purec).

### Round to integer away from zero

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_roundq1_cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_roundq1_purec](../quad#sleef_roundq1_purec).

### Round to integer, ties round to even

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_rintq1_cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_rintq1_purec](../quad#sleef_rintq1_purec).

### Absolute value

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_fabsq1_cuda( Sleef_quadx1 a );
```

These are the vectorized functions
of [Sleef_fabsq1_purec](../quad#sleef_fabsq1_purec).

### Copy sign of a number

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_copysignq1_cuda( Sleef_quadx1 x, Sleef_quadx1 y );
```

These are the vectorized functions
of [Sleef_copysignq1_purec](../quad#sleef_copysignq1_purec).

### Maximum of two numbers

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_fmaxq1_cuda( Sleef_quadx1 x, Sleef_quadx1 y );
```

These are the vectorized functions
of [Sleef_fmaxq1_purec](../quad#sleef_fmaxq1_purec).

### Minimum of two numbers

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_fminq1_cuda( Sleef_quadx1 x, Sleef_quadx1 y );
```

These are the vectorized functions
of [Sleef_fminq1_purec](../quad#sleef_fminq1_purec).

### Positive difference

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_fdimq1_u05cuda( Sleef_quadx1 x, Sleef_quadx1 y );
```

These are the vectorized functions
of [Sleef_fdimq1_u05purec](../quad#sleef_fdimq1_u05purec).

### Floating point remainder

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_fmodq1_cuda( Sleef_quadx1 x, Sleef_quadx1 y );
```

These are the vectorized functions
of [Sleef_fmodq1_purec](../quad#sleef_fmodq1_purec).

### Floating point remainder

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_remainderq1_cuda( Sleef_quadx1 x, Sleef_quadx1 y );
```

These are the vectorized functions
of [Sleef_remainderq1_purec](../quad#sleef_remainderq1_purec).

### Split a number to fractional and integral components

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_frexpq1_cuda( Sleef_quadx1 x, int32_t * ptr );
```

These are the vectorized functions
of [Sleef_frexpq1_purec](../quad#sleef_frexpq1_purec).

### Break a number into integral and fractional parts

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_modfq1_cuda( Sleef_quadx1 x, Sleef_quadx1 * ptr );
```

These are the vectorized functions
of [Sleef_modfq1_purec](../quad#sleef_modfq1_purec).

### 2D Euclidian distance

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_hypotq1_u05cuda( Sleef_quadx1 x, Sleef_quadx1 y );
```

These are the vectorized functions
of [Sleef_hypotq1_u05purec](../quad#sleef_hypotq1_u05purec).

### Fused multiply and accumulate

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_fmaq1_u05cuda( Sleef_quadx1 x, Sleef_quadx1 y, Sleef_quadx1 z );
```

These are the vectorized functions
of [Sleef_fmaq1_u05purec](../quad#sleef_fmaq1_u05purec).

### Multiply by integral power of 2

```c
#include <sleefquadinline_cuda.h>

__device__ Sleef_quadx1 Sleef_ldexpq1_cuda( Sleef_quadx1 x, int32_t e );
```

These are the vectorized functions
of [Sleef_ldexpq1_purec](../quad#sleef_ldexpq1_purec).

### Integer exponent of an FP number

```c
#include <sleefquadinline_cuda.h>

__device__ int32_t Sleef_ilogbq1_cuda( Sleef_quadx1 x );
```

These are the vectorized functions
of [Sleef_ilogbq1_purec](../quad#sleef_ilogbq1_purec).

