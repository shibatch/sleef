---
layout: default
title: CUDA
parent: Single & Double Precision
grand_parent: References
permalink: /2-references/libm/cuda
---

<h1>Single & Double Precision Math library reference (CUDA)</h1>

<h2>Table of contents</h2>

* [Tutorial](#tutorial)
* [Trigonometric functions](#trig)
* [Power, exponential, and logarithmic functions](#pow)
* [Inverse trigonometric functions](#invtrig)
* [Hyperbolic functions and inverse hyperbolic functions](#hyp)
* [Error and gamma functions](#eg)
* [Nearest integer functions](#nearint)
* [Other functions](#other)

<h2 id="tutorial">Tutorial</h2>

The CUDA functions in SLEEF are provided as an [inlinable include
file](../../3-extra#inline). Below is a [test code](../../src/hellocuda.cu) for
the CUDA functions.

```c
#include <iostream>
#include <math.h>

#include "sleefinline_cuda.h"

// Based on the tutorial code at https://developer.nvidia.com/blog/even-easier-introduction-cuda/

__global__ void pow_gpu(int n, double *r, double *x, double *y)
{
  int index = threadIdx.x, stride = blockDim.x;

  for (int i = index; i < n; i += stride)
    r[i] = Sleef_powd1_u10cuda(x[i], y[i]);
}

int main(void)
{
  int N = 1 << 20;

  double *r, *x, *y;
  cudaMallocManaged(&amp;r, N*sizeof(double));
  cudaMallocManaged(&amp;x, N*sizeof(double));
  cudaMallocManaged(&amp;y, N*sizeof(double));

  for (int i = 0; i < N; i++) {
    r[i] = 0.0;
    x[i] = 1.00001;
    y[i] = i;
  }

  pow_gpu<<<1, 256>>>(N, r, x, y);

  cudaDeviceSynchronize();

  double maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(r[i]-pow(x[i], y[i])));
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(y);
  cudaFree(x);
  cudaFree(r);

  return 0;
}
```
<p style="text-align:center;">
  <a class="underlined" href="../../src/hellocuda.cu">Source code for testing CUDA functions</a>
</p>

<h2 id="trig">Trigonometric Functions</h2>

### Vectorized double precision sine function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_sind1_u10cuda(double a);
```

This is the CUDA function of [Sleef_sin_u10](../libm#sleef_sin_u10) with the same accuracy specification.

### Vectorized single precision sine function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_sinf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_sinf_u10](../libm#sleef_sinf_u10) with the same accuracy specification.

### Vectorized double precision sine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_sind1_u35cuda(double a);
```

This is the CUDA function of [Sleef_sin_u35](../libm#sleef_sin_u35) with the same accuracy specification.

### Vectorized single precision sine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_sinf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_sinf_u35](../libm#sleef_sinf_u35) with the same accuracy specification.

### Vectorized double precision cosine function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_cosd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_cos_u10](../libm#sleef_cos_u10) with the same accuracy specification.

### Vectorized single precision cosine function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_cosf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_cosf_u10](../libm#sleef_cosf_u10) with the same accuracy specification.

### Vectorized double precision cosine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_cosd1_u35cuda(double a);
```

This is the CUDA function of [Sleef_cos_u35](../libm#sleef_cos_u35) with the same accuracy specification.

### Vectorized single precision cosine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_cosf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_cosf_u35](../libm#sleef_cosf_u35) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double2 Sleef_sincosd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_sincos_u10](../libm#sleef_sincos_u10) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float2 Sleef_sincosf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_sincosf_u10](../libm#sleef_sincosf_u10) with the same accuracy specification.

### Vectorized double precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double2 Sleef_sincosd1_u35cuda(double a);
```

This is the CUDA function of [Sleef_sincos_u35](../libm#sleef_sincos_u35) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float2 Sleef_sincosf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_sincosf_u35](../libm#sleef_sincosf_u35) with the same accuracy specification.

### Vectorized double precision sine function with 0.506 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_sinpid1_u05cuda(double a);
```

This is the CUDA function of [Sleef_sinpi_u05](../libm#sleef_sinpi_u05) with the same accuracy specification.

### Vectorized single precision sine function with 0.506 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_sinpif1_u05cuda(float a);
```

This is the CUDA function of [Sleef_sinpif_u05](../libm#sleef_sinpif_u05) with the same accuracy specification.

### Vectorized double precision cosine function with 0.506 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_cospid1_u05cuda(double a);
```

This is the CUDA function of [Sleef_cospi_u05](../libm#sleef_cospi_u05) with the same accuracy specification.

### Vectorized single precision cosine function with 0.506 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_cospif1_u05cuda(float a);
```

This is the CUDA function of [Sleef_cospif_u05](../libm#sleef_cospif_u05) with the same accuracy specification.

### Vectorized double precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double2 Sleef_sincospid1_u05cuda(double a);
```

This is the CUDA function of [Sleef_sincospi_u05](../libm#sleef_sincospi_u05) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float2 Sleef_sincospif1_u05cuda(float a);
```

This is the CUDA function of [Sleef_sincospif_u05](../libm#sleef_sincospif_u05) with the same accuracy specification.

### Vectorized double precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double2 Sleef_sincospid1_u35cuda(double a);
```

This is the CUDA function of [Sleef_sincospi_u35](../libm#sleef_sincospi_u35) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float2 Sleef_sincospif1_u35cuda(float a);
```

This is the CUDA function of [Sleef_sincospif_u35](../libm#sleef_sincospif_u35) with the same accuracy specification.

### Vectorized double precision tangent function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_tand1_u10cuda(double a);
```

This is the CUDA function of [Sleef_tan_u10](../libm#sleef_tan_u10) with the same accuracy specification.

### Vectorized single precision tangent function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_tanf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_tanf_u10](../libm#sleef_tanf_u10) with the same accuracy specification.

### Vectorized double precision tangent function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_tand1_u35cuda(double a);
```

This is the CUDA function of [Sleef_tan_u35](../libm#sleef_tan_u35) with the same accuracy specification.

### Vectorized single precision tangent function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_tanf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_tanf_u35](../libm#sleef_tanf_u35) with the same accuracy specification.

<h2 id="pow">Power, exponential, and logarithmic function</h2>

### Vectorized double precision power function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_powd1_u10cuda(double a, double b);
```

This is the CUDA function of [Sleef_pow_u10](../libm#sleef_pow_u10) with the same accuracy specification.

### Vectorized single precision power function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_powf1_u10cuda(float a, float b);
```

This is the CUDA function of [Sleef_powf_u10](../libm#sleef_powf_u10) with the same accuracy specification.

### Vectorized double precision natural logarithmic function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_logd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_log_u10](../libm#sleef_log_u10) with the same accuracy specification.

### Vectorized single precision natural logarithmic function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_logf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_logf_u10](../libm#sleef_logf_u10) with the same accuracy specification.

### Vectorized double precision natural logarithmic function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_logd1_u35cuda(double a);
```

This is the CUDA function of [Sleef_log_u35](../libm#sleef_log_u35) with the same accuracy specification.

### Vectorized single precision natural logarithmic function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_logf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_logf_u35](../libm#sleef_logf_u35) with the same accuracy specification.

### Vectorized double precision base-10 logarithmic function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_log10d1_u10cuda(double a);
```

This is the CUDA function of [Sleef_log10_u10](../libm#sleef_log10_u10) with the same accuracy specification.

### Vectorized single precision base-10 logarithmic function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_log10f1_u10cuda(float a);
```

This is the CUDA function of [Sleef_log10f_u10](../libm#sleef_log10f_u10) with the same accuracy specification.

### Vectorized double precision base-2 logarithmic function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_log2d1_u10cuda(double a);
```

This is the CUDA function of [Sleef_log2_u10](../libm#sleef_log2_u10) with the same accuracy specification.

### Vectorized single precision base-2 logarithmic function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_log2f1_u10cuda(float a);
```

This is the CUDA function of [Sleef_log2f_u10](../libm#sleef_log2f_u10) with the same accuracy specification.

### Vectorized double precision logarithm of one plus argument with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_log1pd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_log1p_u10](../libm#sleef_log1p_u10) with the same accuracy specification.

### Vectorized single precision logarithm of one plus argument with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_log1pf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_log1pf_u10](../libm#sleef_log1pf_u10) with the same accuracy specification.

### Vectorized double precision base-<i>e</i> exponential function function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_expd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_exp_u10](../libm#sleef_exp_u10) with the same accuracy specification.

### Vectorized single precision base-<i>e</i> exponential function function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_expf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_expf_u10](../libm#sleef_expf_u10) with the same accuracy specification.

### Vectorized double precision base-<i>2</i> exponential function function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_exp2d1_u10cuda(double a);
```

This is the CUDA function of [Sleef_exp2_u10](../libm#sleef_exp2_u10) with the same accuracy specification.

### Vectorized single precision base-<i>2</i> exponential function function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_exp2f1_u10cuda(float a);
```

This is the CUDA function of [Sleef_exp2f_u10](../libm#sleef_exp2f_u10) with the same accuracy specification.

### Vectorized double precision base-10 exponential function function with 1.09 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_exp10d1_u10cuda(double a);
```

This is the CUDA function of [Sleef_exp10_u10](../libm#sleef_exp10_u10) with the same accuracy specification.

### Vectorized single precision base-10 exponential function function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_exp10f1_u10cuda(float a);
```

This is the CUDA function of [Sleef_exp10f_u10](../libm#sleef_exp10f_u10) with the same accuracy specification.

### Vectorized double precision base-<i>e</i> exponential function minus 1 with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_expm1d1_u10cuda(double a);
```

This is the CUDA function of [Sleef_expm1_u10](../libm#sleef_expm1_u10) with the same accuracy specification.

### Vectorized single precision base-<i>e</i> exponential function minus 1 with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_expm1f1_u10cuda(float a);
```

This is the CUDA function of [Sleef_expm1f_u10](../libm#sleef_expm1f_u10) with the same accuracy specification.

### Vectorized double precision square root function with 0.5001 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_sqrtd1_u05cuda(double a);
```

This is the CUDA function of [Sleef_sqrt_u05](../libm#sleef_sqrt_u05) with the same accuracy specification.

### Vectorized single precision square root function with 0.5001 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_sqrtf1_u05cuda(float a);
```

This is the CUDA function of [Sleef_sqrtf_u05](../libm#sleef_sqrtf_u05) with the same accuracy specification.

### Vectorized double precision square root function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_sqrtd1_u35cuda(double a);
```

This is the CUDA function of [Sleef_sqrt_u35](../libm#sleef_sqrt_u35) with the same accuracy specification.

### Vectorized single precision square root function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_sqrtf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_sqrtf_u35](../libm#sleef_sqrtf_u35) with the same accuracy specification.

### Vectorized double precision cubic root function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_cbrtd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_cbrt_u10](../libm#sleef_cbrt_u10) with the same accuracy specification.

### Vectorized single precision cubic root function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_cbrtf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_cbrtf_u10](../libm#sleef_cbrtf_u10) with the same accuracy specification.

### Vectorized double precision cubic root function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_cbrtd1_u35cuda(double a);
```

This is the CUDA function of [Sleef_cbrt_u35](../libm#sleef_cbrt_u35) with the same accuracy specification.

### Vectorized single precision cubic root function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_cbrtf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_cbrtf_u35](../libm#sleef_cbrtf_u35) with the same accuracy specification.

### Vectorized double precision 2D Euclidian distance function with 0.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_hypotd1_u05cuda(double a, double b);
```

This is the CUDA function of [Sleef_hypot_u05](../libm#sleef_hypot_u05) with the same accuracy specification.

### Vectorized single precision 2D Euclidian distance function with 0.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_hypotf1_u05cuda(float a, float b);
```

This is the CUDA function of [Sleef_hypotf_u05](../libm#sleef_hypotf_u05) with the same accuracy specification.

### Vectorized double precision 2D Euclidian distance function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_hypotd1_u35cuda(double a, double b);
```

This is the CUDA function of [Sleef_hypot_u35](../libm#sleef_hypot_u35) with the same accuracy specification.

### Vectorized single precision 2D Euclidian distance function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_hypotf1_u35cuda(float a, float b);
```

This is the CUDA function of [Sleef_hypotf_u35](../libm#sleef_hypotf_u35) with the same accuracy specification.

<h2 id="invtrig">Inverse Trigonometric Functions</h2>

### Vectorized double precision arc sine function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_asind1_u10cuda(double a);
```

This is the CUDA function of [Sleef_asin_u10](../libm#sleef_asin_u10) with the same accuracy specification.

### Vectorized single precision arc sine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_asinf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_asinf_u10](../libm#sleef_asinf_u10) with the same accuracy specification.

### Vectorized double precision arc sine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_asind1_u35cuda(double a);
```

This is the CUDA function of [Sleef_asin_u35](../libm#sleef_asin_u35) with the same accuracy specification.

### Vectorized single precision arc sine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_asinf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_asinf_u35](../libm#sleef_asinf_u35) with the same accuracy specification.

### Vectorized double precision arc cosine function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_acosd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_acos_u10](../libm#sleef_acos_u10) with the same accuracy specification.

### Vectorized single precision arc cosine function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_acosf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_acosf_u10](../libm#sleef_acosf_u10) with the same accuracy specification.

### Vectorized double precision arc cosine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_acosd1_u35cuda(double a);
```

This is the CUDA function of [Sleef_acos_u35](../libm#sleef_acos_u35) with the same accuracy specification.

### Vectorized single precision arc cosine function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_acosf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_acosf_u35](../libm#sleef_acosf_u35) with the same accuracy specification.

### Vectorized double precision arc tangent function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_atand1_u10cuda(double a);
```

This is the CUDA function of [Sleef_atan_u10](../libm#sleef_atan_u10) with the same accuracy specification.

### Vectorized single precision arc tangent function with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_atanf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_atanf_u10](../libm#sleef_atanf_u10) with the same accuracy specification.

### Vectorized double precision arc tangent function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_atand1_u35cuda(double a);
```

This is the CUDA function of [Sleef_atan_u35](../libm#sleef_atan_u35) with the same accuracy specification.

### Vectorized single precision arc tangent function with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_atanf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_atanf_u35](../libm#sleef_atanf_u35) with the same accuracy specification.

### Vectorized double precision arc tangent function of two variables with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_atan2d1_u10cuda(double a, double b);
```

This is the CUDA function of [Sleef_atan2_u10](../libm#sleef_atan2_u10) with the same accuracy specification.

### Vectorized single precision arc tangent function of two variables with 1.0 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_atan2f1_u10cuda(float a, float b);
```

This is the CUDA function of [Sleef_atan2f_u10](../libm#sleef_atan2f_u10) with the same accuracy specification.

### Vectorized double precision arc tangent function of two variables with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_atan2d1_u35cuda(double a, double b);
```

This is the CUDA function of [Sleef_atan2_u35](../libm#sleef_atan2_u35) with the same accuracy specification.

### Vectorized single precision arc tangent function of two variables with 3.5 ULP error bound

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_atan2f1_u35cuda(float a, float b);
```

This is the CUDA function of [Sleef_atan2f_u35](../libm#sleef_atan2f_u35) with the same accuracy specification.

<h2 id="hyp">Hyperbolic function and inverse hyperbolic function</h2>

### Vectorized double precision hyperbolic sine function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_sinhd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_sinh_u10](../libm#sleef_sinh_u10) with the same accuracy specification.

### Vectorized single precision hyperbolic sine function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_sinhf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_sinhf_u10](../libm#sleef_sinhf_u10) with the same accuracy specification.

### Vectorized double precision hyperbolic sine function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_sinhd1_u35cuda(double a);
```

This is the CUDA function of [Sleef_sinh_u35](../libm#sleef_sinh_u35) with the same accuracy specification.

### Vectorized single precision hyperbolic sine function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_sinhf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_sinhf_u35](../libm#sleef_sinhf_u35) with the same accuracy specification.

### Vectorized double precision hyperbolic cosine function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_coshd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_cosh_u10](../libm#sleef_cosh_u10) with the same accuracy specification.

### Vectorized single precision hyperbolic cosine function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_coshf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_coshf_u10](../libm#sleef_coshf_u10) with the same accuracy specification.

### Vectorized double precision hyperbolic cosine function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_coshd1_u35cuda(double a);
```

This is the CUDA function of [Sleef_cosh_u35](../libm#sleef_cosh_u35) with the same accuracy specification.

### Vectorized single precision hyperbolic cosine function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_coshf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_coshf_u35](../libm#sleef_coshf_u35) with the same accuracy specification.

### Vectorized double precision hyperbolic tangent function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_tanhd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_tanh_u10](../libm#sleef_tanh_u10) with the same accuracy specification.

### Vectorized single precision hyperbolic tangent function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_tanhf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_tanhf_u10](../libm#sleef_tanhf_u10) with the same accuracy specification.

### Vectorized double precision hyperbolic tangent function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_tanhd1_u35cuda(double a);
```

This is the CUDA function of [Sleef_tanh_u35](../libm#sleef_tanh_u35) with the same accuracy specification.

### Vectorized single precision hyperbolic tangent function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_tanhf1_u35cuda(float a);
```

This is the CUDA function of [Sleef_tanhf_u35](../libm#sleef_tanhf_u35) with the same accuracy specification.

### Vectorized double precision inverse hyperbolic sine function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_asinhd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_asinh_u10](../libm#sleef_asinh_u10) with the same accuracy specification.

### Vectorized single precision inverse hyperbolic sine function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_asinhf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_asinhf_u10](../libm#sleef_asinhf_u10) with the same accuracy specification.

### Vectorized double precision inverse hyperbolic cosine function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_acoshd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_acosh_u10](../libm#sleef_acosh_u10) with the same accuracy specification.

### Vectorized single precision inverse hyperbolic cosine function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_acoshf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_acoshf_u10](../libm#sleef_acoshf_u10) with the same accuracy specification.

### Vectorized double precision inverse hyperbolic tangent function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_atanhd1_u10cuda(double a);
```

This is the CUDA function of [Sleef_atanh_u10](../libm#sleef_atanh_u10) with the same accuracy specification.

### Vectorized single precision inverse hyperbolic tangent function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_atanhf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_atanhf_u10](../libm#sleef_atanhf_u10) with the same accuracy specification.

<h2 id="eg">Error and gamma function</h2>

### Vectorized double precision error function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_erfd1_u10cuda(float a);
```

This is the CUDA function of [Sleef_erf_u10](../libm#sleef_erf_u10) with the same accuracy specification.

### Vectorized single precision error function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_erff1_u10cuda(float a);
```

This is the CUDA function of [Sleef_erff_u10](../libm#sleef_erff_u10) with the same accuracy specification.

### Vectorized double precision complementary error function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_erfcd1_u15cuda(double a);
```

This is the CUDA function of [Sleef_erfc_u15](../libm#sleef_erfc_u15) with the same accuracy specification.

### Vectorized single precision complementary error function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_erfcf1_u15cuda(float a);
```

This is the CUDA function of [Sleef_erfcf_u15](../libm#sleef_erfcf_u15) with the same accuracy specification.

### Vectorized double precision gamma function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_tgammad1_u10cuda(double a);
```

This is the CUDA function of [Sleef_tgamma_u10](../libm#sleef_tgamma_u10) with the same accuracy specification.

### Vectorized single precision gamma function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_tgammaf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_tgammaf_u10](../libm#sleef_tgammaf_u10) with the same accuracy specification.

### Vectorized double precision log gamma function

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_lgammad1_u10cuda(double a);
```

This is the CUDA function of [Sleef_lgamma_u10](../libm#sleef_lgamma_u10) with the same accuracy specification.

### Vectorized single precision log gamma function

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_lgammaf1_u10cuda(float a);
```

This is the CUDA function of [Sleef_lgammaf_u10](../libm#sleef_lgammaf_u10) with the same accuracy specification.

<h2 id="nearint">Nearest integer function</h2>

### Vectorized double precision function for rounding to integer towards zero

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_truncd1_cuda(double a);
```

This is the CUDA function of [Sleef_trunc](../libm#sleef_trunc) with the same accuracy specification.

### Vectorized single precision function for rounding to integer towards zero

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_truncf1_cuda(float a);
```

This is the CUDA function of [Sleef_truncf](../libm#sleef_truncf) with the same accuracy specification.

### Vectorized double precision function for rounding to integer towards negative infinity

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_floord1_cuda(double a);
```

This is the CUDA function of [Sleef_floor](../libm#sleef_floor) with the same accuracy specification.

### Vectorized single precision function for rounding to integer towards negative infinity

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_floorf1_cuda(float a);
```

This is the CUDA function of [Sleef_floorf](../libm#sleef_floorf) with the same accuracy specification.

### Vectorized double precision function for rounding to integer towards positive infinity

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_ceild1_cuda(double a);
```

This is the CUDA function of [Sleef_ceil](../libm#sleef_ceil) with the same accuracy specification.

### Vectorized single precision function for rounding to integer towards positive infinity

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_ceilf1_cuda(float a);
```

This is the CUDA function of [Sleef_ceilf](../libm#sleef_ceilf) with the same accuracy specification.

### Vectorized double precision function for rounding to nearest integer

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_roundd1_cuda(double a);
```

This is the CUDA function of [Sleef_round](../libm#sleef_round) with the same accuracy specification.

### Vectorized single precision function for rounding to nearest integer

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_roundf1_cuda(float a);
```

This is the CUDA function of [Sleef_roundf](../libm#sleef_roundf) with the same accuracy specification.

### Vectorized double precision function for rounding to nearest integer

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_rintd1_cuda(double a);
```

This is the CUDA function of [Sleef_rint](../libm#sleef_rint) with the same accuracy specification.

### Vectorized single precision function for rounding to nearest integer

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_rintf1_cuda(float a);
```

This is the CUDA function of [Sleef_rintf](../libm#sleef_rintf) with the same accuracy specification.

<h2 id="other">Other function</h2>

### Vectorized double precision function for fused multiply-accumulation

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_fmad1_cuda(double a, double b, double c);
```

This is the CUDA function of [Sleef_fma](../libm#sleef_fma) with the same accuracy specification.

### Vectorized single precision function for fused multiply-accumulation

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_fmaf1_cuda(float a, float b, float c);
```

This is the CUDA function of [Sleef_fmaf](../libm#sleef_fmaf) with the same accuracy specification.

### Vectorized double precision FP remainder

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_fmodd1_cuda(double a, double b);
```

This is the CUDA function of [Sleef_fmod](../libm#sleef_fmod) with the same accuracy specification.

### Vectorized single precision FP remainder

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_fmodf1_cuda(float a, float b);
```

This is the CUDA function of [Sleef_fmodf](../libm#sleef_fmodf) with the same accuracy specification.

### Vectorized double precision FP remainder

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_remainderd1_cuda(double a, double b);
```

This is the CUDA function of [Sleef_remainder](../libm#sleef_remainder) with the same accuracy specification.

### Vectorized single precision FP remainder

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_remainderf1_cuda(float a, float b);
```

This is the CUDA function of [Sleef_remainderf](../libm#sleef_remainderf) with the same accuracy specification.

### Vectorized double precision function for multiplying by integral power of 2

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_ldexpd1_cuda(double a, int32x2_t b);
```

This is the CUDA function of [Sleef_ldexp](../libm#sleef_ldexp) with the same accuracy specification.

### Vectorized double precision function for obtaining fractional component of an FP number

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_frfrexpd1_cuda(double a);
```

This is the CUDA function of [Sleef_frfrexp](../libm#sleef_frfrexp) with the same accuracy specification.

### Vectorized single precision function for obtaining fractional component of an FP number

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_frfrexpf1_cuda(float a);
```

This is the CUDA function of [Sleef_frfrexpf](../libm#sleef_frfrexpf) with the same accuracy specification.

### Vectorized double precision function for obtaining integral component of an FP number

```c
#include <sleefinline_cuda.h>

__device__ int32x2_t Sleef_expfrexpd1_cuda(double a);
```

This is the CUDA function of [Sleef_expfrexp](../libm#sleef_expfrexp) with the same accuracy specification.

### Vectorized double precision function for getting integer exponent

```c
#include <sleefinline_cuda.h>

__device__ int32x2_t Sleef_ilogbd1_cuda(double a);
```

This is the CUDA function of [Sleef_ilogb](../libm#sleef_ilogb) with the same accuracy specification.

### Vectorized double precision signed integral and fractional values

```c
#include <sleefinline_cuda.h>

__device__ Sleef_double_2 Sleef_modfd1_cuda(double a);
```

This is the CUDA function of [Sleef_modf](../libm#sleef_modf) with the same accuracy specification.

### Vectorized single precision signed integral and fractional values

```c
#include <sleefinline_cuda.h>

__device__ Sleef_float_2 Sleef_modff1_cuda(float a);
```

This is the CUDA function of [Sleef_modff](../libm#sleef_modff) with the same accuracy specification.

### Vectorized double precision function for calculating the absolute value

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_fabsd1_cuda(double a);
```

This is the CUDA function of [Sleef_fabs](../libm#sleef_fabs) with the same accuracy specification.

### Vectorized single precision function for calculating the absolute value

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_fabsf1_cuda(float a);
```

This is the CUDA function of [Sleef_fabsf](../libm#sleef_fabsf) with the same accuracy specification.

### Vectorized double precision function for copying signs

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_copysignd1_cuda(double a, double b);
```

This is the CUDA function of [Sleef_copysign](../libm#sleef_copysign) with the same accuracy specification.

### Vectorized single precision function for copying signs

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_copysignf1_cuda(float a, float b);
```

This is the CUDA function of [Sleef_copysignf](../libm#sleef_copysignf) with the same accuracy specification.

### Vectorized double precision function for determining maximum of two values

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_fmaxd1_cuda(double a, double b);
```

This is the CUDA function of [Sleef_fmax](../libm#sleef_fmax) with the same accuracy specification.

### Vectorized single precision function for determining maximum of two values

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_fmaxf1_cuda(float a, float b);
```

This is the CUDA function of [Sleef_fmaxf](../libm#sleef_fmaxf) with the same accuracy specification.

### Vectorized double precision function for determining minimum of two values

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_fmind1_cuda(double a, double b);
```

This is the CUDA function of [Sleef_fmin](../libm#sleef_fmin) with the same accuracy specification.

### Vectorized single precision function for determining minimum of two values

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_fminf1_cuda(float a, float b);
```

This is the CUDA function of [Sleef_fminf](../libm#sleef_fminf) with the same accuracy specification.

### Vectorized double precision function to calculate positive difference of two values

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_fdimd1_cuda(double a, double b);
```

This is the CUDA function of [Sleef_fdim](../libm#sleef_fdim) with the same accuracy specification.

### Vectorized single precision function to calculate positive difference of two values

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_fdimf1_cuda(float a, float b);
```

This is the CUDA function of [Sleef_fdimf](../libm#sleef_fdimf) with the same accuracy specification.

### Vectorized double precision function for obtaining the next representable FP value

```c
#include <sleefinline_cuda.h>

__device__ double Sleef_nextafterd1_cuda(double a, double b);
```

This is the CUDA function of [Sleef_nextafter](../libm#sleef_nextafter) with the same accuracy specification.

### Vectorized single precision function for obtaining the next representable FP value

```c
#include <sleefinline_cuda.h>

__device__ float Sleef_nextafterf1_cuda(float a, float b);
```

This is the CUDA function of [Sleef_nextafterf](../libm#sleef_nextafterf) with the same accuracy specification.
