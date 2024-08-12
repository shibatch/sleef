---
layout: default
title: AArch64
parent: Single & Double Precision
grand_parent: References
permalink: /2-references/libm/aarch64
---

<h1>Single & Double Precision Math library reference (AArch64)</h1>

<h2>Table of contents</h2>

* [Data types](#datatypes)
* [Trigonometric functions](#trig)
* [Power, exponential, and logarithmic functions](#pow)
* [Inverse trigonometric functions](#invtrig)
* [Hyperbolic functions and inverse hyperbolic functions](#hyp)
* [Error and gamma functions](#eg)
* [Nearest integer functions](#nearint)
* [Other functions](#other)

<h2 id="datatypes">Data types for AArch64 architecture</h2>

### Sleef_float32x4_t_2

`Sleef_float32x4_t_2` is a data type for storing two `float32x4_t` values,
which is defined in sleef.h as follows:

```c
typedef struct {
  float32x4_t x, y;
} Sleef_float32x4_t_2;
```

### Sleef_float64x2_t_2

`Sleef_float64x2_t_2` is a data type for storing two `float64x2_t` values,
which is defined in sleef.h as follows:

```c
typedef struct {
  float64x2_t x, y;
} Sleef_float64x2_t_2;
```

### Sleef_svfloat32_t_2

`Sleef_svfloat32_t_2` is a data type for storing two `svfloat32_t` values,
which is defined in sleef.h as follows:

```c
typedef struct {
  svfloat32_t x, y;
} Sleef_svfloat32_t_2;
```

### Sleef_svfloat64_t_2

`Sleef_svfloat64_t_2` is a data type for storing two `svfloat64_t` values,
which is defined in sleef.h as follows:

```c
typedef struct {
  svfloat64_t x, y;
} Sleef_svfloat64_t_2;
```

<h2 id="trig">Trigonometric Functions</h2>

### Vectorized double precision sine function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_sind2_u10(float64x2_t a);
float64x2_t Sleef_sind2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_sind2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_sind2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_sindx_u10sve(svfloat64_t a);
svfloat64_t Sleef_sindx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_sindx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_sindx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sin_u10](../libm#sleef_sin_u10) with the same accuracy specification.

### Vectorized single precision sine function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sinf4_u10(float32x4_t a);
float32x4_t Sleef_sinf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_sinf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_sinf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_sinfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_sinfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_sinfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_sinfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinf_u10](../libm#sleef_sinf_u10) with the same accuracy specification.

### Vectorized double precision sine function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_sind2_u35(float64x2_t a);
float64x2_t Sleef_sind2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_sind2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_sind2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_sindx_u35sve(svfloat64_t a);
svfloat64_t Sleef_sindx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_sindx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_sindx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sin_u35](../libm#sleef_sin_u35) with the same accuracy specification.

### Vectorized single precision sine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sinf4_u35(float32x4_t a);
float32x4_t Sleef_sinf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_sinf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_sinf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_sinfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_sinfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_sinfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_sinfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinf_u35](../libm#sleef_sinf_u35) with the same accuracy specification.

### Vectorized double precision cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_cosd2_u10(float64x2_t a);
float64x2_t Sleef_cosd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_cosd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_cosd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_cosdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_cosdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_cosdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_cosdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cos_u10](../libm#sleef_cos_u10) with the same accuracy specification.

### Vectorized single precision cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cosf4_u10(float32x4_t a);
float32x4_t Sleef_cosf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_cosf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_cosf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_cosfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_cosfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_cosfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_cosfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosf_u10](../libm#sleef_cosf_u10) with the same accuracy specification.

### Vectorized double precision cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_cosd2_u35(float64x2_t a);
float64x2_t Sleef_cosd2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_cosd2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_cosd2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_cosdx_u35sve(svfloat64_t a);
svfloat64_t Sleef_cosdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_cosdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_cosdx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cos_u35](../libm#sleef_cos_u35) with the same accuracy specification.

### Vectorized single precision cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cosf4_u35(float32x4_t a);
float32x4_t Sleef_cosf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_cosf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_cosf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_cosfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_cosfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_cosfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_cosfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosf_u35](../libm#sleef_cosf_u35) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

Sleef_float64x2_t_2 Sleef_sincosd2_u10(float64x2_t a);
Sleef_float64x2_t_2 Sleef_sincosd2_u10advsimd(float64x2_t a);
Sleef_float64x2_t_2 Sleef_cinz_sincosd2_u10advsimdnofma(float64x2_t a);
Sleef_float64x2_t_2 Sleef_finz_sincosd2_u10advsimd(float64x2_t a);

Sleef_svfloat64_t_2 Sleef_sincosdx_u10sve(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_sincosdx_u10svenofma(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_cinz_sincosdx_u10svenofma(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_finz_sincosdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincos_u10](../libm#sleef_sincos_u10) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_sincosf4_u10(float32x4_t a);
Sleef_float32x4_t_2 Sleef_sincosf4_u10advsimd(float32x4_t a);
Sleef_float32x4_t_2 Sleef_cinz_sincosf4_u10advsimdnofma(float32x4_t a);
Sleef_float32x4_t_2 Sleef_finz_sincosf4_u10advsimd(float32x4_t a);

Sleef_svfloat32_t_2 Sleef_sincosfx_u10sve(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_sincosfx_u10svenofma(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_cinz_sincosfx_u10svenofma(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_finz_sincosfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincosf_u10](../libm#sleef_sincosf_u10) with the same accuracy specification.

### Vectorized double precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_float64x2_t_2 Sleef_sincosd2_u35(float64x2_t a);
Sleef_float64x2_t_2 Sleef_sincosd2_u35advsimd(float64x2_t a);
Sleef_float64x2_t_2 Sleef_cinz_sincosd2_u35advsimdnofma(float64x2_t a);
Sleef_float64x2_t_2 Sleef_finz_sincosd2_u35advsimd(float64x2_t a);

Sleef_svfloat64_t_2 Sleef_sincosdx_u35sve(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_sincosdx_u35svenofma(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_cinz_sincosdx_u35svenofma(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_finz_sincosdx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincos_u35](../libm#sleef_sincos_u35) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_sincosf4_u35(float32x4_t a);
Sleef_float32x4_t_2 Sleef_sincosf4_u35advsimd(float32x4_t a);
Sleef_float32x4_t_2 Sleef_cinz_sincosf4_u35advsimdnofma(float32x4_t a);
Sleef_float32x4_t_2 Sleef_finz_sincosf4_u35advsimd(float32x4_t a);

Sleef_svfloat32_t_2 Sleef_sincosfx_u35sve(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_sincosfx_u35svenofma(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_cinz_sincosfx_u35svenofma(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_finz_sincosfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincosf_u35](../libm#sleef_sincosf_u35) with the same accuracy specification.

### Vectorized double precision sine function with 0.506 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_sinpid2_u05(float64x2_t a);
float64x2_t Sleef_sinpid2_u05advsimd(float64x2_t a);
float64x2_t Sleef_cinz_sinpid2_u05advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_sinpid2_u05advsimd(float64x2_t a);

svfloat64_t Sleef_sinpidx_u05sve(svfloat64_t a);
svfloat64_t Sleef_sinpidx_u05svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_sinpidx_u05svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_sinpidx_u05sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinpi_u05](../libm#sleef_sinpi_u05) with the same accuracy specification.

### Vectorized single precision sine function with 0.506 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sinpif4_u05(float32x4_t a);
float32x4_t Sleef_sinpif4_u05advsimd(float32x4_t a);
float32x4_t Sleef_cinz_sinpif4_u05advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_sinpif4_u05advsimd(float32x4_t a);

svfloat32_t Sleef_sinpifx_u05sve(svfloat32_t a);
svfloat32_t Sleef_sinpifx_u05svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_sinpifx_u05svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_sinpifx_u05sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinpif_u05](../libm#sleef_sinpif_u05) with the same accuracy specification.

### Vectorized double precision cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_cospid2_u05(float64x2_t a);
float64x2_t Sleef_cospid2_u05advsimd(float64x2_t a);
float64x2_t Sleef_cinz_cospid2_u05advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_cospid2_u05advsimd(float64x2_t a);

svfloat64_t Sleef_cospidx_u05sve(svfloat64_t a);
svfloat64_t Sleef_cospidx_u05svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_cospidx_u05svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_cospidx_u05sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cospi_u05](../libm#sleef_cospi_u05) with the same accuracy specification.

### Vectorized single precision cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cospif4_u05(float32x4_t a);
float32x4_t Sleef_cospif4_u05advsimd(float32x4_t a);
float32x4_t Sleef_cinz_cospif4_u05advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_cospif4_u05advsimd(float32x4_t a);

svfloat32_t Sleef_cospifx_u05sve(svfloat32_t a);
svfloat32_t Sleef_cospifx_u05svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_cospifx_u05svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_cospifx_u05sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cospif_u05](../libm#sleef_cospif_u05) with the same accuracy specification.

### Vectorized double precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

Sleef_float64x2_t_2 Sleef_sincospid2_u05(float64x2_t a);
Sleef_float64x2_t_2 Sleef_sincospid2_u05advsimd(float64x2_t a);
Sleef_float64x2_t_2 Sleef_cinz_sincospid2_u05advsimdnofma(float64x2_t a);
Sleef_float64x2_t_2 Sleef_finz_sincospid2_u05advsimd(float64x2_t a);

Sleef_svfloat64_t_2 Sleef_sincospidx_u05sve(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_sincospidx_u05svenofma(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_cinz_sincospidx_u05svenofma(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_finz_sincospidx_u05sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospi_u05](../libm#sleef_sincospi_u05) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_sincospif4_u05(float32x4_t a);
Sleef_float32x4_t_2 Sleef_sincospif4_u05advsimd(float32x4_t a);
Sleef_float32x4_t_2 Sleef_cinz_sincospif4_u05advsimdnofma(float32x4_t a);
Sleef_float32x4_t_2 Sleef_finz_sincospif4_u05advsimd(float32x4_t a);

Sleef_svfloat32_t_2 Sleef_sincospifx_u05sve(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_sincospifx_u05svenofma(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_cinz_sincospifx_u05svenofma(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_finz_sincospifx_u05sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospif_u05](../libm#sleef_sincospif_u05) with the same accuracy specification.

### Vectorized double precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_float64x2_t_2 Sleef_sincospid2_u35(float64x2_t a);
Sleef_float64x2_t_2 Sleef_sincospid2_u35advsimd(float64x2_t a);
Sleef_float64x2_t_2 Sleef_cinz_sincospid2_u35advsimdnofma(float64x2_t a);
Sleef_float64x2_t_2 Sleef_finz_sincospid2_u35advsimd(float64x2_t a);

Sleef_svfloat64_t_2 Sleef_sincospidx_u35sve(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_sincospidx_u35svenofma(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_cinz_sincospidx_u35svenofma(svfloat64_t a);
Sleef_svfloat64_t_2 Sleef_finz_sincospidx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospi_u35](../libm#sleef_sincospi_u35) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_sincospif4_u35(float32x4_t a);
Sleef_float32x4_t_2 Sleef_sincospif4_u35advsimd(float32x4_t a);
Sleef_float32x4_t_2 Sleef_cinz_sincospif4_u35advsimdnofma(float32x4_t a);
Sleef_float32x4_t_2 Sleef_finz_sincospif4_u35advsimd(float32x4_t a);

Sleef_svfloat32_t_2 Sleef_sincospifx_u35sve(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_sincospifx_u35svenofma(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_cinz_sincospifx_u35svenofma(svfloat32_t a);
Sleef_svfloat32_t_2 Sleef_finz_sincospifx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospif_u35](../libm#sleef_sincospif_u35) with the same accuracy specification.

### Vectorized double precision tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_tand2_u10(float64x2_t a);
float64x2_t Sleef_tand2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_tand2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_tand2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_tandx_u10sve(svfloat64_t a);
svfloat64_t Sleef_tandx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_tandx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_tandx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tan_u10](../libm#sleef_tan_u10) with the same accuracy specification.

### Vectorized single precision tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_tanf4_u10(float32x4_t a);
float32x4_t Sleef_tanf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_tanf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_tanf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_tanfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_tanfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_tanfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_tanfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanf_u10](../libm#sleef_tanf_u10) with the same accuracy specification.

### Vectorized double precision tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_tand2_u35(float64x2_t a);
float64x2_t Sleef_tand2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_tand2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_tand2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_tandx_u35sve(svfloat64_t a);
svfloat64_t Sleef_tandx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_tandx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_tandx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tan_u35](../libm#sleef_tan_u35) with the same accuracy specification.

### Vectorized single precision tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_tanf4_u35(float32x4_t a);
float32x4_t Sleef_tanf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_tanf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_tanf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_tanfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_tanfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_tanfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_tanfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanf_u35](../libm#sleef_tanf_u35) with the same accuracy specification.

<h2 id="pow">Power, exponential, and logarithmic function</h2>

### Vectorized double precision power function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_powd2_u10(float64x2_t a, float64x2_t b);
float64x2_t Sleef_powd2_u10advsimd(float64x2_t a, float64x2_t b);
float64x2_t Sleef_cinz_powd2_u10advsimdnofma(float64x2_t a, float64x2_t b);
float64x2_t Sleef_finz_powd2_u10advsimd(float64x2_t a, float64x2_t b);

svfloat64_t Sleef_powdx_u10sve(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_powdx_u10svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_cinz_powdx_u10svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_finz_powdx_u10sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_pow_u10](../libm#sleef_pow_u10) with the same accuracy specification.

### Vectorized single precision power function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_powf4_u10(float32x4_t a, float32x4_t b);
float32x4_t Sleef_powf4_u10advsimd(float32x4_t a, float32x4_t b);
float32x4_t Sleef_cinz_powf4_u10advsimdnofma(float32x4_t a, float32x4_t b);
float32x4_t Sleef_finz_powf4_u10advsimd(float32x4_t a, float32x4_t b);

svfloat32_t Sleef_powfx_u10sve(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_powfx_u10svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_cinz_powfx_u10svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_finz_powfx_u10sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_powf_u10](../libm#sleef_powf_u10) with the same accuracy specification.

### Vectorized double precision natural logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_logd2_u10(float64x2_t a);
float64x2_t Sleef_logd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_logd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_logd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_logdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_logdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_logdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_logdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log_u10](../libm#sleef_log_u10) with the same accuracy specification.

### Vectorized single precision natural logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_logf4_u10(float32x4_t a);
float32x4_t Sleef_logf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_logf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_logf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_logfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_logfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_logfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_logfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_logf_u10](../libm#sleef_logf_u10) with the same accuracy specification.

### Vectorized double precision natural logarithmic function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_logd2_u35(float64x2_t a);
float64x2_t Sleef_logd2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_logd2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_logd2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_logdx_u35sve(svfloat64_t a);
svfloat64_t Sleef_logdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_logdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_logdx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log_u35](../libm#sleef_log_u35) with the same accuracy specification.

### Vectorized single precision natural logarithmic function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_logf4_u35(float32x4_t a);
float32x4_t Sleef_logf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_logf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_logf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_logfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_logfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_logfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_logfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_logf_u35](../libm#sleef_logf_u35) with the same accuracy specification.

### Vectorized double precision base-10 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_log10d2_u10(float64x2_t a);
float64x2_t Sleef_log10d2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_log10d2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_log10d2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_log10dx_u10sve(svfloat64_t a);
svfloat64_t Sleef_log10dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_log10dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_log10dx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log10_u10](../libm#sleef_log10_u10) with the same accuracy specification.

### Vectorized single precision base-10 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_log10f4_u10(float32x4_t a);
float32x4_t Sleef_log10f4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_log10f4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_log10f4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_log10fx_u10sve(svfloat32_t a);
svfloat32_t Sleef_log10fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_log10fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_log10fx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log10f_u10](../libm#sleef_log10f_u10) with the same accuracy specification.

### Vectorized double precision base-2 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_log2d2_u10(float64x2_t a);
float64x2_t Sleef_log2d2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_log2d2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_log2d2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_log2dx_u10sve(svfloat64_t a);
svfloat64_t Sleef_log2dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_log2dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_log2dx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log2_u10](../libm#sleef_log2_u10) with the same accuracy specification.

### Vectorized single precision base-2 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_log2f4_u10(float32x4_t a);
float32x4_t Sleef_log2f4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_log2f4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_log2f4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_log2fx_u10sve(svfloat32_t a);
svfloat32_t Sleef_log2fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_log2fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_log2fx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log2f_u10](../libm#sleef_log2f_u10) with the same accuracy specification.

### Vectorized double precision logarithm of one plus argument with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_log1pd2_u10(float64x2_t a);
float64x2_t Sleef_log1pd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_log1pd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_log1pd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_log1pdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_log1pdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_log1pdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_log1pdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log1p_u10](../libm#sleef_log1p_u10) with the same accuracy specification.

### Vectorized single precision logarithm of one plus argument with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_log1pf4_u10(float32x4_t a);
float32x4_t Sleef_log1pf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_log1pf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_log1pf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_log1pfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_log1pfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_log1pfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_log1pfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log1pf_u10](../libm#sleef_log1pf_u10) with the same accuracy specification.

### Vectorized double precision base-<i>e</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_expd2_u10(float64x2_t a);
float64x2_t Sleef_expd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_expd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_expd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_expdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_expdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_expdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_expdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp_u10](../libm#sleef_exp_u10) with the same accuracy specification.

### Vectorized single precision base-<i>e</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_expf4_u10(float32x4_t a);
float32x4_t Sleef_expf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_expf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_expf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_expfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_expfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_expfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_expfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expf_u10](../libm#sleef_expf_u10) with the same accuracy specification.

### Vectorized double precision base-<i>2</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_exp2d2_u10(float64x2_t a);
float64x2_t Sleef_exp2d2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_exp2d2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_exp2d2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_exp2dx_u10sve(svfloat64_t a);
svfloat64_t Sleef_exp2dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_exp2dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_exp2dx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp2_u10](../libm#sleef_exp2_u10) with the same accuracy specification.

### Vectorized single precision base-<i>2</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_exp2f4_u10(float32x4_t a);
float32x4_t Sleef_exp2f4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_exp2f4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_exp2f4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_exp2fx_u10sve(svfloat32_t a);
svfloat32_t Sleef_exp2fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_exp2fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_exp2fx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp2f_u10](../libm#sleef_exp2f_u10) with the same accuracy specification.

### Vectorized double precision base-10 exponential function function with 1.09 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_exp10d2_u10(float64x2_t a);
float64x2_t Sleef_exp10d2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_exp10d2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_exp10d2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_exp10dx_u10sve(svfloat64_t a);
svfloat64_t Sleef_exp10dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_exp10dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_exp10dx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp10_u10](../libm#sleef_exp10_u10) with the same accuracy specification.

### Vectorized single precision base-10 exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_exp10f4_u10(float32x4_t a);
float32x4_t Sleef_exp10f4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_exp10f4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_exp10f4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_exp10fx_u10sve(svfloat32_t a);
svfloat32_t Sleef_exp10fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_exp10fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_exp10fx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp10f_u10](../libm#sleef_exp10f_u10) with the same accuracy specification.

### Vectorized double precision base-<i>e</i> exponential function minus 1 with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_expm1d2_u10(float64x2_t a);
float64x2_t Sleef_expm1d2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_expm1d2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_expm1d2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_expm1dx_u10sve(svfloat64_t a);
svfloat64_t Sleef_expm1dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_expm1dx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_expm1dx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expm1_u10](../libm#sleef_expm1_u10) with the same accuracy specification.

### Vectorized single precision base-<i>e</i> exponential function minus 1 with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_expm1f4_u10(float32x4_t a);
float32x4_t Sleef_expm1f4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_expm1f4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_expm1f4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_expm1fx_u10sve(svfloat32_t a);
svfloat32_t Sleef_expm1fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_expm1fx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_expm1fx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expm1f_u10](../libm#sleef_expm1f_u10) with the same accuracy specification.

### Vectorized double precision square root function with 0.5001 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_sqrtd2_u05(float64x2_t a);
float64x2_t Sleef_sqrtd2_u05advsimd(float64x2_t a);
float64x2_t Sleef_cinz_sqrtd2_u05advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_sqrtd2_u05advsimd(float64x2_t a);

svfloat64_t Sleef_sqrtdx_u05sve(svfloat64_t a);
svfloat64_t Sleef_sqrtdx_u05svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_sqrtdx_u05svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_sqrtdx_u05sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrt_u05](../libm#sleef_sqrt_u05) with the same accuracy specification.

### Vectorized single precision square root function with 0.5001 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sqrtf4_u05(float32x4_t a);
float32x4_t Sleef_sqrtf4_u05advsimd(float32x4_t a);
float32x4_t Sleef_cinz_sqrtf4_u05advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_sqrtf4_u05advsimd(float32x4_t a);

svfloat32_t Sleef_sqrtfx_u05sve(svfloat32_t a);
svfloat32_t Sleef_sqrtfx_u05svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_sqrtfx_u05svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_sqrtfx_u05sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrtf_u05](../libm#sleef_sqrtf_u05) with the same accuracy specification.

### Vectorized double precision square root function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_sqrtd2_u35(float64x2_t a);
float64x2_t Sleef_sqrtd2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_sqrtd2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_sqrtd2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_sqrtdx_u35sve(svfloat64_t a);
svfloat64_t Sleef_sqrtdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_sqrtdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_sqrtdx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrt_u35](../libm#sleef_sqrt_u35) with the same accuracy specification.

### Vectorized single precision square root function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sqrtf4_u35(float32x4_t a);
float32x4_t Sleef_sqrtf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_sqrtf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_sqrtf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_sqrtfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_sqrtfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_sqrtfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_sqrtfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrtf_u35](../libm#sleef_sqrtf_u35) with the same accuracy specification.

### Vectorized double precision cubic root function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_cbrtd2_u10(float64x2_t a);
float64x2_t Sleef_cbrtd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_cbrtd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_cbrtd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_cbrtdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_cbrtdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_cbrtdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_cbrtdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrt_u10](../libm#sleef_cbrt_u10) with the same accuracy specification.

### Vectorized single precision cubic root function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cbrtf4_u10(float32x4_t a);
float32x4_t Sleef_cbrtf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_cbrtf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_cbrtf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_cbrtfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_cbrtfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_cbrtfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_cbrtfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrtf_u10](../libm#sleef_cbrtf_u10) with the same accuracy specification.

### Vectorized double precision cubic root function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_cbrtd2_u35(float64x2_t a);
float64x2_t Sleef_cbrtd2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_cbrtd2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_cbrtd2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_cbrtdx_u35sve(svfloat64_t a);
svfloat64_t Sleef_cbrtdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_cbrtdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_cbrtdx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrt_u35](../libm#sleef_cbrt_u35) with the same accuracy specification.

### Vectorized single precision cubic root function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cbrtf4_u35(float32x4_t a);
float32x4_t Sleef_cbrtf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_cbrtf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_cbrtf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_cbrtfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_cbrtfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_cbrtfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_cbrtfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrtf_u35](../libm#sleef_cbrtf_u35) with the same accuracy specification.

### Vectorized double precision 2D Euclidian distance function with 0.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_hypotd2_u05(float64x2_t a, float64x2_t b);
float64x2_t Sleef_hypotd2_u05advsimd(float64x2_t a, float64x2_t b);
float64x2_t Sleef_cinz_hypotd2_u05advsimdnofma(float64x2_t a, float64x2_t b);
float64x2_t Sleef_finz_hypotd2_u05advsimd(float64x2_t a, float64x2_t b);

svfloat64_t Sleef_hypotdx_u05sve(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_hypotdx_u05svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_cinz_hypotdx_u05svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_finz_hypotdx_u05sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypot_u05](../libm#sleef_hypot_u05) with the same accuracy specification.

### Vectorized single precision 2D Euclidian distance function with 0.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_hypotf4_u05(float32x4_t a, float32x4_t b);
float32x4_t Sleef_hypotf4_u05advsimd(float32x4_t a, float32x4_t b);
float32x4_t Sleef_cinz_hypotf4_u05advsimdnofma(float32x4_t a, float32x4_t b);
float32x4_t Sleef_finz_hypotf4_u05advsimd(float32x4_t a, float32x4_t b);

svfloat32_t Sleef_hypotfx_u05sve(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_hypotfx_u05svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_cinz_hypotfx_u05svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_finz_hypotfx_u05sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypotf_u05](../libm#sleef_hypotf_u05) with the same accuracy specification.

### Vectorized double precision 2D Euclidian distance function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_hypotd2_u35(float64x2_t a, float64x2_t b);
float64x2_t Sleef_hypotd2_u35advsimd(float64x2_t a, float64x2_t b);
float64x2_t Sleef_cinz_hypotd2_u35advsimdnofma(float64x2_t a, float64x2_t b);
float64x2_t Sleef_finz_hypotd2_u35advsimd(float64x2_t a, float64x2_t b);

svfloat64_t Sleef_hypotdx_u35sve(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_hypotdx_u35svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_cinz_hypotdx_u35svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_finz_hypotdx_u35sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypot_u35](../libm#sleef_hypot_u35) with the same accuracy specification.

### Vectorized single precision 2D Euclidian distance function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_hypotf4_u35(float32x4_t a, float32x4_t b);
float32x4_t Sleef_hypotf4_u35advsimd(float32x4_t a, float32x4_t b);
float32x4_t Sleef_cinz_hypotf4_u35advsimdnofma(float32x4_t a, float32x4_t b);
float32x4_t Sleef_finz_hypotf4_u35advsimd(float32x4_t a, float32x4_t b);

svfloat32_t Sleef_hypotfx_u35sve(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_hypotfx_u35svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_cinz_hypotfx_u35svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_finz_hypotfx_u35sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypotf_u35](../libm#sleef_hypotf_u35) with the same accuracy specification.

<h2 id="invtrig">Inverse Trigonometric Functions</h2>

### Vectorized double precision arc sine function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_asind2_u10(float64x2_t a);
float64x2_t Sleef_asind2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_asind2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_asind2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_asindx_u10sve(svfloat64_t a);
svfloat64_t Sleef_asindx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_asindx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_asindx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asin_u10](../libm#sleef_asin_u10) with the same accuracy specification.

### Vectorized single precision arc sine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_asinf4_u10(float32x4_t a);
float32x4_t Sleef_asinf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_asinf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_asinf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_asinfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_asinfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_asinfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_asinfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinf_u10](../libm#sleef_asinf_u10) with the same accuracy specification.

### Vectorized double precision arc sine function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_asind2_u35(float64x2_t a);
float64x2_t Sleef_asind2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_asind2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_asind2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_asindx_u35sve(svfloat64_t a);
svfloat64_t Sleef_asindx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_asindx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_asindx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asin_u35](../libm#sleef_asin_u35) with the same accuracy specification.

### Vectorized single precision arc sine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_asinf4_u35(float32x4_t a);
float32x4_t Sleef_asinf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_asinf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_asinf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_asinfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_asinfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_asinfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_asinfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinf_u35](../libm#sleef_asinf_u35) with the same accuracy specification.

### Vectorized double precision arc cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_acosd2_u10(float64x2_t a);
float64x2_t Sleef_acosd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_acosd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_acosd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_acosdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_acosdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_acosdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_acosdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acos_u10](../libm#sleef_acos_u10) with the same accuracy specification.

### Vectorized single precision arc cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_acosf4_u10(float32x4_t a);
float32x4_t Sleef_acosf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_acosf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_acosf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_acosfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_acosfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_acosfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_acosfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acosf_u10](../libm#sleef_acosf_u10) with the same accuracy specification.

### Vectorized double precision arc cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_acosd2_u35(float64x2_t a);
float64x2_t Sleef_acosd2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_acosd2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_acosd2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_acosdx_u35sve(svfloat64_t a);
svfloat64_t Sleef_acosdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_acosdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_acosdx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acos_u35](../libm#sleef_acos_u35) with the same accuracy specification.

### Vectorized single precision arc cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_acosf4_u35(float32x4_t a);
float32x4_t Sleef_acosf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_acosf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_acosf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_acosfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_acosfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_acosfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_acosfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acosf_u35](../libm#sleef_acosf_u35) with the same accuracy specification.

### Vectorized double precision arc tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_atand2_u10(float64x2_t a);
float64x2_t Sleef_atand2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_atand2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_atand2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_atandx_u10sve(svfloat64_t a);
svfloat64_t Sleef_atandx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_atandx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_atandx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan_u10](../libm#sleef_atan_u10) with the same accuracy specification.

### Vectorized single precision arc tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_atanf4_u10(float32x4_t a);
float32x4_t Sleef_atanf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_atanf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_atanf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_atanfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_atanfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_atanfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_atanfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanf_u10](../libm#sleef_atanf_u10) with the same accuracy specification.

### Vectorized double precision arc tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_atand2_u35(float64x2_t a);
float64x2_t Sleef_atand2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_atand2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_atand2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_atandx_u35sve(svfloat64_t a);
svfloat64_t Sleef_atandx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_atandx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_atandx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan_u35](../libm#sleef_atan_u35) with the same accuracy specification.

### Vectorized single precision arc tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_atanf4_u35(float32x4_t a);
float32x4_t Sleef_atanf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_atanf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_atanf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_atanfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_atanfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_atanfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_atanfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanf_u35](../libm#sleef_atanf_u35) with the same accuracy specification.

### Vectorized double precision arc tangent function of two variables with 1.0 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_atan2d2_u10(float64x2_t a, float64x2_t b);
float64x2_t Sleef_atan2d2_u10advsimd(float64x2_t a, float64x2_t b);
float64x2_t Sleef_cinz_atan2d2_u10advsimdnofma(float64x2_t a, float64x2_t b);
float64x2_t Sleef_finz_atan2d2_u10advsimd(float64x2_t a, float64x2_t b);

svfloat64_t Sleef_atan2dx_u10sve(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_atan2dx_u10svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_cinz_atan2dx_u10svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_finz_atan2dx_u10sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2_u10](../libm#sleef_atan2_u10) with the same accuracy specification.

### Vectorized single precision arc tangent function of two variables with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_atan2f4_u10(float32x4_t a, float32x4_t b);
float32x4_t Sleef_atan2f4_u10advsimd(float32x4_t a, float32x4_t b);
float32x4_t Sleef_cinz_atan2f4_u10advsimdnofma(float32x4_t a, float32x4_t b);
float32x4_t Sleef_finz_atan2f4_u10advsimd(float32x4_t a, float32x4_t b);

svfloat32_t Sleef_atan2fx_u10sve(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_atan2fx_u10svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_cinz_atan2fx_u10svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_finz_atan2fx_u10sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2f_u10](../libm#sleef_atan2f_u10) with the same accuracy specification.

### Vectorized double precision arc tangent function of two variables with 3.5 ULP error bound

```c
#include <sleef.h>

float64x2_t Sleef_atan2d2_u35(float64x2_t a, float64x2_t b);
float64x2_t Sleef_atan2d2_u35advsimd(float64x2_t a, float64x2_t b);
float64x2_t Sleef_cinz_atan2d2_u35advsimdnofma(float64x2_t a, float64x2_t b);
float64x2_t Sleef_finz_atan2d2_u35advsimd(float64x2_t a, float64x2_t b);

svfloat64_t Sleef_atan2dx_u35sve(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_atan2dx_u35svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_cinz_atan2dx_u35svenofma(svfloat64_t a, svfloat64_t b);
svfloat64_t Sleef_finz_atan2dx_u35sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2_u35](../libm#sleef_atan2_u35) with the same accuracy specification.

### Vectorized single precision arc tangent function of two variables with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_atan2f4_u35(float32x4_t a, float32x4_t b);
float32x4_t Sleef_atan2f4_u35advsimd(float32x4_t a, float32x4_t b);
float32x4_t Sleef_cinz_atan2f4_u35advsimdnofma(float32x4_t a, float32x4_t b);
float32x4_t Sleef_finz_atan2f4_u35advsimd(float32x4_t a, float32x4_t b);

svfloat32_t Sleef_atan2fx_u35sve(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_atan2fx_u35svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_cinz_atan2fx_u35svenofma(svfloat32_t a, svfloat32_t b);
svfloat32_t Sleef_finz_atan2fx_u35sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2f_u35](../libm#sleef_atan2f_u35) with the same accuracy specification.

<h2 id="hyp">Hyperbolic function and inverse hyperbolic function</h2>

### Vectorized double precision hyperbolic sine function

```c
#include <sleef.h>

float64x2_t Sleef_sinhd2_u10(float64x2_t a);
float64x2_t Sleef_sinhd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_sinhd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_sinhd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_sinhdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_sinhdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_sinhdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_sinhdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinh_u10](../libm#sleef_sinh_u10) with the same accuracy specification.

### Vectorized single precision hyperbolic sine function

```c
#include <sleef.h>

float32x4_t Sleef_sinhf4_u10(float32x4_t a);
float32x4_t Sleef_sinhf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_sinhf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_sinhf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_sinhfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_sinhfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_sinhfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_sinhfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinhf_u10](../libm#sleef_sinhf_u10) with the same accuracy specification.

### Vectorized double precision hyperbolic sine function

```c
#include <sleef.h>

float64x2_t Sleef_sinhd2_u35(float64x2_t a);
float64x2_t Sleef_sinhd2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_sinhd2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_sinhd2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_sinhdx_u35sve(svfloat64_t a);
svfloat64_t Sleef_sinhdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_sinhdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_sinhdx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinh_u35](../libm#sleef_sinh_u35) with the same accuracy specification.

### Vectorized single precision hyperbolic sine function

```c
#include <sleef.h>

float32x4_t Sleef_sinhf4_u35(float32x4_t a);
float32x4_t Sleef_sinhf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_sinhf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_sinhf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_sinhfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_sinhfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_sinhfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_sinhfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinhf_u35](../libm#sleef_sinhf_u35) with the same accuracy specification.

### Vectorized double precision hyperbolic cosine function

```c
#include <sleef.h>

float64x2_t Sleef_coshd2_u10(float64x2_t a);
float64x2_t Sleef_coshd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_coshd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_coshd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_coshdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_coshdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_coshdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_coshdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosh_u10](../libm#sleef_cosh_u10) with the same accuracy specification.

### Vectorized single precision hyperbolic cosine function

```c
#include <sleef.h>

float32x4_t Sleef_coshf4_u10(float32x4_t a);
float32x4_t Sleef_coshf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_coshf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_coshf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_coshfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_coshfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_coshfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_coshfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_coshf_u10](../libm#sleef_coshf_u10) with the same accuracy specification.

### Vectorized double precision hyperbolic cosine function

```c
#include <sleef.h>

float64x2_t Sleef_coshd2_u35(float64x2_t a);
float64x2_t Sleef_coshd2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_coshd2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_coshd2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_coshdx_u35sve(svfloat64_t a);
svfloat64_t Sleef_coshdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_coshdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_coshdx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosh_u35](../libm#sleef_cosh_u35) with the same accuracy specification.

### Vectorized single precision hyperbolic cosine function

```c
#include <sleef.h>

float32x4_t Sleef_coshf4_u35(float32x4_t a);
float32x4_t Sleef_coshf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_coshf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_coshf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_coshfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_coshfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_coshfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_coshfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_coshf_u35](../libm#sleef_coshf_u35) with the same accuracy specification.

### Vectorized double precision hyperbolic tangent function

```c
#include <sleef.h>

float64x2_t Sleef_tanhd2_u10(float64x2_t a);
float64x2_t Sleef_tanhd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_tanhd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_tanhd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_tanhdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_tanhdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_tanhdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_tanhdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanh_u10](../libm#sleef_tanh_u10) with the same accuracy specification.

### Vectorized single precision hyperbolic tangent function

```c
#include <sleef.h>

float32x4_t Sleef_tanhf4_u10(float32x4_t a);
float32x4_t Sleef_tanhf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_tanhf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_tanhf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_tanhfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_tanhfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_tanhfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_tanhfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanhf_u10](../libm#sleef_tanhf_u10) with the same accuracy specification.

### Vectorized double precision hyperbolic tangent function

```c
#include <sleef.h>

float64x2_t Sleef_tanhd2_u35(float64x2_t a);
float64x2_t Sleef_tanhd2_u35advsimd(float64x2_t a);
float64x2_t Sleef_cinz_tanhd2_u35advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_tanhd2_u35advsimd(float64x2_t a);

svfloat64_t Sleef_tanhdx_u35sve(svfloat64_t a);
svfloat64_t Sleef_tanhdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_tanhdx_u35svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_tanhdx_u35sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanh_u35](../libm#sleef_tanh_u35) with the same accuracy specification.

### Vectorized single precision hyperbolic tangent function

```c
#include <sleef.h>

float32x4_t Sleef_tanhf4_u35(float32x4_t a);
float32x4_t Sleef_tanhf4_u35advsimd(float32x4_t a);
float32x4_t Sleef_cinz_tanhf4_u35advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_tanhf4_u35advsimd(float32x4_t a);

svfloat32_t Sleef_tanhfx_u35sve(svfloat32_t a);
svfloat32_t Sleef_tanhfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_tanhfx_u35svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_tanhfx_u35sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanhf_u35](../libm#sleef_tanhf_u35) with the same accuracy specification.

### Vectorized double precision inverse hyperbolic sine function

```c
#include <sleef.h>

float64x2_t Sleef_asinhd2_u10(float64x2_t a);
float64x2_t Sleef_asinhd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_asinhd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_asinhd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_asinhdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_asinhdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_asinhdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_asinhdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinh_u10](../libm#sleef_asinh_u10) with the same accuracy specification.

### Vectorized single precision inverse hyperbolic sine function

```c
#include <sleef.h>

float32x4_t Sleef_asinhf4_u10(float32x4_t a);
float32x4_t Sleef_asinhf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_asinhf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_asinhf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_asinhfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_asinhfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_asinhfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_asinhfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinhf_u10](../libm#sleef_asinhf_u10) with the same accuracy specification.

### Vectorized double precision inverse hyperbolic cosine function

```c
#include <sleef.h>

float64x2_t Sleef_acoshd2_u10(float64x2_t a);
float64x2_t Sleef_acoshd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_acoshd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_acoshd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_acoshdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_acoshdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_acoshdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_acoshdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acosh_u10](../libm#sleef_acosh_u10) with the same accuracy specification.

### Vectorized single precision inverse hyperbolic cosine function

```c
#include <sleef.h>

float32x4_t Sleef_acoshf4_u10(float32x4_t a);
float32x4_t Sleef_acoshf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_acoshf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_acoshf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_acoshfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_acoshfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_acoshfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_acoshfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acoshf_u10](../libm#sleef_acoshf_u10) with the same accuracy specification.

### Vectorized double precision inverse hyperbolic tangent function

```c
#include <sleef.h>

float64x2_t Sleef_atanhd2_u10(float64x2_t a);
float64x2_t Sleef_atanhd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_atanhd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_atanhd2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_atanhdx_u10sve(svfloat64_t a);
svfloat64_t Sleef_atanhdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_atanhdx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_atanhdx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanh_u10](../libm#sleef_atanh_u10) with the same accuracy specification.

### Vectorized single precision inverse hyperbolic tangent function

```c
#include <sleef.h>

float32x4_t Sleef_atanhf4_u10(float32x4_t a);
float32x4_t Sleef_atanhf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_atanhf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_atanhf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_atanhfx_u10sve(svfloat32_t a);
svfloat32_t Sleef_atanhfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_atanhfx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_atanhfx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanhf_u10](../libm#sleef_atanhf_u10) with the same accuracy specification.

<h2 id="eg">Error and gamma function</h2>

### Vectorized double precision error function

```c
#include <sleef.h>

float64x2_t Sleef_erfd2_u10(float64x2_t a);
float64x2_t Sleef_erfd2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_erfd2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_erfd2_u10advsimd(float64x2_t a);

svfloat32_t Sleef_erfdx_u10sve(svfloat32_t a);
svfloat32_t Sleef_erfdx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_erfdx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_erfdx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erf_u10](../libm#sleef_erf_u10) with the same accuracy specification.

### Vectorized single precision error function

```c
#include <sleef.h>

float32x4_t Sleef_erff4_u10(float32x4_t a);
float32x4_t Sleef_erff4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_erff4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_erff4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_erffx_u10sve(svfloat32_t a);
svfloat32_t Sleef_erffx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_erffx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_erffx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erff_u10](../libm#sleef_erff_u10) with the same accuracy specification.

### Vectorized double precision complementary error function

```c
#include <sleef.h>

float64x2_t Sleef_erfcd2_u15(float64x2_t a);
float64x2_t Sleef_erfcd2_u15advsimd(float64x2_t a);
float64x2_t Sleef_cinz_erfcd2_u15advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_erfcd2_u15advsimd(float64x2_t a);

svfloat64_t Sleef_erfcdx_u15sve(svfloat64_t a);
svfloat64_t Sleef_erfcdx_u15svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_erfcdx_u15svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_erfcdx_u15sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erfc_u15](../libm#sleef_erfc_u15) with the same accuracy specification.

### Vectorized single precision complementary error function

```c
#include <sleef.h>

float32x4_t Sleef_erfcf4_u15(float32x4_t a);
float32x4_t Sleef_erfcf4_u15advsimd(float32x4_t a);
float32x4_t Sleef_cinz_erfcf4_u15advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_erfcf4_u15advsimd(float32x4_t a);

svfloat32_t Sleef_erfcfx_u15sve(svfloat32_t a);
svfloat32_t Sleef_erfcfx_u15svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_erfcfx_u15svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_erfcfx_u15sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erfcf_u15](../libm#sleef_erfcf_u15) with the same accuracy specification.

### Vectorized double precision gamma function

```c
#include <sleef.h>

float64x2_t Sleef_tgammad2_u10(float64x2_t a);
float64x2_t Sleef_tgammad2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_tgammad2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_tgammad2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_tgammadx_u10sve(svfloat64_t a);
svfloat64_t Sleef_tgammadx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_tgammadx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_tgammadx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tgamma_u10](../libm#sleef_tgamma_u10) with the same accuracy specification.

### Vectorized single precision gamma function

```c
#include <sleef.h>

float32x4_t Sleef_tgammaf4_u10(float32x4_t a);
float32x4_t Sleef_tgammaf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_cinz_tgammaf4_u10advsimdnofma(float32x4_t a);
float32x4_t Sleef_finz_tgammaf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_tgammafx_u10sve(svfloat32_t a);
svfloat32_t Sleef_tgammafx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_tgammafx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_tgammafx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tgammaf_u10](../libm#sleef_tgammaf_u10) with the same accuracy specification.

### Vectorized double precision log gamma function

```c
#include <sleef.h>

float64x2_t Sleef_lgammad2_u10(float64x2_t a);
float64x2_t Sleef_lgammad2_u10advsimd(float64x2_t a);
float64x2_t Sleef_cinz_lgammad2_u10advsimdnofma(float64x2_t a);
float64x2_t Sleef_finz_lgammad2_u10advsimd(float64x2_t a);

svfloat64_t Sleef_lgammadx_u10sve(svfloat64_t a);
svfloat64_t Sleef_lgammadx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_cinz_lgammadx_u10svenofma(svfloat64_t a);
svfloat64_t Sleef_finz_lgammadx_u10sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_lgamma_u10](../libm#sleef_lgamma_u10) with the same accuracy specification.

### Vectorized single precision log gamma function

```c
#include <sleef.h>

float32x4_t Sleef_lgammaf4_u10(float32x4_t a);
float32x4_t Sleef_lgammaf4_u10advsimd(float32x4_t a);
float32x4_t Sleef_finz_lgammaf4_u10advsimd(float32x4_t a);

svfloat32_t Sleef_lgammafx_u10sve(svfloat32_t a);
svfloat32_t Sleef_lgammafx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_cinz_lgammafx_u10svenofma(svfloat32_t a);
svfloat32_t Sleef_finz_lgammafx_u10sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_lgammaf_u10](../libm#sleef_lgammaf_u10) with the same accuracy specification.

<h2 id="nearint">Nearest integer function</h2>

### Vectorized double precision function for rounding to integer towards zero

```c
#include <sleef.h>

float64x2_t Sleef_truncd2(float64x2_t a);
float64x2_t Sleef_truncd2_advsimd(float64x2_t a);
svfloat64_t Sleef_truncdx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_trunc](../libm#sleef_trunc) with the same accuracy specification.

### Vectorized single precision function for rounding to integer towards zero

```c
#include <sleef.h>

float32x4_t Sleef_truncf4(float32x4_t a);
float32x4_t Sleef_truncf4_advsimd(float32x4_t a);
svfloat32_t Sleef_truncfx_sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_truncf](../libm#sleef_truncf) with the same accuracy specification.

### Vectorized double precision function for rounding to integer towards negative infinity

```c
#include <sleef.h>

float64x2_t Sleef_floord2(float64x2_t a);
float64x2_t Sleef_floord2_advsimd(float64x2_t a);
svfloat64_t Sleef_floordx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_floor](../libm#sleef_floor) with the same accuracy specification.

### Vectorized single precision function for rounding to integer towards negative infinity

```c
#include <sleef.h>

float32x4_t Sleef_floorf4(float32x4_t a);
float32x4_t Sleef_floorf4_advsimd(float32x4_t a);
svfloat32_t Sleef_floorfx_sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_floorf](../libm#sleef_floorf) with the same accuracy specification.

### Vectorized double precision function for rounding to integer towards positive infinity

```c
#include <sleef.h>

float64x2_t Sleef_ceild2(float64x2_t a);
float64x2_t Sleef_ceild2_advsimd(float64x2_t a);
svfloat64_t Sleef_ceildx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_ceil](../libm#sleef_ceil) with the same accuracy specification.

### Vectorized single precision function for rounding to integer towards positive infinity

```c
#include <sleef.h>

float32x4_t Sleef_ceilf4(float32x4_t a);
float32x4_t Sleef_ceilf4_advsimd(float32x4_t a);
svfloat32_t Sleef_ceilfx_sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_ceilf](../libm#sleef_ceilf) with the same accuracy specification.

### Vectorized double precision function for rounding to nearest integer

```c
#include <sleef.h>

float64x2_t Sleef_roundd2(float64x2_t a);
float64x2_t Sleef_roundd2_advsimd(float64x2_t a);
svfloat64_t Sleef_rounddx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_round](../libm#sleef_round) with the same accuracy specification.

### Vectorized single precision function for rounding to nearest integer

```c
#include <sleef.h>

float32x4_t Sleef_roundf4(float32x4_t a);
float32x4_t Sleef_roundf4_advsimd(float32x4_t a);
svfloat32_t Sleef_roundfx_sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_roundf](../libm#sleef_roundf) with the same accuracy specification.

### Vectorized double precision function for rounding to nearest integer

```c
#include <sleef.h>

float64x2_t Sleef_rintd2(float64x2_t a);
float64x2_t Sleef_rintd2_advsimd(float64x2_t a);
svfloat64_t Sleef_rintdx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_rint](../libm#sleef_rint) with the same accuracy specification.

### Vectorized single precision function for rounding to nearest integer

```c
#include <sleef.h>

float32x4_t Sleef_rintf4(float32x4_t a);
float32x4_t Sleef_rintf4_advsimd(float32x4_t a);
svfloat32_t Sleef_rintfx_sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_rintf](../libm#sleef_rintf) with the same accuracy specification.

<h2 id="other">Other function</h2>

### Vectorized double precision function for fused multiply-accumulation

```c
#include <sleef.h>

float64x2_t Sleef_fmad2(float64x2_t a, float64x2_t b, float64x2_t c);
float64x2_t Sleef_fmad2_advsimd(float64x2_t a, float64x2_t b, float64x2_t c);
svfloat64_t Sleef_fmadx_sve(svfloat64_t a, svfloat64_t b, svfloat64_t c);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fma](../libm#sleef_fma) with the same accuracy specification.

### Vectorized single precision function for fused multiply-accumulation

```c
#include <sleef.h>

float32x4_t Sleef_fmaf4(float32x4_t a, float32x4_t b, float32x4_t c);
float32x4_t Sleef_fmaf4_advsimd(float32x4_t a, float32x4_t b, svfloat32_t c);
svfloat32_t Sleef_fmafx_sve(svfloat32_t a, svfloat32_t b, svfloat32_t c);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmaf](../libm#sleef_fmaf) with the same accuracy specification.

### Vectorized double precision FP remainder

```c
#include <sleef.h>

float64x2_t Sleef_fmodd2(float64x2_t a, float64x2_t b);
float64x2_t Sleef_fmodd2_advsimd(float64x2_t a, float64x2_t b);
svfloat64_t Sleef_fmoddx_sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmod](../libm#sleef_fmod) with the same accuracy specification.

### Vectorized single precision FP remainder

```c
#include <sleef.h>

float32x4_t Sleef_fmodf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_fmodf4_advsimd(float32x4_t a, float32x4_t b);
svfloat32_t Sleef_fmodfx_sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmodf](../libm#sleef_fmodf) with the same accuracy specification.

### Vectorized double precision FP remainder

```c
#include <sleef.h>

float64x2_t Sleef_remainderd2(float64x2_t a, float64x2_t b);
float64x2_t Sleef_remainderd2_advsimd(float64x2_t a, float64x2_t b);
svfloat64_t Sleef_remainderdx_sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_remainder](../libm#sleef_remainder) with the same accuracy specification.

### Vectorized single precision FP remainder

```c
#include <sleef.h>

float32x4_t Sleef_remainderf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_remainderf4_advsimd(float32x4_t a, float32x4_t b);
svfloat32_t Sleef_remainderfx_sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_remainderf](../libm#sleef_remainderf) with the same accuracy specification.

### Vectorized double precision function for multiplying by integral power of 2

```c
#include <sleef.h>

float64x2_t Sleef_ldexpd2(float64x2_t a, int32x2_t b);
float64x2_t Sleef_ldexpd2_advsimd(float64x2_t a, int32x2_t b);
svfloat64_t Sleef_ldexpdx_sve(svfloat64_t a, svint32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_ldexp](../libm#sleef_ldexp) with the same accuracy specification.

### Vectorized double precision function for obtaining fractional component of an FP number

```c
#include <sleef.h>

float64x2_t Sleef_frfrexpd2(float64x2_t a);
float64x2_t Sleef_frfrexpd2_advsimd(float64x2_t a);
svfloat64_t Sleef_frfrexpdx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_frfrexp](../libm#sleef_frfrexp) with the same accuracy specification.

### Vectorized single precision function for obtaining fractional component of an FP number

```c
#include <sleef.h>

float32x4_t Sleef_frfrexpf4(float32x4_t a);
float32x4_t Sleef_frfrexpf4_advsimd(float32x4_t a);
svfloat32_t Sleef_frfrexpfx_sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_frfrexpf](../libm#sleef_frfrexpf) with the same accuracy specification.

### Vectorized double precision function for obtaining integral component of an FP number

```c
#include <sleef.h>

int32x2_t Sleef_expfrexpd2(float64x2_t a);
int32x2_t Sleef_expfrexpd2_advsimd(float64x2_t a);
svint32_t Sleef_expfrexpdx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expfrexp](../libm#sleef_expfrexp) with the same accuracy specification.

### Vectorized double precision function for getting integer exponent

```c
#include <sleef.h>

int32x2_t Sleef_ilogbd2(float64x2_t a);
int32x2_t Sleef_ilogbd2_advsimd(float64x2_t a);
svint32_t Sleef_ilogbdx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_ilogb](../libm#sleef_ilogb) with the same accuracy specification.

### Vectorized double precision signed integral and fractional values

```c
#include <sleef.h>

Sleef_float64x2_t_2 Sleef_modfd2(float64x2_t a);
Sleef_float64x2_t_2 Sleef_modfd2_advsimd(float64x2_t a);
Sleef_svfloat64_t_2 Sleef_modfdx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_modf](../libm#sleef_modf) with the same accuracy specification.

### Vectorized single precision signed integral and fractional values

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_modff4(float32x4_t a);
Sleef_float32x4_t_2 Sleef_modff4_advsimd(float32x4_t a);
Sleef_svfloat32_t_2 Sleef_modffx_sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_modff](../libm#sleef_modff) with the same accuracy specification.

### Vectorized double precision function for calculating the absolute value

```c
#include <sleef.h>

float64x2_t Sleef_fabsd2(float64x2_t a);
float64x2_t Sleef_fabsd2_advsimd(float64x2_t a);
svfloat64_t Sleef_fabsdx_sve(svfloat64_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fabs](../libm#sleef_fabs) with the same accuracy specification.

### Vectorized single precision function for calculating the absolute value

```c
#include <sleef.h>

float32x4_t Sleef_fabsf4(float32x4_t a);
float32x4_t Sleef_fabsf4_advsimd(float32x4_t a);
svfloat32_t Sleef_fabsfx_sve(svfloat32_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fabsf](../libm#sleef_fabsf) with the same accuracy specification.

### Vectorized double precision function for copying signs

```c
#include <sleef.h>

float64x2_t Sleef_copysignd2(float64x2_t a, float64x2_t b);
float64x2_t Sleef_copysignd2_advsimd(float64x2_t a, float64x2_t b);
svfloat64_t Sleef_copysigndx_sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_copysign](../libm#sleef_copysign) with the same accuracy specification.

### Vectorized single precision function for copying signs

```c
#include <sleef.h>

float32x4_t Sleef_copysignf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_copysignf4_advsimd(float32x4_t a, float32x4_t b);
svfloat32_t Sleef_copysignfx_sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_copysignf](../libm#sleef_copysignf) with the same accuracy specification.

### Vectorized double precision function for determining maximum of two values

```c
#include <sleef.h>

float64x2_t Sleef_fmaxd2(float64x2_t a, float64x2_t b);
float64x2_t Sleef_fmaxd2_advsimd(float64x2_t a, float64x2_t b);
svfloat64_t Sleef_fmaxdx_sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmax](../libm#sleef_fmax) with the same accuracy specification.

### Vectorized single precision function for determining maximum of two values

```c
#include <sleef.h>

float32x4_t Sleef_fmaxf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_fmaxf4_advsimd(float32x4_t a, float32x4_t b);
svfloat32_t Sleef_fmaxfx_sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmaxf](../libm#sleef_fmaxf) with the same accuracy specification.

### Vectorized double precision function for determining minimum of two values

```c
#include <sleef.h>

float64x2_t Sleef_fmind2(float64x2_t a, float64x2_t b);
float64x2_t Sleef_fmind2_advsimd(float64x2_t a, float64x2_t b);
svfloat64_t Sleef_fmindx_sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmin](../libm#sleef_fmin) with the same accuracy specification.

### Vectorized single precision function for determining minimum of two values

```c
#include <sleef.h>

float32x4_t Sleef_fminf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_fminf4_advsimd(float32x4_t a, float32x4_t b);
svfloat32_t Sleef_fminfx_sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fminf](../libm#sleef_fminf) with the same accuracy specification.

### Vectorized double precision function to calculate positive difference of two values

```c
#include <sleef.h>

float64x2_t Sleef_fdimd2(float64x2_t a, float64x2_t b);
float64x2_t Sleef_fdimd2_advsimd(float64x2_t a, float64x2_t b);
svfloat64_t Sleef_fdimdx_sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fdim](../libm#sleef_fdim) with the same accuracy specification.

### Vectorized single precision function to calculate positive difference of two values

```c
#include <sleef.h>

float32x4_t Sleef_fdimf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_fdimf4_advsimd(float32x4_t a, float32x4_t b);
svfloat32_t Sleef_fdimfx_sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fdimf](../libm#sleef_fdimf) with the same accuracy specification.

### Vectorized double precision function for obtaining the next representable FP value

```c
#include <sleef.h>

float64x2_t Sleef_nextafterd2(float64x2_t a, float64x2_t b);
float64x2_t Sleef_nextafterd2_advsimd(float64x2_t a, float64x2_t b);
svfloat64_t Sleef_nextafterdx_sve(svfloat64_t a, svfloat64_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_nextafter](../libm#sleef_nextafter) with the same accuracy specification.

### Vectorized single precision function for obtaining the next representable FP value

```c
#include <sleef.h>

float32x4_t Sleef_nextafterf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_nextafterf4_advsimd(float32x4_t a, float32x4_t b);
svfloat32_t Sleef_nextafterfx_sve(svfloat32_t a, svfloat32_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_nextafterf](../libm#sleef_nextafterf) with the same accuracy specification.
