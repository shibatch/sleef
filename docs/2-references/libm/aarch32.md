---
layout: default
title: AArch32
parent: Single & Double Precision
grand_parent: References
permalink: /2-references/libm/aarch32
---

<h1>Single & Double Precision Math library reference (AArch32)</h1>

<h2>Table of contents</h2>

* [Data types](#datatypes)
* [Trigonometric functions](#trig)
* [Power, exponential, and logarithmic functions](#pow)
* [Inverse trigonometric functions](#invtrig)
* [Hyperbolic functions and inverse hyperbolic functions](#hyp)
* [Error and gamma functions](#eg)
* [Nearest integer functions](#nearint)
* [Other functions](#other)

<h2 id="datatypes">Data types for AArch32 architecture</h2>

### Sleef_float32x4_t_2

`Sleef_float32x4_t_2` is a data type for storing two `float32x4_t` values,
which is defined in sleef.h as follows:

```c
typedef struct {
  float32x4_t x, y;
} Sleef_float32x4_t_2;
```

<h2 id="trig">Trigonometric Functions</h2>

### Vectorized single precision sine function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sinf4_u10(float32x4_t a);
float32x4_t Sleef_sinf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinf_u10](../libm#sleef_sinf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision sine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sinf4_u35(float32x4_t a);
float32x4_t Sleef_sinf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinf_u35](../libm#sleef_sinf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cosf4_u10(float32x4_t a);
float32x4_t Sleef_cosf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosf_u10](../libm#sleef_cosf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cosf4_u35(float32x4_t a);
float32x4_t Sleef_cosf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosf_u35](../libm#sleef_cosf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision combined sine and cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_sincosf4_u10(float32x4_t a);
Sleef_float32x4_t_2 Sleef_sincosf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincosf_u10](../libm#sleef_sincosf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_sincosf4_u35(float32x4_t a);
Sleef_float32x4_t_2 Sleef_sincosf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincosf_u35](../libm#sleef_sincosf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision sine function with 0.506 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sinpif4_u05(float32x4_t a);
float32x4_t Sleef_sinpif4_u05neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinpif_u05](../libm#sleef_sinpif_u05). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cospif4_u05(float32x4_t a);
float32x4_t Sleef_cospif4_u05neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cospif_u05](../libm#sleef_cospif_u05). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_sincospif4_u05(float32x4_t a);
Sleef_float32x4_t_2 Sleef_sincospif4_u05neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospif_u05](../libm#sleef_sincospif_u05). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_sincospif4_u35(float32x4_t a);
Sleef_float32x4_t_2 Sleef_sincospif4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospif_u35](../libm#sleef_sincospif_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_tanf4_u10(float32x4_t a);
float32x4_t Sleef_tanf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanf_u10](../libm#sleef_tanf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_tanf4_u35(float32x4_t a);
float32x4_t Sleef_tanf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanf_u35](../libm#sleef_tanf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

<h2 id="pow">Power, exponential, and logarithmic function</h2>

### Vectorized single precision power function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_powf4_u10(float32x4_t a, float32x4_t b);
float32x4_t Sleef_powf4_u10neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_powf_u10](../libm#sleef_powf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision natural logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_logf4_u10(float32x4_t a);
float32x4_t Sleef_logf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_logf_u10](../libm#sleef_logf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision natural logarithmic function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_logf4_u35(float32x4_t a);
float32x4_t Sleef_logf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_logf_u35](../libm#sleef_logf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision base-10 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_log10f4_u10(float32x4_t a);
float32x4_t Sleef_log10f4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log10f_u10](../libm#sleef_log10f_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision base-2 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_log2f4_u10(float32x4_t a);
float32x4_t Sleef_log2f4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log2f_u10](../libm#sleef_log2f_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision logarithm of one plus argument with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_log1pf4_u10(float32x4_t a);
float32x4_t Sleef_log1pf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log1pf_u10](../libm#sleef_log1pf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision base-<i>e</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_expf4_u10(float32x4_t a);
float32x4_t Sleef_expf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expf_u10](../libm#sleef_expf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision base-<i>2</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_exp2f4_u10(float32x4_t a);
float32x4_t Sleef_exp2f4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp2f_u10](../libm#sleef_exp2f_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision base-10 exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_exp10f4_u10(float32x4_t a);
float32x4_t Sleef_exp10f4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp10f_u10](../libm#sleef_exp10f_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision base-<i>e</i> exponential function minus 1 with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_expm1f4_u10(float32x4_t a);
float32x4_t Sleef_expm1f4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expm1f_u10](../libm#sleef_expm1f_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision square root function with 0.5001 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sqrtf4(float32x4_t a);
float32x4_t Sleef_sqrtf4_neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrtf_u05](../libm#sleef_sqrtf_u05). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision square root function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_sqrtf4_u35(float32x4_t a);
float32x4_t Sleef_sqrtf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrtf_u35](../libm#sleef_sqrtf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision cubic root function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cbrtf4_u10(float32x4_t a);
float32x4_t Sleef_cbrtf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrtf_u10](../libm#sleef_cbrtf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision cubic root function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_cbrtf4_u35(float32x4_t a);
float32x4_t Sleef_cbrtf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrtf_u35](../libm#sleef_cbrtf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision 2D Euclidian distance function with 0.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_hypotf4_u05(float32x4_t a, float32x4_t b);
float32x4_t Sleef_hypotf4_u05neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypotf_u05](../libm#sleef_hypotf_u05). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision 2D Euclidian distance function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_hypotf4_u35(float32x4_t a, float32x4_t b);
float32x4_t Sleef_hypotf4_u35neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypotf_u35](../libm#sleef_hypotf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

<h2 id="invtrig">Inverse Trigonometric Functions</h2>

### Vectorized single precision arc sine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_asinf4_u10(float32x4_t a);
float32x4_t Sleef_asinf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinf_u10](../libm#sleef_asinf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision arc sine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_asinf4_u35(float32x4_t a);
float32x4_t Sleef_asinf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinf_u35](../libm#sleef_asinf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision arc cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_acosf4_u10(float32x4_t a);
float32x4_t Sleef_acosf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acosf_u10](../libm#sleef_acosf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision arc cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_acosf4_u35(float32x4_t a);
float32x4_t Sleef_acosf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acosf_u35](../libm#sleef_acosf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision arc tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_atanf4_u10(float32x4_t a);
float32x4_t Sleef_atanf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanf_u10](../libm#sleef_atanf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision arc tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_atanf4_u35(float32x4_t a);
float32x4_t Sleef_atanf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanf_u35](../libm#sleef_atanf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision arc tangent function of two variables with 1.0 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_atan2f4_u10(float32x4_t a, float32x4_t b);
float32x4_t Sleef_atan2f4_u10neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2f_u10](../libm#sleef_atan2f_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision arc tangent function of two variables with 3.5 ULP error bound

```c
#include <sleef.h>

float32x4_t Sleef_atan2f4_u35(float32x4_t a, float32x4_t b);
float32x4_t Sleef_atan2f4_u35neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2f_u35](../libm#sleef_atan2f_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

<h2 id="hyp">Hyperbolic function and inverse hyperbolic function</h2>

### Vectorized single precision hyperbolic sine function

```c
#include <sleef.h>

float32x4_t Sleef_sinhf4_u10(float32x4_t a);
float32x4_t Sleef_sinhf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinhf_u10](../libm#sleef_sinhf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision hyperbolic sine function

```c
#include <sleef.h>

float32x4_t Sleef_sinhf4_u35(float32x4_t a);
float32x4_t Sleef_sinhf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinhf_u35](../libm#sleef_sinhf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision hyperbolic cosine function

```c
#include <sleef.h>

float32x4_t Sleef_coshf4_u10(float32x4_t a);
float32x4_t Sleef_coshf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_coshf_u10](../libm#sleef_coshf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision hyperbolic cosine function

```c
#include <sleef.h>

float32x4_t Sleef_coshf4_u35(float32x4_t a);
float32x4_t Sleef_coshf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_coshf_u35](../libm#sleef_coshf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision hyperbolic tangent function

```c
#include <sleef.h>

float32x4_t Sleef_tanhf4_u10(float32x4_t a);
float32x4_t Sleef_tanhf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanhf_u10](../libm#sleef_tanhf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision hyperbolic tangent function

```c
#include <sleef.h>

float32x4_t Sleef_tanhf4_u35(float32x4_t a);
float32x4_t Sleef_tanhf4_u35neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanhf_u35](../libm#sleef_tanhf_u35). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision inverse hyperbolic sine function

```c
#include <sleef.h>

float32x4_t Sleef_asinhf4_u10(float32x4_t a);
float32x4_t Sleef_asinhf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinhf_u10](../libm#sleef_asinhf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision inverse hyperbolic cosine function

```c
#include <sleef.h>

float32x4_t Sleef_acoshf4_u10(float32x4_t a);
float32x4_t Sleef_acoshf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acoshf_u10](../libm#sleef_acoshf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision inverse hyperbolic tangent function

```c
#include <sleef.h>

float32x4_t Sleef_atanhf4_u10(float32x4_t a);
float32x4_t Sleef_atanhf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanhf_u10](../libm#sleef_atanhf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

<h2 id="eg">Error and gamma function</h2>

### Vectorized single precision error function

```c
#include <sleef.h>

float32x4_t Sleef_erff4_u10(float32x4_t a);
float32x4_t Sleef_erff4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erff_u10](../libm#sleef_erff_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision complementary error function

```c
#include <sleef.h>

float32x4_t Sleef_erfcf4_u15(float32x4_t a);
float32x4_t Sleef_erfcf4_u15neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erfcf_u15](../libm#sleef_erfcf_u15). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision gamma function

```c
#include <sleef.h>

float32x4_t Sleef_tgammaf4_u10(float32x4_t a);
float32x4_t Sleef_tgammaf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tgammaf_u10](../libm#sleef_tgammaf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision log gamma function

```c
#include <sleef.h>

float32x4_t Sleef_lgammaf4_u10(float32x4_t a);
float32x4_t Sleef_lgammaf4_u10neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_lgammaf_u10](../libm#sleef_lgammaf_u10). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

<h2 id="nearint">Nearest integer function</h2>

### Vectorized single precision function for rounding to integer towards zero

```c
#include <sleef.h>

float32x4_t Sleef_truncf4(float32x4_t a);
float32x4_t Sleef_truncf4_neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_truncf](../libm#sleef_truncf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for rounding to integer towards negative infinity

```c
#include <sleef.h>

float32x4_t Sleef_floorf4(float32x4_t a);
float32x4_t Sleef_floorf4_neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_floorf](../libm#sleef_floorf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for rounding to integer towards positive infinity

```c
#include <sleef.h>

float32x4_t Sleef_ceilf4(float32x4_t a);
float32x4_t Sleef_ceilf4_neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_ceilf](../libm#sleef_ceilf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for rounding to nearest integer

```c
#include <sleef.h>

float32x4_t Sleef_roundf4(float32x4_t a);
float32x4_t Sleef_roundf4_neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_roundf](../libm#sleef_roundf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for rounding to nearest integer

```c
#include <sleef.h>

float32x4_t Sleef_rintf4(float32x4_t a);
float32x4_t Sleef_rintf4_neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_rintf](../libm#sleef_rintf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

<h2 id="other">Other function</h2>

### Vectorized single precision function for fused multiply-accumulation

```c
#include <sleef.h>

float32x4_t Sleef_fmaf4(float32x4_t a, float32x4_t b, float32x4_t c);
float32x4_t Sleef_fmaf4_neon(float32x4_t a, float32x4_t b, float32x4_t c);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmaf](../libm#sleef_fmaf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision FP remainder

```c
#include <sleef.h>

float32x4_t Sleef_fmodf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_fmodf4_neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmodf](../libm#sleef_fmodf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision FP remainder

```c
#include <sleef.h>

float32x4_t Sleef_remainderf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_remainderf4_neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_remainderf](../libm#sleef_remainderf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for obtaining fractional component of an FP number

```c
#include <sleef.h>

float32x4_t Sleef_frfrexpf4(float32x4_t a);
float32x4_t Sleef_frfrexpf4_neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_frfrexpf](../libm#sleef_frfrexpf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision signed integral and fractional values

```c
#include <sleef.h>

Sleef_float32x4_t_2 Sleef_modff4(float32x4_t a);
Sleef_float32x4_t_2 Sleef_modff4_neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_modff](../libm#sleef_modff). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for calculating the absolute value

```c
#include <sleef.h>

float32x4_t Sleef_fabsf4(float32x4_t a);
float32x4_t Sleef_fabsf4_neon(float32x4_t a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fabsf](../libm#sleef_fabsf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for copying signs

```c
#include <sleef.h>

float32x4_t Sleef_copysignf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_copysignf4_neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_copysignf](../libm#sleef_copysignf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for determining maximum of two values

```c
#include <sleef.h>

float32x4_t Sleef_fmaxf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_fmaxf4_neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmaxf](../libm#sleef_fmaxf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for determining minimum of two values

```c
#include <sleef.h>

float32x4_t Sleef_fminf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_fminf4_neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fminf](../libm#sleef_fminf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function to calculate positive difference of two values

```c
#include <sleef.h>

float32x4_t Sleef_fdimf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_fdimf4_neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fdimf](../libm#sleef_fdimf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.

### Vectorized single precision function for obtaining the next representable FP value

```c
#include <sleef.h>

float32x4_t Sleef_nextafterf4(float32x4_t a, float32x4_t b);
float32x4_t Sleef_nextafterf4_neon(float32x4_t a, float32x4_t b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_nextafterf](../libm#sleef_nextafterf). This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
