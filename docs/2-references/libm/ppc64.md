---
layout: default
Title: PowerPC64
parent: Single & Double Precision
grand_parent: References
permalink: /2-references/libm/ppc64
---

<h1>Single & Double Precision Math library reference (PowerPC64)</h1>

<h2>Table of contents</h2>

* [Data types](#datatypes)
* [Trigonometric functions](#trig)
* [Power, exponential, and logarithmic functions](#pow)
* [Inverse trigonometric functions](#invtrig)
* [Hyperbolic functions and inverse hyperbolic functions](#hyp)
* [Error and gamma functions](#eg)
* [Nearest integer functions](#nearint)
* [Other functions](#other)

<h2 id="datatypes">Data types for PowerPC 64 architecture</h2>

### Sleef_vector_float_2

`Sleef_vector_float_2` is a data type for storing two `vector float` values,
which is defined in sleef.h as follows:

```c
typedef struct {
  vector float x, y;
} Sleef_vector_float_2;
```

### Sleef_vector_double_2

`Sleef_vector_double_2` is a data type for storing two `vector double` values,
which is defined in sleef.h as follows:

```c
typedef struct {
  vector double x, y;
} Sleef_vector_double_2;
```

<h2 id="trig">Trigonometric Functions</h2>

### Vectorized double precision sine function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_sind2_u10(vector double a);
vector double Sleef_sind2_u10vsx(vector double a);
vector double Sleef_sind2_u10vsx3(vector double a);
vector double Sleef_cinz_sind2_u10vsxnofma(vector double a);
vector double Sleef_cinz_sind2_u10vsx3nofma(vector double a);
vector double Sleef_finz_sind2_u10vsx(vector double a);
vector double Sleef_finz_sind2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sin_u10](../libm#sleef_sin_u10) with the same accuracy specification.

### Vectorized single precision sine function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_sinf4_u10(vector float a);
vector float Sleef_sinf4_u10vsx(vector float a);
vector float Sleef_sinf4_u10vsx3(vector float a);
vector float Sleef_cinz_sinf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_sinf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_sinf4_u10vsx(vector float a);
vector float Sleef_finz_sinf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinf_u10](../libm#sleef_sinf_u10) with the same accuracy specification.

### Vectorized double precision sine function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_sind2_u35(vector double a);
vector double Sleef_sind2_u35vsx(vector double a);
vector double Sleef_sind2_u35vsx3(vector double a);
vector double Sleef_cinz_sind2_u35vsxnofma(vector double a);
vector double Sleef_cinz_sind2_u35vsx3nofma(vector double a);
vector double Sleef_finz_sind2_u35vsx(vector double a);
vector double Sleef_finz_sind2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sin_u35](../libm#sleef_sin_u35) with the same accuracy specification.

### Vectorized single precision sine function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_sinf4_u35(vector float a);
vector float Sleef_sinf4_u35vsx(vector float a);
vector float Sleef_sinf4_u35vsx3(vector float a);
vector float Sleef_cinz_sinf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_sinf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_sinf4_u35vsx(vector float a);
vector float Sleef_finz_sinf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinf_u35](../libm#sleef_sinf_u35) with the same accuracy specification.

### Vectorized double precision cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_cosd2_u10(vector double a);
vector double Sleef_cosd2_u10vsx(vector double a);
vector double Sleef_cosd2_u10vsx3(vector double a);
vector double Sleef_cinz_cosd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_cosd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_cosd2_u10vsx(vector double a);
vector double Sleef_finz_cosd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cos_u10](../libm#sleef_cos_u10) with the same accuracy specification.

### Vectorized single precision cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_cosf4_u10(vector float a);
vector float Sleef_cosf4_u10vsx(vector float a);
vector float Sleef_cosf4_u10vsx3(vector float a);
vector float Sleef_cinz_cosf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_cosf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_cosf4_u10vsx(vector float a);
vector float Sleef_finz_cosf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosf_u10](../libm#sleef_cosf_u10) with the same accuracy specification.

### Vectorized double precision cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_cosd2_u35(vector double a);
vector double Sleef_cosd2_u35vsx(vector double a);
vector double Sleef_cosd2_u35vsx3(vector double a);
vector double Sleef_cinz_cosd2_u35vsxnofma(vector double a);
vector double Sleef_cinz_cosd2_u35vsx3nofma(vector double a);
vector double Sleef_finz_cosd2_u35vsx(vector double a);
vector double Sleef_finz_cosd2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cos_u35](../libm#sleef_cos_u35) with the same accuracy specification.

### Vectorized single precision cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_cosf4_u35(vector float a);
vector float Sleef_cosf4_u35vsx(vector float a);
vector float Sleef_cosf4_u35vsx3(vector float a);
vector float Sleef_cinz_cosf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_cosf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_cosf4_u35vsx(vector float a);
vector float Sleef_finz_cosf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosf_u35](../libm#sleef_cosf_u35) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

Sleef_vector_double_2 Sleef_sincosd2_u10(vector double a);
Sleef_vector_double_2 Sleef_sincosd2_u10vsx(vector double a);
Sleef_vector_double_2 Sleef_sincosd2_u10vsx3(vector double a);
Sleef_vector_double_2 Sleef_cinz_sincosd2_u10vsxnofma(vector double a);
Sleef_vector_double_2 Sleef_cinz_sincosd2_u10vsx3nofma(vector double a);
Sleef_vector_double_2 Sleef_finz_sincosd2_u10vsx(vector double a);
Sleef_vector_double_2 Sleef_finz_sincosd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincos_u10](../libm#sleef_sincos_u10) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

Sleef_vector_float_2 Sleef_sincosf4_u10(vector float a);
Sleef_vector_float_2 Sleef_sincosf4_u10vsx(vector float a);
Sleef_vector_float_2 Sleef_sincosf4_u10vsx3(vector float a);
Sleef_vector_float_2 Sleef_cinz_sincosf4_u10vsxnofma(vector float a);
Sleef_vector_float_2 Sleef_cinz_sincosf4_u10vsx3nofma(vector float a);
Sleef_vector_float_2 Sleef_finz_sincosf4_u10vsx(vector float a);
Sleef_vector_float_2 Sleef_finz_sincosf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincosf_u10](../libm#sleef_sincosf_u10) with the same accuracy specification.

### Vectorized double precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_vector_double_2 Sleef_sincosd2_u35(vector double a);
Sleef_vector_double_2 Sleef_sincosd2_u35vsx(vector double a);
Sleef_vector_double_2 Sleef_sincosd2_u35vsx3(vector double a);
Sleef_vector_double_2 Sleef_cinz_sincosd2_u35vsxnofma(vector double a);
Sleef_vector_double_2 Sleef_cinz_sincosd2_u35vsx3nofma(vector double a);
Sleef_vector_double_2 Sleef_finz_sincosd2_u35vsx(vector double a);
Sleef_vector_double_2 Sleef_finz_sincosd2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincos_u35](../libm#sleef_sincos_u35) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_vector_float_2 Sleef_sincosf4_u35(vector float a);
Sleef_vector_float_2 Sleef_sincosf4_u35vsx(vector float a);
Sleef_vector_float_2 Sleef_sincosf4_u35vsx3(vector float a);
Sleef_vector_float_2 Sleef_cinz_sincosf4_u35vsxnofma(vector float a);
Sleef_vector_float_2 Sleef_cinz_sincosf4_u35vsx3nofma(vector float a);
Sleef_vector_float_2 Sleef_finz_sincosf4_u35vsx(vector float a);
Sleef_vector_float_2 Sleef_finz_sincosf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincosf_u35](../libm#sleef_sincosf_u35) with the same accuracy specification.

### Vectorized double precision sine function with 0.506 ULP error bound

```c
#include <sleef.h>

vector double Sleef_sinpid2_u05(vector double a);
vector double Sleef_sinpid2_u05vsx(vector double a);
vector double Sleef_sinpid2_u05vsx3(vector double a);
vector double Sleef_cinz_sinpid2_u05vsxnofma(vector double a);
vector double Sleef_cinz_sinpid2_u05vsx3nofma(vector double a);
vector double Sleef_finz_sinpid2_u05vsx(vector double a);
vector double Sleef_finz_sinpid2_u05vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinpi_u05](../libm#sleef_sinpi_u05) with the same accuracy specification.

### Vectorized single precision sine function with 0.506 ULP error bound

```c
#include <sleef.h>

vector float Sleef_sinpif4_u05(vector float a);
vector float Sleef_sinpif4_u05vsx(vector float a);
vector float Sleef_sinpif4_u05vsx3(vector float a);
vector float Sleef_cinz_sinpif4_u05vsxnofma(vector float a);
vector float Sleef_cinz_sinpif4_u05vsx3nofma(vector float a);
vector float Sleef_finz_sinpif4_u05vsx(vector float a);
vector float Sleef_finz_sinpif4_u05vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinpif_u05](../libm#sleef_sinpif_u05) with the same accuracy specification.

### Vectorized double precision cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

vector double Sleef_cospid2_u05(vector double a);
vector double Sleef_cospid2_u05vsx(vector double a);
vector double Sleef_cospid2_u05vsx3(vector double a);
vector double Sleef_cinz_cospid2_u05vsxnofma(vector double a);
vector double Sleef_cinz_cospid2_u05vsx3nofma(vector double a);
vector double Sleef_finz_cospid2_u05vsx(vector double a);
vector double Sleef_finz_cospid2_u05vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cospi_u05](../libm#sleef_cospi_u05) with the same accuracy specification.

### Vectorized single precision cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

vector float Sleef_cospif4_u05(vector float a);
vector float Sleef_cospif4_u05vsx(vector float a);
vector float Sleef_cospif4_u05vsx3(vector float a);
vector float Sleef_cinz_cospif4_u05vsxnofma(vector float a);
vector float Sleef_cinz_cospif4_u05vsx3nofma(vector float a);
vector float Sleef_finz_cospif4_u05vsx(vector float a);
vector float Sleef_finz_cospif4_u05vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cospif_u05](../libm#sleef_cospif_u05) with the same accuracy specification.

### Vectorized double precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

Sleef_vector_double_2 Sleef_sincospid2_u05(vector double a);
Sleef_vector_double_2 Sleef_sincospid2_u05vsx(vector double a);
Sleef_vector_double_2 Sleef_sincospid2_u05vsx3(vector double a);
Sleef_vector_double_2 Sleef_cinz_sincospid2_u05vsxnofma(vector double a);
Sleef_vector_double_2 Sleef_cinz_sincospid2_u05vsx3nofma(vector double a);
Sleef_vector_double_2 Sleef_finz_sincospid2_u05vsx(vector double a);
Sleef_vector_double_2 Sleef_finz_sincospid2_u05vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospi_u05](../libm#sleef_sincospi_u05) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 0.506 ULP error bound

```c
#include <sleef.h>

Sleef_vector_float_2 Sleef_sincospif4_u05(vector float a);
Sleef_vector_float_2 Sleef_sincospif4_u05vsx(vector float a);
Sleef_vector_float_2 Sleef_sincospif4_u05vsx3(vector float a);
Sleef_vector_float_2 Sleef_cinz_sincospif4_u05vsxnofma(vector float a);
Sleef_vector_float_2 Sleef_cinz_sincospif4_u05vsx3nofma(vector float a);
Sleef_vector_float_2 Sleef_finz_sincospif4_u05vsx(vector float a);
Sleef_vector_float_2 Sleef_finz_sincospif4_u05vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospif_u05](../libm#sleef_sincospif_u05) with the same accuracy specification.

### Vectorized double precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_vector_double_2 Sleef_sincospid2_u35(vector double a);
Sleef_vector_double_2 Sleef_sincospid2_u35vsx(vector double a);
Sleef_vector_double_2 Sleef_sincospid2_u35vsx3(vector double a);
Sleef_vector_double_2 Sleef_cinz_sincospid2_u35vsxnofma(vector double a);
Sleef_vector_double_2 Sleef_cinz_sincospid2_u35vsx3nofma(vector double a);
Sleef_vector_double_2 Sleef_finz_sincospid2_u35vsx(vector double a);
Sleef_vector_double_2 Sleef_finz_sincospid2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospi_u35](../libm#sleef_sincospi_u35) with the same accuracy specification.

### Vectorized single precision combined sine and cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

Sleef_vector_float_2 Sleef_sincospif4_u35(vector float a);
Sleef_vector_float_2 Sleef_sincospif4_u35vsx(vector float a);
Sleef_vector_float_2 Sleef_sincospif4_u35vsx3(vector float a);
Sleef_vector_float_2 Sleef_cinz_sincospif4_u35vsxnofma(vector float a);
Sleef_vector_float_2 Sleef_cinz_sincospif4_u35vsx3nofma(vector float a);
Sleef_vector_float_2 Sleef_finz_sincospif4_u35vsx(vector float a);
Sleef_vector_float_2 Sleef_finz_sincospif4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sincospif_u35](../libm#sleef_sincospif_u35) with the same accuracy specification.

### Vectorized double precision tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_tand2_u10(vector double a);
vector double Sleef_tand2_u10vsx(vector double a);
vector double Sleef_tand2_u10vsx3(vector double a);
vector double Sleef_cinz_tand2_u10vsxnofma(vector double a);
vector double Sleef_cinz_tand2_u10vsx3nofma(vector double a);
vector double Sleef_finz_tand2_u10vsx(vector double a);
vector double Sleef_finz_tand2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tan_u10](../libm#sleef_tan_u10) with the same accuracy specification.

### Vectorized single precision tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_tanf4_u10(vector float a);
vector float Sleef_tanf4_u10vsx(vector float a);
vector float Sleef_tanf4_u10vsx3(vector float a);
vector float Sleef_cinz_tanf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_tanf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_tanf4_u10vsx(vector float a);
vector float Sleef_finz_tanf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanf_u10](../libm#sleef_tanf_u10) with the same accuracy specification.

### Vectorized double precision tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_tand2_u35(vector double a);
vector double Sleef_tand2_u35vsx(vector double a);
vector double Sleef_tand2_u35vsx3(vector double a);
vector double Sleef_cinz_tand2_u35vsxnofma(vector double a);
vector double Sleef_cinz_tand2_u35vsx3nofma(vector double a);
vector double Sleef_finz_tand2_u35vsx(vector double a);
vector double Sleef_finz_tand2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tan_u35](../libm#sleef_tan_u35) with the same accuracy specification.

### Vectorized single precision tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_tanf4_u35(vector float a);
vector float Sleef_tanf4_u35vsx(vector float a);
vector float Sleef_tanf4_u35vsx3(vector float a);
vector float Sleef_cinz_tanf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_tanf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_tanf4_u35vsx(vector float a);
vector float Sleef_finz_tanf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanf_u35](../libm#sleef_tanf_u35) with the same accuracy specification.

<h2 id="pow">Power, exponential, and logarithmic function</h2>

### Vectorized double precision power function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_powd2_u10(vector double a, vector double b);
vector double Sleef_powd2_u10vsx(vector double a, vector double b);
vector double Sleef_powd2_u10vsx3(vector double a, vector double b);
vector double Sleef_cinz_powd2_u10vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_powd2_u10vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_powd2_u10vsx(vector double a, vector double b);
vector double Sleef_finz_powd2_u10vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_pow_u10](../libm#sleef_pow_u10) with the same accuracy specification.

### Vectorized single precision power function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_powf4_u10(vector float a, vector float b);
vector float Sleef_powf4_u10vsx(vector float a, vector float b);
vector float Sleef_powf4_u10vsx3(vector float a, vector float b);
vector float Sleef_cinz_powf4_u10vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_powf4_u10vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_powf4_u10vsx(vector float a, vector float b);
vector float Sleef_finz_powf4_u10vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_powf_u10](../libm#sleef_powf_u10) with the same accuracy specification.

### Vectorized double precision natural logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_logd2_u10(vector double a);
vector double Sleef_logd2_u10vsx(vector double a);
vector double Sleef_logd2_u10vsx3(vector double a);
vector double Sleef_cinz_logd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_logd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_logd2_u10vsx(vector double a);
vector double Sleef_finz_logd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log_u10](../libm#sleef_log_u10) with the same accuracy specification.

### Vectorized single precision natural logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_logf4_u10(vector float a);
vector float Sleef_logf4_u10vsx(vector float a);
vector float Sleef_logf4_u10vsx3(vector float a);
vector float Sleef_cinz_logf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_logf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_logf4_u10vsx(vector float a);
vector float Sleef_finz_logf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_logf_u10](../libm#sleef_logf_u10) with the same accuracy specification.

### Vectorized double precision natural logarithmic function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_logd2_u35(vector double a);
vector double Sleef_logd2_u35vsx(vector double a);
vector double Sleef_logd2_u35vsx3(vector double a);
vector double Sleef_cinz_logd2_u35vsxnofma(vector double a);
vector double Sleef_cinz_logd2_u35vsx3nofma(vector double a);
vector double Sleef_finz_logd2_u35vsx(vector double a);
vector double Sleef_finz_logd2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log_u35](../libm#sleef_log_u35) with the same accuracy specification.

### Vectorized single precision natural logarithmic function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_logf4_u35(vector float a);
vector float Sleef_logf4_u35vsx(vector float a);
vector float Sleef_logf4_u35vsx3(vector float a);
vector float Sleef_cinz_logf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_logf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_logf4_u35vsx(vector float a);
vector float Sleef_finz_logf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_logf_u35](../libm#sleef_logf_u35) with the same accuracy specification.

### Vectorized double precision base-10 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_log10d2_u10(vector double a);
vector double Sleef_log10d2_u10vsx(vector double a);
vector double Sleef_log10d2_u10vsx3(vector double a);
vector double Sleef_cinz_log10d2_u10vsxnofma(vector double a);
vector double Sleef_cinz_log10d2_u10vsx3nofma(vector double a);
vector double Sleef_finz_log10d2_u10vsx(vector double a);
vector double Sleef_finz_log10d2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log10_u10](../libm#sleef_log10_u10) with the same accuracy specification.

### Vectorized single precision base-10 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_log10f4_u10(vector float a);
vector float Sleef_log10f4_u10vsx(vector float a);
vector float Sleef_log10f4_u10vsx3(vector float a);
vector float Sleef_cinz_log10f4_u10vsxnofma(vector float a);
vector float Sleef_cinz_log10f4_u10vsx3nofma(vector float a);
vector float Sleef_finz_log10f4_u10vsx(vector float a);
vector float Sleef_finz_log10f4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log10f_u10](../libm#sleef_log10f_u10) with the same accuracy specification.

### Vectorized double precision base-2 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_log2d2_u10(vector double a);
vector double Sleef_log2d2_u10vsx(vector double a);
vector double Sleef_log2d2_u10vsx3(vector double a);
vector double Sleef_cinz_log2d2_u10vsxnofma(vector double a);
vector double Sleef_cinz_log2d2_u10vsx3nofma(vector double a);
vector double Sleef_finz_log2d2_u10vsx(vector double a);
vector double Sleef_finz_log2d2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log2_u10](../libm#sleef_log2_u10) with the same accuracy specification.

### Vectorized single precision base-2 logarithmic function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_log2f4_u10(vector float a);
vector float Sleef_log2f4_u10vsx(vector float a);
vector float Sleef_log2f4_u10vsx3(vector float a);
vector float Sleef_cinz_log2f4_u10vsxnofma(vector float a);
vector float Sleef_cinz_log2f4_u10vsx3nofma(vector float a);
vector float Sleef_finz_log2f4_u10vsx(vector float a);
vector float Sleef_finz_log2f4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log2f_u10](../libm#sleef_log2f_u10) with the same accuracy specification.

### Vectorized double precision logarithm of one plus argument with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_log1pd2_u10(vector double a);
vector double Sleef_log1pd2_u10vsx(vector double a);
vector double Sleef_log1pd2_u10vsx3(vector double a);
vector double Sleef_cinz_log1pd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_log1pd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_log1pd2_u10vsx(vector double a);
vector double Sleef_finz_log1pd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log1p_u10](../libm#sleef_log1p_u10) with the same accuracy specification.

### Vectorized single precision logarithm of one plus argument with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_log1pf4_u10(vector float a);
vector float Sleef_log1pf4_u10vsx(vector float a);
vector float Sleef_log1pf4_u10vsx3(vector float a);
vector float Sleef_cinz_log1pf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_log1pf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_log1pf4_u10vsx(vector float a);
vector float Sleef_finz_log1pf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_log1pf_u10](../libm#sleef_log1pf_u10) with the same accuracy specification.

### Vectorized double precision base-<i>e</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_expd2_u10(vector double a);
vector double Sleef_expd2_u10vsx(vector double a);
vector double Sleef_expd2_u10vsx3(vector double a);
vector double Sleef_cinz_expd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_expd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_expd2_u10vsx(vector double a);
vector double Sleef_finz_expd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp_u10](../libm#sleef_exp_u10) with the same accuracy specification.

### Vectorized single precision base-<i>e</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_expf4_u10(vector float a);
vector float Sleef_expf4_u10vsx(vector float a);
vector float Sleef_expf4_u10vsx3(vector float a);
vector float Sleef_cinz_expf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_expf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_expf4_u10vsx(vector float a);
vector float Sleef_finz_expf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expf_u10](../libm#sleef_expf_u10) with the same accuracy specification.

### Vectorized double precision base-<i>2</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_exp2d2_u10(vector double a);
vector double Sleef_exp2d2_u10vsx(vector double a);
vector double Sleef_exp2d2_u10vsx3(vector double a);
vector double Sleef_cinz_exp2d2_u10vsxnofma(vector double a);
vector double Sleef_cinz_exp2d2_u10vsx3nofma(vector double a);
vector double Sleef_finz_exp2d2_u10vsx(vector double a);
vector double Sleef_finz_exp2d2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp2_u10](../libm#sleef_exp2_u10) with the same accuracy specification.

### Vectorized single precision base-<i>2</i> exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_exp2f4_u10(vector float a);
vector float Sleef_exp2f4_u10vsx(vector float a);
vector float Sleef_exp2f4_u10vsx3(vector float a);
vector float Sleef_cinz_exp2f4_u10vsxnofma(vector float a);
vector float Sleef_cinz_exp2f4_u10vsx3nofma(vector float a);
vector float Sleef_finz_exp2f4_u10vsx(vector float a);
vector float Sleef_finz_exp2f4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp2f_u10](../libm#sleef_exp2f_u10) with the same accuracy specification.

### Vectorized double precision base-10 exponential function function with 1.09 ULP error bound

```c
#include <sleef.h>

vector double Sleef_exp10d2_u10(vector double a);
vector double Sleef_exp10d2_u10vsx(vector double a);
vector double Sleef_exp10d2_u10vsx3(vector double a);
vector double Sleef_cinz_exp10d2_u10vsxnofma(vector double a);
vector double Sleef_cinz_exp10d2_u10vsx3nofma(vector double a);
vector double Sleef_finz_exp10d2_u10vsx(vector double a);
vector double Sleef_finz_exp10d2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp10_u10](../libm#sleef_exp10_u10) with the same accuracy specification.

### Vectorized single precision base-10 exponential function function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_exp10f4_u10(vector float a);
vector float Sleef_exp10f4_u10vsx(vector float a);
vector float Sleef_exp10f4_u10vsx3(vector float a);
vector float Sleef_cinz_exp10f4_u10vsxnofma(vector float a);
vector float Sleef_cinz_exp10f4_u10vsx3nofma(vector float a);
vector float Sleef_finz_exp10f4_u10vsx(vector float a);
vector float Sleef_finz_exp10f4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_exp10f_u10](../libm#sleef_exp10f_u10) with the same accuracy specification.

### Vectorized double precision base-<i>e</i> exponential function minus 1 with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_expm1d2_u10(vector double a);
vector double Sleef_expm1d2_u10vsx(vector double a);
vector double Sleef_expm1d2_u10vsx3(vector double a);
vector double Sleef_cinz_expm1d2_u10vsxnofma(vector double a);
vector double Sleef_cinz_expm1d2_u10vsx3nofma(vector double a);
vector double Sleef_finz_expm1d2_u10vsx(vector double a);
vector double Sleef_finz_expm1d2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expm1_u10](../libm#sleef_expm1_u10) with the same accuracy specification.

### Vectorized single precision base-<i>e</i> exponential function minus 1 with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_expm1f4_u10(vector float a);
vector float Sleef_expm1f4_u10vsx(vector float a);
vector float Sleef_expm1f4_u10vsx3(vector float a);
vector float Sleef_cinz_expm1f4_u10vsxnofma(vector float a);
vector float Sleef_cinz_expm1f4_u10vsx3nofma(vector float a);
vector float Sleef_finz_expm1f4_u10vsx(vector float a);
vector float Sleef_finz_expm1f4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expm1f_u10](../libm#sleef_expm1f_u10) with the same accuracy specification.

### Vectorized double precision square root function with 0.5001 ULP error bound

```c
#include <sleef.h>

vector double Sleef_sqrtd2_u05(vector double a);
vector double Sleef_sqrtd2_u05vsx(vector double a);
vector double Sleef_sqrtd2_u05vsx3(vector double a);
vector double Sleef_cinz_sqrtd2_u05vsxnofma(vector double a);
vector double Sleef_cinz_sqrtd2_u05vsx3nofma(vector double a);
vector double Sleef_finz_sqrtd2_u05vsx(vector double a);
vector double Sleef_finz_sqrtd2_u05vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrt_u05](../libm#sleef_sqrt_u05) with the same accuracy specification.

### Vectorized single precision square root function with 0.5001 ULP error bound

```c
#include <sleef.h>

vector float Sleef_sqrtf4_u05(vector float a);
vector float Sleef_sqrtf4_u05vsx(vector float a);
vector float Sleef_sqrtf4_u05vsx3(vector float a);
vector float Sleef_cinz_sqrtf4_u05vsxnofma(vector float a);
vector float Sleef_cinz_sqrtf4_u05vsx3nofma(vector float a);
vector float Sleef_finz_sqrtf4_u05vsx(vector float a);
vector float Sleef_finz_sqrtf4_u05vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrtf_u05](../libm#sleef_sqrtf_u05) with the same accuracy specification.

### Vectorized double precision square root function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_sqrtd2_u35(vector double a);
vector double Sleef_sqrtd2_u35vsx(vector double a);
vector double Sleef_sqrtd2_u35vsx3(vector double a);
vector double Sleef_cinz_sqrtd2_u35vsxnofma(vector double a);
vector double Sleef_cinz_sqrtd2_u35vsx3nofma(vector double a);
vector double Sleef_finz_sqrtd2_u35vsx(vector double a);
vector double Sleef_finz_sqrtd2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrt_u35](../libm#sleef_sqrt_u35) with the same accuracy specification.

### Vectorized single precision square root function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_sqrtf4_u35(vector float a);
vector float Sleef_sqrtf4_u35vsx(vector float a);
vector float Sleef_sqrtf4_u35vsx3(vector float a);
vector float Sleef_cinz_sqrtf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_sqrtf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_sqrtf4_u35vsx(vector float a);
vector float Sleef_finz_sqrtf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sqrtf_u35](../libm#sleef_sqrtf_u35) with the same accuracy specification.

### Vectorized double precision cubic root function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_cbrtd2_u10(vector double a);
vector double Sleef_cbrtd2_u10vsx(vector double a);
vector double Sleef_cbrtd2_u10vsx3(vector double a);
vector double Sleef_cinz_cbrtd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_cbrtd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_cbrtd2_u10vsx(vector double a);
vector double Sleef_finz_cbrtd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrt_u10](../libm#sleef_cbrt_u10) with the same accuracy specification.

### Vectorized single precision cubic root function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_cbrtf4_u10(vector float a);
vector float Sleef_cbrtf4_u10vsx(vector float a);
vector float Sleef_cbrtf4_u10vsx3(vector float a);
vector float Sleef_cinz_cbrtf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_cbrtf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_cbrtf4_u10vsx(vector float a);
vector float Sleef_finz_cbrtf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrtf_u10](../libm#sleef_cbrtf_u10) with the same accuracy specification.

### Vectorized double precision cubic root function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_cbrtd2_u35(vector double a);
vector double Sleef_cbrtd2_u35vsx(vector double a);
vector double Sleef_cbrtd2_u35vsx3(vector double a);
vector double Sleef_cinz_cbrtd2_u35vsxnofma(vector double a);
vector double Sleef_cinz_cbrtd2_u35vsx3nofma(vector double a);
vector double Sleef_finz_cbrtd2_u35vsx(vector double a);
vector double Sleef_finz_cbrtd2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrt_u35](../libm#sleef_cbrt_u35) with the same accuracy specification.

### Vectorized single precision cubic root function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_cbrtf4_u35(vector float a);
vector float Sleef_cbrtf4_u35vsx(vector float a);
vector float Sleef_cbrtf4_u35vsx3(vector float a);
vector float Sleef_cinz_cbrtf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_cbrtf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_cbrtf4_u35vsx(vector float a);
vector float Sleef_finz_cbrtf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cbrtf_u35](../libm#sleef_cbrtf_u35) with the same accuracy specification.

### Vectorized double precision 2D Euclidian distance function with 0.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_hypotd2_u05(vector double a, vector double b);
vector double Sleef_hypotd2_u05vsx(vector double a, vector double b);
vector double Sleef_hypotd2_u05vsx3(vector double a, vector double b);
vector double Sleef_cinz_hypotd2_u05vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_hypotd2_u05vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_hypotd2_u05vsx(vector double a, vector double b);
vector double Sleef_finz_hypotd2_u05vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypot_u05](../libm#sleef_hypot_u05) with the same accuracy specification.

### Vectorized single precision 2D Euclidian distance function with 0.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_hypotf4_u05(vector float a, vector float b);
vector float Sleef_hypotf4_u05vsx(vector float a, vector float b);
vector float Sleef_hypotf4_u05vsx3(vector float a, vector float b);
vector float Sleef_cinz_hypotf4_u05vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_hypotf4_u05vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_hypotf4_u05vsx(vector float a, vector float b);
vector float Sleef_finz_hypotf4_u05vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypotf_u05](../libm#sleef_hypotf_u05) with the same accuracy specification.

### Vectorized double precision 2D Euclidian distance function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_hypotd2_u35(vector double a, vector double b);
vector double Sleef_hypotd2_u35vsx(vector double a, vector double b);
vector double Sleef_hypotd2_u35vsx3(vector double a, vector double b);
vector double Sleef_cinz_hypotd2_u35vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_hypotd2_u35vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_hypotd2_u35vsx(vector double a, vector double b);
vector double Sleef_finz_hypotd2_u35vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypot_u35](../libm#sleef_hypot_u35) with the same accuracy specification.

### Vectorized single precision 2D Euclidian distance function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_hypotf4_u35(vector float a, vector float b);
vector float Sleef_hypotf4_u35vsx(vector float a, vector float b);
vector float Sleef_hypotf4_u35vsx3(vector float a, vector float b);
vector float Sleef_cinz_hypotf4_u35vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_hypotf4_u35vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_hypotf4_u35vsx(vector float a, vector float b);
vector float Sleef_finz_hypotf4_u35vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_hypotf_u35](../libm#sleef_hypotf_u35) with the same accuracy specification.

<h2 id="invtrig">Inverse Trigonometric Functions</h2>

### Vectorized double precision arc sine function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_asind2_u10(vector double a);
vector double Sleef_asind2_u10vsx(vector double a);
vector double Sleef_asind2_u10vsx3(vector double a);
vector double Sleef_cinz_asind2_u10vsxnofma(vector double a);
vector double Sleef_cinz_asind2_u10vsx3nofma(vector double a);
vector double Sleef_finz_asind2_u10vsx(vector double a);
vector double Sleef_finz_asind2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asin_u10](../libm#sleef_asin_u10) with the same accuracy specification.

### Vectorized single precision arc sine function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_asinf4_u10(vector float a);
vector float Sleef_asinf4_u10vsx(vector float a);
vector float Sleef_asinf4_u10vsx3(vector float a);
vector float Sleef_cinz_asinf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_asinf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_asinf4_u10vsx(vector float a);
vector float Sleef_finz_asinf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinf_u10](../libm#sleef_asinf_u10) with the same accuracy specification.

### Vectorized double precision arc sine function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_asind2_u35(vector double a);
vector double Sleef_asind2_u35vsx(vector double a);
vector double Sleef_asind2_u35vsx3(vector double a);
vector double Sleef_cinz_asind2_u35vsxnofma(vector double a);
vector double Sleef_cinz_asind2_u35vsx3nofma(vector double a);
vector double Sleef_finz_asind2_u35vsx(vector double a);
vector double Sleef_finz_asind2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asin_u35](../libm#sleef_asin_u35) with the same accuracy specification.

### Vectorized single precision arc sine function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_asinf4_u35(vector float a);
vector float Sleef_asinf4_u35vsx(vector float a);
vector float Sleef_asinf4_u35vsx3(vector float a);
vector float Sleef_cinz_asinf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_asinf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_asinf4_u35vsx(vector float a);
vector float Sleef_finz_asinf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinf_u35](../libm#sleef_asinf_u35) with the same accuracy specification.

### Vectorized double precision arc cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_acosd2_u10(vector double a);
vector double Sleef_acosd2_u10vsx(vector double a);
vector double Sleef_acosd2_u10vsx3(vector double a);
vector double Sleef_cinz_acosd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_acosd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_acosd2_u10vsx(vector double a);
vector double Sleef_finz_acosd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acos_u10](../libm#sleef_acos_u10) with the same accuracy specification.

### Vectorized single precision arc cosine function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_acosf4_u10(vector float a);
vector float Sleef_acosf4_u10vsx(vector float a);
vector float Sleef_acosf4_u10vsx3(vector float a);
vector float Sleef_cinz_acosf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_acosf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_acosf4_u10vsx(vector float a);
vector float Sleef_finz_acosf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acosf_u10](../libm#sleef_acosf_u10) with the same accuracy specification.

### Vectorized double precision arc cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_acosd2_u35(vector double a);
vector double Sleef_acosd2_u35vsx(vector double a);
vector double Sleef_acosd2_u35vsx3(vector double a);
vector double Sleef_cinz_acosd2_u35vsxnofma(vector double a);
vector double Sleef_cinz_acosd2_u35vsx3nofma(vector double a);
vector double Sleef_finz_acosd2_u35vsx(vector double a);
vector double Sleef_finz_acosd2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acos_u35](../libm#sleef_acos_u35) with the same accuracy specification.

### Vectorized single precision arc cosine function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_acosf4_u35(vector float a);
vector float Sleef_acosf4_u35vsx(vector float a);
vector float Sleef_acosf4_u35vsx3(vector float a);
vector float Sleef_cinz_acosf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_acosf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_acosf4_u35vsx(vector float a);
vector float Sleef_finz_acosf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acosf_u35](../libm#sleef_acosf_u35) with the same accuracy specification.

### Vectorized double precision arc tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_atand2_u10(vector double a);
vector double Sleef_atand2_u10vsx(vector double a);
vector double Sleef_atand2_u10vsx3(vector double a);
vector double Sleef_cinz_atand2_u10vsxnofma(vector double a);
vector double Sleef_cinz_atand2_u10vsx3nofma(vector double a);
vector double Sleef_finz_atand2_u10vsx(vector double a);
vector double Sleef_finz_atand2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan_u10](../libm#sleef_atan_u10) with the same accuracy specification.

### Vectorized single precision arc tangent function with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_atanf4_u10(vector float a);
vector float Sleef_atanf4_u10vsx(vector float a);
vector float Sleef_atanf4_u10vsx3(vector float a);
vector float Sleef_cinz_atanf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_atanf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_atanf4_u10vsx(vector float a);
vector float Sleef_finz_atanf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanf_u10](../libm#sleef_atanf_u10) with the same accuracy specification.

### Vectorized double precision arc tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_atand2_u35(vector double a);
vector double Sleef_atand2_u35vsx(vector double a);
vector double Sleef_atand2_u35vsx3(vector double a);
vector double Sleef_cinz_atand2_u35vsxnofma(vector double a);
vector double Sleef_cinz_atand2_u35vsx3nofma(vector double a);
vector double Sleef_finz_atand2_u35vsx(vector double a);
vector double Sleef_finz_atand2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan_u35](../libm#sleef_atan_u35) with the same accuracy specification.

### Vectorized single precision arc tangent function with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_atanf4_u35(vector float a);
vector float Sleef_atanf4_u35vsx(vector float a);
vector float Sleef_atanf4_u35vsx3(vector float a);
vector float Sleef_cinz_atanf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_atanf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_atanf4_u35vsx(vector float a);
vector float Sleef_finz_atanf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanf_u35](../libm#sleef_atanf_u35) with the same accuracy specification.

### Vectorized double precision arc tangent function of two variables with 1.0 ULP error bound

```c
#include <sleef.h>

vector double Sleef_atan2d2_u10(vector double a, vector double b);
vector double Sleef_atan2d2_u10vsx(vector double a, vector double b);
vector double Sleef_atan2d2_u10vsx3(vector double a, vector double b);
vector double Sleef_cinz_atan2d2_u10vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_atan2d2_u10vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_atan2d2_u10vsx(vector double a, vector double b);
vector double Sleef_finz_atan2d2_u10vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2_u10](../libm#sleef_atan2_u10) with the same accuracy specification.

### Vectorized single precision arc tangent function of two variables with 1.0 ULP error bound

```c
#include <sleef.h>

vector float Sleef_atan2f4_u10(vector float a, vector float b);
vector float Sleef_atan2f4_u10vsx(vector float a, vector float b);
vector float Sleef_atan2f4_u10vsx3(vector float a, vector float b);
vector float Sleef_cinz_atan2f4_u10vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_atan2f4_u10vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_atan2f4_u10vsx(vector float a, vector float b);
vector float Sleef_finz_atan2f4_u10vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2f_u10](../libm#sleef_atan2f_u10) with the same accuracy specification.

### Vectorized double precision arc tangent function of two variables with 3.5 ULP error bound

```c
#include <sleef.h>

vector double Sleef_atan2d2_u35(vector double a, vector double b);
vector double Sleef_atan2d2_u35vsx(vector double a, vector double b);
vector double Sleef_atan2d2_u35vsx3(vector double a, vector double b);
vector double Sleef_cinz_atan2d2_u35vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_atan2d2_u35vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_atan2d2_u35vsx(vector double a, vector double b);
vector double Sleef_finz_atan2d2_u35vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2_u35](../libm#sleef_atan2_u35) with the same accuracy specification.

### Vectorized single precision arc tangent function of two variables with 3.5 ULP error bound

```c
#include <sleef.h>

vector float Sleef_atan2f4_u35(vector float a, vector float b);
vector float Sleef_atan2f4_u35vsx(vector float a, vector float b);
vector float Sleef_atan2f4_u35vsx3(vector float a, vector float b);
vector float Sleef_cinz_atan2f4_u35vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_atan2f4_u35vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_atan2f4_u35vsx(vector float a, vector float b);
vector float Sleef_finz_atan2f4_u35vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atan2f_u35](../libm#sleef_atan2f_u35) with the same accuracy specification.

<h2 id="hyp">Hyperbolic function and inverse hyperbolic function</h2>

### Vectorized double precision hyperbolic sine function

```c
#include <sleef.h>

vector double Sleef_sinhd2_u10(vector double a);
vector double Sleef_sinhd2_u10vsx(vector double a);
vector double Sleef_sinhd2_u10vsx3(vector double a);
vector double Sleef_cinz_sinhd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_sinhd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_sinhd2_u10vsx(vector double a);
vector double Sleef_finz_sinhd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinh_u10](../libm#sleef_sinh_u10) with the same accuracy specification.

### Vectorized single precision hyperbolic sine function

```c
#include <sleef.h>

vector float Sleef_sinhf4_u10(vector float a);
vector float Sleef_sinhf4_u10vsx(vector float a);
vector float Sleef_sinhf4_u10vsx3(vector float a);
vector float Sleef_cinz_sinhf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_sinhf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_sinhf4_u10vsx(vector float a);
vector float Sleef_finz_sinhf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinhf_u10](../libm#sleef_sinhf_u10) with the same accuracy specification.

### Vectorized double precision hyperbolic sine function

```c
#include <sleef.h>

vector double Sleef_sinhd2_u35(vector double a);
vector double Sleef_sinhd2_u35vsx(vector double a);
vector double Sleef_sinhd2_u35vsx3(vector double a);
vector double Sleef_cinz_sinhd2_u35vsxnofma(vector double a);
vector double Sleef_cinz_sinhd2_u35vsx3nofma(vector double a);
vector double Sleef_finz_sinhd2_u35vsx(vector double a);
vector double Sleef_finz_sinhd2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinh_u35](../libm#sleef_sinh_u35) with the same accuracy specification.

### Vectorized single precision hyperbolic sine function

```c
#include <sleef.h>

vector float Sleef_sinhf4_u35(vector float a);
vector float Sleef_sinhf4_u35vsx(vector float a);
vector float Sleef_sinhf4_u35vsx3(vector float a);
vector float Sleef_cinz_sinhf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_sinhf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_sinhf4_u35vsx(vector float a);
vector float Sleef_finz_sinhf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_sinhf_u35](../libm#sleef_sinhf_u35) with the same accuracy specification.

### Vectorized double precision hyperbolic cosine function

```c
#include <sleef.h>

vector double Sleef_coshd2_u10(vector double a);
vector double Sleef_coshd2_u10vsx(vector double a);
vector double Sleef_coshd2_u10vsx3(vector double a);
vector double Sleef_cinz_coshd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_coshd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_coshd2_u10vsx(vector double a);
vector double Sleef_finz_coshd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosh_u10](../libm#sleef_cosh_u10) with the same accuracy specification.

### Vectorized single precision hyperbolic cosine function

```c
#include <sleef.h>

vector float Sleef_coshf4_u10(vector float a);
vector float Sleef_coshf4_u10vsx(vector float a);
vector float Sleef_coshf4_u10vsx3(vector float a);
vector float Sleef_cinz_coshf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_coshf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_coshf4_u10vsx(vector float a);
vector float Sleef_finz_coshf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_coshf_u10](../libm#sleef_coshf_u10) with the same accuracy specification.

### Vectorized double precision hyperbolic cosine function

```c
#include <sleef.h>

vector double Sleef_coshd2_u35(vector double a);
vector double Sleef_coshd2_u35vsx(vector double a);
vector double Sleef_coshd2_u35vsx3(vector double a);
vector double Sleef_cinz_coshd2_u35vsxnofma(vector double a);
vector double Sleef_cinz_coshd2_u35vsx3nofma(vector double a);
vector double Sleef_finz_coshd2_u35vsx(vector double a);
vector double Sleef_finz_coshd2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_cosh_u35](../libm#sleef_cosh_u35) with the same accuracy specification.

### Vectorized single precision hyperbolic cosine function

```c
#include <sleef.h>

vector float Sleef_coshf4_u35(vector float a);
vector float Sleef_coshf4_u35vsx(vector float a);
vector float Sleef_coshf4_u35vsx3(vector float a);
vector float Sleef_cinz_coshf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_coshf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_coshf4_u35vsx(vector float a);
vector float Sleef_finz_coshf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_coshf_u35](../libm#sleef_coshf_u35) with the same accuracy specification.

### Vectorized double precision hyperbolic tangent function

```c
#include <sleef.h>

vector double Sleef_tanhd2_u10(vector double a);
vector double Sleef_tanhd2_u10vsx(vector double a);
vector double Sleef_tanhd2_u10vsx3(vector double a);
vector double Sleef_cinz_tanhd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_tanhd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_tanhd2_u10vsx(vector double a);
vector double Sleef_finz_tanhd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanh_u10](../libm#sleef_tanh_u10) with the same accuracy specification.

### Vectorized single precision hyperbolic tangent function

```c
#include <sleef.h>

vector float Sleef_tanhf4_u10(vector float a);
vector float Sleef_tanhf4_u10vsx(vector float a);
vector float Sleef_tanhf4_u10vsx3(vector float a);
vector float Sleef_cinz_tanhf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_tanhf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_tanhf4_u10vsx(vector float a);
vector float Sleef_finz_tanhf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanhf_u10](../libm#sleef_tanhf_u10) with the same accuracy specification.

### Vectorized double precision hyperbolic tangent function

```c
#include <sleef.h>

vector double Sleef_tanhd2_u35(vector double a);
vector double Sleef_tanhd2_u35vsx(vector double a);
vector double Sleef_tanhd2_u35vsx3(vector double a);
vector double Sleef_cinz_tanhd2_u35vsxnofma(vector double a);
vector double Sleef_cinz_tanhd2_u35vsx3nofma(vector double a);
vector double Sleef_finz_tanhd2_u35vsx(vector double a);
vector double Sleef_finz_tanhd2_u35vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanh_u35](../libm#sleef_tanh_u35) with the same accuracy specification.

### Vectorized single precision hyperbolic tangent function

```c
#include <sleef.h>

vector float Sleef_tanhf4_u35(vector float a);
vector float Sleef_tanhf4_u35vsx(vector float a);
vector float Sleef_tanhf4_u35vsx3(vector float a);
vector float Sleef_cinz_tanhf4_u35vsxnofma(vector float a);
vector float Sleef_cinz_tanhf4_u35vsx3nofma(vector float a);
vector float Sleef_finz_tanhf4_u35vsx(vector float a);
vector float Sleef_finz_tanhf4_u35vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tanhf_u35](../libm#sleef_tanhf_u35) with the same accuracy specification.

### Vectorized double precision inverse hyperbolic sine function

```c
#include <sleef.h>

vector double Sleef_asinhd2_u10(vector double a);
vector double Sleef_asinhd2_u10vsx(vector double a);
vector double Sleef_asinhd2_u10vsx3(vector double a);
vector double Sleef_cinz_asinhd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_asinhd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_asinhd2_u10vsx(vector double a);
vector double Sleef_finz_asinhd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinh_u10](../libm#sleef_asinh_u10) with the same accuracy specification.

### Vectorized single precision inverse hyperbolic sine function

```c
#include <sleef.h>

vector float Sleef_asinhf4_u10(vector float a);
vector float Sleef_asinhf4_u10vsx(vector float a);
vector float Sleef_asinhf4_u10vsx3(vector float a);
vector float Sleef_cinz_asinhf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_asinhf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_asinhf4_u10vsx(vector float a);
vector float Sleef_finz_asinhf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_asinhf_u10](../libm#sleef_asinhf_u10) with the same accuracy specification.

### Vectorized double precision inverse hyperbolic cosine function

```c
#include <sleef.h>

vector double Sleef_acoshd2_u10(vector double a);
vector double Sleef_acoshd2_u10vsx(vector double a);
vector double Sleef_acoshd2_u10vsx3(vector double a);
vector double Sleef_cinz_acoshd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_acoshd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_acoshd2_u10vsx(vector double a);
vector double Sleef_finz_acoshd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acosh_u10](../libm#sleef_acosh_u10) with the same accuracy specification.

### Vectorized single precision inverse hyperbolic cosine function

```c
#include <sleef.h>

vector float Sleef_acoshf4_u10(vector float a);
vector float Sleef_acoshf4_u10vsx(vector float a);
vector float Sleef_acoshf4_u10vsx3(vector float a);
vector float Sleef_cinz_acoshf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_acoshf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_acoshf4_u10vsx(vector float a);
vector float Sleef_finz_acoshf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_acoshf_u10](../libm#sleef_acoshf_u10) with the same accuracy specification.

### Vectorized double precision inverse hyperbolic tangent function

```c
#include <sleef.h>

vector double Sleef_atanhd2_u10(vector double a);
vector double Sleef_atanhd2_u10vsx(vector double a);
vector double Sleef_atanhd2_u10vsx3(vector double a);
vector double Sleef_cinz_atanhd2_u10vsxnofma(vector double a);
vector double Sleef_cinz_atanhd2_u10vsx3nofma(vector double a);
vector double Sleef_finz_atanhd2_u10vsx(vector double a);
vector double Sleef_finz_atanhd2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanh_u10](../libm#sleef_atanh_u10) with the same accuracy specification.

### Vectorized single precision inverse hyperbolic tangent function

```c
#include <sleef.h>

vector float Sleef_atanhf4_u10(vector float a);
vector float Sleef_atanhf4_u10vsx(vector float a);
vector float Sleef_atanhf4_u10vsx3(vector float a);
vector float Sleef_cinz_atanhf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_atanhf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_atanhf4_u10vsx(vector float a);
vector float Sleef_finz_atanhf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_atanhf_u10](../libm#sleef_atanhf_u10) with the same accuracy specification.

<h2 id="eg">Error and gamma function</h2>

### Vectorized double precision error function

```c
#include <sleef.h>

__vector double Sleef_erfd2_u10(__vector double a);
__vector double Sleef_erfd2_u10vsx(__vector double a);
__vector double Sleef_erfd2_u10vsx3(__vector double a);
__vector double Sleef_cinz_erfd2_u10vsxnofma(__vector double a);
__vector double Sleef_cinz_erfd2_u10vsx3nofma(__vector double a);
__vector double Sleef_finz_erfd2_u10vsx(__vector double a);
__vector double Sleef_finz_erfd2_u10vsx3(__vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erf_u10](../libm#sleef_erf_u10) with the same accuracy specification.

### Vectorized single precision error function

```c
#include <sleef.h>

vector float Sleef_erff4_u10(vector float a);
vector float Sleef_erff4_u10vsx(vector float a);
vector float Sleef_erff4_u10vsx3(vector float a);
vector float Sleef_cinz_erff4_u10vsxnofma(vector float a);
vector float Sleef_cinz_erff4_u10vsx3nofma(vector float a);
vector float Sleef_finz_erff4_u10vsx(vector float a);
vector float Sleef_finz_erff4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erff_u10](../libm#sleef_erff_u10) with the same accuracy specification.

### Vectorized double precision complementary error function

```c
#include <sleef.h>

vector double Sleef_erfcd2_u15(vector double a);
vector double Sleef_erfcd2_u15vsx(vector double a);
vector double Sleef_erfcd2_u15vsx3(vector double a);
vector double Sleef_cinz_erfcd2_u15vsxnofma(vector double a);
vector double Sleef_cinz_erfcd2_u15vsx3nofma(vector double a);
vector double Sleef_finz_erfcd2_u15vsx(vector double a);
vector double Sleef_finz_erfcd2_u15vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erfc_u15](../libm#sleef_erfc_u15) with the same accuracy specification.

### Vectorized single precision complementary error function

```c
#include <sleef.h>

vector float Sleef_erfcf4_u15(vector float a);
vector float Sleef_erfcf4_u15vsx(vector float a);
vector float Sleef_erfcf4_u15vsx3(vector float a);
vector float Sleef_cinz_erfcf4_u15vsxnofma(vector float a);
vector float Sleef_cinz_erfcf4_u15vsx3nofma(vector float a);
vector float Sleef_finz_erfcf4_u15vsx(vector float a);
vector float Sleef_finz_erfcf4_u15vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_erfcf_u15](../libm#sleef_erfcf_u15) with the same accuracy specification.

### Vectorized double precision gamma function

```c
#include <sleef.h>

vector double Sleef_tgammad2_u10(vector double a);
vector double Sleef_tgammad2_u10vsx(vector double a);
vector double Sleef_tgammad2_u10vsx3(vector double a);
vector double Sleef_cinz_tgammad2_u10vsxnofma(vector double a);
vector double Sleef_cinz_tgammad2_u10vsx3nofma(vector double a);
vector double Sleef_finz_tgammad2_u10vsx(vector double a);
vector double Sleef_finz_tgammad2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tgamma_u10](../libm#sleef_tgamma_u10) with the same accuracy specification.

### Vectorized single precision gamma function

```c
#include <sleef.h>

vector float Sleef_tgammaf4_u10(vector float a);
vector float Sleef_tgammaf4_u10vsx(vector float a);
vector float Sleef_tgammaf4_u10vsx3(vector float a);
vector float Sleef_cinz_tgammaf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_tgammaf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_tgammaf4_u10vsx(vector float a);
vector float Sleef_finz_tgammaf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_tgammaf_u10](../libm#sleef_tgammaf_u10) with the same accuracy specification.

### Vectorized double precision log gamma function

```c
#include <sleef.h>

vector double Sleef_lgammad2_u10(vector double a);
vector double Sleef_lgammad2_u10vsx(vector double a);
vector double Sleef_lgammad2_u10vsx3(vector double a);
vector double Sleef_cinz_lgammad2_u10vsxnofma(vector double a);
vector double Sleef_cinz_lgammad2_u10vsx3nofma(vector double a);
vector double Sleef_finz_lgammad2_u10vsx(vector double a);
vector double Sleef_finz_lgammad2_u10vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_lgamma_u10](../libm#sleef_lgamma_u10) with the same accuracy specification.

### Vectorized single precision log gamma function

```c
#include <sleef.h>

vector float Sleef_lgammaf4_u10(vector float a);
vector float Sleef_lgammaf4_u10vsx(vector float a);
vector float Sleef_lgammaf4_u10vsx3(vector float a);
vector float Sleef_cinz_lgammaf4_u10vsxnofma(vector float a);
vector float Sleef_cinz_lgammaf4_u10vsx3nofma(vector float a);
vector float Sleef_finz_lgammaf4_u10vsx(vector float a);
vector float Sleef_finz_lgammaf4_u10vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_lgammaf_u10](../libm#sleef_lgammaf_u10) with the same accuracy specification.

<h2 id="nearint">Nearest integer function</h2>

### Vectorized double precision function for rounding to integer towards zero

```c
#include <sleef.h>

vector double Sleef_truncd2(vector double a);
vector double Sleef_truncd2_vsx(vector double a);
vector double Sleef_truncd2_vsx3(vector double a);
vector double Sleef_cinz_truncd2_vsxnofma(vector double a);
vector double Sleef_cinz_truncd2_vsx3nofma(vector double a);
vector double Sleef_finz_truncd2_vsx(vector double a);
vector double Sleef_finz_truncd2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_trunc](../libm#sleef_trunc) with the same accuracy specification.

### Vectorized single precision function for rounding to integer towards zero

```c
#include <sleef.h>

vector float Sleef_truncf4(vector float a);
vector float Sleef_truncf4_vsx(vector float a);
vector float Sleef_truncf4_vsx3(vector float a);
vector float Sleef_cinz_truncf4_vsxnofma(vector float a);
vector float Sleef_cinz_truncf4_vsx3nofma(vector float a);
vector float Sleef_finz_truncf4_vsx(vector float a);
vector float Sleef_finz_truncf4_vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_truncf](../libm#sleef_truncf) with the same accuracy specification.

### Vectorized double precision function for rounding to integer towards negative infinity

```c
#include <sleef.h>

vector double Sleef_floord2(vector double a);
vector double Sleef_floord2_vsx(vector double a);
vector double Sleef_floord2_vsx3(vector double a);
vector double Sleef_cinz_floord2_vsxnofma(vector double a);
vector double Sleef_cinz_floord2_vsx3nofma(vector double a);
vector double Sleef_finz_floord2_vsx(vector double a);
vector double Sleef_finz_floord2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_floor](../libm#sleef_floor) with the same accuracy specification.

### Vectorized single precision function for rounding to integer towards negative infinity

```c
#include <sleef.h>

vector float Sleef_floorf4(vector float a);
vector float Sleef_floorf4_vsx(vector float a);
vector float Sleef_floorf4_vsx3(vector float a);
vector float Sleef_cinz_floorf4_vsxnofma(vector float a);
vector float Sleef_cinz_floorf4_vsx3nofma(vector float a);
vector float Sleef_finz_floorf4_vsx(vector float a);
vector float Sleef_finz_floorf4_vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_floorf](../libm#sleef_floorf) with the same accuracy specification.

### Vectorized double precision function for rounding to integer towards positive infinity

```c
#include <sleef.h>

vector double Sleef_ceild2(vector double a);
vector double Sleef_ceild2_vsx(vector double a);
vector double Sleef_ceild2_vsx3(vector double a);
vector double Sleef_cinz_ceild2_vsxnofma(vector double a);
vector double Sleef_cinz_ceild2_vsx3nofma(vector double a);
vector double Sleef_finz_ceild2_vsx(vector double a);
vector double Sleef_finz_ceild2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_ceil](../libm#sleef_ceil) with the same accuracy specification.

### Vectorized single precision function for rounding to integer towards positive infinity

```c
#include <sleef.h>

vector float Sleef_ceilf4(vector float a);
vector float Sleef_ceilf4_vsx(vector float a);
vector float Sleef_ceilf4_vsx3(vector float a);
vector float Sleef_cinz_ceilf4_vsxnofma(vector float a);
vector float Sleef_cinz_ceilf4_vsx3nofma(vector float a);
vector float Sleef_finz_ceilf4_vsx(vector float a);
vector float Sleef_finz_ceilf4_vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_ceilf](../libm#sleef_ceilf) with the same accuracy specification.

### Vectorized double precision function for rounding to nearest integer

```c
#include <sleef.h>

vector double Sleef_roundd2(vector double a);
vector double Sleef_roundd2_vsx(vector double a);
vector double Sleef_roundd2_vsx3(vector double a);
vector double Sleef_cinz_roundd2_vsxnofma(vector double a);
vector double Sleef_cinz_roundd2_vsx3nofma(vector double a);
vector double Sleef_finz_roundd2_vsx(vector double a);
vector double Sleef_finz_roundd2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_round](../libm#sleef_round) with the same accuracy specification.

### Vectorized single precision function for rounding to nearest integer

```c
#include <sleef.h>

vector float Sleef_roundf4(vector float a);
vector float Sleef_roundf4_vsx(vector float a);
vector float Sleef_roundf4_vsx3(vector float a);
vector float Sleef_cinz_roundf4_vsxnofma(vector float a);
vector float Sleef_cinz_roundf4_vsx3nofma(vector float a);
vector float Sleef_finz_roundf4_vsx(vector float a);
vector float Sleef_finz_roundf4_vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_roundf](../libm#sleef_roundf) with the same accuracy specification.

### Vectorized double precision function for rounding to nearest integer

```c
#include <sleef.h>

vector double Sleef_rintd2(vector double a);
vector double Sleef_rintd2_vsx(vector double a);
vector double Sleef_rintd2_vsx3(vector double a);
vector double Sleef_cinz_rintd2_vsxnofma(vector double a);
vector double Sleef_cinz_rintd2_vsx3nofma(vector double a);
vector double Sleef_finz_rintd2_vsx(vector double a);
vector double Sleef_finz_rintd2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_rint](../libm#sleef_rint) with the same accuracy specification.

### Vectorized single precision function for rounding to nearest integer

```c
#include <sleef.h>

vector float Sleef_rintf4(vector float a);
vector float Sleef_rintf4_vsx(vector float a);
vector float Sleef_rintf4_vsx3(vector float a);
vector float Sleef_cinz_rintf4_vsxnofma(vector float a);
vector float Sleef_cinz_rintf4_vsx3nofma(vector float a);
vector float Sleef_finz_rintf4_vsx(vector float a);
vector float Sleef_finz_rintf4_vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_rintf](../libm#sleef_rintf) with the same accuracy specification.

<h2 id="other">Other function</h2>

### Vectorized double precision function for fused multiply-accumulation

```c
#include <sleef.h>

vector double Sleef_fmad2(vector double a, vector double b, vector double c);
vector double Sleef_fmad2_vsx(vector double a, vector double b, vector double c);
vector double Sleef_fmad2_vsx3(vector double a, vector double b, vector double c);
vector double Sleef_cinz_fmad2_vsxnofma(vector double a, vector double b, vector double c);
vector double Sleef_cinz_fmad2_vsx3nofma(vector double a, vector double b, vector double c);
vector double Sleef_finz_fmad2_vsx(vector double a, vector double b, vector double c);
vector double Sleef_finz_fmad2_vsx3(vector double a, vector double b, vector double c);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fma](../libm#sleef_fma) with the same accuracy specification.

### Vectorized single precision function for fused multiply-accumulation

```c
#include <sleef.h>

vector float Sleef_fmaf4(vector float a, vector float b, vector float c);
vector float Sleef_fmaf4_vsx(vector float a, vector float b, vector float c);
vector float Sleef_fmaf4_vsx3(vector float a, vector float b, vector float c);
vector float Sleef_cinz_fmaf4_vsxnofma(vector float a, vector float b, vector float c);
vector float Sleef_cinz_fmaf4_vsx3nofma(vector float a, vector float b, vector float c);
vector float Sleef_finz_fmaf4_vsx(vector float a, vector float b, vector float c);
vector float Sleef_finz_fmaf4_vsx3(vector float a, vector float b, vector float c);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmaf](../libm#sleef_fmaf) with the same accuracy specification.

### Vectorized double precision FP remainder

```c
#include <sleef.h>

vector double Sleef_fmodd2(vector double a, vector double b);
vector double Sleef_fmodd2_vsx(vector double a, vector double b);
vector double Sleef_fmodd2_vsx3(vector double a, vector double b);
vector double Sleef_cinz_fmodd2_vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_fmodd2_vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_fmodd2_vsx(vector double a, vector double b);
vector double Sleef_finz_fmodd2_vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmod](../libm#sleef_fmod) with the same accuracy specification.

### Vectorized single precision FP remainder

```c
#include <sleef.h>

vector float Sleef_fmodf4(vector float a, vector float b);
vector float Sleef_fmodf4_vsx(vector float a, vector float b);
vector float Sleef_fmodf4_vsx3(vector float a, vector float b);
vector float Sleef_cinz_fmodf4_vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_fmodf4_vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_fmodf4_vsx(vector float a, vector float b);
vector float Sleef_finz_fmodf4_vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmodf](../libm#sleef_fmodf) with the same accuracy specification.

### Vectorized double precision FP remainder

```c
#include <sleef.h>

vector double Sleef_remainderd2(vector double a, vector double b);
vector double Sleef_remainderd2_vsx(vector double a, vector double b);
vector double Sleef_remainderd2_vsx3(vector double a, vector double b);
vector double Sleef_cinz_remainderd2_vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_remainderd2_vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_remainderd2_vsx(vector double a, vector double b);
vector double Sleef_finz_remainderd2_vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_remainder](../libm#sleef_remainder) with the same accuracy specification.

### Vectorized single precision FP remainder

```c
#include <sleef.h>

vector float Sleef_remainderf4(vector float a, vector float b);
vector float Sleef_remainderf4_vsx(vector float a, vector float b);
vector float Sleef_remainderf4_vsx3(vector float a, vector float b);
vector float Sleef_cinz_remainderf4_vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_remainderf4_vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_remainderf4_vsx(vector float a, vector float b);
vector float Sleef_finz_remainderf4_vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_remainderf](../libm#sleef_remainderf) with the same accuracy specification.

### Vectorized double precision function for multiplying by integral power of 2

```c
#include <sleef.h>

vector double Sleef_ldexpd2(vector double a, vector int b);
vector double Sleef_ldexpd2_vsx(vector double a, vector int b);
vector double Sleef_ldexpd2_vsx3(vector double a, vector int b);
vector double Sleef_cinz_ldexpd2_vsxnofma(vector double a, vector int b);
vector double Sleef_cinz_ldexpd2_vsx3nofma(vector double a, vector int b);
vector double Sleef_finz_ldexpd2_vsx(vector double a, vector int b);
vector double Sleef_finz_ldexpd2_vsx3(vector double a, vector int b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_ldexp](../libm#sleef_ldexp) with the same accuracy specification.

### Vectorized double precision function for obtaining fractional component of an FP number

```c
#include <sleef.h>

vector double Sleef_frfrexpd2(vector double a);
vector double Sleef_frfrexpd2_vsx(vector double a);
vector double Sleef_frfrexpd2_vsx3(vector double a);
vector double Sleef_cinz_frfrexpd2_vsxnofma(vector double a);
vector double Sleef_cinz_frfrexpd2_vsx3nofma(vector double a);
vector double Sleef_finz_frfrexpd2_vsx(vector double a);
vector double Sleef_finz_frfrexpd2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_frfrexp](../libm#sleef_frfrexp) with the same accuracy specification.

### Vectorized single precision function for obtaining fractional component of an FP number

```c
#include <sleef.h>

vector float Sleef_frfrexpf4(vector float a);
vector float Sleef_frfrexpf4_vsx(vector float a);
vector float Sleef_frfrexpf4_vsx3(vector float a);
vector float Sleef_cinz_frfrexpf4_vsxnofma(vector float a);
vector float Sleef_cinz_frfrexpf4_vsx3nofma(vector float a);
vector float Sleef_finz_frfrexpf4_vsx(vector float a);
vector float Sleef_finz_frfrexpf4_vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_frfrexpf](../libm#sleef_frfrexpf) with the same accuracy specification.

### Vectorized double precision function for obtaining integral component of an FP number

```c
#include <sleef.h>

vector int Sleef_expfrexpd2(vector double a);
vector int Sleef_expfrexpd2_vsx(vector double a);
vector int Sleef_expfrexpd2_vsx3(vector double a);
vector int Sleef_cinz_expfrexpd2_vsxnofma(vector double a);
vector int Sleef_cinz_expfrexpd2_vsx3nofma(vector double a);
vector int Sleef_finz_expfrexpd2_vsx(vector double a);
vector int Sleef_finz_expfrexpd2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_expfrexp](../libm#sleef_expfrexp) with the same accuracy specification.

### Vectorized double precision function for getting integer exponent

```c
#include <sleef.h>

vector int Sleef_ilogbd2(vector double a);
vector int Sleef_ilogbd2_vsx(vector double a);
vector int Sleef_ilogbd2_vsx3(vector double a);
vector int Sleef_cinz_ilogbd2_vsxnofma(vector double a);
vector int Sleef_cinz_ilogbd2_vsx3nofma(vector double a);
vector int Sleef_finz_ilogbd2_vsx(vector double a);
vector int Sleef_finz_ilogbd2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_ilogb](../libm#sleef_ilogb) with the same accuracy specification.

### Vectorized double precision signed integral and fractional values

```c
#include <sleef.h>

Sleef_vector_double_2 Sleef_modfd2(vector double a);
Sleef_vector_double_2 Sleef_modfd2_vsx(vector double a);
Sleef_vector_double_2 Sleef_modfd2_vsx3(vector double a);
Sleef_vector_double_2 Sleef_cinz_modfd2_vsxnofma(vector double a);
Sleef_vector_double_2 Sleef_cinz_modfd2_vsx3nofma(vector double a);
Sleef_vector_double_2 Sleef_finz_modfd2_vsx(vector double a);
Sleef_vector_double_2 Sleef_finz_modfd2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_modf](../libm#sleef_modf) with the same accuracy specification.

### Vectorized single precision signed integral and fractional values

```c
#include <sleef.h>

Sleef_vector_float_2 Sleef_modff4(vector float a);
Sleef_vector_float_2 Sleef_modff4_vsx(vector float a);
Sleef_vector_float_2 Sleef_modff4_vsx3(vector float a);
Sleef_vector_float_2 Sleef_cinz_modff4_vsxnofma(vector float a);
Sleef_vector_float_2 Sleef_cinz_modff4_vsx3nofma(vector float a);
Sleef_vector_float_2 Sleef_finz_modff4_vsx(vector float a);
Sleef_vector_float_2 Sleef_finz_modff4_vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_modff](../libm#sleef_modff) with the same accuracy specification.

### Vectorized double precision function for calculating the absolute value

```c
#include <sleef.h>

vector double Sleef_fabsd2(vector double a);
vector double Sleef_fabsd2_vsx(vector double a);
vector double Sleef_fabsd2_vsx3(vector double a);
vector double Sleef_cinz_fabsd2_vsxnofma(vector double a);
vector double Sleef_cinz_fabsd2_vsx3nofma(vector double a);
vector double Sleef_finz_fabsd2_vsx(vector double a);
vector double Sleef_finz_fabsd2_vsx3(vector double a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fabs](../libm#sleef_fabs) with the same accuracy specification.

### Vectorized single precision function for calculating the absolute value

```c
#include <sleef.h>

vector float Sleef_fabsf4(vector float a);
vector float Sleef_fabsf4_vsx(vector float a);
vector float Sleef_fabsf4_vsx3(vector float a);
vector float Sleef_cinz_fabsf4_vsxnofma(vector float a);
vector float Sleef_cinz_fabsf4_vsx3nofma(vector float a);
vector float Sleef_finz_fabsf4_vsx(vector float a);
vector float Sleef_finz_fabsf4_vsx3(vector float a);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fabsf](../libm#sleef_fabsf) with the same accuracy specification.

### Vectorized double precision function for copying signs

```c
#include <sleef.h>

vector double Sleef_copysignd2(vector double a, vector double b);
vector double Sleef_copysignd2_vsx(vector double a, vector double b);
vector double Sleef_copysignd2_vsx3(vector double a, vector double b);
vector double Sleef_cinz_copysignd2_vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_copysignd2_vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_copysignd2_vsx(vector double a, vector double b);
vector double Sleef_finz_copysignd2_vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_copysign](../libm#sleef_copysign) with the same accuracy specification.

### Vectorized single precision function for copying signs

```c
#include <sleef.h>

vector float Sleef_copysignf4(vector float a, vector float b);
vector float Sleef_copysignf4_vsx(vector float a, vector float b);
vector float Sleef_copysignf4_vsx3(vector float a, vector float b);
vector float Sleef_cinz_copysignf4_vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_copysignf4_vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_copysignf4_vsx(vector float a, vector float b);
vector float Sleef_finz_copysignf4_vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_copysignf](../libm#sleef_copysignf) with the same accuracy specification.

### Vectorized double precision function for determining maximum of two values

```c
#include <sleef.h>

vector double Sleef_fmaxd2(vector double a, vector double b);
vector double Sleef_fmaxd2_vsx(vector double a, vector double b);
vector double Sleef_fmaxd2_vsx3(vector double a, vector double b);
vector double Sleef_cinz_fmaxd2_vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_fmaxd2_vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_fmaxd2_vsx(vector double a, vector double b);
vector double Sleef_finz_fmaxd2_vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmax](../libm#sleef_fmax) with the same accuracy specification.

### Vectorized single precision function for determining maximum of two values

```c
#include <sleef.h>

vector float Sleef_fmaxf4(vector float a, vector float b);
vector float Sleef_fmaxf4_vsx(vector float a, vector float b);
vector float Sleef_fmaxf4_vsx3(vector float a, vector float b);
vector float Sleef_cinz_fmaxf4_vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_fmaxf4_vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_fmaxf4_vsx(vector float a, vector float b);
vector float Sleef_finz_fmaxf4_vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmaxf](../libm#sleef_fmaxf) with the same accuracy specification.

### Vectorized double precision function for determining minimum of two values

```c
#include <sleef.h>

vector double Sleef_fmind2(vector double a, vector double b);
vector double Sleef_fmind2_vsx(vector double a, vector double b);
vector double Sleef_fmind2_vsx3(vector double a, vector double b);
vector double Sleef_cinz_fmind2_vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_fmind2_vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_fmind2_vsx(vector double a, vector double b);
vector double Sleef_finz_fmind2_vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fmin](../libm#sleef_fmin) with the same accuracy specification.

### Vectorized single precision function for determining minimum of two values

```c
#include <sleef.h>

vector float Sleef_fminf4(vector float a, vector float b);
vector float Sleef_fminf4_vsx(vector float a, vector float b);
vector float Sleef_fminf4_vsx3(vector float a, vector float b);
vector float Sleef_cinz_fminf4_vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_fminf4_vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_fminf4_vsx(vector float a, vector float b);
vector float Sleef_finz_fminf4_vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fminf](../libm#sleef_fminf) with the same accuracy specification.

### Vectorized double precision function to calculate positive difference of two values

```c
#include <sleef.h>

vector double Sleef_fdimd2(vector double a, vector double b);
vector double Sleef_fdimd2_vsx(vector double a, vector double b);
vector double Sleef_fdimd2_vsx3(vector double a, vector double b);
vector double Sleef_cinz_fdimd2_vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_fdimd2_vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_fdimd2_vsx(vector double a, vector double b);
vector double Sleef_finz_fdimd2_vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fdim](../libm#sleef_fdim) with the same accuracy specification.

### Vectorized single precision function to calculate positive difference of two values

```c
#include <sleef.h>

vector float Sleef_fdimf4(vector float a, vector float b);
vector float Sleef_fdimf4_vsx(vector float a, vector float b);
vector float Sleef_fdimf4_vsx3(vector float a, vector float b);
vector float Sleef_cinz_fdimf4_vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_fdimf4_vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_fdimf4_vsx(vector float a, vector float b);
vector float Sleef_finz_fdimf4_vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_fdimf](../libm#sleef_fdimf) with the same accuracy specification.

### Vectorized double precision function for obtaining the next representable FP value

```c
#include <sleef.h>

vector double Sleef_nextafterd2(vector double a, vector double b);
vector double Sleef_nextafterd2_vsx(vector double a, vector double b);
vector double Sleef_nextafterd2_vsx3(vector double a, vector double b);
vector double Sleef_cinz_nextafterd2_vsxnofma(vector double a, vector double b);
vector double Sleef_cinz_nextafterd2_vsx3nofma(vector double a, vector double b);
vector double Sleef_finz_nextafterd2_vsx(vector double a, vector double b);
vector double Sleef_finz_nextafterd2_vsx3(vector double a, vector double b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_nextafter](../libm#sleef_nextafter) with the same accuracy specification.

### Vectorized single precision function for obtaining the next representable FP value

```c
#include <sleef.h>

vector float Sleef_nextafterf4(vector float a, vector float b);
vector float Sleef_nextafterf4_vsx(vector float a, vector float b);
vector float Sleef_nextafterf4_vsx3(vector float a, vector float b);
vector float Sleef_cinz_nextafterf4_vsxnofma(vector float a, vector float b);
vector float Sleef_cinz_nextafterf4_vsx3nofma(vector float a, vector float b);
vector float Sleef_finz_nextafterf4_vsx(vector float a, vector float b);
vector float Sleef_finz_nextafterf4_vsx3(vector float a, vector float b);
```
Link with `-lsleef`.

These are the vectorized functions of [Sleef_nextafterf](../libm#sleef_nextafterf) with the same accuracy specification.
