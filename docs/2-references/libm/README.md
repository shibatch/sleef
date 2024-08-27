---
layout: default
title: Single & Double Precision
parent: References
has_children: true
permalink: /2-references/libm/
---

<h1>Single & Double Precision Math library reference</h1>

<h2>Table of contents</h2>

* [Function naming convention](#naming)
* [Data types](#datatypes)
* [Trigonometric functions](#trig)
* [Power, exponential, and logarithmic functions](#pow)
* [Inverse trigonometric functions](#invtrig)
* [Hyperbolic functions and inverse hyperbolic functions](#hyp)
* [Error and gamma functions](#eg)
* [Nearest integer functions](#nearint)
* [Other functions](#other)

<h2 id="naming">Function naming convention</h2>

The functions whose names end with `purecfma` and `purec` are implemented with
and without using
[FMA](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation#Fused_multiply%E2%80%93add)
instructions, respectively. Functions with FMA instructions are only available
on architectures with FMA instructions. The two digits number after `_u` is 10
times the maximum error for typical input domain in [ULP](../../3-extra#ulp).
The functions whose names contain `finz` and `cinz` are functions that give
bit-wise consistent results across all platforms. The `f` attribute indicates
that it utilizes FMA instructions, while `c` means no FMA.

<h2 id="datatypes">Data types</h2>

### Sleef_double2

#### Description

`Sleef_double2` is a generic data type for storing two double-precision floating
point values, which is defined in `sleef.h` as follows:

```c
typedef struct {
double x, y;
} Sleef_double2;
```

### Sleef_float2

#### Description

`Sleef_float2` is a generic data type for storing two single-precision floating
point values, which is defined in `sleef.h` as follows:

```c
typedef struct {
float x, y;
} Sleef_float2;
```

<h2 id="trig">Trigonometric Functions</h2>

### Sleef_sin_u10
### Sleef_sinf_u10

sine functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_sin_u10(double a);
float Sleef_sinf_u10(float a);

double Sleef_sind1_u10(double a);
double Sleef_sind1_u10purec(double a);
double Sleef_sind1_u10purecfma(double a);
double Sleef_cinz_sind1_u10purec(double a);
double Sleef_finz_sind1_u10purecfma(double a);

float Sleef_sinf1_u10(float a);
float Sleef_sinf1_u10purec(float a);
float Sleef_sinf1_u10purecfma(float a);
float Sleef_cinz_sinf1_u10purec(float a);
float Sleef_finz_sinf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the sine function of a value in a. The error bound of
the returned value is 1.0 ULP.  These functions treat the non-number arguments
and return non-numbers as specified in the C99 specification. These functions
do not set errno nor raise an exception.

### Sleef_sin_u35
### Sleef_sinf_u35

sine functions with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_sin_u35(double a);
float Sleef_sinf_u35(float a);

double Sleef_sind1_u35(double a);
double Sleef_sind1_u35purec(double a);
double Sleef_sind1_u35purecfma(double a);
double Sleef_cinz_sind1_u35purec(double a);
double Sleef_finz_sind1_u35purecfma(double a);

float Sleef_sinf1_u35(float a);
float Sleef_sinf1_u35purec(float a);
float Sleef_sinf1_u35purecfma(float a);
float Sleef_cinz_sinf1_u35purec(float a);
float Sleef_finz_sinf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the sine function of a value in a. The error bound of
the returned value is 3.5 ULP. These functions treat the non-number arguments
and return non-numbers as specified in the C99 specification. These functions
do not set errno nor raise an exception.

### Sleef_cos_u10
### Sleef_cosf_u10

cosine functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_cos_u10(double a);
float Sleef_cosf_u10(float a);

double Sleef_cosd1_u10(double a);
double Sleef_cosd1_u10purec(double a);
double Sleef_cosd1_u10purecfma(double a);
double Sleef_cinz_cosd1_u10purec(double a);
double Sleef_finz_cosd1_u10purecfma(double a);

float Sleef_cosf1_u10(float a);
float Sleef_cosf1_u10purec(float a);
float Sleef_cosf1_u10purecfma(float a);
float Sleef_cinz_cosf1_u10purec(float a);
float Sleef_finz_cosf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the cosine function of a value in a. The error bound
of the returned value is 1.0 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_cos_u35
### Sleef_cosf_u35

cosine functions with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_cos_u35(double a);
float Sleef_cosf_u35(float a);

double Sleef_cosd1_u35(double a);
double Sleef_cosd1_u35purec(double a);
double Sleef_cosd1_u35purecfma(double a);
double Sleef_cinz_cosd1_u35purec(double a);
double Sleef_finz_cosd1_u35purecfma(double a);

float Sleef_cosf1_u35(float a);
float Sleef_cosf1_u35purec(float a);
float Sleef_cosf1_u35purecfma(float a);
float Sleef_cinz_cosf1_u35purec(float a);
float Sleef_finz_cosf1_u35purecfma(float a);

```
Link with `-lsleef`.

#### Description

These functions evaluate the cosine function of a value in a. The error bound
of the returned value is 3.5 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_sincos_u10
### Sleef_sincosf_u10

evaluate sine and cosine functions simultaneously with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

Sleef_double2 Sleef_sincos_u10(double a)
Sleef_float2 Sleef_sincosf_u10(float a)

Sleef_double2 Sleef_sincosd1_u10(double a);
Sleef_double2 Sleef_sincosd1_u10purec(double a);
Sleef_double2 Sleef_sincosd1_u10purecfma(double a);
Sleef_double2 Sleef_cinz_sincosd1_u10purec(double a);
Sleef_double2 Sleef_finz_sincosd1_u10purecfma(double a);

Sleef_float2 Sleef_sincosf1_u10(float a);
Sleef_float2 Sleef_sincosf1_u10purec(float a);
Sleef_float2 Sleef_sincosf1_u10purecfma(float a);
Sleef_float2 Sleef_cinz_sincosf1_u10purec(float a);
Sleef_float2 Sleef_finz_sincosf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

Evaluates the sine and cosine functions of a value in a at a time, and store
the two values in x and y elements in the returned value, respectively. The
error bound of the returned values is 1.0 ULP. If a is a NaN or infinity, a NaN
is returned.

### Sleef_sincos_u35
### Sleef_sincosf_u35

evaluate sine and cosine functions simultaneously with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

Sleef_double2 Sleef_sincos_u35(double a)
Sleef_float2 Sleef_sincosf_u35(float a)

Sleef_double2 Sleef_sincosd1_u35(double a);
Sleef_double2 Sleef_sincosd1_u35purec(double a);
Sleef_double2 Sleef_sincosd1_u35purecfma(double a);
Sleef_double2 Sleef_cinz_sincosd1_u35purec(double a);
Sleef_double2 Sleef_finz_sincosd1_u35purecfma(double a);

Sleef_float2 Sleef_sincosf1_u35(float a);
Sleef_float2 Sleef_sincosf1_u35purec(float a);
Sleef_float2 Sleef_sincosf1_u35purecfma(float a);
Sleef_float2 Sleef_cinz_sincosf1_u35purec(float a);
Sleef_float2 Sleef_finz_sincosf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

Evaluates the sine and cosine functions of a value in a at a time, and store
the two values in x and y elements in the returned value, respectively. The
error bound of the returned values is 3.5 ULP. If a is a NaN or infinity, a NaN
is returned.

### Sleef_sincospi_u05"
### Sleef_sincospil_u05

evaluate `sin( &pi;a )` and `cos( &pi;a )` for given a simultaneously with
0.506 ULP error bound

#### Synopsis

```c
#include <sleef.h>

Sleef_double2 Sleef_sincospi_u05(double a)
Sleef_float2 Sleef_sincospif_u05(float a)

Sleef_double2 Sleef_sincospid1_u05(double a);
Sleef_double2 Sleef_sincospid1_u05purec(double a);
Sleef_double2 Sleef_sincospid1_u05purecfma(double a);
Sleef_double2 Sleef_cinz_sincospid1_u05purec(double a);
Sleef_double2 Sleef_finz_sincospid1_u05purecfma(double a);

Sleef_float2 Sleef_sincospif1_u05(float a);
Sleef_float2 Sleef_sincospif1_u05purec(float a);
Sleef_float2 Sleef_sincospif1_u05purecfma(float a);
Sleef_float2 Sleef_cinz_sincospif1_u05purec(float a);
Sleef_float2 Sleef_finz_sincospif1_u05purecfma(float a);
```
Link with `-lsleef`.

#### Description

Evaluates the sine and cosine functions of &pi;a  at a time, and store the two
values in x and y elements in the returned value, respectively. The error bound
of the returned value are max(0.506 ULP, DBL_MIN) if a is in [-1e+9, 1e+9] for
double-precision function, or max(0.506 ULP, FLT_MIN) if [-1e+7, 1e+7] for the
single-precision function. If a is a finite value out of this range, an
arbitrary value within [-1, 1] is returned. If a is a NaN or infinity, a NaN is
returned.

### Sleef_sincospi_u35
### Sleef_sincospif_u35

evaluate `sin( &pi;a )` and `cos( &pi;a )` for given a simultaneously with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

Sleef_double2 Sleef_sincospi_u35(double a)
Sleef_float2 Sleef_sincospif_u35(float a)

Sleef_double2 Sleef_sincospid1_u35(double a);
Sleef_double2 Sleef_sincospid1_u35purec(double a);
Sleef_double2 Sleef_sincospid1_u35purecfma(double a);
Sleef_double2 Sleef_cinz_sincospid1_u35purec(double a);
Sleef_double2 Sleef_finz_sincospid1_u35purecfma(double a);

Sleef_float2 Sleef_sincospif1_u35(float a);
Sleef_float2 Sleef_sincospif1_u35purec(float a);
Sleef_float2 Sleef_sincospif1_u35purecfma(float a);
Sleef_float2 Sleef_cinz_sincospif1_u35purec(float a);
Sleef_float2 Sleef_finz_sincospif1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

Evaluates the sine and cosine functions of &pi;a  at a time, and store the two
values in x and y elements in the returned value, respectively. The error bound
of the returned values is 3.5 ULP if a is in [-1e+9, 1e+9] for double-precision
function or [-1e+7, 1e+7] for the single-precision function. If a is a finite
value out of this range, an arbitrary value within [-1, 1] is returned. If a is
a NaN or infinity, a NaN is returned.

### Sleef_sinpi_u05
### Sleef_sinpif_u05

evaluate `sin( &pi;a )` for given a with 0.506 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_sinpi_u05(double a);
float Sleef_sinpif_u05(float a);

double Sleef_sinpid1_u05(double a);
double Sleef_sinpid1_u05purec(double a);
double Sleef_sinpid1_u05purecfma(double a);
double Sleef_cinz_sinpid1_u05purec(double a);
double Sleef_finz_sinpid1_u05purecfma(double a);

float Sleef_sinpif1_u05(float a);
float Sleef_sinpif1_u05purec(float a);
float Sleef_sinpif1_u05purecfma(float a);
float Sleef_cinz_sinpif1_u05purec(float a);
float Sleef_finz_sinpif1_u05purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the sine functions of &pi;a . The error bound of the
returned value are max(0.506 ULP, DBL_MIN) if a is in [-1e+9, 1e+9] for
double-precision function, or max(0.506 ULP, FLT_MIN) if [-1e+7, 1e+7] for the
single-precision function. If a is a finite value out of this range, an
arbitrary value within [-1, 1] is returned. If a is a NaN or infinity, a NaN is
returned.

### Sleef_cospi_u05
### Sleef_cospif_u05

evaluate `cos( &pi;a )` for given a with 0.506 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_cospi_u05(double a);
float Sleef_cospif_u05(float a);

double Sleef_cospid1_u05(double a);
double Sleef_cospid1_u05purec(double a);
double Sleef_cospid1_u05purecfma(double a);
double Sleef_cinz_cospid1_u05purec(double a);
double Sleef_finz_cospid1_u05purecfma(double a);

float Sleef_cospif1_u05(float a);
float Sleef_cospif1_u05purec(float a);
float Sleef_cospif1_u05purecfma(float a);
float Sleef_cinz_cospif1_u05purec(float a);
float Sleef_finz_cospif1_u05purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the cosine functions of &pi;a . The error bound of the
returned value are max(0.506 ULP, DBL_MIN) if a is in [-1e+9, 1e+9] for
double-precision function, or max(0.506 ULP, FLT_MIN) if [-1e+7, 1e+7] for the
single-precision function. If a is a finite value out of this range, an
arbitrary value within [-1, 1] is returned. If a is a NaN or infinity, a NaN is
returned.

### Sleef_tan_u10
### Sleef_tan_u10

tangent functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_tan_u10(double a);
float Sleef_tanf_u10(float a);

double Sleef_tand1_u10(double a);
double Sleef_tand1_u10purec(double a);
double Sleef_tand1_u10purecfma(double a);
double Sleef_cinz_tand1_u10purec(double a);
double Sleef_finz_tand1_u10purecfma(double a);

float Sleef_tanf1_u10(float a);
float Sleef_tanf1_u10purec(float a);
float Sleef_tanf1_u10purecfma(float a);
float Sleef_cinz_tanf1_u10purec(float a);
float Sleef_finz_tanf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the tangent function of a value in a. The error bound
of the returned value is 1.0 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_tan_u10
### Sleef_tan_u10

tangent functions with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_tan_u35(double a);
float Sleef_tanf_u35(float a);

double Sleef_tand1_u35(double a);
double Sleef_tand1_u35purec(double a);
double Sleef_tand1_u35purecfma(double a);
double Sleef_cinz_tand1_u35purec(double a);
double Sleef_finz_tand1_u35purecfma(double a);

float Sleef_tanf1_u35(float a);
float Sleef_tanf1_u35purec(float a);
float Sleef_tanf1_u35purecfma(float a);
float Sleef_cinz_tanf1_u35purec(float a);
float Sleef_finz_tanf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the tangent function of a value in a. The error bound
of the returned value is 3.5 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

<h2 id="pow">Power, exponential, and logarithmic functions</h2>

### Sleef_pow_u10
### Sleef_powf_u10

power functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_pow_u10(double x, double y);
float Sleef_powf_u10(float x, float y);

double Sleef_powd1_u10(double a, double b);
double Sleef_powd1_u10purec(double a, double b);
double Sleef_powd1_u10purecfma(double a, double b);
double Sleef_cinz_powd1_u10purec(double a, double b);
double Sleef_finz_powd1_u10purecfma(double a, double b);

float Sleef_powf1_u10(float a, float b);
float Sleef_powf1_u10purec(float a, float b);
float Sleef_powf1_u10purecfma(float a, float b);
float Sleef_cinz_powf1_u10purec(float a, float b);
float Sleef_finz_powf1_u10purecfma(float a, float b);
```
Link with `-lsleef`.

#### Description

These functions return the value of x raised to the power of y. The error bound
of the returned value is 1.0 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_log_u10
### Sleef_logf_u10

natural logarithmic functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_log_u10(double a);
float Sleef_logf_u10(float a);

double Sleef_logd1_u10(double a);
double Sleef_logd1_u10purec(double a);
double Sleef_logd1_u10purecfma(double a);
double Sleef_cinz_logd1_u10purec(double a);
double Sleef_finz_logd1_u10purecfma(double a);

float Sleef_logf1_u10(float a);
float Sleef_logf1_u10purec(float a);
float Sleef_logf1_u10purecfma(float a);
float Sleef_cinz_logf1_u10purec(float a);
float Sleef_finz_logf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the natural logarithm of a. The error bound of the
returned value is 1.0 ULP.  These functions treat the non-number arguments and
return non-numbers as specified in the C99 specification. These functions do
not set errno nor raise an exception.

### Sleef_log_u35
### Sleef_logf_u35

natural logarithmic functions with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_log_u35(double a);
float Sleef_logf_u35(float a);

double Sleef_logd1_u35(double a);
double Sleef_logd1_u35purec(double a);
double Sleef_logd1_u35purecfma(double a);
double Sleef_cinz_logd1_u35purec(double a);
double Sleef_finz_logd1_u35purecfma(double a);

float Sleef_logf1_u35(float a);
float Sleef_logf1_u35purec(float a);
float Sleef_logf1_u35purecfma(float a);
float Sleef_cinz_logf1_u35purec(float a);
float Sleef_finz_logf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the natural logarithm of a. The error bound of the
returned value is 3.5 ULP.  These functions treat the non-number arguments and
return non-numbers as specified in the C99 specification. These functions do
not set errno nor raise an exception.

### Sleef_log10_u10
### Sleef_log10f_u10

base-10 logarithmic functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_log10_u10(double a);
float Sleef_log10f_u10(float a);

double Sleef_log10d1_u10(double a);
double Sleef_log10d1_u10purec(double a);
double Sleef_log10d1_u10purecfma(double a);
double Sleef_cinz_log10d1_u10purec(double a);
double Sleef_finz_log10d1_u10purecfma(double a);

float Sleef_log10f1_u10(float a);
float Sleef_log10f1_u10purec(float a);
float Sleef_log10f1_u10purecfma(float a);
float Sleef_cinz_log10f1_u10purec(float a);
float Sleef_finz_log10f1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the base-10 logarithm of a. The error bound of the
returned value is 1.0 ULP.  These functions treat the non-number arguments and
return non-numbers as specified in the C99 specification. These functions do
not set errno nor raise an exception.

### Sleef_log2_u10
### Sleef_log2f_u10

base-2 logarithmic functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_log2_u10(double a);
float Sleef_log2f_u10(float a);

double Sleef_log2d1_u10(double a);
double Sleef_log2d1_u10purec(double a);
double Sleef_log2d1_u10purecfma(double a);
double Sleef_cinz_log2d1_u10purec(double a);
double Sleef_finz_log2d1_u10purecfma(double a);

float Sleef_log2f1_u10(float a);
float Sleef_log2f1_u10purec(float a);
float Sleef_log2f1_u10purecfma(float a);
float Sleef_cinz_log2f1_u10purec(float a);
float Sleef_finz_log2f1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the base-2 logarithm of a. The error bound of the
returned value is 1.0 ULP.  These functions treat the non-number arguments and
return non-numbers as specified in the C99 specification. These functions do
not set errno nor raise an exception.

### Sleef_log1p_u10
### Sleef_log1pf_u10

logarithm of one plus argument

#### Synopsis

```c
#include <sleef.h>

double Sleef_log1p_u10(double a);
float Sleef_log1pf_u10(float a);

double Sleef_log1pd1_u10(double a);
double Sleef_log1pd1_u10purec(double a);
double Sleef_log1pd1_u10purecfma(double a);
double Sleef_cinz_log1pd1_u10purec(double a);
double Sleef_finz_log1pd1_u10purecfma(double a);

float Sleef_log1pf1_u10(float a);
float Sleef_log1pf1_u10purec(float a);
float Sleef_log1pf1_u10purecfma(float a);
float Sleef_cinz_log1pf1_u10purec(float a);
float Sleef_finz_log1pf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the natural logarithm of (1+a). The error bound of the
returned value is 1.0 ULP.  These functions treat the non-number arguments and
return non-numbers as specified in the C99 specification. These functions do
not set errno nor raise an exception.

### Sleef_exp_u10
### Sleef_expf_u10

base-<i>e exponential functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_exp_u10(double a);
float Sleef_expf_u10(float a);

double Sleef_expd1_u10(double a);
double Sleef_expd1_u10purec(double a);
double Sleef_expd1_u10purecfma(double a);
double Sleef_cinz_expd1_u10purec(double a);
double Sleef_finz_expd1_u10purecfma(double a);

float Sleef_expf1_u10(float a);
float Sleef_expf1_u10purec(float a);
float Sleef_expf1_u10purecfma(float a);
float Sleef_cinz_expf1_u10purec(float a);
float Sleef_finz_expf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value of <i>e raised to a. The error bound of the
returned value is 1.0 ULP.  These functions treat the non-number arguments and
return non-numbers as specified in the C99 specification. These functions do
not set errno nor raise an exception.

### Sleef_exp2_u10
### Sleef_exp2f_u10

base-2 exponential functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_exp2_u10(double a);
float Sleef_exp2f_u10(float a);

double Sleef_exp2d1_u10(double a);
double Sleef_exp2d1_u10purec(double a);
double Sleef_exp2d1_u10purecfma(double a);
double Sleef_cinz_exp2d1_u10purec(double a);
double Sleef_finz_exp2d1_u10purecfma(double a);

float Sleef_exp2f1_u10(float a);
float Sleef_exp2f1_u10purec(float a);
float Sleef_exp2f1_u10purecfma(float a);
float Sleef_cinz_exp2f1_u10purec(float a);
float Sleef_finz_exp2f1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return 2 raised to a. The error bound of the returned value is
1.0 ULP.  These functions treat the non-number arguments and return non-numbers
as specified in the C99 specification. These functions do not set errno nor
raise an exception.

### Sleef_exp10_u10
### Sleef_exp10f_u10

base-10 exponential functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_exp10_u10(double a);
float Sleef_exp10f_u10(float a);

double Sleef_exp10d1_u10(double a);
double Sleef_exp10d1_u10purec(double a);
double Sleef_exp10d1_u10purecfma(double a);
double Sleef_cinz_exp10d1_u10purec(double a);
double Sleef_finz_exp10d1_u10purecfma(double a);

float Sleef_exp10f1_u10(float a);
float Sleef_exp10f1_u10purec(float a);
float Sleef_exp10f1_u10purecfma(float a);
float Sleef_cinz_exp10f1_u10purec(float a);
float Sleef_finz_exp10f1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return 10 raised to a. The error bound of the returned value is
1.09 ULP.  These functions treat the non-number arguments and return
non-numbers as specified in the C99 specification. These functions do not set
errno nor raise an exception.

### Sleef_expm1_u10
### Sleef_expm1f_u10

base-<i>e exponential functions minus 1

#### Synopsis

```c
#include <sleef.h>

double Sleef_expm1_u10(double a);
float Sleef_expm1f_u10(float a);

double Sleef_expm1d1_u10(double a);
double Sleef_expm1d1_u10purec(double a);
double Sleef_expm1d1_u10purecfma(double a);
double Sleef_cinz_expm1d1_u10purec(double a);
double Sleef_finz_expm1d1_u10purecfma(double a);

float Sleef_expm1f1_u10(float a);
float Sleef_expm1f1_u10purec(float a);
float Sleef_expm1f1_u10purecfma(float a);
float Sleef_cinz_expm1f1_u10purec(float a);
float Sleef_finz_expm1f1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value one less than <i>e raised to a. The error
bound of the returned value is 1.0 ULP.  These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_sqrt_u05
### Sleef_sqrtf_u05

square root function with 0.5001 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_sqrt_u05(double x);
float Sleef_sqrtf_u05(float x);

double Sleef_sqrtd1_u05(double a);
double Sleef_sqrtd1_u05purec(double a);
double Sleef_sqrtd1_u05purecfma(double a);
double Sleef_cinz_sqrtd1_u05purec(double a);
double Sleef_finz_sqrtd1_u05purecfma(double a);

float Sleef_sqrtf1_u05(float a);
float Sleef_sqrtf1_u05purec(float a);
float Sleef_sqrtf1_u05purecfma(float a);
float Sleef_cinz_sqrtf1_u05purec(float a);
float Sleef_finz_sqrtf1_u05purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of sqrt
and sqrtf functions. The error bound of the returned value is 0.5001 ULP.
These functions do not set errno nor raise an exception.

### Sleef_sqrt_u35
### Sleef_sqrtf_u35

square root function with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_sqrt_u35(double x);
float Sleef_sqrtf_u35(float x);

double Sleef_sqrtd1_u35(double a);
double Sleef_sqrtd1_u35purec(double a);
double Sleef_sqrtd1_u35purecfma(double a);
double Sleef_cinz_sqrtd1_u35purec(double a);
double Sleef_finz_sqrtd1_u35purecfma(double a);

float Sleef_sqrtf1_u35(float a);
float Sleef_sqrtf1_u35purec(float a);
float Sleef_sqrtf1_u35purecfma(float a);
float Sleef_cinz_sqrtf1_u35purec(float a);
float Sleef_finz_sqrtf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of sqrt
and sqrtf functions. The error bound of the returned value is 3.5 ULP.  These
functions do not set errno nor raise an exception.

### Sleef_cbrt_u10
### Sleef_cbrtf_u10

cube root function with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_cbrt_u10(double a);
float Sleef_cbrtf_u10(float a);

double Sleef_cbrtd1_u10(double a);
double Sleef_cbrtd1_u10purec(double a);
double Sleef_cbrtd1_u10purecfma(double a);
double Sleef_cinz_cbrtd1_u10purec(double a);
double Sleef_finz_cbrtd1_u10purecfma(double a);

float Sleef_cbrtf1_u10(float a);
float Sleef_cbrtf1_u10purec(float a);
float Sleef_cbrtf1_u10purecfma(float a);
float Sleef_cinz_cbrtf1_u10purec(float a);
float Sleef_finz_cbrtf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the real cube root of a. The error bound of the returned
value is 1.0 ULP.  These functions treat the non-number arguments and return
non-numbers as specified in the C99 specification. These functions do not set
errno nor raise an exception.

### Sleef_cbrt_u35
### Sleef_cbrtf_u35

cube root function with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_cbrt_u35(double a);
float Sleef_cbrtf_u35(float a);

double Sleef_cbrtd1_u35(double a);
double Sleef_cbrtd1_u35purec(double a);
double Sleef_cbrtd1_u35purecfma(double a);
double Sleef_cinz_cbrtd1_u35purec(double a);
double Sleef_finz_cbrtd1_u35purecfma(double a);

float Sleef_cbrtf1_u35(float a);
float Sleef_cbrtf1_u35purec(float a);
float Sleef_cbrtf1_u35purecfma(float a);
float Sleef_cinz_cbrtf1_u35purec(float a);
float Sleef_finz_cbrtf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the real cube root of a. The error bound of the returned
value is 1.0 ULP.  These functions treat the non-number arguments and return
non-numbers as specified in the C99 specification. These functions do not set
errno nor raise an exception.

### Sleef_hypot_u05
### Sleef_hypotf_u05

2D Euclidian distance function with 0.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_hypot_u05(double x, double y);
float Sleef_hypotf_u05(float x, float y);

double Sleef_hypotd1_u05(double a, double b);
double Sleef_hypotd1_u05purec(double a, double b);
double Sleef_hypotd1_u05purecfma(double a, double b);
double Sleef_cinz_hypotd1_u05purec(double a, double b);
double Sleef_finz_hypotd1_u05purecfma(double a, double b);

float Sleef_hypotf1_u05(float a, float b);
float Sleef_hypotf1_u05purec(float a, float b);
float Sleef_hypotf1_u05purecfma(float a, float b);
float Sleef_cinz_hypotf1_u05purec(float a, float b);
float Sleef_finz_hypotf1_u05purecfma(float a, float b);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of hypot
and hypotf functions. The error bound of the returned value is 0.5001 ULP.
These functions do not set errno nor raise an exception.

### Sleef_hypot_u35
### Sleef_hypotf_u35

2D Euclidian distance function with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_hypot_u35(double x, double y);
float Sleef_hypotf_u35(float x, float y);

double Sleef_hypotd1_u35(double a, double b);
double Sleef_hypotd1_u35purec(double a, double b);
double Sleef_hypotd1_u35purecfma(double a, double b);
double Sleef_cinz_hypotd1_u35purec(double a, double b);
double Sleef_finz_hypotd1_u35purecfma(double a, double b);

float Sleef_hypotf1_u35(float a, float b);
float Sleef_hypotf1_u35purec(float a, float b);
float Sleef_hypotf1_u35purecfma(float a, float b);
float Sleef_cinz_hypotf1_u35purec(float a, float b);
float Sleef_finz_hypotf1_u35purecfma(float a, float b);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of hypot
and hypotf functions. The error bound of the returned value is 0.5001 ULP.
These functions do not set errno nor raise an exception.

<h2 id="invtrig">Inverse Trigonometric Functions</h2>

### Sleef_asin_u10
### Sleef_asinf_u10

arc sine functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_asin_u10(double a);
float Sleef_asinf_u10(float a);

double Sleef_asind1_u10(double a);
double Sleef_asind1_u10purec(double a);
double Sleef_asind1_u10purecfma(double a);
double Sleef_cinz_asind1_u10purec(double a);
double Sleef_finz_asind1_u10purecfma(double a);

float Sleef_asinf1_u10(float a);
float Sleef_asinf1_u10purec(float a);
float Sleef_asinf1_u10purecfma(float a);
float Sleef_cinz_asinf1_u10purec(float a);
float Sleef_finz_asinf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the arc sine function of a value in a. The error bound
of the returned value is 1.0 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_asin_u35
### Sleef_asinf_u35

arc sine functions with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_asin_u35(double a);
float Sleef_asinf_u35(float a);

double Sleef_asind1_u35(double a);
double Sleef_asind1_u35purec(double a);
double Sleef_asind1_u35purecfma(double a);
double Sleef_cinz_asind1_u35purec(double a);
double Sleef_finz_asind1_u35purecfma(double a);

float Sleef_asinf1_u35(float a);
float Sleef_asinf1_u35purec(float a);
float Sleef_asinf1_u35purecfma(float a);
float Sleef_cinz_asinf1_u35purec(float a);
float Sleef_finz_asinf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the arc sine function of a value in a. The error bound
of the returned value is 3.5 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_acos_u10
### Sleef_acosf_u10

arc cosine functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_acos_u10(double a);
float Sleef_acosf_u10(float a);

double Sleef_acosd1_u10(double a);
double Sleef_acosd1_u10purec(double a);
double Sleef_acosd1_u10purecfma(double a);
double Sleef_cinz_acosd1_u10purec(double a);
double Sleef_finz_acosd1_u10purecfma(double a);

float Sleef_acosf1_u10(float a);
float Sleef_acosf1_u10purec(float a);
float Sleef_acosf1_u10purecfma(float a);
float Sleef_cinz_acosf1_u10purec(float a);
float Sleef_finz_acosf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the arc cosine function of a value in a. The error
bound of the returned value is 1.0 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_acos_u35
### Sleef_acosf_u35

arc cosine functions with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_acos_u35(double a);
float Sleef_acosf_u35(float a);

double Sleef_acosd1_u35(double a);
double Sleef_acosd1_u35purec(double a);
double Sleef_acosd1_u35purecfma(double a);
double Sleef_cinz_acosd1_u35purec(double a);
double Sleef_finz_acosd1_u35purecfma(double a);

float Sleef_acosf1_u35(float a);
float Sleef_acosf1_u35purec(float a);
float Sleef_acosf1_u35purecfma(float a);
float Sleef_cinz_acosf1_u35purec(float a);
float Sleef_finz_acosf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the arc cosine function of a value in a. The error
bound of the returned value is 3.5 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_atan_u10
### Sleef_atanf_u10

arc tangent functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_atan_u10(double a);
float Sleef_atanf_u10(float a);

double Sleef_atand1_u10(double a);
double Sleef_atand1_u10purec(double a);
double Sleef_atand1_u10purecfma(double a);
double Sleef_cinz_atand1_u10purec(double a);
double Sleef_finz_atand1_u10purecfma(double a);

float Sleef_atanf1_u10(float a);
float Sleef_atanf1_u10purec(float a);
float Sleef_atanf1_u10purecfma(float a);
float Sleef_cinz_atanf1_u10purec(float a);
float Sleef_finz_atanf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the arc tangent function of a value in a. The error
bound of the returned value is 1.0 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_atan_u35
### Sleef_atanf_u35

arc tangent functions with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_atan_u35(double a);
float Sleef_atanf_u35(float a);

double Sleef_atand1_u35(double a);
double Sleef_atand1_u35purec(double a);
double Sleef_atand1_u35purecfma(double a);
double Sleef_cinz_atand1_u35purec(double a);
double Sleef_finz_atand1_u35purecfma(double a);

float Sleef_atanf1_u35(float a);
float Sleef_atanf1_u35purec(float a);
float Sleef_atanf1_u35purecfma(float a);
float Sleef_cinz_atanf1_u35purec(float a);
float Sleef_finz_atanf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the arc tangent function of a value in a. The error
bound of the returned value is 3.5 ULP. These functions treat the non-number
arguments and return non-numbers as specified in the C99 specification. These
functions do not set errno nor raise an exception.

### Sleef_atan2_u10
### Sleef_atan2f_u10

arc tangent functions of two variables with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_atan2_u10(double y, double x);
float Sleef_atan2f_u10(float y, float x);

double Sleef_atan2d1_u10(double a, double b);
double Sleef_atan2d1_u10purec(double a, double b);
double Sleef_atan2d1_u10purecfma(double a, double b);
double Sleef_cinz_atan2d1_u10purec(double a, double b);
double Sleef_finz_atan2d1_u10purecfma(double a, double b);

float Sleef_atan2f1_u10(float a, float b);
float Sleef_atan2f1_u10purec(float a, float b);
float Sleef_atan2f1_u10purecfma(float a, float b);
float Sleef_cinz_atan2f1_u10purec(float a, float b);
float Sleef_finz_atan2f1_u10purecfma(float a, float b);
```
Link with `-lsleef`.

#### Description

These functions evaluate the arc tangent function of (y / x).  The quadrant of
the result is determined according to the signs of x and y.  The error bounds
of the returned values are max(1.0 ULP, DBL_MIN) and max(1.0 ULP, FLT_MIN),
respectively. These functions treat the non-number arguments and return
non-numbers as specified in the C99 specification. These functions do not set
errno nor raise an exception.

### Sleef_atan2_u35
### Sleef_atan2f_u35

arc tangent functions of two variables with 3.5 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_atan2_u35(double y, double x);
float Sleef_atan2f_u35(float y, float x);

double Sleef_atan2d1_u35(double a, double b);
double Sleef_atan2d1_u35purec(double a, double b);
double Sleef_atan2d1_u35purecfma(double a, double b);
double Sleef_cinz_atan2d1_u35purec(double a, double b);
double Sleef_finz_atan2d1_u35purecfma(double a, double b);

float Sleef_atan2f1_u35(float a, float b);
float Sleef_atan2f1_u35purec(float a, float b);
float Sleef_atan2f1_u35purecfma(float a, float b);
float Sleef_cinz_atan2f1_u35purec(float a, float b);
float Sleef_finz_atan2f1_u35purecfma(float a, float b);
```
Link with `-lsleef`.

#### Description

These functions evaluate the arc tangent function of (y / x).  The quadrant of
the result is determined according to the signs of x and y.  The error bound of
the returned value is 3.5 ULP. These functions treat the non-number arguments
and return non-numbers as specified in the C99 specification. These functions
do not set errno nor raise an exception.

<h2 id="hyp">Hyperbolic functions and inverse hyperbolic functions</h2>

### Sleef_sinh_u10
### Sleef_sinhf_u10

hyperbolic sine functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_sinh_u10(double a);
float Sleef_sinhf_u10(float a);

double Sleef_sinhd1_u10(double a);
double Sleef_sinhd1_u10purec(double a);
double Sleef_sinhd1_u10purecfma(double a);
double Sleef_cinz_sinhd1_u10purec(double a);
double Sleef_finz_sinhd1_u10purecfma(double a);

float Sleef_sinhf1_u10(float a);
float Sleef_sinhf1_u10purec(float a);
float Sleef_sinhf1_u10purecfma(float a);
float Sleef_cinz_sinhf1_u10purec(float a);
float Sleef_finz_sinhf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the hyperbolic sine function of a value in a. The
error bound of the returned value is 1.0 ULP if a is in [-709, 709] for the
double-precision function or [-88.5, 88.5] for the single-precision function .
If a is a finite value out of this range, infinity with a correct sign or a
correct value with 1.0 ULP error bound is returned. These functions treat the
non-number arguments and return non-numbers as specified in the C99
specification. These functions do not set errno nor raise an exception.

### Sleef_sinh_u35
### Sleef_sinhf_u35

hyperbolic sine functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_sinh_u35(double a);
float Sleef_sinhf_u35(float a);

double Sleef_sinhd1_u35(double a);
double Sleef_sinhd1_u35purec(double a);
double Sleef_sinhd1_u35purecfma(double a);
double Sleef_cinz_sinhd1_u35purec(double a);
double Sleef_finz_sinhd1_u35purecfma(double a);

float Sleef_sinhf1_u35(float a);
float Sleef_sinhf1_u35purec(float a);
float Sleef_sinhf1_u35purecfma(float a);
float Sleef_cinz_sinhf1_u35purec(float a);
float Sleef_finz_sinhf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the hyperbolic sine function of a value in a. The
error bound of the returned value is 3.5 ULP if a is in [-709, 709] for the
double-precision function or [-88, 88] for the single-precision function . If a
is a finite value out of this range, infinity with a correct sign or a correct
value with 3.5 ULP error bound is returned. These functions treat the
non-number arguments and return non-numbers as specified in the C99
specification. These functions do not set errno nor raise an exception.

### Sleef_cosh_u10
### Sleef_coshf_u10

hyperbolic cosine functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_cosh_u10(double a);
float Sleef_coshf_u10(float a);

double Sleef_coshd1_u10(double a);
double Sleef_coshd1_u10purec(double a);
double Sleef_coshd1_u10purecfma(double a);
double Sleef_cinz_coshd1_u10purec(double a);
double Sleef_finz_coshd1_u10purecfma(double a);

float Sleef_coshf1_u10(float a);
float Sleef_coshf1_u10purec(float a);
float Sleef_coshf1_u10purecfma(float a);
float Sleef_cinz_coshf1_u10purec(float a);
float Sleef_finz_coshf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the hyperbolic cosine function of a value in a. The
error bound of the returned value is 1.0 ULP if a is in [-709, 709] for the
double-precision function or [-88.5, 88.5] for the single-precision function .
If a is a finite value out of this range, infinity with a correct sign or a
correct value with 1.0 ULP error bound is returned. These functions treat the
non-number arguments and return non-numbers as specified in the C99
specification. These functions do not set errno nor raise an exception.

### Sleef_cosh_u35
### Sleef_coshf_u35

hyperbolic cosine functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_cosh_u35(double a);
float Sleef_coshf_u35(float a);

double Sleef_coshd1_u35(double a);
double Sleef_coshd1_u35purec(double a);
double Sleef_coshd1_u35purecfma(double a);
double Sleef_cinz_coshd1_u35purec(double a);
double Sleef_finz_coshd1_u35purecfma(double a);

float Sleef_coshf1_u35(float a);
float Sleef_coshf1_u35purec(float a);
float Sleef_coshf1_u35purecfma(float a);
float Sleef_cinz_coshf1_u35purec(float a);
float Sleef_finz_coshf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the hyperbolic cosine function of a value in a. The
error bound of the returned value is 3.5 ULP if a is in [-709, 709] for the
double-precision function or [-88, 88] for the single-precision function . If a
is a finite value out of this range, infinity with a correct sign or a correct
value with 3.5 ULP error bound is returned. These functions treat the
non-number arguments and return non-numbers as specified in the C99
specification. These functions do not set errno nor raise an exception.

### Sleef_tanh_u10
### Sleef_tanhf_u10

hyperbolic tangent functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_tanh_u10(double a);
float Sleef_tanhf_u10(float a);

double Sleef_tanhd1_u10(double a);
double Sleef_tanhd1_u10purec(double a);
double Sleef_tanhd1_u10purecfma(double a);
double Sleef_cinz_tanhd1_u10purec(double a);
double Sleef_finz_tanhd1_u10purecfma(double a);

float Sleef_tanhf1_u10(float a);
float Sleef_tanhf1_u10purec(float a);
float Sleef_tanhf1_u10purecfma(float a);
float Sleef_cinz_tanhf1_u10purec(float a);
float Sleef_finz_tanhf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the hyperbolic tangent function of a value in a. The
error bound of the returned value is 1.0 ULP for the double-precision function
or 1.0001 ULP for the single-precision function. These functions treat the
non-number arguments and return non-numbers as specified in the C99
specification. These functions do not set errno nor raise an exception.

### Sleef_tanh_u35
### Sleef_tanhf_u35

hyperbolic tangent functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_tanh_u35(double a);
float Sleef_tanhf_u35(float a);

double Sleef_tanhd1_u35(double a);
double Sleef_tanhd1_u35purec(double a);
double Sleef_tanhd1_u35purecfma(double a);
double Sleef_cinz_tanhd1_u35purec(double a);
double Sleef_finz_tanhd1_u35purecfma(double a);

float Sleef_tanhf1_u35(float a);
float Sleef_tanhf1_u35purec(float a);
float Sleef_tanhf1_u35purecfma(float a);
float Sleef_cinz_tanhf1_u35purec(float a);
float Sleef_finz_tanhf1_u35purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the hyperbolic tangent function of a value in a. The
error bound of the returned value is 3.5 ULP for the double-precision function
or 3.5 ULP for the single-precision function. These functions treat the
non-number arguments and return non-numbers as specified in the C99
specification. These functions do not set errno nor raise an exception.

### Sleef_asinh_u10
### Sleef_asinhf_u10

inverse hyperbolic sine functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_asinh_u10(double a);
float Sleef_asinhf_u10(float a);

double Sleef_asinhd1_u10(double a);
double Sleef_asinhd1_u10purec(double a);
double Sleef_asinhd1_u10purecfma(double a);
double Sleef_cinz_asinhd1_u10purec(double a);
double Sleef_finz_asinhd1_u10purecfma(double a);

float Sleef_asinhf1_u10(float a);
float Sleef_asinhf1_u10purec(float a);
float Sleef_asinhf1_u10purecfma(float a);
float Sleef_cinz_asinhf1_u10purec(float a);
float Sleef_finz_asinhf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the inverse hyperbolic sine function of a value in a.
The error bound of the returned value is 1.0 ULP if a is in [-1.34e+154,
1.34e+154] for the double-precision function or 1.001 ULP if a is in
[-1.84e+19, 1.84e+19] for the single-precision function . If a is a finite
value out of this range, infinity with a correct sign or a correct value with
1.0 ULP error bound is returned. These functions treat the non-number arguments
and return non-numbers as specified in the C99 specification. These functions
do not set errno nor raise an exception.

### Sleef_acosh_u10
### Sleef_acoshf_u10

inverse hyperbolic cosine functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_acosh_u10(double a);
float Sleef_acoshf_u10(float a);

double Sleef_acoshd1_u10(double a);
double Sleef_acoshd1_u10purec(double a);
double Sleef_acoshd1_u10purecfma(double a);
double Sleef_cinz_acoshd1_u10purec(double a);
double Sleef_finz_acoshd1_u10purecfma(double a);

float Sleef_acoshf1_u10(float a);
float Sleef_acoshf1_u10purec(float a);
float Sleef_acoshf1_u10purecfma(float a);
float Sleef_cinz_acoshf1_u10purec(float a);
float Sleef_finz_acoshf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the inverse hyperbolic cosine function of a value in a.
The error bound of the returned value is 1.0 ULP if a is in [1, 1.34e+154] for
the double-precision function or 1.001 ULP if a is in [1, 1.84e+19] for the
single-precision function. If a is lower than 1, nan is returned (including for
the negative infinity case). If the argument is positive (including positive
infinity) but lives outside the ranges defined above, then positive infinity is
returned. These functions treat the non-number arguments and return non-numbers
as specified in the C99 specification. These functions do not set errno nor raise
an exception.

### Sleef_atanh_u10
### Sleef_atanhf_u10

inverse hyperbolic tangent functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_atanh_u10(double a);
float Sleef_atanhf_u10(float a);

double Sleef_atanhd1_u10(double a);
double Sleef_atanhd1_u10purec(double a);
double Sleef_atanhd1_u10purecfma(double a);
double Sleef_cinz_atanhd1_u10purec(double a);
double Sleef_finz_atanhd1_u10purecfma(double a);

float Sleef_atanhf1_u10(float a);
float Sleef_atanhf1_u10purec(float a);
float Sleef_atanhf1_u10purecfma(float a);
float Sleef_cinz_atanhf1_u10purec(float a);
float Sleef_finz_atanhf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions evaluate the inverse hyperbolic tangent function of a value in
a. The error bound of the returned value is 1.0 ULP for the double-precision
function or 1.0001 ULP for the single-precision function. These functions treat
the non-number arguments and return non-numbers as specified in the C99
specification. These functions do not set errno nor raise an exception.

<h2 id="eg">Error and gamma functions</h2>

### Sleef_erf_u10
### Sleef_erff_u10

error functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_erf_u10(double x);
float Sleef_erff_u10(float x);

double Sleef_erfd1_u10(double a);
double Sleef_erfd1_u10purec(double a);
double Sleef_erfd1_u10purecfma(double a);
double Sleef_cinz_erfd1_u10purec(double a);
double Sleef_finz_erfd1_u10purecfma(double a);

float Sleef_erff1_u10(float a);
float Sleef_erff1_u10purec(float a);
float Sleef_erff1_u10purecfma(float a);
float Sleef_cinz_erff1_u10purec(float a);
float Sleef_finz_erff1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of erf
and erff functions. The error bound of the returned value is 1.0 ULP.  These
functions do not set errno nor raise an exception.

### Sleef_erf
### Sleef_erff

complementary error functions

#### Synopsis

```c
#include <sleef.h>

double Sleef_erfc_u15(double x);
float Sleef_erfcf_u15(float x);

double Sleef_erfcd1_u15(double a);
double Sleef_erfcd1_u15purec(double a);
double Sleef_erfcd1_u15purecfma(double a);
double Sleef_cinz_erfcd1_u15purec(double a);
double Sleef_finz_erfcd1_u15purecfma(double a);

float Sleef_erfcf1_u15(float a);
float Sleef_erfcf1_u15purec(float a);
float Sleef_erfcf1_u15purecfma(float a);
float Sleef_cinz_erfcf1_u15purec(float a);
float Sleef_finz_erfcf1_u15purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of erfc
and erfcf functions. The error bound of the returned value for the DP function
is max(1.5 ULP, DBL_MIN) if the argument is less than 26.2, and max(2.5 ULP,
DBL_MIN) otherwise. For the SP function, the error bound is max(1.5 ULP,
FLT_MIN). These functions do not set errno nor raise an exception.

### Sleef_tgamma_u10
### Sleef_tgammaf_u10

gamma functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_tgamma_u10(double x);
float Sleef_tgammaf_u10(float x);

double Sleef_tgammad1_u10(double a);
double Sleef_tgammad1_u10purec(double a);
double Sleef_tgammad1_u10purecfma(double a);
double Sleef_cinz_tgammad1_u10purec(double a);
double Sleef_finz_tgammad1_u10purecfma(double a);

float Sleef_tgammaf1_u10(float a);
float Sleef_tgammaf1_u10purec(float a);
float Sleef_tgammaf1_u10purecfma(float a);
float Sleef_cinz_tgammaf1_u10purec(float a);
float Sleef_finz_tgammaf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of
tgamma and tgammaf functions. The error bound of the returned value is 1.0 ULP.
These functions do not set errno nor raise an exception.

### Sleef_lgamma_u10
### Sleef_lgammaf_u10

log gamma functions with 1.0 ULP error bound

#### Synopsis

```c
#include <sleef.h>

double Sleef_lgamma_u10(double x);
float Sleef_lgammaf_u10(float x);

double Sleef_lgammad1_u10(double a);
double Sleef_lgammad1_u10purec(double a);
double Sleef_lgammad1_u10purecfma(double a);
double Sleef_cinz_lgammad1_u10purec(double a);
double Sleef_finz_lgammad1_u10purecfma(double a);

float Sleef_lgammaf1_u10(float a);
float Sleef_lgammaf1_u10purec(float a);
float Sleef_lgammaf1_u10purecfma(float a);
float Sleef_cinz_lgammaf1_u10purec(float a);
float Sleef_finz_lgammaf1_u10purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of
lgamma and lgammaf functions. The error bound of the returned value is 1.0 ULP
if the argument is positive. If the argument is larger than 2e+305 for the DP
function and 4e+36 for the SP function, it may return infinity instead of the
correct value.  The error bound is max(1 ULP, 1e-15) for the DP function and
max(1 ULP and 1e-8) for the SP function, if the argument is negative.  These
functions do not set errno nor raise an exception.

<h2 id="nearint">Nearest integer functions</h2>

### Sleef_trunc
### Sleef_truncf

round to integer towards zero

#### Synopsis

```c
#include <sleef.h>

double Sleef_trunc(double x);
float Sleef_truncf(float x);

double Sleef_truncd1(double a);
double Sleef_truncd1_purec(double a);
double Sleef_truncd1_purecfma(double a);
double Sleef_cinz_truncd1_purec(double a);
double Sleef_finz_truncd1_purecfma(double a);

float Sleef_truncf1(float a);
float Sleef_truncf1_purec(float a);
float Sleef_truncf1_purecfma(float a);
float Sleef_cinz_truncf1_purec(float a);
float Sleef_finz_truncf1_purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of trunc
and truncf functions. These functions do not set errno nor raise an exception.

### Sleef_floor
### Sleef_floorf

round to integer towards minus infinity

#### Synopsis

```c
#include <sleef.h>

double Sleef_floor(double x);
float Sleef_floorf(float x);

double Sleef_floord1(double a);
double Sleef_floord1_purec(double a);
double Sleef_floord1_purecfma(double a);
double Sleef_cinz_floord1_purec(double a);
double Sleef_finz_floord1_purecfma(double a);

float Sleef_floorf1(float a);
float Sleef_floorf1_purec(float a);
float Sleef_floorf1_purecfma(float a);
float Sleef_cinz_floorf1_purec(float a);
float Sleef_finz_floorf1_purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of floor
and floorf functions. These functions do not set errno nor raise an exception.

### Sleef_ceil
### Sleef_ceilf

round to integer towards plus infinity

#### Synopsis

```c
#include <sleef.h>

double Sleef_ceil(double x);
float Sleef_ceilf(float x);

double Sleef_ceild1(double a);
double Sleef_ceild1_purec(double a);
double Sleef_ceild1_purecfma(double a);
double Sleef_cinz_ceild1_purec(double a);
double Sleef_finz_ceild1_purecfma(double a);

float Sleef_ceilf1(float a);
float Sleef_ceilf1_purec(float a);
float Sleef_ceilf1_purecfma(float a);
float Sleef_cinz_ceilf1_purec(float a);
float Sleef_finz_ceilf1_purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of ceil
and ceilf functions. These functions do not set errno nor raise an exception.

### Sleef_round
### Sleef_roundf

round to integer away from zero

#### Synopsis

```c
#include <sleef.h>

double Sleef_round(double x);
float Sleef_roundf(float x);

double Sleef_roundd1(double a);
double Sleef_roundd1_purec(double a);
double Sleef_roundd1_purecfma(double a);
double Sleef_cinz_roundd1_purec(double a);
double Sleef_finz_roundd1_purecfma(double a);

float Sleef_roundf1(float a);
float Sleef_roundf1_purec(float a);
float Sleef_roundf1_purecfma(float a);
float Sleef_cinz_roundf1_purec(float a);
float Sleef_finz_roundf1_purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of round
and roundf functions. These functions do not set errno nor raise an exception.

### Sleef_rint
### Sleef_rintf

round to integer, ties round to even

#### Synopsis

```c
#include <sleef.h>

double Sleef_rint(double x);
float Sleef_rintf(float x);

double Sleef_rintd1(double a);
double Sleef_rintd1_purec(double a);
double Sleef_rintd1_purecfma(double a);
double Sleef_cinz_rintd1_purec(double a);
double Sleef_finz_rintd1_purecfma(double a);

float Sleef_rintf1(float a);
float Sleef_rintf1_purec(float a);
float Sleef_rintf1_purecfma(float a);
float Sleef_cinz_rintf1_purec(float a);
float Sleef_finz_rintf1_purecfma(float a);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of rint
and rintf functions. These functions do not set errno nor raise an exception.

<h2 id="other">Other functions</h2>

### Sleef_fma
### Sleef_fmaf

fused multiply and accumulate

#### Synopsis

```c
#include <sleef.h>

double Sleef_fma(double x, double y, double z);
float Sleef_fmaf(float x, float y, float z);

double Sleef_fmad1(double x, double y, double z);
double Sleef_fmad1_purec(double x, double y, double z);
double Sleef_fmad1_purecfma(double x, double y, double z);
double Sleef_cinz_fmad1_purec(double x, double y, double z);
double Sleef_finz_fmad1_purecfma(double x, double y, double z);

float Sleef_fmaf1(float x, float y, float z);
float Sleef_fmaf1_purec(float x, float y, float z);
float Sleef_fmaf1_purecfma(float x, float y, float z);
float Sleef_cinz_fmaf1_purec(float x, float y, float z);
float Sleef_finz_fmaf1_purecfma(float x, float y, float z);
```
Link with `-lsleef`.

#### Description

These functions compute (`x * y + z`) without rounding, and then return
the rounded value of the result. These functions may return infinity with a
correct sign if the absolute value of the correct return value is greater than
1e+300 and 1e+33, respectively.  The error bounds of the returned values are
0.5 ULP and max(0.50001 ULP, FLT_MIN), respectively.

### Sleef_fmod
### Sleef_fmodf

FP remainder

#### Synopsis

```c
#include <sleef.h>

double Sleef_fmod(double x, double y);
float Sleef_fmodf(float x, float y);

double Sleef_fmodd1(double x, double y);
double Sleef_fmodd1_purec(double x, double y);
double Sleef_fmodd1_purecfma(double x, double y);
double Sleef_cinz_fmodd1_purec(double x, double y);
double Sleef_finz_fmodd1_purecfma(double x, double y);

float Sleef_fmodf1(float x, float y);
float Sleef_fmodf1_purec(float x, float y);
float Sleef_fmodf1_purecfma(float x, float y);
float Sleef_cinz_fmodf1_purec(float x, float y);
float Sleef_finz_fmodf1_purecfma(float x, float y);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of fmod
and fmodf functions, if `|x / y|` is smaller than 1e+300 and 1e+38,
respectively. The returned value is undefined, otherwise. These functions do
not set errno nor raise an exception.

### Sleef_remainder
### Sleef_remainderf

FP remainder

#### Synopsis

```c
#include <sleef.h>

double Sleef_remainder(double x, double y);
float Sleef_remainderf(float x, float y);

double Sleef_remainderd1(double x, double y);
double Sleef_remainderd1_purec(double x, double y);
double Sleef_remainderd1_purecfma(double x, double y);
double Sleef_cinz_remainderd1_purec(double x, double y);
double Sleef_finz_remainderd1_purecfma(double x, double y);

float Sleef_remainderf1(float x, float y);
float Sleef_remainderf1_purec(float x, float y);
float Sleef_remainderf1_purecfma(float x, float y);
float Sleef_cinz_remainderf1_purec(float x, float y);
float Sleef_finz_remainderf1_purecfma(float x, float y);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of
remainder and remainderf functions, if <i>|x / y| is smaller than 1e+300 and
1e+38, respectively. The returned value is undefined, otherwise. These
functions do not set errno nor raise an exception.

### Sleef_ldexp
### Sleef_ldexpf

multiply by integral power of 2

#### Synopsis

```c
#include <sleef.h>

double Sleef_ldexp(double m, int x);
float Sleef_ldexpf(float m, int x);

double Sleef_ldexpd1(double m, int x);
double Sleef_ldexpd1_purec(double m, int x);
double Sleef_ldexpd1_purecfma(double m, int x);
double Sleef_cinz_ldexpd1_purec(double m, int x);
double Sleef_finz_ldexpd1_purecfma(double m, int x);

float Sleef_ldexpf1(float m, int x);
float Sleef_ldexpf1_purec(float m, int x);
float Sleef_ldexpf1_purecfma(float m, int x);
float Sleef_cinz_ldexpf1_purec(float m, int x);
float Sleef_finz_ldexpf1_purecfma(float m, int x);
```
Link with `-lsleef`.

#### Description

These functions return the result of multiplying m by 2 raised to the power x.
These functions treat the non-number arguments and return non-numbers as
specified in the C99 specification. These functions do not set errno nor raise
an exception.

### Sleef_frfrexp
### Sleef_frfrexpf

fractional component of an FP number

#### Synopsis

```c
#include <sleef.h>

double Sleef_frfrexp(double x);
float Sleef_frfrexpf(float x);

double Sleef_frfrexpd1(double x);
double Sleef_frfrexpd1_purec(double x);
double Sleef_frfrexpd1_purecfma(double x);
double Sleef_cinz_frfrexpd1_purec(double x);
double Sleef_finz_frfrexpd1_purecfma(double x);

float Sleef_frfrexpf1(float x);
float Sleef_frfrexpf1_purec(float x);
float Sleef_frfrexpf1_purecfma(float x);
float Sleef_cinz_frfrexpf1_purec(float x);
float Sleef_finz_frfrexpf1_purecfma(float x);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of frexp
and frexpf functions. These functions do not set errno nor raise an exception.

### Sleef_expfrexp
### Sleef_expfrexpf

exponent of an FP number

#### Synopsis

```c
#include <sleef.h>

int Sleef_expfrexp(double x);
int Sleef_expfrexpf(float x);

int Sleef_expfrexpd1(double x);
int Sleef_expfrexpd1_purec(double x);
int Sleef_expfrexpd1_purecfma(double x);
int Sleef_cinz_expfrexpd1_purec(double x);
int Sleef_finz_expfrexpd1_purecfma(double x);

int Sleef_expfrexpf1(float x);
int Sleef_expfrexpf1_purec(float x);
int Sleef_expfrexpf1_purecfma(float x);
int Sleef_cinz_expfrexpf1_purec(float x);
int Sleef_finz_expfrexpf1_purecfma(float x);
```
Link with `-lsleef`.

#### Description

These functions return the exponent returned by frexp and frexpf functions as
specified in the C99 specification. These functions do not set errno nor raise
an exception.

### Sleef_ilogb
### Sleef_ilogbf

integer exponent of an FP number

#### Synopsis

```c
#include <sleef.h>

int Sleef_ilogb(double m, int x);
int Sleef_ilogbf(float m, int x);

int Sleef_ilogbd1(double m, int x);
int Sleef_ilogbd1_purec(double m, int x);
int Sleef_ilogbd1_purecfma(double m, int x);
int Sleef_cinz_ilogbd1_purec(double m, int x);
int Sleef_finz_ilogbd1_purecfma(double m, int x);

int Sleef_ilogbf1(float m, int x);
int Sleef_ilogbf1_purec(float m, int x);
int Sleef_ilogbf1_purecfma(float m, int x);
int Sleef_cinz_ilogbf1_purec(float m, int x);
int Sleef_finz_ilogbf1_purecfma(float m, int x);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of ilogb
and ilogbf functions. These functions do not set errno nor raise an exception.

### Sleef_modf
### Sleef_modff

integral and fractional value of FP number

#### Synopsis

```c
#include <sleef.h>

Sleef_double2 Sleef_modf(double x);
Sleef_float2 Sleef_modff(float x);

Sleef_double2 Sleef_modfd1(double x);
Sleef_double2 Sleef_modfd1_purec(double x);
Sleef_double2 Sleef_modfd1_purecfma(double x);
Sleef_double2 Sleef_cinz_modfd1_purec(double x);
Sleef_double2 Sleef_finz_modfd1_purecfma(double x);

Sleef_float2 Sleef_modff1(float x);
Sleef_float2 Sleef_modff1_purec(float x);
Sleef_float2 Sleef_modff1_purecfma(float x);
Sleef_float2 Sleef_cinz_modff1_purec(float x);
Sleef_float2 Sleef_finz_modff1_purecfma(float x);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of modf
and modff functions. These functions do not set errno nor raise an exception.

### Sleef_fabs
### Sleef_fabsf

absolute value

#### Synopsis

```c
#include <sleef.h>

double Sleef_fabs(double x);
float Sleef_fabsf(float x);

double Sleef_fabsd1(double x);
double Sleef_fabsd1_purec(double x);
double Sleef_fabsd1_purecfma(double x);
double Sleef_cinz_fabsd1_purec(double x);
double Sleef_finz_fabsd1_purecfma(double x);

float Sleef_fabsf1(float x);
float Sleef_fabsf1_purec(float x);
float Sleef_fabsf1_purecfma(float x);
float Sleef_cinz_fabsf1_purec(float x);
float Sleef_finz_fabsf1_purecfma(float x);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of fabs
and fabsf functions. These functions do not set errno nor raise an exception.

### Sleef_fmax
### Sleef_fmaxf

maximum of two numbers

#### Synopsis

```c
#include <sleef.h>

double Sleef_fmax(double x, double y);
float Sleef_fmaxf(float x, float y);

double Sleef_fmaxd1(double x, double y);
double Sleef_fmaxd1_purec(double x, double y);
double Sleef_fmaxd1_purecfma(double x, double y);
double Sleef_cinz_fmaxd1_purec(double x, double y);
double Sleef_finz_fmaxd1_purecfma(double x, double y);

float Sleef_fmaxf1(float x, float y);
float Sleef_fmaxf1_purec(float x, float y);
float Sleef_fmaxf1_purecfma(float x, float y);
float Sleef_cinz_fmaxf1_purec(float x, float y);
float Sleef_finz_fmaxf1_purecfma(float x, float y);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of fmax
and fmaxf functions. These functions do not set errno nor raise an exception.

### Sleef_fmin
### Sleef_fminf

minimum of two numbers

#### Synopsis

```c
#include <sleef.h>

double Sleef_fmin(double x, double y);
float Sleef_fminf(float x, float y);

double Sleef_fmind1(double x, double y);
double Sleef_fmind1_purec(double x, double y);
double Sleef_fmind1_purecfma(double x, double y);
double Sleef_cinz_fmind1_purec(double x, double y);
double Sleef_finz_fmind1_purecfma(double x, double y);

float Sleef_fminf1(float x, float y);
float Sleef_fminf1_purec(float x, float y);
float Sleef_fminf1_purecfma(float x, float y);
float Sleef_cinz_fminf1_purec(float x, float y);
float Sleef_finz_fminf1_purecfma(float x, float y);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of fmin
and fminf functions. These functions do not set errno nor raise an exception.

### Sleef_fdim
### Sleef_fdimf

positive difference

#### Synopsis

```c
#include <sleef.h>

double Sleef_fdim(double x, double y);
float Sleef_fdimf(float x, float y);

double Sleef_fdimd1(double x, double y);
double Sleef_fdimd1_purec(double x, double y);
double Sleef_fdimd1_purecfma(double x, double y);
double Sleef_cinz_fdimd1_purec(double x, double y);
double Sleef_finz_fdimd1_purecfma(double x, double y);

float Sleef_fdimf1(float x, float y);
float Sleef_fdimf1_purec(float x, float y);
float Sleef_fdimf1_purecfma(float x, float y);
float Sleef_cinz_fdimf1_purec(float x, float y);
float Sleef_finz_fdimf1_purecfma(float x, float y);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of fdim
and fdimf functions. These functions do not set errno nor raise an exception.

### Sleef_copysign
### Sleef_copysignf

copy sign of a number

#### Synopsis

```c
#include <sleef.h>

double Sleef_copysign(double x, double y);
float Sleef_copysignf(float x, float y);

double Sleef_copysignd1(double x, double y);
double Sleef_copysignd1_purec(double x, double y);
double Sleef_copysignd1_purecfma(double x, double y);
double Sleef_cinz_copysignd1_purec(double x, double y);
double Sleef_finz_copysignd1_purecfma(double x, double y);

float Sleef_copysignf1(float x, float y);
float Sleef_copysignf1_purec(float x, float y);
float Sleef_copysignf1_purecfma(float x, float y);
float Sleef_cinz_copysignf1_purec(float x, float y);
float Sleef_finz_copysignf1_purecfma(float x, float y);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of
copysign and copysignf functions. These functions do not set errno nor raise an
exception.

### Sleef_nextafter
### Sleef_nextafterf

find the next representable FP value

#### Synopsis

```c
#include <sleef.h>

double Sleef_nextafter(double x, double y);
float Sleef_nextafterf(float x, float y);

double Sleef_nextafterd1(double x, double y);
double Sleef_nextafterd1_purec(double x, double y);
double Sleef_nextafterd1_purecfma(double x, double y);
double Sleef_cinz_nextafterd1_purec(double x, double y);
double Sleef_finz_nextafterd1_purecfma(double x, double y);

float Sleef_nextafterf1(float x, float y);
float Sleef_nextafterf1_purec(float x, float y);
float Sleef_nextafterf1_purecfma(float x, float y);
float Sleef_cinz_nextafterf1_purec(float x, float y);
float Sleef_finz_nextafterf1_purecfma(float x, float y);
```
Link with `-lsleef`.

#### Description

These functions return the value as specified in the C99 specification of
nextafter and nextafterf functions. These functions do not set errno nor raise
an exception.

