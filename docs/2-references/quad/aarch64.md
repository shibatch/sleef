---
layout: default
title: AArch64
parent: Quadruple Precision
grand_parent: References
permalink: /2-references/quad/aarch64
---

<h1>Quadruple precision Math Library reference (AArch64)</h1>

<h2>Table of contents</h2>

* [Data types](#datatypes)
* [Functions for accessing elements inside vector variable](#access)
* [Conversion functions](#conversion)
* [Comparison and selection functions](#comparison)
* [Math functions](#mathfunctions)

<h2 id="datatypes">Data types</h2>

### Sleef_quadx2

`Sleef_quadx2` is a data type for retaining two QP FP numbers. Please use
[Sleef_loadq2](#load), [Sleef_storeq2](#store), [Sleef_getq2](#get) and
[Sleef_setq2](#set) functions to access the data inside variables in this data
type.

### Sleef_svquad

`Sleef_svquad` is a data type for retaining a vector of QP FP numbers.  Please
use [Sleef_loadqx_sve](#load), [Sleef_storeqx_sve](#store),
[Sleef_getqx_sve](#get) and [Sleef_setqx_sve](#set) functions to access the
data inside variables in this data type.

<h2 id="access">Functions for accessing elements inside vector variable</h2>

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_loadq2(Sleef_quad * ptr);
Sleef_quadx2 Sleef_loadq2_advsimd(Sleef_quad * ptr);
Sleef_svquad Sleef_loadqx_sve(Sleef_quad * ptr);
```
Link with `-lsleefquad`.

These functions load QP FP values from memory and store the results in a vector
variable.

```c
#include <sleefquad.h>

void Sleef_storeq2(Sleef_quad * ptr, Sleef_quadx2 v);
void Sleef_storeq2_advsimd(Sleef_quad * ptr, Sleef_quadx2 v);
void Sleef_storeqx_sve(Sleef_quad * ptr, Sleef_svquad v);
```
Link with `-lsleefquad`.

These functions store QP FP values in the given variable to memory.

```c
#include <sleefquad.h>

Sleef_quad Sleef_getq2(Sleef_quadx2 v, int index);
Sleef_quad Sleef_getq2_advsimd(Sleef_quadx2 v, int index);
Sleef_quad Sleef_getqx_sve(Sleef_svquad v, int index);
```
Link with `-lsleefquad`.

These functions extract the `index`-th element from a QP FP vector variable.

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_setq2(Sleef_quadx2 v, int index, Sleef_quad e);
Sleef_quadx2 Sleef_setq2_advsimd(Sleef_quadx2 v, int index, Sleef_quad e);
Sleef_svquad Sleef_setqx_sve(Sleef_svquad v, int index, Sleef_quad e);
```
Link with `-lsleefquad`.

These functions return a QP FP vector in which the `index`-th element is `e`,
and other elements are the same as `v`.

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_splatq2(Sleef_quad e);
Sleef_quadx2 Sleef_splatq2_advsimd(Sleef_quad e);
Sleef_svquad Sleef_splatqx_sve(Sleef_quad e);
```
Link with `-lsleefquad`.

These functions return a QP FP vector in which all elements are set to `e`.

<h2 id="conversion">Conversion functions</h2>

### Convert QP number to double-precision number

```c
#include <sleefquad.h>

float64x2_t Sleef_cast_to_doubleq2(Sleef_quadx2 a);
float64x2_t Sleef_cast_to_doubleq2_advsimd(Sleef_quadx2 a);
svfloat64_t Sleef_cast_to_doubleqx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These functions convert a QP FP value to a double-precision value.

### Convert double-precision number to QP number

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_cast_from_doubleq2(float64x2_t a);
Sleef_quadx2 Sleef_cast_from_doubleq2_advsimd(float64x2_t a);
Sleef_svquad Sleef_cast_from_doubleqx_sve(svfloat64_t a);
```
Link with `-lsleefquad`.

These functions convert a double-precision value to a QP FP value.

### Convert QP number to 64-bit signed integer

```c
#include <sleefquad.h>

int64x2_t Sleef_cast_to_int64q2(Sleef_quadx2 a);
int64x2_t Sleef_cast_to_int64q2_advsimd(Sleef_quadx2 a);
svint64_t Sleef_cast_to_int64qx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These functions convert a QP FP value to a 64-bit signed integer.

### Convert 64-bit signed integer to QP number

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_cast_from_int64q2(int64x2_t a);
Sleef_quadx2 Sleef_cast_from_int64q2_advsimd(int64x2_t a);
Sleef_svquad Sleef_cast_from_int64qx_sve(svint64_t a);
```
Link with `-lsleefquad`.

These functions convert a 64-bit signed integer to a QP FP value.

### Convert QP number to 64-bit unsigned integer

```c
#include <sleefquad.h>

uint64x2_t Sleef_cast_to_uint64q2(Sleef_quadx2 a);
uint64x2_t Sleef_cast_to_uint64q2_advsimd(Sleef_quadx2 a);
svuint64_t Sleef_cast_to_uint64qx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These functions convert a QP FP value to a 64-bit signed integer.

### Convert 64-bit unsigned integer to QP number

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_cast_from_uint64q2(uint64x2_t a);
Sleef_quadx2 Sleef_cast_from_uint64q2_advsimd(uint64x2_t a);
Sleef_svquad Sleef_cast_from_uint64qx_sve(svuint64_t a);
```
Link with `-lsleefquad`.

These functions convert a 64-bit unsigned integer to a QP FP value.

<h2 id="comparison">Comparison and selection functions</h2>

### QP comparison functions

```c
#include <sleefquad.h>

int32x2_t Sleef_icmpltq2(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpleq2(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpgtq2(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpgeq2(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpeqq2(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpneq2(Sleef_quadx2 a, Sleef_quadx2 b);

int32x2_t Sleef_icmpltq2_advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpleq2_advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpgtq2_advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpgeq2_advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpeqq2_advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpneq2_advsimd(Sleef_quadx2 a, Sleef_quadx2 b);

svint32_t Sleef_icmpltqx_sve(Sleef_svquad a, Sleef_svquad b);
svint32_t Sleef_icmpleqx_sve(Sleef_svquad a, Sleef_svquad b);
svint32_t Sleef_icmpgtqx_sve(Sleef_svquad a, Sleef_svquad b);
svint32_t Sleef_icmpgeqx_sve(Sleef_svquad a, Sleef_svquad b);
svint32_t Sleef_icmpeqqx_sve(Sleef_svquad a, Sleef_svquad b);
svint32_t Sleef_icmpneqx_sve(Sleef_svquad a, Sleef_svquad b);
```
Link with `-lsleefquad`.

These are the vectorized functions of [comparison
functions](../quad#basicComparison).

### QP comparison functions of the second kind

```c
#include <sleefquad.h>

int32x2_t Sleef_icmpq2(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_icmpq2_advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
svint32_t Sleef_icmpqx_sve(Sleef_svquad a, Sleef_svquad b);
```
Link with `-lsleefquad`.

These are the vectorized functions of
[Sleef_icmpq1_purec](../quad#sleef_icmpq1_purec).

### Check orderedness

```c
#include <sleefquad.h>

int32x2_t Sleef_iunordq2(Sleef_quadx2 a, Sleef_quadx2 b);
int32x2_t Sleef_iunordq2_advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
svint32_t Sleef_iunordqx_sve(Sleef_svquad a, Sleef_svquad b);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_iunordq1_purec](../quad#sleef_iunordq1_purec).

### Select elements

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_iselectq2(int32x2_t c, Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_quadx2 Sleef_iselectq2_advsimd(int32x2_t c, Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_svquad Sleef_iselectqx_sve(svint32_t c, Sleef_svquad a, Sleef_svquad b);
```
Link with `-lsleefquad`.

These are the vectorized functions that operate in the same way as the ternary
operator.

<h2 id="mathfunctions">Math functions</h2>

### QP functions for basic arithmetic operations

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_addq2_u05(Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_quadx2 Sleef_subq2_u05(Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_quadx2 Sleef_mulq2_u05(Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_quadx2 Sleef_divq2_u05(Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_quadx2 Sleef_negq2(Sleef_quadx2 a);

Sleef_quadx2 Sleef_addq2_u05advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_quadx2 Sleef_subq2_u05advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_quadx2 Sleef_mulq2_u05advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_quadx2 Sleef_divq2_u05advsimd(Sleef_quadx2 a, Sleef_quadx2 b);
Sleef_quadx2 Sleef_negq2_advsimd(Sleef_quadx2 a);

Sleef_svquad Sleef_addqx_u05sve(Sleef_svquad a, Sleef_svquad b);
Sleef_svquad Sleef_subqx_u05sve(Sleef_svquad a, Sleef_svquad b);
Sleef_svquad Sleef_mulqx_u05sve(Sleef_svquad a, Sleef_svquad b);
Sleef_svquad Sleef_divqx_u05sve(Sleef_svquad a, Sleef_svquad b);
Sleef_svquad Sleef_negqx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions of [the basic arithmetic
operations](../quad#basicArithmetic).

### Square root functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_sqrtq2_u05(Sleef_quadx2 a);
Sleef_quadx2 Sleef_sqrtq2_u05advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_sqrtqx_u05sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_sqrtq1_u05purec](../quad#sleef_sqrtq1_u05purec).

### Sine functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_sinq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_sinq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_sinqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_sinq1_u10purec](../quad#sleef_sinq1_u10purec).

### Cosine functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_cosq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_cosq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_cosqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_cosq1_u10purec](../quad#sleef_cosq1_u10purec).

### Tangent functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_tanq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_tanq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_tanqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_tanq1_u10purec](../quad#sleef_tanq1_u10purec).

### Arc sine functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_asinq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_asinq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_asinqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_asinq1_u10purec](../quad#sleef_asinq1_u10purec).

### Arc cosine functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_acosq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_acosq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_acosqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_acosq1_u10purec](../quad#sleef_acosq1_u10purec).

### Arc tangent functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_atanq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_atanq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_atanqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_atanq1_u10purec](../quad#sleef_atanq1_u10purec).

### Base-<i>e</i> exponential functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_expq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_expq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_expqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_expq1_u10purec](../quad#sleef_expq1_u10purec).

### Base-2 exponential functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_exp2q2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_exp2q2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_exp2qx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_exp2q1_u10purec](../quad#sleef_exp2q1_u10purec).

### Base-10 exponentail

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_exp10q2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_exp10q2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_exp10qx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_exp10q1_u10purec](../quad#sleef_exp10q1_u10purec).

### Base-<i>e</i> exponential functions minus 1

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_expm1q2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_expm1q2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_expm1qx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_expm1q1_u10purec](../quad#sleef_expm1q1_u10purec).

### Natural logarithmic functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_logq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_logq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_logqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_logq1_u10purec](../quad#sleef_logq1_u10purec).

### Base-2 logarithmic functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_log2q2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_log2q2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_log2qx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_log2q1_u10purec](../quad#sleef_log2q1_u10purec).

### Base-10 logarithmic functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_log10q2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_log10q2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_log10qx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_log10q1_u10purec](../quad#sleef_log10q1_u10purec).

### Logarithm of one plus argument

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_log1pq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_log1pq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_log1pqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_log1pq1_u10purec](../quad#sleef_log1pq1_u10purec).

### Power functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_powq2_u10(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_quadx2 Sleef_powq2_u10advsimd(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_svquad Sleef_powqx_u10sve(Sleef_svquad x, Sleef_svquad y);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_powq1_u10purec](../quad#sleef_powq1_u10purec).

### Hyperbolic sine functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_sinhq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_sinhq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_sinhqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_sinhq1_u10purec](../quad#sleef_sinhq1_u10purec).

### Hyperbolic cosine functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_coshq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_coshq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_coshqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_coshq1_u10purec](../quad#sleef_coshq1_u10purec).

### Hyperbolic tangent functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_tanhq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_tanhq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_tanhqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_tanhq1_u10purec](../quad#sleef_tanhq1_u10purec).

### Inverse hyperbolic sine functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_asinhq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_asinhq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_asinhqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_asinhq1_u10purec](../quad#sleef_asinhq1_u10purec).

### Inverse hyperbolic cosine functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_acoshq2_u10(Sleef_quadx2 a);
Sleef_quadx2 Sleef_acoshq2_u10advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_acoshqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_acoshq1_u10purec](../quad#sleef_acoshq1_u10purec).

### Inverse hyperbolic tangent functions

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_atanhq2_u10adv(Sleef_quadx2 a);
Sleef_quadx2 Sleef_atanhq2_u10advadvsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_atanhqx_u10sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_atanhq1_u10purec](../quad#sleef_atanhq1_u10purec).

### Round to integer towards zero

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_truncq2(Sleef_quadx2 a);
Sleef_quadx2 Sleef_truncq2_advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_truncqx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_truncq1_purec](../quad#sleef_truncq1_purec).

### Round to integer towards minus infinity

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_floorq2(Sleef_quadx2 a);
Sleef_quadx2 Sleef_floorq2_advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_floorqx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_floorq1_purec](../quad#sleef_floorq1_purec).

### Round to integer towards plus infinity

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_ceilq2(Sleef_quadx2 a);
Sleef_quadx2 Sleef_ceilq2_advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_ceilqx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_ceilq1_purec](../quad#sleef_ceilq1_purec).

### Round to integer away from zero

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_roundq2(Sleef_quadx2 a);
Sleef_quadx2 Sleef_roundq2_advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_roundqx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_roundq1_purec](../quad#sleef_roundq1_purec).

### Round to integer, ties round to even

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_rintq2(Sleef_quadx2 a);
Sleef_quadx2 Sleef_rintq2_advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_rintqx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_rintq1_purec](../quad#sleef_rintq1_purec).

### Absolute value

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_fabsq2(Sleef_quadx2 a);
Sleef_quadx2 Sleef_fabsq2_advsimd(Sleef_quadx2 a);
Sleef_svquad Sleef_fabsqx_sve(Sleef_svquad a);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_fabsq1_purec](../quad#sleef_fabsq1_purec).

### Copy sign of a number

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_copysignq2(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_quadx2 Sleef_copysignq2_advsimd(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_svquad Sleef_copysignqx_sve(Sleef_svquad x, Sleef_svquad y);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_copysignq1_purec](../quad#sleef_copysignq1_purec).

### Maximum of two numbers

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_fmaxq2(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_quadx2 Sleef_fmaxq2_advsimd(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_svquad Sleef_fmaxqx_sve(Sleef_svquad x, Sleef_svquad y);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_fmaxq1_purec](../quad#sleef_fmaxq1_purec).

### Minimum of two numbers

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_fminq2(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_quadx2 Sleef_fminq2_advsimd(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_svquad Sleef_fminqx_sve(Sleef_svquad x, Sleef_svquad y);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_fminq1_purec](../quad#sleef_fminq1_purec).

### Positive difference

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_fdimq2_u05(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_quadx2 Sleef_fdimq2_u05advsimd(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_svquad Sleef_fdimqx_u05sve(Sleef_svquad x, Sleef_svquad y);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_fdimq1_u05purec](../quad#sleef_fdimq1_u05purec).

### Floating point remainder

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_fmodq2(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_quadx2 Sleef_fmodq2_advsimd(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_svquad Sleef_fmodqx_sve(Sleef_svquad x, Sleef_svquad y);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_fmodq1_purec](../quad#sleef_fmodq1_purec).

### Floating point remainder

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_remainderq2(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_quadx2 Sleef_remainderq2_advsimd(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_svquad Sleef_remainderqx_sve(Sleef_svquad x, Sleef_svquad y);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_remainderq1_purec](../quad#sleef_remainderq1_purec).

### Split a number to fractional and integral components

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_frexpq2(Sleef_quadx2 x, int32x2_t * ptr);
Sleef_quadx2 Sleef_frexpq2_advsimd(Sleef_quadx2 x, int32x2_t * ptr);
Sleef_svquad Sleef_frexpqx_sve(Sleef_svquad x, svint32_t * ptr);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_frexpq1_purec](../quad#sleef_frexpq1_purec).

### Break a number into integral and fractional parts

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_modfq2(Sleef_quadx2 x, Sleef_quadx2 * ptr);
Sleef_quadx2 Sleef_modfq2_advsimd(Sleef_quadx2 x, Sleef_quadx2 * ptr);
Sleef_svquad Sleef_modfqx_sve(Sleef_svquad x, Sleef_svquad * ptr);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_modfq1_purec](../quad#sleef_modfq1_purec).

### 2D Euclidian distance

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_hypotq2_u05(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_quadx2 Sleef_hypotq2_u05advsimd(Sleef_quadx2 x, Sleef_quadx2 y);
Sleef_svquad Sleef_hypotqx_u05sve(Sleef_svquad x, Sleef_svquad y);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_hypotq1_u05purec](../quad#sleef_hypotq1_u05purec).

### Fused multiply and accumulate

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_fmaq2_u05(Sleef_quadx2 x, Sleef_quadx2 y, Sleef_quadx2 z);
Sleef_quadx2 Sleef_fmaq2_u05advsimd(Sleef_quadx2 x, Sleef_quadx2 y, Sleef_quadx2 z);
Sleef_svquad Sleef_fmaqx_u05sve(Sleef_svquad x, Sleef_svquad y, Sleef_svquad z);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_fmaq1_u05purec](../quad#sleef_fmaq1_u05purec).

### Multiply by integral power of 2

```c
#include <sleefquad.h>

Sleef_quadx2 Sleef_ldexpq2(Sleef_quadx2 x, int32x2_t e);
Sleef_quadx2 Sleef_ldexpq2_advsimd(Sleef_quadx2 x, int32x2_t e);
Sleef_svquad Sleef_ldexpqx_sve(Sleef_svquad x, svint32_t e);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_ldexpq1_purec](../quad#sleef_ldexpq1_purec).

### Integer exponent of an FP number

```c
#include <sleefquad.h>

int32x2_t Sleef_ilogbq2(Sleef_quadx2 x);
int32x2_t Sleef_ilogbq2_advsimd(Sleef_quadx2 x);
svint32_t Sleef_ilogbqx_sve(Sleef_svquad x);
```
Link with `-lsleefquad`.

These are the vectorized functions
of [Sleef_ilogbq1_purec](../quad#sleef_ilogbq1_purec).

