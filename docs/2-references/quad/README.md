---
layout: default
title: Quadruple Precision
parent: References
has_children: true
permalink: /2-references/quad/
---

<h1>Quadruple Precision Math Library reference</h1>

<h2>Table of contents</h2>

* [Introduction](#introduction)
* [Data types](#datatypes)
* [Convenience macros and constants](#macro)
* [Conversion and output functions](#conversion)
* [Comparison functions](#comparison)
* [Math functions](#mathfunctions)
* [Tutorial](#tutorial)

<h2 id="introduction">Introduction</h2>

As of version 3.6, SLEEF includes a quad-precision math library. This library
includes various functions for computing with [IEEE 754 quadruple-precision(QP)
floating-point
numbers](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format).
If the compiler natively supports IEEE 754 QP FP type, numbers in this natively
supported QP data type can be directly passed to the library functions.
Otherwise, the library defines a data type for retaining numbers in IEEE 754 QP
data type.

Many of the functions are fully vectorized to maximize the throughput of
computation. For these functions, two vector registers are combined to retain a
vector of QP FP numbers, where one of these registers holds the upper part of
QP numbers, and the other register holds the lower part. Dedicated functions
are provided to load to or store from variables in vector QP data types.

<h2 id="datatypes">Data types</h2>

### `Sleef_quad`

`Sleef_quad` is a data type for retaining a single QP FP number. If the
compiler natively supports IEEE 754 quad precision FP type, then `Sleef_quad`
is an alias of that data type.

```c
typedef __float128 Sleef_quad;
typedef long double Sleef_quad;    // On AArch64 and System/390 systems
```

Otherwise, the library defines a 128-bit data type for retaining a number in
IEEE 754 QP data format.

<h2 id="macro">Convenience function, macro and constants</h2>

### `sleef_q`

Function to define any QP FP constant

```c
#include <sleefquad.h>

Sleef_quad sleef_q (int64_t upper, uint64_t lower, int e);
```
This is a function for defining any QP FP constant. This function can be used
to represent a QP FP constant on compilers without support for a QP FP data
type. This function is small and thus the code is likely be optimized so that
the immediate QP FP value is available at runtime. You can convert any QP FP
value to this representation using `qutil` utility program.

### `SLEEF_QUAD_C`

Macro for appending the correct suffix to a QP FP literal

```c
#include <sleefquad.h>

Sleef_quad SLEEF_QUAD_C (literal);
```
This is a macro for appending the correct suffix to a QP FP literal. The data
type for representing a QP FP number is not available on all systems, and the
data type differs between architectures. This macro can be used to append the
correct suffix regardless of the architecture. This macro is defined only if
the compiler supports a data type for representing a QP FP number.

Below is the table of QP constants defined in sleefquad.h. These are also
defined in the inline headers. These constants are defined as QP literals if
the compiler supports them. Otherwise, they are defined with `sleef_q` function.

<table style="text-align:center;" align="center">
<tr align="center">
<td>
<table class="lt">
<tr>
<td class="lt-hl"></td>
<td class="lt-hl"></td>
<td class="lt-hl"></td>
</tr>
<tr>
<td class="lt-br">Symbol</td>
<td class="lt-br">Meaning</td>
<td class="lt-bl">Value</td>
</tr>
<tr>
<td class="lt-hl"></td>
<td class="lt-hl"></td>
<td class="lt-hl"></td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_Eq</td>
<td class="lt-r" align="left">The base `e` of natural logarithm</td>
<td class="lt-l" align="left">2.7182818284590452353...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_LOG2Eq</td>
<td class="lt-r" align="left">The logarithm to base 2 of `e`</td>
<td class="lt-l" align="left">1.4426950408889634073...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_LOG10Eq</td>
<td class="lt-r" align="left">The logarithm to base 10 of `e`</td>
<td class="lt-l" align="left">0.4342944819032518276...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_LN2q</td>
<td class="lt-r" align="left">The natural logarithm of 2</td>
<td class="lt-l" align="left">0.6931471805599453094...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_LN10q</td>
<td class="lt-r" align="left">The natural logarithm of 10</td>
<td class="lt-l" align="left">2.3025850929940456840...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_PIq</td>
<td class="lt-r" align="left">&pi;, the ratio of a circle’s circumference to its diameter</td>
<td class="lt-l" align="left">3.1415926535897932384...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_PI_2q</td>
<td class="lt-r" align="left">&pi; divided by two</td>
<td class="lt-l" align="left">1.5707963267948966192...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_PI_4q</td>
<td class="lt-r" align="left">&pi; divided by four</td>
<td class="lt-l" align="left">0.7853981633974483096...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_1_PIq</td>
<td class="lt-r" align="left">1/&pi;, the receiprocal of &pi;</td>
<td class="lt-l" align="left">0.3183098861837906715...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_2_PIq</td>
<td class="lt-r" align="left">2/&pi;, two times the receiprocal of &pi;</td>
<td class="lt-l" align="left">0.6366197723675813430...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_2_SQRTPIq</td>
<td class="lt-r" align="left">Two times the reciprocal of the square root of &pi;</td>
<td class="lt-l" align="left">1.1283791670955125738...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_SQRT2q</td>
<td class="lt-r" align="left">The square root of two</td>
<td class="lt-l" align="left">1.4142135623730950488...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_SQRT3q</td>
<td class="lt-r" align="left">The square root of three</td>
<td class="lt-l" align="left">1.7320508075688772935...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_INV_SQRT3q</td>
<td class="lt-r" align="left">The reciprocal of the square root of three</td>
<td class="lt-l" align="left">0.5773502691896257645...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_SQRT1_2q</td>
<td class="lt-r" align="left">The reciprocal of the square root of two</td>
<td class="lt-l" align="left">0.7071067811865475244...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_INV_SQRTPIq</td>
<td class="lt-r" align="left">The reciprocal of the square root of &pi;</td>
<td class="lt-l" align="left">0.5641895835477562869...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_EGAMMAq</td>
<td class="lt-r" align="left">The Euler-Mascheroni constant</td>
<td class="lt-l" align="left">0.5772156649015328606...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_M_PHIq</td>
<td class="lt-r" align="left">The golden ratio constant</td>
<td class="lt-l" align="left">1.6180339887498948482...</td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_QUAD_MAX</td>
<td class="lt-r" align="left">The largest finite QP FP number</td>
<td class="lt-l" align="left">2<sup>16383</sup> &times; (2 - 2<sup>-112</sup>) = 1.18973... &times; 10<sup>4932</sup></td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_QUAD_MIN</td>
<td class="lt-r" align="left">The smallest positive normal QP FP number</td>
<td class="lt-l" align="left">2<sup>-16382</sup> &times; (1 - 2<sup>-112</sup>) = 3.36210... &times; 10<sup>-4932</sup></td>
</tr>
<tr>
<td class="lt-r" align="left">SLEEF_QUAD_EPSILON</td>
<td class="lt-r" align="left">The difference between 1 and the least QP FP number greater than 1</td>
<td class="lt-l" align="left">2<sup>-112</sup> = 1.92592... &times; 10<sup>-34</sup></td>
</tr>
<tr>
<td class="lt-br" align="left">SLEEF_QUAD_DENORM_MIN</td>
<td class="lt-br" align="left">The smallest positive QP FP number</td>
<td class="lt-bl" align="left">2<sup>-16494</sup> = 6.47517... &times; 10<sup>-4966</sup></td>
</tr>
</table>
</td>
</tr>
</table>

<h2 id="conversion">Conversion and output functions</h2>

### `Sleef_strtoq`

convert ASCII string to QP FP number

```c
#include <sleefquad.h>

Sleef_quad Sleef_strtoq(const char * str, char ** endptr);
```
Link with `-lsleefquad`.

This is a QP version of the strtod function. It converts the given string to a
QP FP number. It supports converting from both decimal and hexadecimal numbers.

### `Sleef_vprintf`
### `Sleef_vfprintf`
### `Sleef_vsnprintf`

Format various data into string and output

```c
#include <sleefquad.h>

int Sleef_printf(const char * fmt, ... );
int Sleef_fprintf(FILE * fp, const char * fmt, ... );
int Sleef_snprintf(char * str, size_t size, const char * fmt, ... );
int Sleef_vprintf(const char * fmt, va_list ap);
int Sleef_vfprintf(FILE * fp, const char * fmt, va_list ap);
int Sleef_vsnprintf(char * str, size_t size, const char * fmt, va_list ap);
```
Link with `-lsleefquad`.

These functions are equivalent to the corresponding printf functions in the C
standard, except for the following extensions. These functions support
converting from QP FP numbers using `Q` and P` modifiers with `a`, e`, f` and
g` conversions. With `Q` modifier, an immediate QP FP value can be passed to
these functions, while a QP FP value can be passed via a pointer with `P`
modifier. These functions only supports the format strings defined in the C
standard. Beware of [this bug](https://bugs.llvm.org/show_bug.cgi?id=47665)
when you use this function with clang. These functions are neither thread safe
nor atomic.

### `Sleef_unregisterPrintfHook`

Register and unregister C library hooks to printf family funcions in the GNU C Library

```c
#include <sleefquad.h>

void Sleef_registerPrintfHook( void );
void Sleef_unregisterPrintfHook( void );
```
Link with `-lsleefquad`.

These functions are defined only on systems with Glibc. Glibc has a
functionality for installing C library hooks to extend conversion capability of
its printf-familiy functions. `Sleef_registerPrintfHook` installs C library
hooks to add support for the `Q` and `P` modifiers.
`Sleef_unregisterPrintfHook` uninstalls the C library hooks.

### `Sleef_cast_to_doubleq1`
### `Sleef_cast_to_doubleq1_purec`
### `Sleef_cast_to_doubleq1_purecfma`

Convert QP number to double-precision number

```c
#include <sleefquad.h>

double Sleef_cast_to_doubleq1(Sleef_quad a);
double Sleef_cast_to_doubleq1_purec(Sleef_quad a);
double Sleef_cast_to_doubleq1_purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions convert a QP FP value to a double-precision value.

### `Sleef_cast_from_doubleq1`
### `Sleef_cast_from_doubleq1_purec`
### `Sleef_cast_from_doubleq1_purecfma`

Convert double-precision number to QP number

```c
#include <sleefquad.h>

Sleef_quad Sleef_cast_from_doubleq1(double a);
Sleef_quad Sleef_cast_from_doubleq1_purec(double a);
Sleef_quad Sleef_cast_from_doubleq1_purecfma(double a);
```
Link with `-lsleefquad`.

These functions convert a double-precision value to a QP FP value.

### `Sleef_cast_to_int64q1`
### `Sleef_cast_to_int64q1_purec`
### `Sleef_cast_to_int64q1_purecfma`

Convert QP number to 64-bit signed integer

```c
#include <sleefquad.h>

int64_t Sleef_cast_to_int64q1(Sleef_quad a);
int64_t Sleef_cast_to_int64q1_purec(Sleef_quad a);
int64_t Sleef_cast_to_int64q1_purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions convert a QP FP value to a 64-bit signed integer.

### `Sleef_cast_from_int64q1`
### `Sleef_cast_from_int64q1_purec`
### `Sleef_cast_from_int64q1_purecfma`

Convert 64-bit signed integer to QP number

```c
#include <sleefquad.h>

Sleef_quad Sleef_cast_from_int64q1(int64_t a);
Sleef_quad Sleef_cast_from_int64q1_purec(int64_t a);
Sleef_quad Sleef_cast_from_int64q1_purecfma(int64_t a);
```
Link with `-lsleefquad`.

These functions convert a 64-bit signed integer to a QP FP value.

### `Sleef_cast_to_uint64q1`
### `Sleef_cast_to_uint64q1_purec`
### `Sleef_cast_to_uint64q1_purecfma`

Convert QP number to 64-bit unsigned integer

```c
#include <sleefquad.h>

uint64_t Sleef_cast_to_uint64q1(Sleef_quad a);
uint64_t Sleef_cast_to_uint64q1_purec(Sleef_quad a);
uint64_t Sleef_cast_to_uint64q1_purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions convert a QP FP value to a 64-bit signed integer.

### `Sleef_cast_from_uint64q1`
### `Sleef_cast_from_uint64q1_purec`
### `Sleef_cast_from_uint64q1_purecfma`

Convert 64-bit unsigned integer to QP number

```c
#include <sleefquad.h>

Sleef_quad Sleef_cast_from_uint64q1(uint64_t a);
Sleef_quad Sleef_cast_from_uint64q1_purec(uint64_t a);
Sleef_quad Sleef_cast_from_uint64q1_purecfma(uint64_t a);
```
Link with `-lsleefquad`.

These functions convert a 64-bit unsigned integer to a QP FP value.

<h2 id="comparison">Comparison functions</h2>

```c
#include <sleefquad.h>

int32_t Sleef_icmpltq1(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpleq1(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpgtq1(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpgeq1(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpeqq1(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpneq1(Sleef_quad a, Sleef_quad b);

int32_t Sleef_icmpltq1_purec(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpleq1_purec(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpgtq1_purec(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpgeq1_purec(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpeqq1_purec(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpneq1_purec(Sleef_quad a, Sleef_quad b);

int32_t Sleef_icmpltq1_purecfma(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpleq1_purecfma(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpgtq1_purecfma(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpgeq1_purecfma(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpeqq1_purecfma(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpneq1_purecfma(Sleef_quad a, Sleef_quad b);
```
Link with `-lsleefquad`.

These functions compare two QP FP values. The `lt`, `le`, `gt`, `ge`, `eq` and
`ne` functions return 1 if and only if `a` is less than, less or equal,
greather than, greater or equal, equal and not equal to `b`, respectively.
Otherwise, 0 is returned.

### `Sleef_icmpq1`
### `Sleef_icmpq1_purec`
### `Sleef_icmpq1_purecfma`

QP comparison

```c
#include <sleefquad.h>

int32_t Sleef_icmpq1(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpq1_purec(Sleef_quad a, Sleef_quad b);
int32_t Sleef_icmpq1_purecfma(Sleef_quad a, Sleef_quad b);
```
Link with `-lsleefquad`.

These functions compare two QP FP values. If `a` is
greater than `b`, they return 1. If `a` is less than `b`, they
return -1. Otherwise, they return 0. If either argument is NaN, they
return 0.

### `Sleef_iunordq1`
### `Sleef_iunordq1_purec`
### `Sleef_iunordq1_purecfma`

Check orderedness

```c
#include <sleefquad.h>

int32_t Sleef_iunordq1(Sleef_quad a, Sleef_quad b);
int32_t Sleef_iunordq1_purec(Sleef_quad a, Sleef_quad b);
int32_t Sleef_iunordq1_purecfma(Sleef_quad a, Sleef_quad b);
```
Link with `-lsleefquad`.

These functions return 1 if either argument is NaN, otherwise 0.

<h2 id="mathfunctions">Math functions</h2>

### Basic Arithmetic

```c
#include <sleefquad.h>

Sleef_quad Sleef_addq1_u05(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_subq1_u05(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_mulq1_u05(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_divq1_u05(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_negq1(Sleef_quad a);

Sleef_quad Sleef_addq1_u05purec(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_subq1_u05purec(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_mulq1_u05purec(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_divq1_u05purec(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_negq1_purec(Sleef_quad a);

Sleef_quad Sleef_addq1_u05purecfma(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_subq1_u05purecfma(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_mulq1_u05purecfma(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_divq1_u05purecfma(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_negq1_purec(Sleef_quad a);
```
Link with `-lsleefquad`.

The add, sub, mul and div functions perform addition, subtraction,
multiplication and division of two QP FP values. The error bound of these
functions are 0.5000000001 ULP. The neg functions return `-a`.

### `Sleef_sqrtq1_u05`
### `Sleef_sqrtq1_u05purec`
### `Sleef_sqrtq1_u05purecfma`

square root functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_sqrtq1_u05(Sleef_quad a);
Sleef_quad Sleef_sqrtq1_u05purec(Sleef_quad a);
Sleef_quad Sleef_sqrtq1_u05purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return nonnegative square root of a value in `a`. The error
bound of the returned value is 0.5000000001 ULP.

### `Sleef_sinq1_u10`
### `Sleef_sinq1_u10purec`
### `Sleef_sinq1_u10purecfma`

sine functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_sinq1_u10(Sleef_quad a);
Sleef_quad Sleef_sinq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_sinq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the sine function of a value in `a`. The error bound
of the returned value is 1.0 ULP.

### `Sleef_cosq1_u10`
### `Sleef_cosq1_u10purec`
### `Sleef_cosq1_u10purecfma`

cosine functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_cosq1_u10(Sleef_quad a);
Sleef_quad Sleef_cosq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_cosq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the cosine function of a value in `a`. The error bound
of the returned value is 1.0 ULP.

### `Sleef_tanq1_u10`
### `Sleef_tanq1_u10purec`
### `Sleef_tanq1_u10purecfma`

tangent functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_tanq1_u10(Sleef_quad a);
Sleef_quad Sleef_tanq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_tanq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the tangent function of a value in `a`. The error
bound of the returned value is 1.0 ULP.

### `Sleef_asinq1_u10`
### `Sleef_asinq1_u10purec`
### `Sleef_asinq1_u10purecfma`

arc sine functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_asinq1_u10(Sleef_quad a);
Sleef_quad Sleef_asinq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_asinq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the arc sine function of a value in `a`. The error
bound of the returned value is 1.0 ULP.

### `Sleef_acosq1_u10`
### `Sleef_acosq1_u10purec`
### `Sleef_acosq1_u10purecfma`

arc cosine functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_acosq1_u10(Sleef_quad a);
Sleef_quad Sleef_acosq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_acosq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the arc cosine function of a value in `a`. The error
bound of the returned value is 1.0 ULP.

### `Sleef_atanq1_u10`
### `Sleef_atanq1_u10purec`
### `Sleef_atanq1_u10purecfma`

arc tangent functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_atanq1_u10(Sleef_quad a);
Sleef_quad Sleef_atanq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_atanq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the arc tangent function of a value in `a`. The error
bound of the returned value is 1.0 ULP.

### `Sleef_atan2q1_u10`
### `Sleef_atan2q1_u10purec`
### `Sleef_atan2q1_u10purecfma`

arc tangent functions of two variables

```c
#include <sleefquad.h>

Sleef_quad Sleef_atan2q1_u10(Sleef_quad y, Sleef_quad x);
Sleef_quad Sleef_atan2q1_u10purec(Sleef_quad y, Sleef_quad x);
Sleef_quad Sleef_atan2q1_u10purecfma(Sleef_quad y, Sleef_quad x);
```
Link with `-lsleefquad`.

These functions evaluate the arc tangent function of ( `y / x` ).  The quadrant
of the result is determined according to the signs of `x` and `y`. The error
bound of the returned value is 1.0 ULP.

### `Sleef_expq1_u10`
### `Sleef_expq1_u10purec`
### `Sleef_expq1_u10purecfma`

base-`e` exponential functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_expq1_u10(Sleef_quad a);
Sleef_quad Sleef_expq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_expq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return the value of `e` raised to `a`. The error bound of the
returned value is 1.0 ULP.

### `Sleef_exp2q1_u10`
### `Sleef_exp2q1_u10purec`
### `Sleef_exp2q1_u10purecfma`

base-2 exponential functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_exp2q1_u10(Sleef_quad a);
Sleef_quad Sleef_exp2q1_u10purec(Sleef_quad a);
Sleef_quad Sleef_exp2q1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return 2 raised to `a`. The error bound of the returned value
is 1.0 ULP.

### `Sleef_exp10q1_u10`
### `Sleef_exp10q1_u10purec`
### `Sleef_exp10q1_u10purecfma`

base-10 exponentail

```c
#include <sleefquad.h>

Sleef_quad Sleef_exp10q1_u10(Sleef_quad a);
Sleef_quad Sleef_exp10q1_u10purec(Sleef_quad a);
Sleef_quad Sleef_exp10q1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return 10 raised to `a`. The error bound of the returned value
is 1.0 ULP.

### `Sleef_expm1q1_u10`
### `Sleef_expm1q1_u10purec`
### `Sleef_expm1q1_u10purecfma`

base-`e` exponential functions minus 1

```c
#include <sleefquad.h>

Sleef_quad Sleef_expm1q1_u10(Sleef_quad a);
Sleef_quad Sleef_expm1q1_u10purec(Sleef_quad a);
Sleef_quad Sleef_expm1q1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return the value one less than `e` raised to `a`. The error
bound of the returned value is 1.0 ULP.

### `Sleef_logq1_u10`
### `Sleef_logq1_u10purec`
### `Sleef_logq1_u10purecfma`

natural logarithmic functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_logq1_u10(Sleef_quad a);
Sleef_quad Sleef_logq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_logq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return the natural logarithm of `a`.  The error bound of the
returned value is 1.0 ULP.

### `Sleef_log2q1_u10`
### `Sleef_log2q1_u10purec`
### `Sleef_log2q1_u10purecfma`

base-2 logarithmic functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_log2q1_u10(Sleef_quad a);
Sleef_quad Sleef_log2q1_u10purec(Sleef_quad a);
Sleef_quad Sleef_log2q1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return the base-2 logarithm of `a`.  The error bound of the
returned value is 1.0 ULP.

### `Sleef_log10q1_u10`
### `Sleef_log10q1_u10purec`
### `Sleef_log10q1_u10purecfma`

base-10 logarithmic functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_log10q1_u10(Sleef_quad a);
Sleef_quad Sleef_log10q1_u10purec(Sleef_quad a);
Sleef_quad Sleef_log10q1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return the base-10 logarithm of `a`.  The error bound of the
returned value is 1.0 ULP.

### `Sleef_log1pq1_u10`
### `Sleef_log1pq1_u10purec`
### `Sleef_log1pq1_u10purecfma`

logarithm of one plus argument

```c
#include <sleefquad.h>

Sleef_quad Sleef_log1pq1_u10(Sleef_quad a);
Sleef_quad Sleef_log1pq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_log1pq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return the natural logarithm of `(1 + a)`.  The error bound
of the returned value is 1.0 ULP.

### `Sleef_powq1_u10`
### `Sleef_powq1_u10purec`
### `Sleef_powq1_u10purecfma`

power functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_powq1_u10(Sleef_quad x, Sleef_quad y);
Sleef_quad Sleef_powq1_u10purec(Sleef_quad x, Sleef_quad y);
Sleef_quad Sleef_powq1_u10purecfma(Sleef_quad x, Sleef_quad y);
```
Link with `-lsleefquad`.

These functions return the value of `x` raised to the power
of `y`. The error bound of the returned value is 1.0 ULP.

### `Sleef_sinhq1_u10`
### `Sleef_sinhq1_u10purec`
### `Sleef_sinhq1_u10purecfma`

hyperbolic sine functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_sinhq1_u10(Sleef_quad a);
Sleef_quad Sleef_sinhq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_sinhq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the hyperbolic sine function of a value in `a`. The
error bound of the returned value is 1.0 ULP.

### `Sleef_coshq1_u10`
### `Sleef_coshq1_u10purec`
### `Sleef_coshq1_u10purecfma`

hyperbolic cosine functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_coshq1_u10(Sleef_quad a);
Sleef_quad Sleef_coshq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_coshq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the hyperbolic cosine function of a value in `a`. The
error bound of the returned value is 1.0 ULP.

### `Sleef_tanhq1_u10`
### `Sleef_tanhq1_u10purec`
### `Sleef_tanhq1_u10purecfma`

hyperbolic tangent functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_tanhq1_u10(Sleef_quad a);
Sleef_quad Sleef_tanhq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_tanhq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the hyperbolic tangent function of a value in `a`. The
error bound of the returned value is 1.0 ULP.

### `Sleef_asinhq1_u10`
### `Sleef_asinhq1_u10purec`
### `Sleef_asinhq1_u10purecfma`

inverse hyperbolic sine functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_asinhq1_u10(Sleef_quad a);
Sleef_quad Sleef_asinhq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_asinhq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the inverse hyperbolic sine function of a value in
`a`. The error bound of the returned value is 1.0 ULP.

### `Sleef_acoshq1_u10`
### `Sleef_acoshq1_u10purec`
### `Sleef_acoshq1_u10purecfma`

inverse hyperbolic cosine functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_acoshq1_u10(Sleef_quad a);
Sleef_quad Sleef_acoshq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_acoshq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the inverse hyperbolic cosine function of a value in
`a`. The error bound of the returned value is 1.0 ULP.

### `Sleef_atanhq1_u10`
### `Sleef_atanhq1_u10purec`
### `Sleef_atanhq1_u10purecfma`

inverse hyperbolic tangent functions

```c
#include <sleefquad.h>

Sleef_quad Sleef_atanhq1_u10(Sleef_quad a);
Sleef_quad Sleef_atanhq1_u10purec(Sleef_quad a);
Sleef_quad Sleef_atanhq1_u10purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions evaluate the inverse hyperbolic tangent function of a value in
`a`. The error bound of the returned value is 1.0 ULP.

### `Sleef_truncq1`
### `Sleef_truncq1_purec`
### `Sleef_truncq1_purecfma`

round to integer towards zero

```c
#include <sleefquad.h>

Sleef_quad Sleef_truncq1(Sleef_quad a);
Sleef_quad Sleef_truncq1_purec(Sleef_quad a);
Sleef_quad Sleef_truncq1_purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions round their argument to the integer value, in floating
format, nearest to but no larger in magnitude than the argument.

### `Sleef_floorq1`
### `Sleef_floorq1_purec`
### `Sleef_floorq1_purecfma`

round to integer towards minus infinity

```c
#include <sleefquad.h>

Sleef_quad Sleef_floorq1(Sleef_quad a);
Sleef_quad Sleef_floorq1_purec(Sleef_quad a);
Sleef_quad Sleef_floorq1_purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return the largest integer value not greater than `a`.

### `Sleef_ceilq1`
### `Sleef_ceilq1_purec`
### `Sleef_ceilq1_purecfma`

round to integer towards plus infinity

```c
#include <sleefquad.h>

Sleef_quad Sleef_ceilq1(Sleef_quad a);
Sleef_quad Sleef_ceilq1_purec(Sleef_quad a);
Sleef_quad Sleef_ceilq1_purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return the smallest integer value not less than `a`.

### `Sleef_roundq1`
### `Sleef_roundq1_purec`
### `Sleef_roundq1_purecfma`

round to integer away from zero

```c
#include <sleefquad.h>

Sleef_quad Sleef_roundq1(Sleef_quad a);
Sleef_quad Sleef_roundq1_purec(Sleef_quad a);
Sleef_quad Sleef_roundq1_purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions round their argument to the nearest integer value, rounding
halfway cases away from zero.

### `Sleef_rintq1`
### `Sleef_rintq1_purec`
### `Sleef_rintq1_purecfma`

round to integer, ties round to even

```c
#include <sleefquad.h>

Sleef_quad Sleef_rintq1(Sleef_quad a);
Sleef_quad Sleef_rintq1_purec(Sleef_quad a);
Sleef_quad Sleef_rintq1_purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions round their argument to an integer value with the round-to-even
method.

### `Sleef_fabsq1`
### `Sleef_fabsq1_purec`
### `Sleef_fabsq1_purecfma`

absolute value

```c
#include <sleefquad.h>

Sleef_quad Sleef_fabsq1(Sleef_quad a);
Sleef_quad Sleef_fabsq1_purec(Sleef_quad a);
Sleef_quad Sleef_fabsq1_purecfma(Sleef_quad a);
```
Link with `-lsleefquad`.

These functions return the absolute value of `a`.

### `Sleef_copysignq1`
### `Sleef_copysignq1_purec`
### `Sleef_copysignq1_purecfma`

copy sign of a number

```c
#include <sleefquad.h>

Sleef_quad Sleef_copysignq1(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_copysignq1_purec(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_copysignq1_purecfma(Sleef_quad a, Sleef_quad b);
```
Link with `-lsleefquad`.

These functions return a value with the magnitude of `a` and the sign of `b`.

### `Sleef_fmaxq1`
### `Sleef_fmaxq1_purec`
### `Sleef_fmaxq1_purecfma`

maximum of two numbers

```c
#include <sleefquad.h>

Sleef_quad Sleef_fmaxq1(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_fmaxq1_purec(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_fmaxq1_purecfma(Sleef_quad a, Sleef_quad b);
```
Link with `-lsleefquad`.

These functions return the larger value of `a` and `b`.

### `Sleef_fminq1`
### `Sleef_fminq1_purec`
### `Sleef_fminq1_purecfma`

minimum of two numbers

```c
#include <sleefquad.h>

Sleef_quad Sleef_fminq1(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_fminq1_purec(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_fminq1_purecfma(Sleef_quad a, Sleef_quad b);
```
Link with `-lsleefquad`.

These functions return the smaller value of `a` and `b`.

### `Sleef_fdimq1_u05`
### `Sleef_fdimq1_u05purec`
### `Sleef_fdimq1_u05purecfma`

positive difference

```c
#include <sleefquad.h>

Sleef_quad Sleef_fdimq1_u05(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_fdimq1_u05purec(Sleef_quad a, Sleef_quad b);
Sleef_quad Sleef_fdimq1_u05purecfma(Sleef_quad a, Sleef_quad b);
```
Link with `-lsleefquad`.

These functions return the positive difference between `a` and `b`. The error
bound of these functions are 0.5000000001 ULP.

### `Sleef_fmodq1`
### `Sleef_fmodq1_purec`
### `Sleef_fmodq1_purecfma`

floating point remainder

```c
#include <sleefquad.h>

Sleef_quad Sleef_fmodq1(Sleef_quad x, Sleef_quad y);
Sleef_quad Sleef_fmodq1_purec(Sleef_quad x, Sleef_quad y);
Sleef_quad Sleef_fmodq1_purecfma(Sleef_quad x, Sleef_quad y);
```
Link with `-lsleefquad`.

These functions return `x - n * y`, where `n` is the quotient of `x / y`,
rounded toward zero to an integer.

### `Sleef_remainderq1`
### `Sleef_remainderq1_purec`
### `Sleef_remainderq1_purecfma`

floatint point remainder

```c
#include <sleefquad.h>

Sleef_quad Sleef_remainderq1(Sleef_quad x, Sleef_quad y);
Sleef_quad Sleef_remainderq1_purec(Sleef_quad x, Sleef_quad y);
Sleef_quad Sleef_remainderq1_purecfma(Sleef_quad x, Sleef_quad y);
```
Link with `-lsleefquad`.

These functions return `x - n * y`, where `n` is the quotient of `x / y`,
rounded to an integer with the round-to-even method.

### `Sleef_frexpq1`
### `Sleef_frexpq1_purec`
### `Sleef_frexpq1_purecfma`

split a number to fractional and integral components

```c
#include <sleefquad.h>

Sleef_quad Sleef_frexpq1(Sleef_quad a, int * ptr);
Sleef_quad Sleef_frexpq1_purec(Sleef_quad a, int * ptr);
Sleef_quad Sleef_frexpq1_purecfma(Sleef_quad a, int * ptr);
```
Link with `-lsleefquad`.

These functions split the argument `a` into a fraction `f` and an exponent `e`,
where `0.5 <= f < 1` and `a = f * 2^e`. The computed exponent `e` will be
stored in `*ptr`, and the functions return `f`.

### `Sleef_modfq1`
### `Sleef_modfq1_purec`
### `Sleef_modfq1_purecfma`

break a number into integral and fractional parts

```c
#include <sleefquad.h>

Sleef_quad Sleef_modfq1(Sleef_quad a, Sleef_quad * ptr);
Sleef_quad Sleef_modfq1_purec(Sleef_quad a, Sleef_quad * ptr);
Sleef_quad Sleef_modfq1_purecfma(Sleef_quad a, Sleef_quad * ptr);
```
Link with `-lsleefquad`.

These functions split the argument `a` into an integral part `k` and a
fractional part `f`, where `k` and `f` have the same sign as `a`. `k` will be
stored in `*ptr`, and the functions return `f`.

### `Sleef_hypotq1_u05`
### `Sleef_hypotq1_u05purec`
### `Sleef_hypotq1_u05purecfma`

2D Euclidian distance

```c
#include <sleefquad.h>

Sleef_quad Sleef_hypotq1_u05(Sleef_quad x, Sleef_quad y);
Sleef_quad Sleef_hypotq1_u05purec(Sleef_quad x, Sleef_quad y);
Sleef_quad Sleef_hypotq1_u05purecfma(Sleef_quad x, Sleef_quad y);
```
Link with `-lsleefquad`.

These functions return the square root of the sum of the squares
of `x` and y`. The error bound of
these functions are 0.5000000001 ULP.

### `Sleef_fmaq1_u05`
### `Sleef_fmaq1_u05purec`
### `Sleef_fmaq1_u05purecfma`

fused multiply and accumulate

```c
#include <sleefquad.h>

Sleef_quad Sleef_fmaq1_u05(Sleef_quad x, Sleef_quad y, Sleef_quad z);
Sleef_quad Sleef_fmaq1_u05purec(Sleef_quad x, Sleef_quad y, Sleef_quad z);
Sleef_quad Sleef_fmaq1_u05purecfma(Sleef_quad x, Sleef_quad y, Sleef_quad z);
```
Link with `-lsleefquad`.

These functions return `x * y + z` with a single rounding. The error bound of
these functions are 0.5000000001 ULP.

### `Sleef_ldexpq1`
### `Sleef_ldexpq1_purec`
### `Sleef_ldexpq1_purecfma`

multiply by integral power of 2

```c
#include <sleefquad.h>

Sleef_quad Sleef_ldexpq1(Sleef_quad x, int e);
Sleef_quad Sleef_ldexpq1_purec(Sleef_quad x, int e);
Sleef_quad Sleef_ldexpq1_purecfma(Sleef_quad x, int e);
```
Link with `-lsleefquad`.

These functions return `x * 2^e`.

### `Sleef_ilogbq1`
### `Sleef_ilogbq1_purec`
### `Sleef_ilogbq1_purecfma`

integer exponent of an FP number

```c
#include <sleefquad.h>

int Sleef_ilogbq1(Sleef_quad x);
int Sleef_ilogbq1_purec(Sleef_quad x);
int Sleef_ilogbq1_purecfma(Sleef_quad x);
```
Link with `-lsleefquad`.

These functions return the exponent of `x`. If x` is zero, infinite and a NaN,
they return `SLEEF_FP_ILOGB0`, `INT_MAX` and `SLEEF_FP_ILOGBNAN`,
respectively.

<h2 id="tutorial">Tutorial</h2>

I would like to show an example of how the vectorized QP functions can be used.
Below is [a source code](../../src/machinx86.c) for computing &pi; with
[Machin's formula](https://en.wikipedia.org/wiki/Machin-like_formula) on x86
GNU systems. This formula has two terms with arc tangent which can be
independently calculated. In this example, these two terms are computed using
vector functions. `__float128` data type is defined on x86,
and you can use literals in this type to initialize the variables. Variables
`q0`, q1`, q2` and `q3` retain two QP FP values each. At line 5, 8 and 11, QP
values in arrays are loaded into these variables with
[Sleef_loadq2_sse2](x86#load) function. Vector operations are carried out from
line 14 to 16, and then each element of the vector variable `q3` is extracted
with [Sleef_getq2_sse2](x86#get) function and subtracted to obtain the result
at line 18. Note that the QP subtract function in the standard library is
called to do this subtraction.  This result is output to the console using
[Sleef_printf](#Sleef_printf) function at line 22. Note that this source code
can be built with clang, on which libquadmath is not available.

```c
#include <sleefquad.h>

int main(int argc, char **argv) {
__float128 a0[] = { 5, 239 };
Sleef_quadx2 q0 = Sleef_loadq2_sse2(a0);

__float128 a1[] = { 1, 1 };
Sleef_quadx2 q1 = Sleef_loadq2_sse2(a1);

__float128 a2[] = { 16, 4 };
Sleef_quadx2 q2 = Sleef_loadq2_sse2(a2);

Sleef_quadx2 q3;
q3 = Sleef_divq2_u05sse2(q1, q0);
q3 = Sleef_atanq2_u10sse2(q3);
q3 = Sleef_mulq2_u05sse2(q3, q2);

__float128 pi = Sleef_getq2_sse2(q3, 0) - Sleef_getq2_sse2(q3, 1);

Sleef_printf("%.40Pg\n", &pi);
}
```
<p style="text-align:center;">
Fig. 4.1: <a href="../../src/machinx86.c">Example source code for x86 computers</a>
</p>

`__float128` data type is not defined on MSVC, and thus we cannot use literals
of this type to initialize QP variables. Sleef provides various conversion
functions for this purpose. In [the following source
code](../../src/machinx86.c),
[Sleef_cast_from_doubleq1_purec](#Sleef_cast_from_doubleq1_purec) function is
used to initialize `q0` from line 4 to 5. [Sleef_strtoq](#Sleef_strtoq)
function is used to initialize `q1` at line 7.  Here,
[Sleef_splatq2_sse2](x86#splat) function is a function for setting the
specified QP FP value to all elements in a vector. Obtaining QP FP constants by
calling these functions may waste some CPU time. From line 9 and 12,
[sleef_q](#sleef_q) function is used to intialize `q2`. [sleef_q](#sleef_q) is
a function for defining a QP FP constant. You can convert any QP FP value to
this representation using `qutil` utility program. At line 20,
[Sleef_subq1_u05purec](#basic-arithmetic) function is used to subtract a scalar
value from another to obtain the result.

```c
#include <sleefquad.h>

int main(int argc, char **argv) {
Sleef_quad a0[] = { Sleef_cast_from_doubleq1_purec(5), Sleef_cast_from_doubleq1_purec(239) };
Sleef_quadx2 q0 = Sleef_loadq2_sse2(a0);

Sleef_quadx2 q1 = Sleef_splatq2_sse2(Sleef_strtoq("1.0", NULL));

Sleef_quadx2 q2 = Sleef_loadq2_sse2((Sleef_quad[]) {
sleef_q(+0x1000000000000LL, 0x0000000000000000ULL, 4), // 16.0
sleef_q(+0x1000000000000LL, 0x0000000000000000ULL, 2), // 4.0
});

Sleef_quadx2 q3;

q3 = Sleef_divq2_u05sse2(q1, q0);
q3 = Sleef_atanq2_u10sse2(q3);
q3 = Sleef_mulq2_u05sse2(q3, q2);

Sleef_quad pi = Sleef_subq1_u05purec(Sleef_getq2_sse2(q3, 0), Sleef_getq2_sse2(q3, 1));

Sleef_printf("%.40Pg\n", &pi);
}
```
<p style="text-align:center;">
Fig. 4.2: <a href="../../src/machinmsvc.c">Example source code for MSVC</a>
<p>

