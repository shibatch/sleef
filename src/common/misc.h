//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671537767526745028724
#endif

#ifndef M_1_PIl
#define M_1_PIl 0.318309886183790671537767526745028724L
#endif

#ifndef M_2_PI
#define M_2_PI 0.636619772367581343075535053490057448
#endif

#ifndef M_2_PIl
#define M_2_PIl 0.636619772367581343075535053490057448L
#endif

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif

typedef long double longdouble;

#ifndef Sleef_double2_DEFINED
#define Sleef_double2_DEFINED
typedef struct {
  double x, y;
} Sleef_double2;
#endif

#ifndef Sleef_float2_DEFINED
#define Sleef_float2_DEFINED
typedef struct {
  float x, y;
} Sleef_float2;
#endif

#ifndef Sleef_longdouble2_DEFINED
#define Sleef_longdouble2_DEFINED
typedef struct {
  long double x, y;
} Sleef_longdouble2;
#endif

#if defined(ENABLEFLOAT128) && !defined(Sleef_quad2_DEFINED)
#define Sleef_quad2_DEFINED
typedef __float128 Sleef_quad;
typedef struct {
  __float128 x, y;
} Sleef_quad2;
#endif

//

#if defined (__GNUC__) || defined (__clang__) || defined(__INTEL_COMPILER)

#define INLINE __attribute__((always_inline))

#if defined(__MINGW32__) || defined(__MINGW64__) || defined(__CYGWIN__)
#define EXPORT __stdcall __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#ifdef INFINITY
#undef INFINITY
#endif

#ifdef NAN
#undef NAN
#endif

#define NAN __builtin_nan("")
#define NANf __builtin_nanf("")
#define NANl __builtin_nanl("")
#define INFINITY __builtin_inf()
#define INFINITYf __builtin_inff()
#define INFINITYl __builtin_infl()

#if defined(__INTEL_COMPILER)
#define INFINITYq __builtin_inf()
#define NANq __builtin_nan("")
#else
#define INFINITYq __builtin_infq()
#define NANq (INFINITYq - INFINITYq)
#endif

#elif defined(_MSC_VER)

#include <math.h>
#include <float.h>

#define INLINE __inline
#define EXPORT __declspec(dllexport)

#define INFINITYf ((float)INFINITY)
#define NANf ((float)NAN)
#define INFINITYl ((long double)INFINITY)
#define NANl ((long double)NAN)

#if (defined(_M_AMD64) || defined(_M_X64))
#ifndef __SSE2__
#define __SSE2__
#define __SSE3__
#define __SSE4_1__
#endif
#elif _M_IX86_FP == 2
#ifndef __SSE2__
#define __SSE2__
#define __SSE3__
#define __SSE4_1__
#endif
#elif _M_IX86_FP == 1
#ifndef __SSE__
#define __SSE__
#endif
#endif

#endif // defined(_MSC_VER)
