//   Copyright Naoki Shibata and contributors 2010 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(SLEEF_GENHEADER)

#if (defined(__SIZEOF_FLOAT128__) && __SIZEOF_FLOAT128__ == 16) || (defined(__linux__) && defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))) || (defined(__PPC64__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8)
#define SLEEF_FLOAT128_IS_IEEEQP
#endif

#if !defined(SLEEF_FLOAT128_IS_IEEEQP) && defined(__SIZEOF_LONG_DOUBLE__) && __SIZEOF_LONG_DOUBLE__ == 16 && (defined(__aarch64__) || defined(__zarch__))
#define SLEEF_LONGDOUBLE_IS_IEEEQP
#endif

#if !defined(Sleef_quad_DEFINED)
#define Sleef_quad_DEFINED
typedef struct { uint64_t x, y; } Sleef_uint64_2t;
#if defined(SLEEF_FLOAT128_IS_IEEEQP)
typedef __float128 Sleef_quad;
#define SLEEF_QUAD_C(x) (x ## Q)
#elif defined(SLEEF_LONGDOUBLE_IS_IEEEQP)
typedef long double Sleef_quad;
#define SLEEF_QUAD_C(x) (x ## L)
#else
typedef Sleef_uint64_2t Sleef_quad;
#endif
#endif

#else // #if !defined(SLEEF_GENHEADER)

&if (defined(SLEEFXXX__SIZEOF_FLOAT128__) && SLEEFXXX__SIZEOF_FLOAT128__ == 16) || (defined(SLEEFXXX__linux__) && defined(SLEEFXXX__GNUC__) && (defined(SLEEFXXX__i386__) || defined(SLEEFXXX__x86_64__))) || (defined(SLEEFXXX__PPC64__) && defined(SLEEFXXX__GNUC__) && !defined(SLEEFXXX__clang__) && SLEEFXXX__GNUC__ >= 8)
&define SLEEFXXXSLEEF_FLOAT128_IS_IEEEQP
&endif

&if !defined(SLEEFXXXSLEEF_FLOAT128_IS_IEEEQP) && defined(SLEEFXXX__SIZEOF_LONG_DOUBLE__) && SLEEFXXX__SIZEOF_LONG_DOUBLE__ == 16 && (defined(SLEEFXXX__aarch64__) || defined(SLEEFXXX__zarch__))
&define SLEEFXXXSLEEF_LONGDOUBLE_IS_IEEEQP
&endif

&if !defined(SLEEFXXXSleef_quad_DEFINED)
&define SLEEFXXXSleef_quad_DEFINED
typedef struct { uint64_t x, y; } Sleef_uint64_2t;
&if defined(SLEEFXXXSLEEF_FLOAT128_IS_IEEEQP)
typedef __float128 Sleef_quad;
&define SLEEFXXXSLEEF_QUAD_C(x) (x ## Q)
&elif defined(SLEEFXXXSLEEF_LONGDOUBLE_IS_IEEEQP)
typedef long double Sleef_quad;
&define SLEEFXXXSLEEF_QUAD_C(x) (x ## L)
&else
typedef Sleef_uint64_2t Sleef_quad;
&endif
&endif

#endif // #if !defined(SLEEF_GENHEADER)
