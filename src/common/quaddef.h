//   Copyright Naoki Shibata and contributors 2010 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if (defined(__SIZEOF_FLOAT128__) && __SIZEOF_FLOAT128__ == 16) || (defined(__linux__) && defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))) || (defined(__PPC64__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8)
#define SLEEF_FLOAT128_IS_IEEEQP
#endif

#if !defined(SLEEF_FLOAT128_IS_IEEEQP) && defined(__SIZEOF_LONG_DOUBLE__) && __SIZEOF_LONG_DOUBLE__ == 16 && (defined(__aarch64__) || defined(__zarch__))
#define SLEEF_LONGDOUBLE_IS_IEEEQP
#endif

#if !defined(Sleef_quad_DEFINED)
#define Sleef_quad_DEFINED
#if defined(SLEEF_FLOAT128_IS_IEEEQP)
typedef __float128 Sleef_quad;
#define SLEEF_QUAD_C(x) (x ## Q)
#elif defined(SLEEF_LONGDOUBLE_IS_IEEEQP)
typedef long double Sleef_quad;
#define SLEEF_QUAD_C(x) (x ## L)
#else
typedef struct { uint64_t x, y; } Sleef_quad;
#endif
#endif
