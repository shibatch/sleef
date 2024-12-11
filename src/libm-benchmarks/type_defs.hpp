//   Copyright Naoki Shibata and contributors 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#pragma once

///////////////////////////////////
// Library Includes and ///////////
/////// Type Definitions //////////
///////////////////////////////////
template <typename T> const inline int vector_len = 1;
template <> const inline int vector_len<float> = 1;
template <> const inline int vector_len<double> = 1;

#if !defined(BENCH_LIBM)

#include <sleef.h>
#if defined(__i386__) || defined(__x86_64__)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#if !defined(ARCH_VECT_LEN) || ARCH_VECT_LEN == 128
#ifdef __SSE2__
typedef __m128d vdouble;
typedef __m128 vfloat;
typedef Sleef___m128d_2 vdouble2;
typedef Sleef___m128_2 vfloat2;
template <> const inline int vector_len<vfloat> = 4;
template <> const inline int vector_len<vdouble> = 2;
#define ENABLE_VECTOR_BENCHMARKS
#endif
// * Bigger precisions:
#elif ARCH_VECT_LEN == 256
#ifdef __AVX__
typedef __m256d vdouble;
typedef __m256 vfloat;
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
template <> const inline int vector_len<vfloat> = 8;
template <> const inline int vector_len<vdouble> = 4;
#define ENABLE_VECTOR_BENCHMARKS
#endif
#elif ARCH_VECT_LEN == 512
#ifdef __AVX512F__
typedef __m512d vdouble;
typedef __m512 vfloat;
typedef Sleef___m512d_2 vdouble2;
typedef Sleef___m512_2 vfloat2;
template <> const inline int vector_len<vfloat> = 16;
template <> const inline int vector_len<vdouble> = 8;
#define ENABLE_VECTOR_BENCHMARKS
#endif
#endif

#elif defined(__ARM_NEON)
#include <arm_neon.h>
typedef float64x2_t vdouble;
typedef float32x4_t vfloat;
typedef Sleef_float64x2_t_2 vdouble2;
typedef Sleef_float32x4_t_2 vfloat2;
template <> const inline int vector_len<vfloat> = 4;
template <> const inline int vector_len<vdouble> = 2;
#define ENABLE_VECTOR_BENCHMARKS

#elif defined(__VSX__)
#include <altivec.h>
typedef __vector double vdouble;
typedef __vector float vfloat;
typedef Sleef_SLEEF_VECTOR_DOUBLE_2 vdouble2;
typedef Sleef_SLEEF_VECTOR_FLOAT_2 vfloat2;
template <> const inline int vector_len<vfloat> = 4;
template <> const inline int vector_len<vdouble> = 2;
#define ENABLE_VECTOR_BENCHMARKS

#elif defined(__VX__)
#include <vecintrin.h>
typedef __vector double vdouble;
typedef __vector float vfloat;
typedef Sleef_SLEEF_VECTOR_DOUBLE_2 vdouble2;
typedef Sleef_SLEEF_VECTOR_FLOAT_2 vfloat2;
template <> const inline int vector_len<vfloat> = 4;
template <> const inline int vector_len<vdouble> = 2;
#define ENABLE_VECTOR_BENCHMARKS
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
typedef svfloat64_t svdouble;
typedef svfloat32_t svfloat;
typedef svfloat64x2_t svdouble2;
typedef svfloat32x2_t svfloat2;
template <> const inline int vector_len<svfloat> = svcntw();
template <> const inline int vector_len<svdouble> = svcntd();
#define ENABLE_SVECTOR_BENCHMARKS
#endif

#endif