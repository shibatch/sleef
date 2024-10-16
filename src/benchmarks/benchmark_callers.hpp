//   Copyright Naoki Shibata and contributors 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#include "benchmark_templates.hpp"

// Define macros that can be used to generate benchmark calls (defined in
// benchmark_templates.hpp).
// Example to generate benchmarks for 1ULP sin(x) for x between 0 and 6.28:
//   BENCH(Sleef_sin_u10, double, 0, 6.28);
// BENCHMARK_CAPTURE is a symbol from the google bench framework
// Note: type is only passed for name filtering reasons
#define BENCH(funname, typefilter, min, max)                                   \
  BENCHMARK_CAPTURE(BM_Sleef_templated_function, #funname, funname, min, max)  \
      ->Name("MB_" #funname "_" #typefilter "_" #min "_" #max);

#define BENCH_SINGLE_SCALAR(fun, ulp, min, max)                                \
  BENCH(Sleef_##fun##f_##ulp, scalarf, min, max);
#define BENCH_DOUBLE_SCALAR(fun, ulp, min, max)                                \
  BENCH(Sleef_##fun##_##ulp, scalard, min, max);
// Generate benchmarks for scalar function implementations
#define BENCH_SCALAR(fun, ulp, min, max)                                       \
  BENCH_SINGLE_SCALAR(fun, ulp, min, max);                                     \
  BENCH_DOUBLE_SCALAR(fun, ulp, min, max);

// Generate benchmarks for vector function implementations
#ifdef ENABLE_VECTOR_BENCHMARKS
#if !defined(ARCH_VECT_LEN) || ARCH_VECT_LEN == 128
#define BENCH_SINGLE_VECTOR(fun, ulp, min, max)                                \
  BENCH(Sleef_##fun##f4_##ulp, vectorf128, min, max);
#define BENCH_DOUBLE_VECTOR(fun, ulp, min, max)                                \
  BENCH(Sleef_##fun##d2_##ulp, vectord128, min, max);
#elif ARCH_VECT_LEN == 256
#define BENCH_SINGLE_VECTOR(fun, ulp, min, max)                                \
  BENCH(Sleef_##fun##f8_##ulp, vectorf256, min, max);
#define BENCH_DOUBLE_VECTOR(fun, ulp, min, max)                                \
  BENCH(Sleef_##fun##d4_##ulp, vectord256, min, max);
#elif ARCH_VECT_LEN == 512
#define BENCH_SINGLE_VECTOR(fun, ulp, min, max)                                \
  BENCH(Sleef_##fun##f16_##ulp, vectorf512, min, max);
#define BENCH_DOUBLE_VECTOR(fun, ulp, min, max)                                \
  BENCH(Sleef_##fun##d8_##ulp, vectord512, min, max);
#endif
#define BENCH_VECTOR(fun, ulp, min, max)                                       \
  BENCH_SINGLE_VECTOR(fun, ulp, min, max);                                     \
  BENCH_DOUBLE_VECTOR(fun, ulp, min, max);
#else
#define BENCH_SINGLE_VECTOR(fun, ulp, min, max)
#define BENCH_DOUBLE_VECTOR(fun, ulp, min, max)
#define BENCH_VECTOR(fun, ulp, min, max)
#endif

// Generate benchmarks for SVE function implementations
#ifdef ENABLE_SVECTOR_BENCHMARKS
#define BENCH_SINGLE_SVE(fun, ulp, min, max)                                   \
  BENCH(Sleef_##fun##fx_##ulp##sve, svef, min, max);
#define BENCH_DOUBLE_SVE(fun, ulp, min, max)                                   \
  BENCH(Sleef_##fun##dx_##ulp##sve, sved, min, max);
#define BENCH_SVE(fun, ulp, min, max)                                          \
  BENCH_SINGLE_SVE(fun, ulp, min, max);                                        \
  BENCH_DOUBLE_SVE(fun, ulp, min, max);
#else
#define BENCH_SINGLE_SVE(fun, ulp, min, max)
#define BENCH_DOUBLE_SVE(fun, ulp, min, max)
#define BENCH_SVE(fun, ulp, min, max)
#endif

// Given a function implemented meeting a specific ulp
// error (present in the name of the function),
// BENCH_ALL_W_FIX_ULP macro will
// generate benchmarks for
// * all vector extensions supported
// * all precisions
// * all vector lengths
#define BENCH_ALL_W_FIX_ULP(fun, ulp, min, max)                                \
  BENCH_SCALAR(fun, ulp, min, max);                                            \
  BENCH_VECTOR(fun, ulp, min, max);                                            \
  BENCH_SVE(fun, ulp, min, max);
#define BENCH_SINGLEP_W_FIX_ULP(fun, ulp, min, max)                            \
  BENCH_SINGLE_SCALAR(fun, ulp, min, max);                                     \
  BENCH_SINGLE_VECTOR(fun, ulp, min, max);                                     \
  BENCH_SINGLE_SVE(fun, ulp, min, max);
#define BENCH_DOUBLEP_W_FIX_ULP(fun, ulp, min, max)                            \
  BENCH_DOUBLE_SCALAR(fun, ulp, min, max);                                     \
  BENCH_DOUBLE_VECTOR(fun, ulp, min, max);                                     \
  BENCH_DOUBLE_SVE(fun, ulp, min, max);

#define BENCH_ALL_SINGLEP(fun, min, max)                                       \
  BENCH_SINGLEP_W_FIX_ULP(fun, u10, min, max);                                 \
  BENCH_SINGLEP_W_FIX_ULP(fun, u35, min, max);
#define BENCH_ALL_DOUBLEP(fun, min, max)                                       \
  BENCH_DOUBLEP_W_FIX_ULP(fun, u10, min, max);                                 \
  BENCH_DOUBLEP_W_FIX_ULP(fun, u35, min, max);

// Given a function, BENCH_ALL macro will
// generate benchmarks for
// * all ulp implementations available (u10 and u35)
// * all vector extensions supported
// * all precisions
// * all vector lengths
#define BENCH_ALL(fun, min, max)                                               \
  BENCH_ALL_W_FIX_ULP(fun, u10, min, max);                                     \
  BENCH_ALL_W_FIX_ULP(fun, u35, min, max);
