//   Copyright Naoki Shibata and contributors 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#include "benchmark_templates.hpp"

// Callers for libm - no separate file as its much simpler
// only interested in scalar (i think?)
#define BENCH(funname, funtype, namefilter, min, max)                                   \
  BENCHMARK_CAPTURE(BM_Sleef_templated_function, #funname, funtype funname, min, max)  \
      ->Name("MB_libm_" #funname "_" #namefilter "_" #min "_" #max);

#define BENCH_SINGLE_SCALAR(fun, min, max)                                \
  BENCH(fun, (float (*) (float)), u10_scalarf, min, max);
#define BENCH_DOUBLE_SCALAR(fun, min, max)                                \
  BENCH(fun, (double (*) (double)), u10_scalard, min, max);

#define BENCH_SCALAR(fun, min, max)                                       \
  BENCH_SINGLE_SCALAR(fun, min, max);                                     \
  BENCH_DOUBLE_SCALAR(fun, min, max);

// special case for pow and atan2
#define BENCH_SINGLE_SCALAR_2ARGS(fun, min, max)                                \
  BENCH(fun, (float (*) (float, float)), u10_scalarf, min, max);
#define BENCH_DOUBLE_SCALAR_2ARGS(fun, min, max)                                \
  BENCH(fun, (double (*) (double, double)), u10_scalard, min, max);

#define BENCH_SCALAR_2ARGS(fun, min, max)                                       \
  BENCH_SINGLE_SCALAR_2ARGS(fun, min, max);                                     \
  BENCH_DOUBLE_SCALAR_2ARGS(fun, min, max);


// special case for sincos
#define BENCH_SINGLE_SCALAR_VOID_3ARGS(fun, min, max)                                \
  BENCH(fun, (void (*)(float, float*, float*)), u10_scalarf, min, max);
#define BENCH_DOUBLE_SCALAR_VOID_3ARGS(fun, min, max)                                \
  BENCH(fun, (void (*)(double, double*, double*)), u10_scalard, min, max);

#define BENCH_SCALAR_VOID_3ARGS(fun, min, max)                                       \
  BENCH_SINGLE_SCALAR_VOID_3ARGS(fun, min, max);                                     \
  BENCH_DOUBLE_SCALAR_VOID_3ARGS(fun, min, max);



