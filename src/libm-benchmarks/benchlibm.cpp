//   Copyright Naoki Shibata and contributors 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "benchmark_callers_libm.hpp"
#include <math.h>

// ======================TRIG==========================
// sin on different intervals
BENCH_SCALAR(sin, 0, 6.28);
BENCH_SCALAR(sin, 0, 1e+6);
BENCH_SINGLE_SCALAR(sin, 0, 1e20);
BENCH_DOUBLE_SCALAR(sin, 0, 1e+100);

// cos on different intervals
BENCH_SCALAR(cos, 0, 6.28);
BENCH_SCALAR(cos, 0, 1e+6);
BENCH_SINGLE_SCALAR(cos, 0, 1e20);
BENCH_DOUBLE_SCALAR(cos, 0, 1e+100);

// tan on different intervals
BENCH_SCALAR(tan, 0, 6.28);
BENCH_SCALAR(tan, 0, 1e+6);
BENCH_SINGLE_SCALAR(tan, 0, 1e20);
BENCH_DOUBLE_SCALAR(tan, 0, 1e+100);

BENCH_SCALAR_VOID_3ARGS(sincos, 0, 6.28);
BENCH_SCALAR_VOID_3ARGS(sincos, 0, 1e+6);
BENCH_SINGLE_SCALAR_VOID_3ARGS(sincos, 0, 1e20);
BENCH_DOUBLE_SCALAR_VOID_3ARGS(sincos, 0, 1e+100);

// inverse trig
BENCH_SCALAR(asin, -1.0, 1.0);
BENCH_SCALAR(acos, -1.0, 1.0);
BENCH_SCALAR(atan, -10, 10);
BENCH_SCALAR_2ARGS(atan2, -10, 10)

// ======================NON TRIG==========================
//  log
BENCH_SINGLE_SCALAR(log, 0, 1e+38);
BENCH_DOUBLE_SCALAR(log, 0, 1e+100);

BENCH_SINGLE_SCALAR(log2, 0, 1e+38);
BENCH_DOUBLE_SCALAR(log2, 0, 1e+100);

BENCH_SINGLE_SCALAR(log10, 0, 1e+38);
BENCH_DOUBLE_SCALAR(log10, 0, 1e+100);

BENCH_SINGLE_SCALAR(log1p, 0, 1e+38);
BENCH_DOUBLE_SCALAR(log1p, 0, 1e+100);

// exp
BENCH_SINGLE_SCALAR(exp, -700, 700);
BENCH_DOUBLE_SCALAR(exp, -700, 700);

BENCH_SINGLE_SCALAR(exp2, -100, 100);
BENCH_DOUBLE_SCALAR(exp2, -700, 700);

BENCH_SINGLE_SCALAR(exp10, -100, 100);
BENCH_DOUBLE_SCALAR(exp10, -700, 700);

BENCH_SINGLE_SCALAR(expm1, -100, 100);
BENCH_DOUBLE_SCALAR(expm1, -700, 700);

// pow
BENCH_SCALAR_2ARGS(pow, -30, 30);

BENCHMARK_MAIN();