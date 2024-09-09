//   Copyright Naoki Shibata and contributors 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "benchmark_callers.hpp"
#include <sleef.h>

// ======================TRIG==========================
// sin on different intervals
BENCH_ALL(sin, 0, 6.28);
BENCH_ALL(sin, 0, 1e+6);
BENCH_ALL_SINGLEP(sin, 0, 1e20);
BENCH_ALL_DOUBLEP(sin, 0, 1e+100);

// cos on different intervals
BENCH_ALL(cos, 0, 6.28);
BENCH_ALL(cos, 0, 1e+6);
BENCH_ALL_SINGLEP(cos, 0, 1e20);
BENCH_ALL_DOUBLEP(cos, 0, 1e+100);

// tan on different intervals
BENCH_ALL(tan, 0, 6.28);
BENCH_ALL(tan, 0, 1e+6);
BENCH_ALL_SINGLEP(tan, 0, 1e20);
BENCH_ALL_DOUBLEP(tan, 0, 1e+100);

// sincos on different intervals
BENCH_ALL(sincos, 0, 6.28);
BENCH_ALL(sincos, 0, 1e+6);
BENCH_ALL_SINGLEP(sincos, 0, 1e20);
BENCH_ALL_DOUBLEP(sincos, 0, 1e+100);

// inverse trig
BENCH_ALL(asin, -1.0, 1.0);
BENCH_ALL(acos, -1.0, 1.0);
BENCH_ALL(atan, -10, 10);
BENCH_ALL(atan2, -10, 10)

// ======================NON TRIG==========================
//  log
BENCH_ALL_SINGLEP(log, 0, 1e+38);
BENCH_ALL_DOUBLEP(log, 0, 1e+100);

BENCH_ALL_SINGLEP(log2, 0, 1e+38);
BENCH_ALL_DOUBLEP(log2, 0, 1e+100);

BENCH_SINGLEP_W_FIX_ULP(log10, u10, 0, 1e+38);
BENCH_DOUBLEP_W_FIX_ULP(log10, u10, 0, 1e+100);

BENCH_SINGLEP_W_FIX_ULP(log1p, u10, 0, 1e+38);
BENCH_DOUBLEP_W_FIX_ULP(log1p, u10, 0, 1e+100);

// exp
BENCH_SINGLEP_W_FIX_ULP(exp, u10, -700, 700);
BENCH_DOUBLEP_W_FIX_ULP(exp, u10, -700, 700);

BENCH_ALL_SINGLEP(exp2, -100, 100);
BENCH_ALL_DOUBLEP(exp2, -700, 700);

BENCH_ALL_SINGLEP(exp10, -100, 100);
BENCH_ALL_DOUBLEP(exp10, -700, 700);

BENCH_SINGLEP_W_FIX_ULP(expm1, u10, -100, 100);
BENCH_DOUBLEP_W_FIX_ULP(expm1, u10, -700, 700);

// pow
BENCH_ALL_W_FIX_ULP(pow, u10, -30, 30);

BENCHMARK_MAIN();