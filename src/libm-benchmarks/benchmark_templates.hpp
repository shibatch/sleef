//   Copyright Naoki Shibata and contributors 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#include "gen_input.hpp"
#include <benchmark/benchmark.h>

///////////////////////////////////
// Function Callers ///////////////
///////////////////////////////////
template <typename F, typename... In_T>
__attribute__((noinline)) void call_fun(F f, In_T... x) {
  f(x...);
}

///////////////////////////////////
// Benchmarkers ///////////////////
///////////////////////////////////
template <typename T, typename Ret>
static void BM_Sleef_templated_function(benchmark::State &state, Ret (*fun)(T),
                                        double min, double max) {
  T x = gen_input<T>(min, max);
  for (auto _ : state) {
    call_fun(fun, x);
  }
  int num_els_processed = state.iterations() * vector_len<T>;
  state.counters["NSperEl"] =
      benchmark::Counter(num_els_processed, benchmark::Counter::kIsRate |
                                                benchmark::Counter::kInvert);
}

template <typename T, typename Ret>
static void BM_Sleef_templated_function(benchmark::State &state,
                                        Ret (*fun)(T, T), double min,
                                        double max) {
  T x0 = gen_input<T>(min, max);
  T x1 = gen_input<T>(min, max);
  for (auto _ : state) {
    call_fun(fun, x0, x1);
  }
  int num_els_processed = state.iterations() * vector_len<T>;
  state.counters["NSperEl"] =
      benchmark::Counter(num_els_processed, benchmark::Counter::kIsRate |
                                                benchmark::Counter::kInvert);
}

// Necessary for libm sincos (which takes 3 arguments)
template <typename T>
static void BM_Sleef_templated_function(benchmark::State &state,
                                        void (*fun)(T, T*, T*), double min,
                                        double max) {
  T p0, p1;
  T x = gen_input<T>(min, max);
  for (auto _ : state) {
    call_fun(fun, x, &p0, &p1);
  }
  int num_els_processed = state.iterations() * vector_len<T>;
  state.counters["NSperEl"] =
      benchmark::Counter(num_els_processed, benchmark::Counter::kIsRate |
                                                benchmark::Counter::kInvert);
}
