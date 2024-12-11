//   Copyright Naoki Shibata and contributors 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#include "type_defs.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

///////////////////////////////////
// Random Generators //////////////
///////////////////////////////////
static_assert(sizeof(uint64_t) == sizeof(double));
inline double asdouble(uint64_t i) {
  union {
    uint64_t i;
    double f;
  } u = {i};
  return u.f;
}

uint64_t seed;
void init_rng() { seed = 0x0123456789abcdef; }

double gen_rand(double lo, double hi) {
  seed = 6364136223846793005ULL * seed + 1;
  return lo + (hi - lo) * (asdouble(seed >> 12 | 0x3ffULL << 52) - 1.0);
}
float gen_randf(double lo, double hi) {
  return static_cast<float>(gen_rand(lo, hi));
}

///////////////////////////////////
// Input Generators ///////////////
///////////////////////////////////
template <typename T> T gen_input(double, double);
template <> float gen_input(double lo, double hi) { return gen_randf(lo, hi); }
template <> double gen_input(double lo, double hi) { return gen_rand(lo, hi); }
#ifdef ENABLE_VECTOR_BENCHMARKS
template <> vfloat gen_input(double lo, double hi) {
  int vlen = vector_len<vfloat>;
  vfloat in;
  for (int i = 0; i < vlen; i++) {
    in[i] = gen_randf(lo, hi);
  }
  return in;
}
template <> vdouble gen_input(double lo, double hi) {
  int vlen = vector_len<vdouble>;
  vdouble in;
  for (int i = 0; i < vlen; i++) {
    in[i] = gen_rand(lo, hi);
  }
  return in;
}
#endif
#ifdef ENABLE_SVECTOR_BENCHMARKS
template <> svfloat gen_input(double lo, double hi) {
  int vlen = vector_len<svfloat>;
  std::vector<float> in(vlen);
  std::generate(in.begin(), in.end(), [&]() { return gen_randf(lo, hi); });
  svbool_t pg = svptrue_b32();
  return (svfloat)svld1(pg, in.data());
}
template <> svdouble gen_input(double lo, double hi) {
  int vlen = vector_len<svdouble>;
  std::vector<double> in(vlen);
  std::generate(in.begin(), in.end(), [&]() { return gen_rand(lo, hi); });
  svbool_t pg = svptrue_b32();
  return (svdouble)svld1(pg, in.data());
}
#endif