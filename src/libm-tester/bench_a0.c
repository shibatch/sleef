//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>

#include <mpfr.h>

#if (defined(__GNUC__) || defined(__CLANG__)) && (defined(__i386__) || defined(__x86_64__))
#include <x86intrin.h>
#endif

#if (defined(_MSC_VER))
#include <intrin.h>
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "misc.h"
#include "sleef.h"
//#include "sleefdft.h"

#define DENORMAL_DBL_MIN (4.9406564584124654418e-324)
#define POSITIVE_INFINITY INFINITY
#define NEGATIVE_INFINITY (-INFINITY)

int isnumber(double x) { return !isinf(x) && !isnan(x); }
int isPlusZero(double x) { return x == 0 && copysign(1, x) == 1; }
int isMinusZero(double x) { return x == 0 && copysign(1, x) == -1; }

mpfr_t fra, frb, frc, frd;

double countULP(double d, mpfr_t c) {
  double c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) return 10000;
  if (isnan(c2) && isnan(d)) return 0;
  if (isnan(c2) || isnan(d)) return 10001;
  if (c2 == POSITIVE_INFINITY && d == POSITIVE_INFINITY) return 0;
  if (c2 == NEGATIVE_INFINITY && d == NEGATIVE_INFINITY) return 0;

  double v = 0;
  if (isinf(d) && !isinfl(mpfr_get_ld(c, GMP_RNDN))) {
    d = copysign(DBL_MAX, c2);
    v = 1;
  }

  //
  
  int e;
  frexpl(mpfr_get_ld(c, GMP_RNDN), &e);
  mpfr_set_ld(frb, fmaxl(ldexpl(1.0, e-53), DENORMAL_DBL_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fra, frd, c, GMP_RNDN);
  mpfr_div(fra, fra, frb, GMP_RNDN);
  double u = fabs(mpfr_get_d(fra, GMP_RNDN));

  return u + v;
}

#ifndef SVMLULP
#define SVMLULP
#endif

char* name[] = {
  "_mm_sin_pd" SVMLULP, 
  "_mm256_sin_pd" SVMLULP, 
  "sin in _mm_sincos_pd" SVMLULP,
  "cos in _mm_sincos_pd" SVMLULP,
  "sin in _mm256_sincos_pd" SVMLULP,
  "cos in _mm256_sincos_pd" SVMLULP,
  "sin", 
  "Sleef_sin_u10", 
  "Sleef_sind2_u10sse2",
  "Sleef_sind4_u10avx",
  "Sleef_sind4_u10avx2",
  "Sleef_sin_u35", 
  "Sleef_sind2_u35sse2",
  "Sleef_sind4_u35avx",
  "Sleef_sind4_u35avx2",
  "sin in Sleef_sincos_u10",
  "cos in Sleef_sincos_u10",
  "sin in Sleef_sincosd2_u10sse2",
  "cos in Sleef_sincosd2_u10sse2",
  "sin in Sleef_sincosd2_u10avx",
  "cos in Sleef_sincosd2_u10avx",
  "sin in Sleef_sincosd2_u10avx2",
  "cos in Sleef_sincosd2_u10avx2",
  "sin in Sleef_sincos_u35",
  "cos in Sleef_sincos_u35",
  "sin in Sleef_sincosd2_u35sse2",
  "cos in Sleef_sincosd2_u35sse2",
  "sin in Sleef_sincosd2_u35avx",
  "cos in Sleef_sincosd2_u35avx",
  "sin in Sleef_sincosd2_u35avx2",
  "cos in Sleef_sincosd2_u35avx2",
  NULL,
};
  
int main(int argc, char **argv) {
  double start = 0.0, finish = 10000, step = 0.01;

  mpfr_set_default_prec(128);
  
  if (argc >= 2) start = atof(argv[1]);
  if (argc >= 3) finish = atof(argv[2]);
  if (argc >= 4) step = atof(argv[3]);

  mpfr_t frt, fru;
  mpfr_inits(fra, frb, frc, frd, frt, fru, NULL);

#define N 100
  
  double max[N], sum[N];
  uint64_t count = 0;

  for(int i=0;i<N;i++) max[i] = sum[i] = 0;

  for(double a = start;a <= finish;a += step) {
    mpfr_set_d(frt, a, GMP_RNDN);
    mpfr_sin(frt, frt, GMP_RNDN);
    mpfr_set_d(fru, a, GMP_RNDN);
    mpfr_cos(fru, fru, GMP_RNDN);

    int idx = 0;
    double e;

    e = countULP(_mm_sin_pd((__m128d){a, a})[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(_mm256_sin_pd((__m256d){a, a, a, a})[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    {
      __m128d c;
      e = countULP(_mm_sincos_pd(&c, (__m128d){a, a})[0], frt);
      sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
      e = countULP(c[0], fru);
      sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    }

    {
      __m256d c;
      e = countULP(_mm256_sincos_pd(&c, (__m256d){a, a, a, a})[0], frt);
      sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
      e = countULP(c[0], fru);
      sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    }

#ifdef SVMLONLY
    count++;
    continue;
#endif

    e = countULP(sin(a), frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    
    e = countULP(Sleef_sin_u10(a), frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    
    e = countULP(Sleef_sind2_u10sse2((__m128d){a, a})[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sind4_u10avx((__m256d){a, a, a, a})[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sind4_u10avx2((__m256d){a, a, a, a})[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sin_u35(a), frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    
    e = countULP(Sleef_sind2_u35sse2((__m128d){a, a})[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sind4_u35avx((__m256d){a, a, a, a})[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sind4_u35avx2((__m256d){a, a, a, a})[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    
    e = countULP(Sleef_sincos_u10(a).x, frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    e = countULP(Sleef_sincos_u10(a).y, fru);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sincosd2_u10sse2((__m128d){a, a}).x[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    e = countULP(Sleef_sincosd2_u10sse2((__m128d){a, a}).y[0], fru);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sincosd4_u10avx((__m256d){a, a, a, a}).x[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    e = countULP(Sleef_sincosd4_u10avx((__m256d){a, a, a, a}).y[0], fru);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sincosd4_u10avx2((__m256d){a, a, a, a}).x[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    e = countULP(Sleef_sincosd4_u10avx2((__m256d){a, a, a, a}).y[0], fru);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    //

    e = countULP(Sleef_sincos_u35(a).x, frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    e = countULP(Sleef_sincos_u35(a).y, fru);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sincosd2_u35sse2((__m128d){a, a}).x[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    e = countULP(Sleef_sincosd2_u35sse2((__m128d){a, a}).y[0], fru);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sincosd4_u35avx((__m256d){a, a, a, a}).x[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    e = countULP(Sleef_sincosd4_u35avx((__m256d){a, a, a, a}).y[0], fru);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    e = countULP(Sleef_sincosd4_u35avx2((__m256d){a, a, a, a}).x[0], frt);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;
    e = countULP(Sleef_sincosd4_u35avx2((__m256d){a, a, a, a}).y[0], fru);
    sum[idx] += e; max[idx] = fmax(max[idx], e); idx++;

    count++;
  }

  for(int i=0;name[i] != NULL;i++) {
    printf("%s, %g, %g\n", name[i], max[i], sum[i] / count);
#ifdef SVMLONLY
    if (i == 5) break;
#endif
  }
}
