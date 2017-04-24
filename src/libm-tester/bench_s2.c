//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>

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
#include "sleefdft.h"

#pragma GCC optimize ("O0") // This is important

#ifndef SVMLULP
#define SVMLULP
#endif

#define callFunc1_1(funcName, arg, type) ({				\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int j=0;j<niter2;j++) {					\
	type *p = (type *)(arg);					\
	for(int i=0;i<niter1;i++) funcName(*p++);			\
      }									\
      printf(#funcName ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      fflush(stdout); \
    })

#define callFunc2_1(funcName, arg, type) ({				\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int j=0;j<niter2;j++) {					\
	type *p = (type *)(arg), c;					\
	for(int i=0;i<niter1;i++) funcName(&c, *p++);			\
      }									\
      printf(#funcName ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      fflush(stdout);							\
    })

#define callFunc1_2(funcName, arg1, arg2, type) ({			\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int j=0;j<niter2;j++) {					\
	type *p1 = (type *)(arg1), *p2 = (type *)(arg2);		\
	for(int i=0;i<niter1;i++) funcName(*p1++, *p2++);		\
      }									\
      printf(#funcName ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      fflush(stdout);							\
    })

#define callFuncS1_1(funcName, arg, type) ({				\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int j=0;j<niter2;j++) {					\
	type *p = (type *)(arg);					\
	for(int i=0;i<niter1;i++) funcName(*p++);			\
      }									\
      printf(#funcName SVMLULP ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      fflush(stdout);							\
    })

#define callFuncS2_1(funcName, arg, type) ({				\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int j=0;j<niter2;j++) {					\
	type *p = (type *)(arg), c;					\
	for(int i=0;i<niter1;i++) funcName(&c, *p++);			\
      }									\
      printf(#funcName SVMLULP ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      fflush(stdout);							\
    })

#define callFuncS1_2(funcName, arg1, arg2, type) ({			\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int j=0;j<niter2;j++) {					\
	type *p1 = (type *)(arg1), *p2 = (type *)(arg2);		\
	for(int i=0;i<niter1;i++) funcName(*p1++, *p2++);		\
      }									\
      printf(#funcName SVMLULP ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      fflush(stdout);							\
    })

static int cpuSupportsAVX2() {
    int32_t reg[4];
    Sleef_x86CpuID(reg, 7, 0);
    return (reg[1] & (1 << 5)) != 0;
}

double doNothing1_1_double(double);
__m128d doNothing1_1_m128d(__m128d);
__m256d doNothing1_1_m256d(__m256d);
double doNothing1_2_double(double, double);
__m128d doNothing1_2_m128d(__m128d, __m128d);
__m256d doNothing1_2_m256d(__m256d, __m256d);
Sleef_double2 doNothing2_1_double(double);
Sleef___m128d_2 doNothing2_1_m128d(__m128d);
Sleef___m256d_2 doNothing2_1_m256d(__m256d);

int main(int argc, char **argv) {
  const int niter1 = 10000, niter2 = 10000, niter = niter1 * niter2;
  
  double a = 0.0, b = 6.28;
  if (argc >= 2) a = atof(argv[1]);
  if (argc >= 3) b = atof(argv[2]);

  const int doAvx2 = cpuSupportsAVX2();
  
  const int veclen = 8;
  double *abuf = Sleef_malloc(sizeof(double) * niter * veclen);
  double *bbuf = Sleef_malloc(sizeof(double) * niter * veclen);

  srandom(time(NULL));

  //

  for(int i=0;i<niter*veclen;i++) {
    abuf[i] = pow(2.1, (double)random() / RAND_MAX * 2000 - 1000);
  }

  // log
  
#ifndef SVMLONLY
  callFunc1_1(log, abuf, double);
#endif

#ifndef NOSVML
  callFuncS1_1(_mm_log_pd, abuf, __m128d);
  callFuncS1_1(_mm256_log_pd, abuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_1(Sleef_log_u10, abuf, double);
  callFunc1_1(Sleef_logd2_u10sse2, abuf, __m128d);
  callFunc1_1(Sleef_logd4_u10avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_logd4_u10avx2, abuf, __m256d);
#endif
  
  printf("\n");
  
  printf("\n");
  printf("\n");
#ifndef SVMLONLY
  callFunc1_1(Sleef_log_u35, abuf, double);
  callFunc1_1(Sleef_logd2_u35sse2, abuf, __m128d);
  callFunc1_1(Sleef_logd4_u35avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_logd4_u35avx2, abuf, __m256d);
#endif
  
  printf("\n");

  //
  
  for(int i=0;i<niter*veclen;i++) {
    abuf[i] = (double)random() / RAND_MAX * 1000 - 500;
  }

  // exp
  
#ifndef SVMLONLY
  callFunc1_1(exp, abuf, double);
#endif
  
#ifndef NOSVML
  callFuncS1_1(_mm_exp_pd, abuf, __m128d);
  callFuncS1_1(_mm256_exp_pd, abuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_1(Sleef_exp_u10, abuf, double);
  callFunc1_1(Sleef_expd2_u10sse2, abuf, __m128d);
  callFunc1_1(Sleef_expd4_u10avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_expd4_u10avx2, abuf, __m256d);
#endif
  
  printf("\n");
  
  printf("\n");
  printf("\n");
  
  printf("\n");
  
  //
  
  for(int i=0;i<niter*veclen;i++) {
    abuf[i] = (double)random() / RAND_MAX * 200 - 100;
    bbuf[i] = (double)random() / RAND_MAX * 100;
  }

  // pow

#ifndef SVMLONLY
  callFunc1_2(pow, abuf, bbuf, double);
#endif
#ifndef NOSVML
  callFuncS1_2(_mm_pow_pd, abuf, bbuf, __m128d);
  callFuncS1_2(_mm256_pow_pd, abuf, bbuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_2(Sleef_pow_u10, abuf, bbuf, double);
  callFunc1_2(Sleef_powd2_u10sse2, abuf, bbuf, __m128d);
  callFunc1_2(Sleef_powd4_u10avx, abuf, bbuf, __m256d);
  if (doAvx2) callFunc1_2(Sleef_powd4_u10avx2, abuf, bbuf, __m256d);
#endif

  printf("\n");

  //

  for(int i=0;i<niter*veclen;i++) {
    abuf[i] = (double)random() / RAND_MAX * 2 - 1;
  }

  // asin
  
#ifndef SVMLONLY
  callFunc1_1(asin, abuf, double);
#endif
  
#ifndef NOSVML
  callFuncS1_1(_mm_asin_pd, abuf, __m128d);
  callFuncS1_1(_mm256_asin_pd, abuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_1(Sleef_asin_u10, abuf, double);
  callFunc1_1(Sleef_asind2_u10sse2, abuf, __m128d);
  callFunc1_1(Sleef_asind4_u10avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_asind4_u10avx2, abuf, __m256d);
#endif
  
  printf("\n");
  
  printf("\n");
  printf("\n");
#ifndef SVMLONLY
  callFunc1_1(Sleef_asin_u35, abuf, double);
  callFunc1_1(Sleef_asind2_u35sse2, abuf, __m128d);
  callFunc1_1(Sleef_asind4_u35avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_asind4_u35avx2, abuf, __m256d);
#endif

  printf("\n");

  // acos
  
#ifndef SVMLONLY
  callFunc1_1(acos, abuf, double);
#endif
  
#ifndef NOSVML
  callFuncS1_1(_mm_acos_pd, abuf, __m128d);
  callFuncS1_1(_mm256_acos_pd, abuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_1(Sleef_acos_u10, abuf, double);
  callFunc1_1(Sleef_acosd2_u10sse2, abuf, __m128d);
  callFunc1_1(Sleef_acosd4_u10avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_acosd4_u10avx2, abuf, __m256d);
#endif
  
  printf("\n");
  
  printf("\n");
  printf("\n");
#ifndef SVMLONLY
  callFunc1_1(Sleef_acos_u35, abuf, double);
  callFunc1_1(Sleef_acosd2_u35sse2, abuf, __m128d);
  callFunc1_1(Sleef_acosd4_u35avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_acosd4_u35avx2, abuf, __m256d);
#endif

  printf("\n");

  //
  
  for(int i=0;i<niter*veclen;i++) {
    abuf[i] = (double)random() / RAND_MAX * 2;
  }

  // atan
  
#ifndef SVMLONLY
  callFunc1_1(atan, abuf, double);
#endif
  
#ifndef NOSVML
  callFuncS1_1(_mm_atan_pd, abuf, __m128d);
  callFuncS1_1(_mm256_atan_pd, abuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_1(Sleef_atan_u10, abuf, double);
  callFunc1_1(Sleef_atand2_u10sse2, abuf, __m128d);
  callFunc1_1(Sleef_atand4_u10avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_atand4_u10avx2, abuf, __m256d);
#endif
  
  printf("\n");
  
  printf("\n");
  printf("\n");
#ifndef SVMLONLY
  callFunc1_1(Sleef_atan_u35, abuf, double);
  callFunc1_1(Sleef_atand2_u35sse2, abuf, __m128d);
  callFunc1_1(Sleef_atand4_u35avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_atand4_u35avx2, abuf, __m256d);
#endif

  printf("\n");

  //
  
  for(int i=0;i<niter*veclen;i++) {
    bbuf[i] = (double)random() / RAND_MAX * 2;
  }

  // atan2
  
#ifndef SVMLONLY
  callFunc1_2(atan2, abuf, bbuf, double);
#endif
  
#ifndef NOSVML
  callFuncS1_2(_mm_atan2_pd, abuf, bbuf, __m128d);
  callFuncS1_2(_mm256_atan2_pd, abuf, bbuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_2(Sleef_atan2_u10, abuf, bbuf, double);
  callFunc1_2(Sleef_atan2d2_u10sse2, abuf, bbuf, __m128d);
  callFunc1_2(Sleef_atan2d4_u10avx, abuf, bbuf, __m256d);
  if (doAvx2) callFunc1_2(Sleef_atan2d4_u10avx2, abuf, bbuf, __m256d);
#endif
  
  printf("\n");
  
  printf("\n");
  printf("\n");
#ifndef SVMLONLY
  callFunc1_2(Sleef_atan2_u35, abuf, bbuf, double);
  callFunc1_2(Sleef_atan2d2_u35sse2, abuf, bbuf, __m128d);
  callFunc1_2(Sleef_atan2d4_u35avx, abuf, bbuf, __m256d);
  if (doAvx2) callFunc1_2(Sleef_atan2d4_u35avx2, abuf, bbuf, __m256d);
#endif

  printf("\n");

  // doNothing

#ifndef SVMLONLY
  callFunc1_1(doNothing1_1_double, abuf, double);
  callFunc1_1(doNothing1_1_m128d, abuf, __m128d);
  callFunc1_1(doNothing1_1_m256d, abuf, __m256d);
  callFunc1_2(doNothing1_2_double, abuf, bbuf, double);
  callFunc1_2(doNothing1_2_m128d, abuf, bbuf, __m128d);
  callFunc1_2(doNothing1_2_m256d, abuf, bbuf, __m256d);
  callFunc1_1(doNothing2_1_double, abuf, double);
  callFunc1_1(doNothing2_1_m128d, abuf, __m128d);
  callFunc1_1(doNothing2_1_m256d, abuf, __m256d);
#endif

  printf("\n");

  //
  
  printf("\n");
  
  exit(0);
}
