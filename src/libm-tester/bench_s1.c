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

#ifdef __INTEL_COMPILER
#define callFuncG1_1(funcName, arg, type) ({				\
      uint64_t t = Sleef_currentTimeMicros();				\
      type acc = 0;							\
      for(int j=0;j<niter2;j++) {					\
	type *p = (type *)(arg);					\
	for(int i=0;i<niter1;i++) funcName(*p++);			\
      }									\
      printf(#funcName ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      fflush(stdout);							\
      acc;								\
    })
#else
#define callFuncG1_1(funcName, arg, type) ({				\
      uint64_t t = Sleef_currentTimeMicros();				\
      type acc = 0;							\
      for(int j=0;j<niter2;j++) {					\
	type *p = (type *)(arg);					\
	for(int i=0;i<niter1;i++) acc += funcName(*p++);		\
      }									\
      printf(#funcName ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      fflush(stdout);							\
      acc;								\
    })
#endif

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

int main(int argc, char **argv) {
  const int niter1 = 10000, niter2 = 10000, niter = niter1 * niter2;
  
  double a = 0.0, b = 6.28;
  if (argc >= 2) a = atof(argv[1]);
  if (argc >= 3) b = atof(argv[2]);

  const int doAvx2 = cpuSupportsAVX2();
  
  const int veclen = 8;
  double *abuf = Sleef_malloc(sizeof(double) * niter * veclen);

  double acc = 0;
  
  srandom(time(NULL));

  //
  
  for(int i=0;i<niter*veclen;i++) {
    abuf[i] = (double)random() / RAND_MAX * (b - a) + a;
  }

  // sin
  
#ifndef SVMLONLY
  acc += callFuncG1_1(sin, abuf, double);
#endif
  
#ifndef NOSVML
  callFuncS1_1(_mm_sin_pd, abuf, __m128d);
  callFuncS1_1(_mm256_sin_pd, abuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_1(Sleef_sin_u10, abuf, double);
  callFunc1_1(Sleef_sind2_u10sse2, abuf, __m128d);
  callFunc1_1(Sleef_sind4_u10avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_sind4_u10avx2, abuf, __m256d);
#endif
  
  printf("\n");
  

  printf("\n");
  printf("\n");
#ifndef SVMLONLY
  callFunc1_1(Sleef_sin_u35, abuf, double);
  callFunc1_1(Sleef_sind2_u35sse2, abuf, __m128d);
  callFunc1_1(Sleef_sind4_u35avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_sind4_u35avx2, abuf, __m256d);
#endif
  
  printf("\n");

  // cos
  
#ifndef SVMLONLY
  acc += callFuncG1_1(cos, abuf, double);
#endif
  
#ifndef NOSVML
  callFuncS1_1(_mm_cos_pd, abuf, __m128d);
  callFuncS1_1(_mm256_cos_pd, abuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_1(Sleef_cos_u10, abuf, double);
  callFunc1_1(Sleef_cosd2_u10sse2, abuf, __m128d);
  callFunc1_1(Sleef_cosd4_u10avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_cosd4_u10avx2, abuf, __m256d);
#endif
  
  printf("\n");
  
  printf("\n");
  printf("\n");
#ifndef SVMLONLY
  callFunc1_1(Sleef_cos_u35, abuf, double);
  callFunc1_1(Sleef_cosd2_u35sse2, abuf, __m128d);
  callFunc1_1(Sleef_cosd4_u35avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_cosd4_u35avx2, abuf, __m256d);
#endif
  
  printf("\n");
  
#ifndef NOSVML
  callFuncS2_1(_mm_sincos_pd, abuf, __m128d);
  callFuncS2_1(_mm256_sincos_pd, abuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_1(Sleef_sincos_u10, abuf, double);
  callFunc1_1(Sleef_sincosd2_u10sse2, abuf, __m128d);
  callFunc1_1(Sleef_sincosd4_u10avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_sincosd4_u10avx2, abuf, __m256d);
#endif
  
  printf("\n");

  // sincos
  
  printf("\n");
  printf("\n");
#ifndef SVMLONLY
  callFunc1_1(Sleef_sincos_u35, abuf, double);
  callFunc1_1(Sleef_sincosd2_u35sse2, abuf, __m128d);
  callFunc1_1(Sleef_sincosd4_u35avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_sincosd4_u35avx2, abuf, __m256d);
#endif
  printf("\n");
  
#ifndef SVMLONLY
  callFunc1_1(Sleef_sincospi_u05, abuf, double);
  callFunc1_1(Sleef_sincospid2_u05sse2, abuf, __m128d);
  callFunc1_1(Sleef_sincospid4_u05avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_sincospid4_u05avx2, abuf, __m256d);

  callFunc1_1(Sleef_sincospi_u35, abuf, double);
  callFunc1_1(Sleef_sincospid2_u35sse2, abuf, __m128d);
  callFunc1_1(Sleef_sincospid4_u35avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_sincospid4_u35avx2, abuf, __m256d);
  printf("\n");
#endif
  
  // tan
  
#ifndef SVMLONLY
  acc += callFuncG1_1(tan, abuf, double);
#endif
  
#ifndef NOSVML
  callFuncS1_1(_mm_tan_pd, abuf, __m128d);
  callFuncS1_1(_mm256_tan_pd, abuf, __m256d);
#endif
#ifndef SVMLONLY
  callFunc1_1(Sleef_tan_u10, abuf, double);
  callFunc1_1(Sleef_tand2_u10sse2, abuf, __m128d);
  callFunc1_1(Sleef_tand4_u10avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_tand4_u10avx2, abuf, __m256d);
#endif
  
  printf("\n");
  
  printf("\n");
  printf("\n");
#ifndef SVMLONLY
  callFunc1_1(Sleef_tan_u35, abuf, double);
  callFunc1_1(Sleef_tand2_u35sse2, abuf, __m128d);
  callFunc1_1(Sleef_tand4_u35avx, abuf, __m256d);
  if (doAvx2) callFunc1_1(Sleef_tand4_u35avx2, abuf, __m256d);
#endif
  
  printf("\n\n");

  printf("dummy value : %g\n", acc);
  
  exit(0);
}
