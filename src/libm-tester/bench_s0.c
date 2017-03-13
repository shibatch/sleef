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

#define callFunc1_1(funcName, arg) ({		\
      uint64_t t = Sleef_currentTimeMicros();	\
      for(int i=0;i<niter;i++) funcName(arg);				\
      printf(#funcName ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      Sleef_currentTimeMicros() - t;					\
    })

#define callFunc2_1(funcName, arg, type) ({	\
      uint64_t t = Sleef_currentTimeMicros();				\
      type c;								\
      for(int i=0;i<niter;i++) funcName(&c, arg);			\
      printf(#funcName ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      Sleef_currentTimeMicros() - t;					\
    })

#define callFunc1_2(funcName, arg1, arg2) ({	\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int i=0;i<niter;i++) funcName(arg1, arg2);			\
      printf(#funcName ", %g\n", (double)(Sleef_currentTimeMicros() - t) / niter); \
      Sleef_currentTimeMicros() - t;					\
    })

int main(int argc, char **argv) {
  const int niter = 20000000;

  double a = 3.0, b = 3.0;
  if (argc >= 2) a = atof(argv[1]);
  if (argc >= 3) b = atof(argv[2]);

  for(int i=0;i<niter/2;i++) Sleef_sin_u35(0); // Warming up

  callFunc1_1(Sleef_sin_u35, a);
  callFunc1_1(Sleef_sind2_u35sse2, ((__m128d){a, a}));
  callFunc1_1(Sleef_sind4_u35avx, ((__m256d){a, a, a, a}));
  callFunc1_1(Sleef_sind4_u35avx2, ((__m256d){a, a, a, a}));

  printf("\n");

#ifdef __INTEL_COMPILER
  callFunc1_1(_mm_sin_pd, ((__m128d){a, a}));
  callFunc1_1(_mm256_sin_pd, ((__m256d){a, a, a, a}));
#endif

  printf("\n");
  
  callFunc1_1(Sleef_sin_u10, a);
  callFunc1_1(Sleef_sind2_u10sse2, ((__m128d){a, a}));
  callFunc1_1(Sleef_sind4_u10avx, ((__m256d){a, a, a, a}));
  callFunc1_1(Sleef_sind4_u10avx2, ((__m256d){a, a, a, a}));

  printf("\n");
  
  callFunc1_1(sin, a);

  printf("\n");
  
  callFunc1_1(Sleef_sincos_u35, a);
  callFunc1_1(Sleef_sincosd2_u35sse2, ((__m128d){a, a}));
  callFunc1_1(Sleef_sincosd4_u35avx, ((__m256d){a, a, a, a}));
  callFunc1_1(Sleef_sincosd4_u35avx2, ((__m256d){a, a, a, a}));

  printf("\n");
  
#ifdef __INTEL_COMPILER
  callFunc2_1(_mm_sincos_pd, ((__m128d){a, a}), __m128d);
  callFunc2_1(_mm256_sincos_pd, ((__m256d){a, a, a, a}), __m256d);
#endif

  printf("\n");
  
  callFunc1_1(Sleef_sincos_u10, a);
  callFunc1_1(Sleef_sincosd2_u10sse2, ((__m128d){a, a}));
  callFunc1_1(Sleef_sincosd4_u10avx, ((__m256d){a, a, a, a}));
  callFunc1_1(Sleef_sincosd4_u10avx2, ((__m256d){a, a, a, a}));
  
  printf("\n");

  callFunc1_1(Sleef_sincospi_u35, a);
  callFunc1_1(Sleef_sincospid2_u35sse2, ((__m128d){a, a}));
  callFunc1_1(Sleef_sincospid4_u35avx, ((__m256d){a, a, a, a}));
  callFunc1_1(Sleef_sincospid4_u35avx2, ((__m256d){a, a, a, a}));

  printf("\n");

  callFunc1_1(Sleef_sincospi_u05, a);
  callFunc1_1(Sleef_sincospid2_u05sse2, ((__m128d){a, a}));
  callFunc1_1(Sleef_sincospid4_u05avx, ((__m256d){a, a, a, a}));
  callFunc1_1(Sleef_sincospid4_u05avx2, ((__m256d){a, a, a, a}));

  printf("\n");
  
  callFunc1_2(Sleef_pow_u10, a, b);
  callFunc1_2(Sleef_powd2_u10sse2, ((__m128d){a, a}), ((__m128d){b, b}));
  callFunc1_2(Sleef_powd4_u10avx, ((__m256d){a, a, a, a}), ((__m256d){b, b, b, b}));
  callFunc1_2(Sleef_powd4_u10avx2, ((__m256d){a, a, a, a}), ((__m256d){b, b, b, b}));

  printf("\n");
  
  callFunc1_2(pow, a, b);
  
#ifdef __INTEL_COMPILER
  callFunc1_2(_mm_pow_pd, ((__m128d){a, a}), ((__m128d){b, b}));
  callFunc1_2(_mm256_pow_pd, ((__m256d){a, a, a, a}), ((__m256d){b, b, b, b}));
#endif

  printf("\n\n");
  
  exit(0);
}
