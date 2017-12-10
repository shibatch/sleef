//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#include "sleef.h"
#include "misc.h"
#include "testerutil.h"

//

static int32_t seed;

// Fill memory with random bits
static inline void memrand(void *p, int size) {
  uint8_t *q = p;
  for(int i=0;i<size;i++) {
    q[i] = (seed >> 16) & 255;
    seed = seed * 1103515245 + 12345; 
  }
}

static inline void memranddp(double *p, int size) {
  for(int i=0;i<size;i++) p[i] = (2.0 * rand() / RAND_MAX - 1) * 4 * M_PI;
}

static inline void memrandsp(float *p, int size) {
  for(int i=0;i<size;i++) p[i] = (2.0 * rand() / RAND_MAX - 1) * 4 * M_PI;
}

//

double setdouble(double d, int r) { return d; }
double set2double(double d, int r) { return d; }
double getdouble(double v, int r) { return v; }
float setfloat(float d, int r) { return d; }
float set2float(float d, int r) { return d; }
float getfloat(float v, int r) { return v; }

#if defined(__i386__) || defined(__x86_64__) || defined(_MSC_VER)
__m128d set__m128d(double d, int r) { double a[2]; memrand(a, sizeof(a)); a[r & 1] = d; return _mm_loadu_pd(a); }
__m128d set2__m128d(double d, int r) { double a[2]; memranddp(a, 2); a[r & 1] = d; return _mm_loadu_pd(a); }
double get__m128d(__m128d v, int r) { double a[2]; _mm_storeu_pd(a, v); return a[r & 1]; }
__m128 set__m128(float d, int r) { float a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return _mm_loadu_ps(a); }
__m128 set2__m128(float d, int r) { float a[4]; memrandsp(a, 4); a[r & 3] = d; return _mm_loadu_ps(a); }
float get__m128(__m128 v, int r) { float a[4]; _mm_storeu_ps(a, v); return a[r & 3]; }

#if defined(__AVX__)
__m256d set__m256d(double d, int r) { double a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return _mm256_loadu_pd(a); }
__m256d set2__m256d(double d, int r) { double a[4]; memranddp(a, 4); a[r & 3] = d; return _mm256_loadu_pd(a); }
double get__m256d(__m256d v, int r) { double a[4]; _mm256_storeu_pd(a, v); return a[r & 3]; }
__m256 set__m256(float d, int r) { float a[8]; memrand(a, sizeof(a)); a[r & 7] = d; return _mm256_loadu_ps(a); }
__m256 set2__m256(float d, int r) { float a[8]; memrandsp(a, 8); a[r & 7] = d; return _mm256_loadu_ps(a); }
float get__m256(__m256 v, int r) { float a[8]; _mm256_storeu_ps(a, v); return a[r & 7]; }
#endif

#if defined(__AVX512F__)
__m512d set__m512d(double d, int r) { double a[8]; memrand(a, sizeof(a)); a[r & 7] = d; return _mm512_loadu_pd(a); }
__m512d set2__m512d(double d, int r) { double a[8]; memranddp(a, 8); a[r & 7] = d; return _mm512_loadu_pd(a); }
double get__m512d(__m512d v, int r) { double a[8]; _mm512_storeu_pd(a, v); return a[r & 7]; }
__m512 set__m512(float d, int r) { float a[16]; memrand(a, sizeof(a)); a[r & 15] = d; return _mm512_loadu_ps(a); }
__m512 set2__m512(float d, int r) { float a[16]; memrandsp(a, 16); a[r & 15] = d; return _mm512_loadu_ps(a); }
float get__m512(__m512 v, int r) { float a[16]; _mm512_storeu_ps(a, v); return a[r & 15]; }
#endif
#endif // #if defined(__i386__) || defined(__x86_64__) || defined(_MSC_VER)

#ifdef __ARM_NEON
#endif

//

#define DENORMAL_DBL_MIN (4.9406564584124654418e-324)
#define DENORMAL_FLT_MIN (1.4012984643248170709e-45f)

static double sndp[] = {
  +0.0, +0.11, +0.5, +1, +1.5, +2, +10.1, +M_PI, +M_PI*10000,
  -0.0, -0.11, -0.5, -1, -1.5, -2, -10.1, -M_PI, -M_PI*10000,
  +1e-10, +1e-310, +1e-300, +DENORMAL_DBL_MIN,
  -1e-10, -1e-310, -1e-300, -DENORMAL_DBL_MIN,
  +1e+10, +1e+100, +1e+300, +DBL_MAX, +INFINITY,
  -1e+10, -1e+100, -1e+300, -DBL_MAX, -INFINITY,
  NAN
};

static double sndp2[] = {
  +0.0, +0.11, +0.5, +1, +1.5, +2, +10.1, +M_PI, +M_PI*4,
  -0.0, -0.11, -0.5, -1, -1.5, -2, -10.1, -M_PI, -M_PI*4,
  +1e-10, +1e-310, +1e-300, +DENORMAL_DBL_MIN,
  -1e-10, -1e-310, -1e-300, -DENORMAL_DBL_MIN,
  +INFINITY, -INFINITY, NAN
};

static float snsp[] = {
  +0.0, +0.11, +0.5, +1, +1.5, +2, +10.1, +M_PI, +M_PI*10000,
  -0.0, -0.11, -0.5, -1, -1.5, -2, -10.1, -M_PI, -M_PI*10000,
  +1e-10, +1e-38, +1e-30, +DENORMAL_FLT_MIN,
  -1e-10, -1e-38, -1e-30, -DENORMAL_FLT_MIN,
  +1e+3, +1e+10, +1e+40, +FLT_MAX, +INFINITY,
  -1e+3, -1e+10, -1e+40, -FLT_MAX, -INFINITY,
  NAN
};

static float snsp2[] = {
  +0.0, +0.11, +0.5, +1, +1.5, +2, +10.1, +M_PI, +M_PI*4,
  -0.0, -0.11, -0.5, -1, -1.5, -2, -10.1, -M_PI, -M_PI*4,
  +1e-10, +1e-38, +1e-30, +DENORMAL_FLT_MIN,
  -1e-10, -1e-38, -1e-30, -DENORMAL_FLT_MIN,
  +INFINITY, -INFINITY, NAN
};

//

#define FUNC(NAME, TYPE, ULP, EXT) NAME ## TYPE ## _ ## ULP ## EXT
#define TYPE2(TYPE) Sleef_ ## TYPE ## _2
#define SET(TYPE) set ## TYPE
#define SET2(TYPE) set2 ## TYPE
#define GET(TYPE) get ## TYPE

//

#define STEPSCALE 10

#define compare_d_d(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, arg) do { \
    int r = rand();							\
    DPTYPEX vx = FUNC(NAME, TSX, ULP, EXTX) (SET(TYPEX) (arg, r));	\
    DPTYPEY vy = FUNC(NAME, TSY, ULP, EXTY) (SET(TYPEY) (arg, r));	\
    double fx = GET(TYPEX)(vx, r), fy = GET(TYPEY)(vy, r);		\
    int success = (isnan(fx) && isnan(fy)) || d2u(fx) == d2u(fy);	\
    if (!success) {							\
      printf("%s %s arg=%.20g x=%.20g(%016llx) y=%.20g(%016llx)\n", #NAME, #ULP, arg, \
	     fx, (long long unsigned int)d2u(fx),			\
	     fy, (long long unsigned int)d2u(fy));			\
      exit(-1);								\
    }									\
  } while(0)

#define test_d_d(NAME, ULP, START, END, STEP) do {			\
    for(int i=0;i<sizeof(sndp)/sizeof(double);i++)			\
      compare_d_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, sndp[i]); \
    for(double d = (START);d < (END);d += (STEP)*STEPSCALE)			\
      compare_d_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, d); \
  } while(0)

//

#define compare2_d_d(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, arg) do { \
    int r = rand();							\
    DPTYPEX vx = FUNC(NAME, TSX, ULP, EXTX) (SET2(TYPEX) (arg, r));	\
    DPTYPEY vy = FUNC(NAME, TSY, ULP, EXTY) (SET2(TYPEY) (arg, r));	\
    double fx = GET(TYPEX)(vx, r), fy = GET(TYPEY)(vy, r);		\
    int success = (isnan(fx) && isnan(fy)) || d2u(fx) == d2u(fy);	\
    if (!success) {							\
      printf("%s %s arg=%.20g x=%.20g(%016llx) y=%.20g(%016llx)\n", #NAME, #ULP, arg, \
	     fx, (long long unsigned int)d2u(fx),			\
	     fy, (long long unsigned int)d2u(fy));			\
      exit(-1);								\
    }									\
  } while(0)

#define test2_d_d(NAME, ULP, START, END, STEP) do {			\
    for(int i=0;i<sizeof(sndp2)/sizeof(double);i++)			\
      compare2_d_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, sndp2[i]); \
    for(double d = (START);d < (END);d += (STEP)*STEPSCALE)			\
      compare2_d_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, d); \
  } while(0)

//

#define compare_d2_d(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, arg) do { \
    int r = rand();							\
    TYPE2(TYPEX) vx2 = FUNC(NAME, TSX, ULP, EXTX) (SET(TYPEX) (arg, r)); \
    TYPE2(TYPEY) vy2 = FUNC(NAME, TSY, ULP, EXTY) (SET(TYPEY) (arg, r)); \
    double fxx = GET(TYPEX)(vx2.x, r), fyx = GET(TYPEY)(vy2.x, r);	\
    double fxy = GET(TYPEX)(vx2.y, r), fyy = GET(TYPEY)(vy2.y, r);	\
    int success = ((isnan(fxx) && isnan(fyx)) || d2u(fxx) == d2u(fyx)) && \
      ((isnan(fxy) && isnan(fyy)) || d2u(fxy) == d2u(fyy));		\
    if (!success) {							\
      printf("%s %s arg=%.20g x.x=%.20g(%016llx) y.x=%.20g(%016llx) x.x=%.20g(%016llx) y.x=%.20g(%016llx)\n", \
	     #NAME, #ULP, arg,						\
	     GET(TYPEX)(vx2.x, r), (long long unsigned int)d2u(GET(TYPEX)(vx2.x, r)), \
	     GET(TYPEY)(vy2.x, r), (long long unsigned int)d2u(GET(TYPEY)(vy2.x, r)), \
	     GET(TYPEX)(vx2.y, r), (long long unsigned int)d2u(GET(TYPEX)(vx2.y, r)), \
	     GET(TYPEY)(vy2.y, r), (long long unsigned int)d2u(GET(TYPEY)(vy2.y, r))); \
      exit(-1);								\
    }									\
  } while(0)

#define test_d2_d(NAME, ULP, START, END, STEP) do {			\
    for(int i=0;i<sizeof(sndp)/sizeof(double);i++)			\
      compare_d2_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, sndp[i]); \
    for(double d = (START);d < (END);d += (STEP)*STEPSCALE)			\
      compare_d2_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, d); \
  } while(0)

//

#define compare2_d2_d(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, arg) do { \
    int r = rand();							\
    TYPE2(TYPEX) vx2 = FUNC(NAME, TSX, ULP, EXTX) (SET2(TYPEX) (arg, r)); \
    TYPE2(TYPEY) vy2 = FUNC(NAME, TSY, ULP, EXTY) (SET2(TYPEY) (arg, r)); \
    double fxx = GET(TYPEX)(vx2.x, r), fyx = GET(TYPEY)(vy2.x, r);	\
    double fxy = GET(TYPEX)(vx2.y, r), fyy = GET(TYPEY)(vy2.y, r);	\
    int success = ((isnan(fxx) && isnan(fyx)) || d2u(fxx) == d2u(fyx)) && \
      ((isnan(fxy) && isnan(fyy)) || d2u(fxy) == d2u(fyy));		\
    if (!success) {							\
      printf("%s %s arg=%.20g x.x=%.20g(%016llx) y.x=%.20g(%016llx) x.x=%.20g(%016llx) y.x=%.20g(%016llx)\n", \
	     #NAME, #ULP, arg,						\
	     GET(TYPEX)(vx2.x, r), (long long unsigned int)d2u(GET(TYPEX)(vx2.x, r)), \
	     GET(TYPEY)(vy2.x, r), (long long unsigned int)d2u(GET(TYPEY)(vy2.x, r)), \
	     GET(TYPEX)(vx2.y, r), (long long unsigned int)d2u(GET(TYPEX)(vx2.y, r)), \
	     GET(TYPEY)(vy2.y, r), (long long unsigned int)d2u(GET(TYPEY)(vy2.y, r))); \
      exit(-1);								\
    }									\
  } while(0)

#define test2_d2_d(NAME, ULP, START, END, STEP) do {			\
    for(int i=0;i<sizeof(sndp2)/sizeof(double);i++)			\
      compare2_d2_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, sndp2[i]); \
    for(double d = (START);d < (END);d += (STEP)*STEPSCALE)			\
      compare2_d2_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, d); \
  } while(0)

//

#define compare_d_d_d(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, argu, argv) do { \
    int r = rand();							\
    DPTYPEX vx = FUNC(NAME, TSX, ULP, EXTX) (SET(TYPEX) (argu, r), SET(TYPEX) (argv, r)); \
    DPTYPEY vy = FUNC(NAME, TSY, ULP, EXTY) (SET(TYPEY) (argu, r), SET(TYPEY) (argv, r)); \
    double fx = GET(TYPEX)(vx, r), fy = GET(TYPEY)(vy, r);		\
    int success = (isnan(fx) && isnan(fy)) || d2u(fx) == d2u(fy);	\
    if (!success) {							\
      printf("%s %s argu=%.20g argv=%.20g x=%.20g(%016llx) y=%.20g(%016llx)\n", #NAME, #ULP, argu, argv, \
	     GET(TYPEX)(vx, r), (long long unsigned int)d2u(GET(TYPEX)(vx, r)), \
	     GET(TYPEY)(vy, r), (long long unsigned int)d2u(GET(TYPEY)(vy, r))); \
      exit(-1);								\
    }									\
  } while(0)

#define test_d_d_d(NAME, ULP, STARTU, ENDU, STEPU, STARTV, ENDV, STEPV) do { \
    for(int i=0;i<sizeof(sndp)/sizeof(double);i++)			\
      for(int j=0;j<sizeof(sndp)/sizeof(double);j++)			\
	compare_d_d_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, sndp[i], sndp[j]); \
    for(double u = (STARTU);u < (ENDU);u += (STEPU) * sqrt(STEPSCALE))	\
      for(double v = (STARTV);v < (ENDV);v += (STEPV) * sqrt(STEPSCALE)) \
	compare_d_d_d(NAME, ULP, DPTYPEX, DPTYPESPECX, EXTSPECX, DPTYPEY, DPTYPESPECY, EXTSPECY, u, v); \
  } while(0)

//

#define compare_f_f(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, arg) do { \
    int r = rand();							\
    SPTYPEX vx = FUNC(NAME, TSX, ULP, EXTX) (SET(TYPEX) (arg, r));	\
    SPTYPEY vy = FUNC(NAME, TSY, ULP, EXTY) (SET(TYPEY) (arg, r));	\
    float fx = GET(TYPEX)(vx, r), fy = GET(TYPEY)(vy, r);		\
    int success = (isnanf(fx) && isnanf(fy)) || f2u(fx) == f2u(fy);	\
    if (!success) {							\
      printf("f %s %s arg=%.20g x=%.20g y=%.20g\n", #NAME, #ULP, arg, GET(TYPEX)(vx, r), GET(TYPEY)(vy, r)); \
      exit(-1);								\
    }									\
  } while(0)

#define test_f_f(NAME, ULP, START, END, STEP) do {			\
    for(int i=0;i<sizeof(snsp)/sizeof(float);i++)			\
      compare_f_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, snsp[i]); \
    for(float d = (START);d < (END);d += (STEP)*STEPSCALE)			\
      compare_f_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, d); \
  } while(0)

//

#define compare2_f_f(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, arg) do { \
    int r = rand();							\
    SPTYPEX vx = FUNC(NAME, TSX, ULP, EXTX) (SET2(TYPEX) (arg, r));	\
    SPTYPEY vy = FUNC(NAME, TSY, ULP, EXTY) (SET2(TYPEY) (arg, r));	\
    float fx = GET(TYPEX)(vx, r), fy = GET(TYPEY)(vy, r);		\
    int success = (isnanf(fx) && isnanf(fy)) || f2u(fx) == f2u(fy);	\
    if (!success) {							\
      printf("f %s %s arg=%.20g x=%.20g y=%.20g\n", #NAME, #ULP, arg, GET(TYPEX)(vx, r), GET(TYPEY)(vy, r)); \
      exit(-1);								\
    }									\
  } while(0)

#define test2_f_f(NAME, ULP, START, END, STEP) do {			\
    for(int i=0;i<sizeof(snsp2)/sizeof(float);i++)			\
      compare2_f_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, snsp2[i]); \
    for(float d = (START);d < (END);d += (STEP)*STEPSCALE)			\
      compare2_f_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, d); \
  } while(0)

//

#define compare_f2_f(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, arg) do { \
    int r = rand();							\
    TYPE2(TYPEX) vx2 = FUNC(NAME, TSX, ULP, EXTX) (SET(TYPEX) (arg, r)); \
    TYPE2(TYPEY) vy2 = FUNC(NAME, TSY, ULP, EXTY) (SET(TYPEY) (arg, r)); \
    double fxx = GET(TYPEX)(vx2.x, r), fyx = GET(TYPEY)(vy2.x, r);	\
    double fxy = GET(TYPEX)(vx2.y, r), fyy = GET(TYPEY)(vy2.y, r);	\
    int success = ((isnanf(fxx) && isnanf(fyx)) || f2u(fxx) == f2u(fyx)) && \
      ((isnanf(fxy) && isnanf(fyy)) || f2u(fxy) == f2u(fyy));		\
    if (!success) {							\
      printf("f %s %s arg=%.20g x.x=%.20g y.x=%.20g x.x=%.20g y.x=%.20g\n", \
	     #NAME, #ULP, arg, GET(TYPEX)(vx2.x, r), GET(TYPEY)(vy2.x, r), GET(TYPEX)(vx2.y, r), GET(TYPEY)(vy2.y, r)); \
      exit(-1);								\
    }									\
  } while(0)

#define test_f2_f(NAME, ULP, START, END, STEP) do {			\
    for(int i=0;i<sizeof(snsp)/sizeof(float);i++)			\
      compare_f2_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, snsp[i]); \
    for(float d = (START);d < (END);d += (STEP)*STEPSCALE)			\
      compare_f2_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, d); \
  } while(0)

//

#define compare2_f2_f(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, arg) do { \
    int r = rand();							\
    TYPE2(TYPEX) vx2 = FUNC(NAME, TSX, ULP, EXTX) (SET2(TYPEX) (arg, r)); \
    TYPE2(TYPEY) vy2 = FUNC(NAME, TSY, ULP, EXTY) (SET2(TYPEY) (arg, r)); \
    double fxx = GET(TYPEX)(vx2.x, r), fyx = GET(TYPEY)(vy2.x, r);	\
    double fxy = GET(TYPEX)(vx2.y, r), fyy = GET(TYPEY)(vy2.y, r);	\
    int success = ((isnanf(fxx) && isnanf(fyx)) || f2u(fxx) == f2u(fyx)) && \
      ((isnanf(fxy) && isnanf(fyy)) || f2u(fxy) == f2u(fyy));		\
    if (!success) {							\
      printf("f %s %s arg=%.20g x.x=%.20g y.x=%.20g x.x=%.20g y.x=%.20g\n", \
	     #NAME, #ULP, arg, GET(TYPEX)(vx2.x, r), GET(TYPEY)(vy2.x, r), GET(TYPEX)(vx2.y, r), GET(TYPEY)(vy2.y, r)); \
      exit(-1);								\
    }									\
  } while(0)

#define test2_f2_f(NAME, ULP, START, END, STEP) do {			\
    for(int i=0;i<sizeof(snsp2)/sizeof(float);i++)			\
      compare2_f2_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, snsp2[i]); \
    for(float d = (START);d < (END);d += (STEP)*STEPSCALE)			\
      compare2_f2_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, d); \
  } while(0)

//

#define compare_f_f_f(NAME, ULP, TYPEX, TSX, EXTX, TYPEY, TSY, EXTY, argu, argv) do { \
    int r = rand();							\
    SPTYPEX vx = FUNC(NAME, TSX, ULP, EXTX) (SET(TYPEX) (argu, r), SET(TYPEX) (argv, r)); \
    SPTYPEY vy = FUNC(NAME, TSY, ULP, EXTY) (SET(TYPEY) (argu, r), SET(TYPEY) (argv, r)); \
    float fx = GET(TYPEX)(vx, r), fy = GET(TYPEY)(vy, r);		\
    int success = (isnanf(fx) && isnanf(fy)) || f2u(fx) == f2u(fy);	\
    if (!success) {							\
      printf("f %s %s argu=%.20g argv=%.20g x=%.20g y=%.20g\n", #NAME, #ULP, argu, argv, GET(TYPEX)(vx, r), GET(TYPEY)(vy, r)); \
      exit(-1);								\
    }									\
  } while(0)

#define test_f_f_f(NAME, ULP, STARTU, ENDU, STEPU, STARTV, ENDV, STEPV) do { \
    for(int i=0;i<sizeof(snsp)/sizeof(float);i++)			\
      for(int j=0;j<sizeof(snsp)/sizeof(float);j++)			\
	compare_f_f_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, snsp[i], snsp[j]); \
    for(float u = (STARTU);u < (ENDU);u += (STEPU) * sqrt(STEPSCALE))	\
      for(float v = (STARTV);v < (ENDV);v += (STEPV) * sqrt(STEPSCALE))	\
	compare_f_f_f(NAME, ULP, SPTYPEX, SPTYPESPECX, EXTSPECX, SPTYPEY, SPTYPESPECY, EXTSPECY, u, v); \
  } while(0)

//

int do_test(int argc, char **argv)
{
  srand(seed = time(NULL));

  test2_d_d(Sleef_sin, u10, -4*M_PI, 4*M_PI, 1.2e-5);
  test_d_d(Sleef_sin, u35, -1e+14, 1e+14, 1e+8+0.1);
  test2_d_d(Sleef_cos, u10, -4*M_PI, 4*M_PI, 1.2e-5);
  test_d_d(Sleef_cos, u35, -1e+14, 1e+14, 1e+8+0.1);
  test2_d_d(Sleef_tan, u10, -4*M_PI, 4*M_PI, 1.2e-5);
  test_d_d(Sleef_tan, u35, -1e+14, 1e+14, 1e+8+0.1);
  test_d_d(Sleef_sinpi, u05, -1e+14, 1e+14, 1e+8+0.1);
  test_d_d(Sleef_cospi, u05, -1e+14, 1e+14, 1e+8+0.1);
  test2_d2_d(Sleef_sincos, u10, -4*M_PI, 4*M_PI, 1.2e-5);
  test_d2_d(Sleef_sincos, u35, -1e+14, 1e+14, 1e+8+0.1);
  test_d2_d(Sleef_sincospi, u05, -1e+14, 1e+14, 1e+8+0.1);
  test_d2_d(Sleef_sincospi, u35, -1e+14, 1e+14, 1e+8+0.1);
  
  test_d_d(Sleef_log, u10, 0, 1e+14, 0.5e+8+0.1);
  test_d_d(Sleef_log, u35, 0, 1e+14, 0.5e+8+0.1);
  test_d_d(Sleef_log10, u10, 0, 1e+14, 0.5e+8+0.1);
  test_d_d(Sleef_log1p, u10, 0, 1e+14, 0.5e+8+0.1);
  test_d_d(Sleef_exp, u10, -1000, 1000, 0.001);
  test_d_d(Sleef_exp2, u10, -1000, 1000, 0.001);
  test_d_d(Sleef_exp10, u10, -1000, 1000, 0.001);
  test_d_d(Sleef_expm1, u10, -1000, 1000, 0.001);
  test_d_d_d(Sleef_pow, u10, -100, 100, 0.19, -100, 100, 0.19);
  
  test_d_d(Sleef_sqrt, u05, 0, 1e+14, 1e+8+0.1);
  test_d_d(Sleef_sqrt, u35, 0, 1e+14, 1e+8+0.1);
  test_d_d(Sleef_cbrt, u10, -1e+14, 1e+14, 1e+8+0.1);
  test_d_d(Sleef_cbrt, u35, -1e+14, 1e+14, 1e+8+0.1);
  test_d_d_d(Sleef_hypot, u05, -1e7, 1e7, 1.51e+4, -1e7, 1e7, 1.51e+4);
  test_d_d_d(Sleef_hypot, u35, -1e7, 1e7, 1.51e+4, -1e7, 1e7, 1.51e+4);

  test_d_d(Sleef_asin, u10, -1, 1, 1.1e-6);
  test_d_d(Sleef_asin, u35, -1, 1, 1.1e-6);
  test_d_d(Sleef_acos, u10, -1, 1, 1.1e-6);
  test_d_d(Sleef_acos, u35, -1, 1, 1.1e-6);
  test_d_d(Sleef_atan, u10, -10000, 10000, 0.011);
  test_d_d(Sleef_atan, u35, -10000, 10000, 0.011);
  test_d_d_d(Sleef_atan2, u10, -10, 10, 0.015, -10, 10, 0.015);
  test_d_d_d(Sleef_atan2, u35, -10, 10, 0.015, -10, 10, 0.015);

  test_d_d(Sleef_sinh, u10, -700, 700, 0.0011);
  test_d_d(Sleef_cosh, u10, -700, 700, 0.0011);
  test_d_d(Sleef_tanh, u10, -700, 700, 0.0011);
  test_d_d(Sleef_asinh, u10, -700, 700, 0.0011);
  test_d_d(Sleef_acosh, u10, 1, 700, 0.00055);
  test_d_d(Sleef_atanh, u10, -700, 700, 0.0011);
  
  test_d_d(Sleef_lgamma, u10, -5000, 5000, 0.0055);
  test_d_d(Sleef_tgamma, u10, -10, 10, 1.1e-5);
  test_d_d(Sleef_erf, u10, -100, 100, 1.1e-4);
  test_d_d(Sleef_erfc, u15, -1, 100, 0.55e-4);

  test_d_d(Sleef_fabs, , -100.5, 100.5, 0.25);
  test_d_d_d(Sleef_copysign, , -1e+10, 1e+10, 1.51e+9, -1e+10, 1e+10, 1.51e+9);
  test_d_d_d(Sleef_fmax, , -1e+10, 1e+10, 1.51e+9, -1e+10, 1e+10, 1.51e+9);
  test_d_d_d(Sleef_fmin, , -1e+10, 1e+10, 1.51e+9, -1e+10, 1e+10, 1.51e+9);
  test_d_d_d(Sleef_fdim, , -1e+10, 1e+10, 1.51e+9, -1e+10, 1e+10, 1.51e+9);
  test_d_d_d(Sleef_fmod, , -1e+10, 1e+10, 1.51e+7, -1e+10, 1e+10, 1.51e+7);
  test_d2_d(Sleef_modf, , -1e+14, 1e+14, 1e+8+0.1);
  test_d_d_d(Sleef_nextafter, , -1e+10, 1e+10, 1.51e+7, -1e+10, 1e+10, 1.51e+7);

  test_d_d(Sleef_trunc, , -100.5, 100.5, 0.25);
  test_d_d(Sleef_floor, , -100.5, 100.5, 0.25);
  test_d_d(Sleef_ceil, , -100.5, 100.5, 0.25);
  test_d_d(Sleef_round, , -100.5, 100.5, 0.25);
  test_d_d(Sleef_rint, , -100.5, 100.5, 0.25);

  //

  test2_f_f(Sleef_sin, u10, -4*M_PI, 4*M_PI, 1.2e-5);
  test_f_f(Sleef_sin, u35, -10000, 10000, 0.011);
  test2_f_f(Sleef_cos, u10, -4*M_PI, 4*M_PI, 1.2e-5);
  test_f_f(Sleef_cos, u35, -10000, 10000, 0.011);
  test2_f_f(Sleef_tan, u10, -4*M_PI, 4*M_PI, 1.2e-5);
  test_f_f(Sleef_tan, u35, -10000, 10000, 0.011);
  test_f_f(Sleef_sinpi, u05, -10000, 10000, 0.011);
  test_f_f(Sleef_cospi, u05, -10000, 10000, 0.011);
  test2_f2_f(Sleef_sincos, u10, -4*M_PI, 4*M_PI, 1.2e-5);
  test_f2_f(Sleef_sincos, u35, -10000, 10000, 0.011);
  test_f2_f(Sleef_sincospi, u05, -10000, 10000, 0.011);
  test_f2_f(Sleef_sincospi, u35, -10000, 10000, 0.011);
  
  test_f_f(Sleef_log, u10, 0, 10000, 0.011);
  test_f_f(Sleef_log, u35, 0, 10000, 0.011);
  test_f_f(Sleef_log10, u10, 0, 10000, 0.011);
  test_f_f(Sleef_log1p, u10, 0, 10000, 0.011);
  test_f_f(Sleef_exp, u10, -1000, 1000, 0.0011);
  test_f_f(Sleef_exp2, u10, -1000, 1000, 0.0011);
  test_f_f(Sleef_exp10, u10, -1000, 1000, 0.0011);
  test_f_f(Sleef_expm1, u10, -1000, 1000, 0.0011);
  test_f_f_f(Sleef_pow, u10, -100, 100, 0.19, -100, 100, 0.19);
  
  test_f_f(Sleef_sqrt, u05, 0, 1e+14, 1e+8+0.1);
  test_f_f(Sleef_sqrt, u35, 0, 1e+14, 1e+8+0.1);
  test_f_f(Sleef_cbrt, u10, -10000, 10000, 0.011);
  test_f_f(Sleef_cbrt, u35, -10000, 10000, 0.011);
  test_f_f_f(Sleef_hypot, u05, -1e7, 1e7, 1.51e+4, -1e7, 1e7, 1.51e+4);
  test_f_f_f(Sleef_hypot, u35, -1e7, 1e7, 1.51e+4, -1e7, 1e7, 1.51e+4);

  test_f_f(Sleef_asin, u10, -1, 1, 1.1e-6);
  test_f_f(Sleef_asin, u35, -1, 1, 1.1e-6);
  test_f_f(Sleef_acos, u10, -1, 1, 1.1e-6);
  test_f_f(Sleef_acos, u35, -1, 1, 1.1e-6);
  test_f_f(Sleef_atan, u10, -10000, 10000, 0.011);
  test_f_f(Sleef_atan, u35, -10000, 10000, 0.011);
  test_f_f_f(Sleef_atan2, u10, -10, 10, 0.15, -10, 10, 0.015);
  test_f_f_f(Sleef_atan2, u35, -10, 10, 0.15, -10, 10, 0.015);

  test_f_f(Sleef_sinh, u10, -88, 88, 8.88e-5);
  test_f_f(Sleef_cosh, u10, -88, 88, 8.88e-5);
  test_f_f(Sleef_tanh, u10, -88, 88, 8.88e-5);
  test_f_f(Sleef_asinh, u10, -88, 88, 8.88e-5);
  test_f_f(Sleef_acosh, u10, 1, 88, 8.88e-5);
  test_f_f(Sleef_atanh, u10, -88, 88, 8.88e-5);
  
  test_f_f(Sleef_lgamma, u10, -5000, 5000, 0.0055);
  test_f_f(Sleef_tgamma, u10, -10, 10, 1.1e-5);
  test_f_f(Sleef_erf, u10, -100, 100, 1.1e-4);
  test_f_f(Sleef_erfc, u15, -1, 100, 0.55e-4);

  test_f_f(Sleef_fabs, , -100.5, 100.5, 0.25);
  test_f_f_f(Sleef_copysign, , -1e+10, 1e+10, 1.51e+9, -1e+10, 1e+10, 1.51e+9);
  test_f_f_f(Sleef_fmax, , -1e+10, 1e+10, 1.51e+9, -1e+10, 1e+10, 1.51e+9);
  test_f_f_f(Sleef_fmin, , -1e+10, 1e+10, 1.51e+9, -1e+10, 1e+10, 1.51e+9);
  test_f_f_f(Sleef_fdim, , -1e+10, 1e+10, 1.51e+9, -1e+10, 1e+10, 1.51e+9);
  test_f_f_f(Sleef_fmod, , -1e+10, 1e+10, 1.51e+7, -1e+10, 1e+10, 1.51e+7);
  test_f2_f(Sleef_modf, , -10000, 10000, 0.0011);
  test_f_f_f(Sleef_nextafter, , -1e+10, 1e+10, 1.51e+7, -1e+10, 1e+10, 1.51e+7);

  test_f_f(Sleef_trunc, , -100.5, 100.5, 0.25);
  test_f_f(Sleef_floor, , -100.5, 100.5, 0.25);
  test_f_f(Sleef_ceil, , -100.5, 100.5, 0.25);
  test_f_f(Sleef_round, , -100.5, 100.5, 0.25);
  test_f_f(Sleef_rint, , -100.5, 100.5, 0.25);

  exit(0);
}
