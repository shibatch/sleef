//          Copyright Naoki Shibata 2010 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include <time.h>

#include "md5.h"

#include "sleef.h"
#include "misc.h"
#include "testerutil.h"

#ifdef __VSX__
typedef vector double vectordouble
typedef vector float  vectorfloat
#endif

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
double get__m128d(__m128d v, int r) { double a[2]; _mm_storeu_pd(a, v); return a[r & 1]; }
__m128 set__m128(float d, int r) { float a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return _mm_loadu_ps(a); }
float get__m128(__m128 v, int r) { float a[4]; _mm_storeu_ps(a, v); return a[r & 3]; }

#if defined(__AVX__)
__m256d set__m256d(double d, int r) { double a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return _mm256_loadu_pd(a); }
double get__m256d(__m256d v, int r) { double a[4]; _mm256_storeu_pd(a, v); return a[r & 3]; }
__m256 set__m256(float d, int r) { float a[8]; memrand(a, sizeof(a)); a[r & 7] = d; return _mm256_loadu_ps(a); }
float get__m256(__m256 v, int r) { float a[8]; _mm256_storeu_ps(a, v); return a[r & 7]; }
#endif

#if defined(__AVX512F__)
__m512d set__m512d(double d, int r) { double a[8]; memrand(a, sizeof(a)); a[r & 7] = d; return _mm512_loadu_pd(a); }
double get__m512d(__m512d v, int r) { double a[8]; _mm512_storeu_pd(a, v); return a[r & 7]; }
__m512 set__m512(float d, int r) { float a[16]; memrand(a, sizeof(a)); a[r & 15] = d; return _mm512_loadu_ps(a); }
float get__m512(__m512 v, int r) { float a[16]; _mm512_storeu_ps(a, v); return a[r & 15]; }
#endif
#endif // #if defined(__i386__) || defined(__x86_64__) || defined(_MSC_VER)

#ifdef __ARM_NEON
float64x2_t setfloat64x2_t(double d, int r) { double a[2]; memrand(a, sizeof(a)); a[r & 1] = d; return vld1q_f64(a); }
double getfloat64x2_t(float64x2_t v, int r) { double a[2]; vst1q_f64(a, v); return a[r & 1]; }
float32x4_t setfloat32x4_t(float d, int r) { float a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return vld1q_f32(a); }
float getfloat32x4_t(float32x4_t v, int r) { float a[4]; vst1q_f32(a, v); return a[r & 3]; }
#endif

#ifdef __ARM_FEATURE_SVE
svfloat64_t setsvfloat64_t(double d, int r) { double a[svcntd()]; memrand(a, sizeof(a)); a[r & (svcntd()-1)] = d; return svld1_f64(svptrue_b8(), a); }
double getsvfloat64_t(svfloat64_t v, int r) { double a[svcntd()]; svst1_f64(svptrue_b8(), a, v); return a[r & (svcntd()-1)]; }
svfloat32_t setsvfloat32_t(float d, int r)  { float  a[svcntw()]; memrand(a, sizeof(a)); a[r & (svcntw()-1)] = d; return svld1_f32(svptrue_b8(), a); }
float getsvfloat32_t(svfloat32_t v, int r)  { float  a[svcntw()]; svst1_f32(svptrue_b8(), a, v); return a[r & (svcntw()-1)]; }
#endif

#ifdef __VSX__
vectordouble setvectordouble(double d, int r) { double a[2]; memrand(a, sizeof(a)); a[r & 1] = d; return (vector double) ( a[0], a[1] ); }
double getvectordouble(vector double v, int r) { double a[2]; return v[r & 1]; }
vectorfloat setvectorfloat(float d, int r) { float a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return (vector float) ( a[0], a[1], a[2], a[3] ); }
float getvectorfloat(vectorfloat v, int r) { float a[4]; return v[r & 3]; }
#endif

//

// ATR = "cinz_", NAME = sin, TYPE = d2, ULP = u35, EXT = sse2
#define FUNC(ATR, NAME, TYPE, ULP, EXT) Sleef_ ## ATR ## NAME ## TYPE ## _ ## ULP ## EXT
#define TYPE2(TYPE) Sleef_ ## TYPE ## _2
#define SET(TYPE) set ## TYPE
#define GET(TYPE) get ## TYPE

//

#define checkDigest(NAME) do {						\
    unsigned char d[16], mes[64], buf[64];				\
    MD5_Final(d, &ctx);							\
    sprintf((char *)mes, "%s %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",	\
	    #NAME, d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],		\
	    d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]);		\
    if (fp != NULL) {							\
      fgets((char *)buf, 60, fp);					\
      puts((char *)mes);						\
      puts((char *)buf);						\
      if (strncmp((char *)mes, (char *)buf, strlen((char *)mes)) != 0) { \
	fprintf(stderr, "%s\n", #NAME); exit(-1);			\
      }									\
    } else puts((char *)mes);						\
  } while(0)

//

#define STEPSCALE 10

#define exec_d_d(ATR, NAME, ULP, TYPE, TSX, EXT, arg) do {		\
    int r = rand();							\
    DPTYPE vx = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (arg, r));	\
    double fx = GET(TYPE)(vx, r);					\
    MD5_Update(&ctx, &fx, sizeof(double));				\
  } while(0)

#define test_d_d(NAME, ULP, START, END, STEP) do {			\
    MD5_CTX ctx;							\
    memset(&ctx, 1, sizeof(MD5_CTX));					\
    MD5_Init(&ctx);							\
    for(double d = (START);d < (END);d += (STEP)*STEPSCALE)		\
      exec_d_d(ATR, NAME, ULP, DPTYPE, DPTYPESPEC, EXTSPEC, d);	\
    checkDigest(NAME);							\
  } while(0)

//

#define exec_d2_d(ATR, NAME, ULP, TYPE, TSX, EXT, arg) do { \
    int r = rand();							\
    TYPE2(TYPE) vx2 = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (arg, r)); \
    double fxx = GET(TYPE)(vx2.x, r), fxy = GET(TYPE)(vx2.y, r);	\
    MD5_Update(&ctx, &fxx, sizeof(double));				\
    MD5_Update(&ctx, &fxy, sizeof(double));				\
  } while(0)

#define test_d2_d(NAME, ULP, START, END, STEP) do {		\
    MD5_CTX ctx;						\
    MD5_Init(&ctx);						\
    for(double d = (START);d < (END);d += (STEP)*STEPSCALE)	\
      exec_d2_d(ATR, NAME, ULP, DPTYPE, DPTYPESPEC, EXTSPEC, d);	\
    checkDigest(NAME);						\
  } while(0)

//

#define exec_d_d_d(ATR, NAME, ULP, TYPE, TSX, EXT, argu, argv) do { \
    int r = rand();							\
    DPTYPE vx = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (argu, r), SET(TYPE) (argv, r)); \
    double fx = GET(TYPE)(vx, r);					\
    MD5_Update(&ctx, &fx, sizeof(double));				\
  } while(0)

#define test_d_d_d(NAME, ULP, STARTU, ENDU, STEPU, STARTV, ENDV, STEPV) do { \
    MD5_CTX ctx;							\
    MD5_Init(&ctx);							\
    for(double u = (STARTU);u < (ENDU);u += (STEPU) * sqrt(STEPSCALE))	\
      for(double v = (STARTV);v < (ENDV);v += (STEPV) * sqrt(STEPSCALE)) \
	exec_d_d_d(ATR, NAME, ULP, DPTYPE, DPTYPESPEC, EXTSPEC, u, v);	\
    checkDigest(NAME);							\
  } while(0)

//

#define exec_f_f(ATR, NAME, ULP, TYPE, TSX, EXT, arg) do { \
    int r = rand();							\
    SPTYPE vx = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (arg, r));	\
    float fx = GET(TYPE)(vx, r);					\
    MD5_Update(&ctx, &fx, sizeof(float));				\
  } while(0)

#define test_f_f(NAME, ULP, START, END, STEP) do {		\
    MD5_CTX ctx;						\
    MD5_Init(&ctx);						\
    for(float d = (START);d < (END);d += (STEP)*STEPSCALE)	\
      exec_f_f(ATR, NAME, ULP, SPTYPE, SPTYPESPEC, EXTSPEC, d);	\
    checkDigest(NAME);						\
  } while(0)

//

#define exec_f2_f(ATR, NAME, ULP, TYPE, TSX, EXT, arg) do { \
    int r = rand();							\
    TYPE2(TYPE) vx2 = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (arg, r)); \
    float fxx = GET(TYPE)(vx2.x, r), fxy = GET(TYPE)(vx2.y, r);	\
    MD5_Update(&ctx, &fxx, sizeof(float));				\
    MD5_Update(&ctx, &fxy, sizeof(float));				\
  } while(0)

#define test_f2_f(NAME, ULP, START, END, STEP) do {		\
    MD5_CTX ctx;						\
    MD5_Init(&ctx);						\
    for(float d = (START);d < (END);d += (STEP)*STEPSCALE)	\
      exec_f2_f(ATR, NAME, ULP, SPTYPE, SPTYPESPEC, EXTSPEC, d);	\
    checkDigest(NAME);						\
  } while(0)

//

#define exec_f_f_f(ATR, NAME, ULP, TYPE, TSX, EXT, argu, argv) do { \
    int r = rand();							\
    SPTYPE vx = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (argu, r), SET(TYPE) (argv, r)); \
    float fx = GET(TYPE)(vx, r);					\
    MD5_Update(&ctx, &fx, sizeof(float));				\
  } while(0)

#define test_f_f_f(NAME, ULP, STARTU, ENDU, STEPU, STARTV, ENDV, STEPV) do { \
    MD5_CTX ctx;							\
    MD5_Init(&ctx);							\
    for(float u = (STARTU);u < (ENDU);u += (STEPU) * sqrt(STEPSCALE))	\
      for(float v = (STARTV);v < (ENDV);v += (STEPV) * sqrt(STEPSCALE))	\
	exec_f_f_f(ATR, NAME, ULP, SPTYPE, SPTYPESPEC, EXTSPEC, u, v);	\
    checkDigest(NAME);							\
  } while(0)

//

int do_test(int argc, char **argv)
{
  FILE *fp = NULL;

  if (argc != 1) {
    fp = fopen(argv[1], "r");
    if (fp == NULL) {
      fprintf(stderr, "Could not open %s\n", argv[1]);
      exit(-1);
    }
  }

  srand(seed = time(NULL));

  test_d_d(asin, u35, -1, 1, 1.1e-6);
  test_d_d(acos, u35, -1, 1, 1.1e-6);

  test_f_f(asin, u35, -1, 1, 1.1e-6);
  test_f_f(acos, u35, -1, 1, 1.1e-6);

  if (fp != NULL) fclose(fp);

  exit(0);
}
