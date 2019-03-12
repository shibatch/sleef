//          Copyright Naoki Shibata 2010 - 2019.
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

#include <openssl/md5.h>

#include "sleef.h"
#include "misc.h"
#include "testerutil.h"

#ifdef __VSX__
typedef vector double vector_double;
typedef vector float  vector_float;
#endif

//

#define XNAN  (((union { long long int u; double d; })  { .u = 0xffffffffffffffffLL }).d)
#define XNANf (((union { long int u; float d; })  { .u = 0xffffffff }).d)

static INLINE double unifyValue(double x) { x = !(x == x) ? XNAN  : x; return x; }
static INLINE float unifyValuef(float  x) { x = !(x == x) ? XNANf : x; return x; }

static INLINE double setdouble(double d, int r) { return d; }
static INLINE double getdouble(double v, int r) { return unifyValue(v); }
static INLINE float setfloat(float d, int r) { return d; }
static INLINE float getfloat(float v, int r) { return unifyValuef(v); }

#if defined(__i386__) || defined(__x86_64__) || defined(_MSC_VER)
static INLINE __m128d set__m128d(double d, int r) { static double a[2]; memrand(a, sizeof(a)); a[r & 1] = d; return _mm_loadu_pd(a); }
static INLINE double get__m128d(__m128d v, int r) { static double a[2]; _mm_storeu_pd(a, v); return unifyValue(a[r & 1]); }
static INLINE __m128 set__m128(float d, int r) { static float a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return _mm_loadu_ps(a); }
static INLINE float get__m128(__m128 v, int r) { static float a[4]; _mm_storeu_ps(a, v); return unifyValuef(a[r & 3]); }

#if defined(__AVX__)
static INLINE __m256d set__m256d(double d, int r) { static double a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return _mm256_loadu_pd(a); }
static INLINE double get__m256d(__m256d v, int r) { static double a[4]; _mm256_storeu_pd(a, v); return unifyValue(a[r & 3]); }
static INLINE __m256 set__m256(float d, int r) { static float a[8]; memrand(a, sizeof(a)); a[r & 7] = d; return _mm256_loadu_ps(a); }
static INLINE float get__m256(__m256 v, int r) { static float a[8]; _mm256_storeu_ps(a, v); return unifyValuef(a[r & 7]); }
#endif

#if defined(__AVX512F__)
static INLINE __m512d set__m512d(double d, int r) { static double a[8]; memrand(a, sizeof(a)); a[r & 7] = d; return _mm512_loadu_pd(a); }
static INLINE double get__m512d(__m512d v, int r) { static double a[8]; _mm512_storeu_pd(a, v); return unifyValue(a[r & 7]); }
static INLINE __m512 set__m512(float d, int r) { static float a[16]; memrand(a, sizeof(a)); a[r & 15] = d; return _mm512_loadu_ps(a); }
static INLINE float get__m512(__m512 v, int r) { static float a[16]; _mm512_storeu_ps(a, v); return unifyValuef(a[r & 15]); }
#endif
#endif // #if defined(__i386__) || defined(__x86_64__) || defined(_MSC_VER)

#if defined(__aarch64__) && defined(__ARM_NEON)
static INLINE VECTOR_CC float64x2_t setfloat64x2_t(double d, int r) { double a[2]; memrand(a, sizeof(a)); a[r & 1] = d; return vld1q_f64(a); }
static INLINE VECTOR_CC double getfloat64x2_t(float64x2_t v, int r) { double a[2]; vst1q_f64(a, v); return unifyValue(a[r & 1]); }
static INLINE VECTOR_CC float32x4_t setfloat32x4_t(float d, int r) { float a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return vld1q_f32(a); }
static INLINE VECTOR_CC float getfloat32x4_t(float32x4_t v, int r) { float a[4]; vst1q_f32(a, v); return unifyValuef(a[r & 3]); }
#endif

#ifdef __ARM_FEATURE_SVE
static INLINE svfloat64_t setsvfloat64_t(double d, int r) { double a[svcntd()]; memrand(a, sizeof(a)); a[r & (svcntd()-1)] = d; return svld1_f64(svptrue_b8(), a); }
static INLINE double getsvfloat64_t(svfloat64_t v, int r) { double a[svcntd()]; svst1_f64(svptrue_b8(), a, v); return unifyValue(a[r & (svcntd()-1)]); }
static INLINE svfloat32_t setsvfloat32_t(float d, int r)  { float  a[svcntw()]; memrand(a, sizeof(a)); a[r & (svcntw()-1)] = d; return svld1_f32(svptrue_b8(), a); }
static INLINE float getsvfloat32_t(svfloat32_t v, int r)  { float  a[svcntw()]; svst1_f32(svptrue_b8(), a, v); return unifyValuef(a[r & (svcntw()-1)]); }
#endif

#ifdef __VSX__
static INLINE vector_double setvector_double(double d, int r) { double a[2]; memrand(a, sizeof(a)); a[r & 1] = d; return (vector double) ( a[0], a[1] ); }
static INLINE double getvector_double(vector double v, int r) { double a[2]; return unifyValue(v[r & 1]); }
static INLINE vector_float setvector_float(float d, int r) { float a[4]; memrand(a, sizeof(a)); a[r & 3] = d; return (vector float) ( a[0], a[1], a[2], a[3] ); }
static INLINE float getvector_float(vector float v, int r) { float a[4]; return unifyValuef(v[r & 3]); }
#endif

//

// ATR = cinz_, NAME = sin, TYPE = d2, ULP = u35, EXT = sse2
#define FUNC(ATR, NAME, TYPE, ULP, EXT) Sleef_ ## ATR ## NAME ## TYPE ## _ ## ULP ## EXT
#define TYPE2(TYPE) Sleef_ ## TYPE ## _2
#define SET(TYPE) set ## TYPE
#define GET(TYPE) get ## TYPE

//

#define checkDigest(NAME, ULP) do {					\
    unsigned char d[16], mes[64], buf[64];				\
    MD5_Final(d, &ctx);							\
    snprintf((char *)mes, 60, "%s %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",	\
	    #NAME " " #ULP, d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],	\
	    d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]);		\
    if (fp != NULL) {							\
      fgets((char *)buf, 60, fp);					\
      if (strncmp((char *)mes, (char *)buf, strlen((char *)mes)) != 0) { \
	puts((char *)mes);						\
	puts((char *)buf);						\
	success = 0;							\
      }									\
    } else puts((char *)mes);						\
  } while(0)

//
//  printf("%.10g %.10g\n", arg, GET(TYPE)(vx, r));

#define exec_d_d(ATR, NAME, ULP, TYPE, TSX, EXT, arg) do {		\
    int r = xrand() & 0xffff;						\
    DPTYPE vx = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (arg, r));	\
    double fx = GET(TYPE)(vx, r);					\
    MD5_Update(&ctx, &fx, sizeof(double));				\
  } while(0)

#define test_d_d(NAME, ULP, START, END, NSTEP) do {		\
    static MD5_CTX ctx;						\
    memset(&ctx, 0, sizeof(MD5_CTX));				\
    MD5_Init(&ctx);						\
    double step = ((double)(END) - (double)(START))/NSTEP;	\
    for(double d = (START);d < (END);d += step)			\
      exec_d_d(ATR, NAME, ULP, DPTYPE, DPTYPESPEC, EXTSPEC, d);	\
    checkDigest(NAME, ULP);					\
  } while(0)

#define testu_d_d(NAME, ULP, START, END, NSTEP) do {			\
    static MD5_CTX ctx;							\
    memset(&ctx, 0, sizeof(MD5_CTX));					\
    MD5_Init(&ctx);							\
    uint64_t step = (d2u(END) - d2u(START))/NSTEP;			\
    for(uint64_t u = d2u(START);u < d2u(END);u += step)			\
      exec_d_d(ATR, NAME, ULP, DPTYPE, DPTYPESPEC, EXTSPEC, u2d(u));	\
    checkDigest(NAME, ULP);						\
  } while(0)

//

#define exec_d2_d(ATR, NAME, ULP, TYPE, TSX, EXT, arg) do {		\
    int r = xrand() & 0xffff;						\
    TYPE2(TYPE) vx2 = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE)(arg, r)); \
    double fxx = GET(TYPE)(vx2.x, r), fxy = GET(TYPE)(vx2.y, r);	\
    MD5_Update(&ctx, &fxx, sizeof(double));				\
    MD5_Update(&ctx, &fxy, sizeof(double));				\
  } while(0)

#define test_d2_d(NAME, ULP, START, END, NSTEP) do {			\
    static MD5_CTX ctx;							\
    MD5_Init(&ctx);							\
    double step = ((double)(END) - (double)(START))/NSTEP;		\
    for(double d = (START);d < (END);d += step)				\
      exec_d2_d(ATR, NAME, ULP, DPTYPE, DPTYPESPEC, EXTSPEC, d);	\
    checkDigest(NAME, ULP);						\
  } while(0)

//

#define exec_d_d_d(ATR, NAME, ULP, TYPE, TSX, EXT, argu, argv) do {	\
    int r = xrand() & 0xffff;						\
    DPTYPE vx = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (argu, r), SET(TYPE) (argv, r)); \
    double fx = GET(TYPE)(vx, r);					\
    MD5_Update(&ctx, &fx, sizeof(double));				\
  } while(0)

#define test_d_d_d(NAME, ULP, STARTU, ENDU, NSTEPU, STARTV, ENDV, NSTEPV) do { \
    static MD5_CTX ctx;							\
    MD5_Init(&ctx);							\
    double stepu = ((double)(ENDU) - (double)(STARTU))/NSTEPU;		\
    double stepv = ((double)(ENDV) - (double)(STARTV))/NSTEPV;		\
    for(double u = (STARTU);u < (ENDU);u += stepu)			\
      for(double v = (STARTV);v < (ENDV);v += stepv)			\
	exec_d_d_d(ATR, NAME, ULP, DPTYPE, DPTYPESPEC, EXTSPEC, u, v);	\
    checkDigest(NAME, ULP);						\
  } while(0)

//

#define exec_f_f(ATR, NAME, ULP, TYPE, TSX, EXT, arg) do {		\
    int r = xrand() & 0xffff;						\
    SPTYPE vx = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (arg, r));	\
    float fx = GET(TYPE)(vx, r);					\
    MD5_Update(&ctx, &fx, sizeof(float));				\
  } while(0)

#define test_f_f(NAME, ULP, START, END, NSTEP) do {		\
    static MD5_CTX ctx;						\
    MD5_Init(&ctx);						\
    float step = ((double)(END) - (double)(START))/NSTEP;	\
    for(float d = (START);d < (END);d += step)			\
      exec_f_f(ATR, NAME, ULP, SPTYPE, SPTYPESPEC, EXTSPEC, d);	\
    checkDigest(NAME ## f, ULP);				\
  } while(0)

#define testu_f_f(NAME, ULP, START, END, NSTEP) do {			\
    static MD5_CTX ctx;							\
    MD5_Init(&ctx);							\
    uint32_t step = (f2u(END) - f2u(START))/NSTEP;			\
    for(uint32_t u = f2u(START);u < f2u(END);u += step)			\
      exec_f_f(ATR, NAME, ULP, SPTYPE, SPTYPESPEC, EXTSPEC, u2f(u));	\
    checkDigest(NAME ## f, ULP);					\
  } while(0)

//

#define exec_f2_f(ATR, NAME, ULP, TYPE, TSX, EXT, arg) do {		\
    int r = xrand() & 0xffff;						\
    TYPE2(TYPE) vx2 = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (arg, r)); \
    float fxx = GET(TYPE)(vx2.x, r), fxy = GET(TYPE)(vx2.y, r);		\
    MD5_Update(&ctx, &fxx, sizeof(float));				\
    MD5_Update(&ctx, &fxy, sizeof(float));				\
  } while(0)

#define test_f2_f(NAME, ULP, START, END, NSTEP) do {			\
    static MD5_CTX ctx;							\
    MD5_Init(&ctx);							\
    float step = ((float)(END) - (float)(START))/NSTEP;			\
    for(float d = (START);d < (END);d += step)				\
      exec_f2_f(ATR, NAME, ULP, SPTYPE, SPTYPESPEC, EXTSPEC, d);	\
    checkDigest(NAME ## f, ULP);					\
  } while(0)

//

#define exec_f_f_f(ATR, NAME, ULP, TYPE, TSX, EXT, argu, argv) do {	\
    int r = xrand() & 0xffff;						\
    SPTYPE vx = FUNC(ATR, NAME, TSX, ULP, EXT) (SET(TYPE) (argu, r), SET(TYPE) (argv, r)); \
    float fx = GET(TYPE)(vx, r);					\
    MD5_Update(&ctx, &fx, sizeof(float));				\
  } while(0)

#define test_f_f_f(NAME, ULP, STARTU, ENDU, NSTEPU, STARTV, ENDV, NSTEPV) do { \
    static MD5_CTX ctx;							\
    MD5_Init(&ctx);							\
    float stepu = ((float)(ENDU) - (float)(STARTU))/NSTEPU;		\
    float stepv = ((float)(ENDV) - (float)(STARTV))/NSTEPV;		\
    for(float u = (STARTU);u < (ENDU);u += stepu)			\
      for(float v = (STARTV);v < (ENDV);v += stepv)			\
	exec_f_f_f(ATR, NAME, ULP, SPTYPE, SPTYPESPEC, EXTSPEC, u, v);	\
    checkDigest(NAME ## f, ULP);					\
  } while(0)

//

int success = 1;

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

  xsrand(time(NULL));

  //

  testu_d_d(sin, u35, 1e-300, 1e+8, 200001);
  testu_d_d(sin, u10, 1e-300, 1e+8, 200001);
  testu_d_d(cos, u35, 1e-300, 1e+8, 200001);
  testu_d_d(cos, u10, 1e-300, 1e+8, 200001);
  testu_d_d(tan, u35, 1e-300, 1e+8, 200001);
  testu_d_d(tan, u10, 1e-300, 1e+8, 200001);
  test_d2_d(sincos, u10, -1e+14, 1e+14, 200001);
  test_d2_d(sincos, u35, -1e+14, 1e+14, 200001);
  test_d2_d(sincospi, u05, -1e+14, 1e+14, 200001);
  test_d2_d(sincospi, u35, -1e+14, 1e+14, 200001);

  testu_d_d(log, u10, 1e-300, 1e+14, 200001);
  testu_d_d(log, u35, 1e-300, 1e+14, 200001);
  testu_d_d(log2, u10, 1e-300, 1e+14, 200001);
  testu_d_d(log2, u35, 1e-300, 1e+14, 200001);
  testu_d_d(log10, u10, 1e-300, 1e+14, 200001);
  testu_d_d(log1p, u10, 1e-300, 1e+14, 200001);
  test_d_d(exp, u10, -1000, 1000, 200001);
  test_d_d(exp2, u10, -1000, 1000, 200001);
  test_d_d(exp2, u35, -1000, 1000, 200001);
  test_d_d(exp10, u10, -1000, 1000, 200001);
  test_d_d(exp10, u35, -1000, 1000, 200001);
  test_d_d(expm1, u10, -1000, 1000, 200001);
  test_d_d_d(pow, u10, -100, 100, 451, -100, 100, 451);
  
  testu_d_d(cbrt, u10, 1e-14, 1e+14, 100001);
  testu_d_d(cbrt, u10, -1e-14, -1e+14, 100001);
  testu_d_d(cbrt, u35, 1e-14, 1e+14, 100001);
  testu_d_d(cbrt, u35, -1e-14, -1e+14, 100001);
  test_d_d_d(hypot, u05, -1e7, 1e7, 451, -1e7, 1e7, 451);
  test_d_d_d(hypot, u35, -1e7, 1e7, 451, -1e7, 1e7, 451);

  test_d_d(asin, u10, -1, 1, 200001);
  test_d_d(asin, u35, -1, 1, 200001);
  test_d_d(acos, u10, -1, 1, 200001);
  test_d_d(acos, u35, -1, 1, 200001);
  testu_d_d(atan, u10,  1e-3,  1e+7, 100001);
  testu_d_d(atan, u10, -1e-2, -1e+8, 100001);
  testu_d_d(atan, u35,  1e-3,  1e+7, 100001);
  testu_d_d(atan, u35, -1e-2, -1e+8, 100001);
  test_d_d_d(atan2, u10, -10, 10, 451, -10, 10, 451);
  test_d_d_d(atan2, u35, -10, 10, 451, -10, 10, 451);

  test_d_d(sinh, u10, -700, 700, 200001);
  test_d_d(cosh, u10, -700, 700, 200001);
  test_d_d(tanh, u10, -700, 700, 200001);
  test_d_d(asinh, u10, -700, 700, 200001);
  test_d_d(acosh, u10, 1, 700, 200001);
  test_d_d(atanh, u10, -700, 700, 200001);
  
  test_d_d(lgamma, u10, -5000, 5000, 200001);
  test_d_d(tgamma, u10, -10, 10, 200001);
  test_d_d(erf, u10, -100, 100, 200001);
  test_d_d(erfc, u15, -1, 100, 200001);

  test_d_d(fabs, , -100.5, 100.5, 200001);
  test_d_d_d(copysign, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_d_d_d(fmax, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_d_d_d(fmin, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_d_d_d(fdim, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_d_d_d(fmod, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_d2_d(modf, , -1e+14, 1e+14, 200001);
  test_d_d_d(nextafter, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);

  test_d_d(trunc, , -100, 100, 800);
  test_d_d(floor, , -100, 100, 800);
  test_d_d(ceil, , -100, 100, 800);
  test_d_d(round, , -100, 100, 800);
  test_d_d(rint, , -100, 100, 800);

  //

  testu_f_f(sin, u35, 1e-30, 1e+8, 200001);
  testu_f_f(sin, u10, 1e-30, 1e+8, 200001);
  testu_f_f(cos, u35, 1e-30, 1e+8, 200001);
  testu_f_f(cos, u10, 1e-30, 1e+8, 200001);
  testu_f_f(tan, u35, 1e-30, 1e+8, 200001);
  testu_f_f(tan, u10, 1e-30, 1e+8, 200001);
  test_f2_f(sincos, u10, -1e+14, 1e+14, 200001);
  test_f2_f(sincos, u35, -1e+14, 1e+14, 200001);
  test_f2_f(sincospi, u05, -10000, 10000, 200001);
  test_f2_f(sincospi, u35, -10000, 10000, 200001);

  testu_f_f(log, u10, 1e-30, 1e+14, 200001);
  testu_f_f(log, u35, 1e-30, 1e+14, 200001);
  testu_f_f(log2, u10, 1e-30, 1e+14, 200001);
  testu_f_f(log2, u35, 1e-30, 1e+14, 200001);
  testu_f_f(log10, u10, 1e-30, 1e+14, 200001);
  testu_f_f(log1p, u10, 1e-30, 1e+14, 200001);
  test_f_f(exp, u10, -1000, 1000, 200001);
  test_f_f(exp2, u10, -1000, 1000, 200001);
  test_f_f(exp2, u35, -1000, 1000, 200001);
  test_f_f(exp10, u10, -1000, 1000, 200001);
  test_f_f(exp10, u35, -1000, 1000, 200001);
  test_f_f(expm1, u10, -1000, 1000, 200001);
  test_f_f_f(pow, u10, -100, 100, 451, -100, 100, 451);
  
  testu_f_f(cbrt, u10, 1e-14, 1e+14, 100001);
  testu_f_f(cbrt, u10, -1e-14, -1e+14, 100001);
  testu_f_f(cbrt, u35, 1e-14, 1e+14, 100001);
  testu_f_f(cbrt, u35, -1e-14, -1e+14, 100001);
  test_f_f_f(hypot, u05, -1e7, 1e7, 451, -1e7, 1e7, 451);
  test_f_f_f(hypot, u35, -1e7, 1e7, 451, -1e7, 1e7, 451);

  test_f_f(asin, u10, -1, 1, 200001);
  test_f_f(asin, u35, -1, 1, 200001);
  test_f_f(acos, u10, -1, 1, 200001);
  test_f_f(acos, u35, -1, 1, 200001);
  testu_f_f(atan, u10,  1e-3,  1e+7, 100001);
  testu_f_f(atan, u10, -1e-2, -1e+8, 100001);
  testu_f_f(atan, u35,  1e-3,  1e+7, 100001);
  testu_f_f(atan, u35, -1e-2, -1e+8, 100001);
  test_f_f_f(atan2, u10, -10, 10, 451, -10, 10, 451);
  test_f_f_f(atan2, u35, -10, 10, 451, -10, 10, 451);

  test_f_f(sinh, u10, -88, 88, 200001);
  test_f_f(cosh, u10, -88, 88, 200001);
  test_f_f(tanh, u10, -88, 88, 200001);
  test_f_f(asinh, u10, -88, 88, 200001);
  test_f_f(acosh, u10, 1, 88, 200001);
  test_f_f(atanh, u10, -88, 88, 200001);
  
  test_f_f(lgamma, u10, -5000, 5000, 200001);
  test_f_f(tgamma, u10, -10, 10, 200001);
  test_f_f(erf, u10, -100, 100, 200001);
  test_f_f(erfc, u15, -1, 100, 200001);

  test_f_f(fabs, , -100.5, 100.5, 200001);
  test_f_f_f(copysign, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_f_f_f(fmax, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_f_f_f(fmin, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_f_f_f(fdim, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_f_f_f(fmod, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);
  test_f2_f(modf, , -1e+14, 1e+14, 200001);
  test_f_f_f(nextafter, , -1e+10, 1e+10, 451, -1e+10, 1e+10, 451);

  test_f_f(trunc, , -100, 100, 800);
  test_f_f(floor, , -100, 100, 800);
  test_f_f(ceil, , -100, 100, 800);
  test_f_f(round, , -100, 100, 800);
  test_f_f(rint, , -100, 100, 800);

  test_f_f(fastsin, u3500, 1e-30, 100, 200001);
  test_f_f(fastcos, u3500, 1e-30, 100, 200001);
  test_f_f_f(fastpow, u3500, 0, 25, 451, -25, 25, 451);

  //

  if (fp != NULL) fclose(fp);

  exit(success ? 0 : -1);
}
