//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpfr.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#ifdef ENABLE_SYS_getrandom
#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/random.h>
#endif

#include "sleef.h"

#if defined(__APPLE__)
static int isinff(float x) { return x == __builtin_inff() || x == -__builtin_inff(); }
#endif

#define DORENAME

#ifdef ENABLE_SSE2
#define CONFIG 2
#include "helpersse2.h"
#include "renamesse2.h"
typedef Sleef___m128d_2 vdouble2;
typedef Sleef___m128_2 vfloat2;
#endif

#ifdef ENABLE_AVX
#define CONFIG 1
#include "helperavx.h"
#include "renameavx.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_FMA4
#define CONFIG 4
#include "helperavx.h"
#include "renamefma4.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_AVX2
#define CONFIG 1
#include "helperavx2.h"
#include "renameavx2.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_AVX512F
#define CONFIG 1
#include "helperavx512f.h"
#include "renameavx512f.h"
typedef Sleef___m512d_2 vdouble2;
typedef Sleef___m512_2 vfloat2;
#endif

#ifdef ENABLE_VECEXT
#define CONFIG 1
#include "helpervecext.h"
#include "norename.h"
#endif

#ifdef ENABLE_PUREC
#define CONFIG 1
#include "helperpurec.h"
#include "norename.h"
#endif

mpfr_t fra, frb, frc, frd, frw, frx, fry, frz;

#define DENORMAL_FLT_MIN (1.40130e-45f)
#define POSITIVE_INFINITYf ((float)INFINITY)
#define NEGATIVE_INFINITYf (-(float)INFINITY)

int isnumber(double x) { return !isinf(x) && !isnan(x); }
int isPlusZero(double x) { return x == 0 && copysign(1, x) == 1; }
int isMinusZero(double x) { return x == 0 && copysign(1, x) == -1; }

double countULP(float d, mpfr_t c) {
  float c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) return 10000;
  //if (isPlusZero(c2) && !isPlusZero(d)) return 10003;
  //if (isMinusZero(c2) && !isMinusZero(d)) return 10004;
  if (isnan(c2) && isnan(d)) return 0;
  if (isnan(c2) || isnan(d)) return 10001;
  if (c2 == POSITIVE_INFINITYf && d == POSITIVE_INFINITYf) return 0;
  if (c2 == NEGATIVE_INFINITYf && d == NEGATIVE_INFINITYf) return 0;
  if (!isnumber(c2) || !isnumber(d)) return 10002;

  //

  int e;
  frexpl(mpfr_get_d(c, GMP_RNDN), &e);
  mpfr_set_ld(frw, fmaxl(ldexpl(1.0, e-24), DENORMAL_FLT_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fry, frd, c, GMP_RNDN);
  mpfr_div(fry, fry, frw, GMP_RNDN);
  double u = fabs(mpfr_get_d(fry, GMP_RNDN));

  return u;
}

double countULP2(float d, mpfr_t c) {
  float c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) return 10000;
  //if (isPlusZero(c2) && !isPlusZero(d)) return 10003;
  //if (isMinusZero(c2) && !isMinusZero(d)) return 10004;
  if (isnan(c2) && isnan(d)) return 0;
  if (isnan(c2) || isnan(d)) return 10001;
  if (c2 == POSITIVE_INFINITYf && d == POSITIVE_INFINITYf) return 0;
  if (c2 == NEGATIVE_INFINITYf && d == NEGATIVE_INFINITYf) return 0;
  if (!isnumber(c2) || !isnumber(d)) return 10002;

  //
  
  int e;
  frexpl(mpfr_get_d(c, GMP_RNDN), &e);
  mpfr_set_ld(frw, fmaxl(ldexpl(1.0, e-24), FLT_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fry, frd, c, GMP_RNDN);
  mpfr_div(fry, fry, frw, GMP_RNDN);
  double u = fabs(mpfr_get_d(fry, GMP_RNDN));

  return u;
}

typedef union {
  double d;
  uint64_t u64;
  int64_t i64;
} conv64_t;

typedef union {
  float f;
  uint32_t u32;
  int32_t i32;
} conv32_t;

float rnd() {
  conv32_t c;
  switch(random() & 15) {
  case 0: return  INFINITY;
  case 1: return -INFINITY;
  }
#ifdef ENABLE_SYS_getrandom
  syscall(SYS_getrandom, &c.u32, sizeof(c.u32), 0);
#else
  c.u32 = (uint32_t)random() | ((uint32_t)random() << 31);
#endif
  return c.f;
}

float rnd_fr() {
  conv32_t c;
  do {
#ifdef ENABLE_SYS_getrandom
    syscall(SYS_getrandom, &c.u32, sizeof(c.u32), 0);
#else
    c.u32 = (uint32_t)random() | ((uint32_t)random() << 31);
#endif
  } while(!isnumber(c.f));
  return c.f;
}

float rnd_zo() {
  conv32_t c;
  do {
#ifdef ENABLE_SYS_getrandom
    syscall(SYS_getrandom, &c.u32, sizeof(c.u32), 0);
#else
    c.u32 = (uint32_t)random() | ((uint32_t)random() << 31);
#endif
  } while(!isnumber(c.f) || c.f < -1 || 1 < c.f);
  return c.f;
}

void sinpifr(mpfr_t ret, double d) {
  mpfr_t frpi, frd;
  mpfr_inits(frpi, frd, NULL);

  mpfr_const_pi(frpi, GMP_RNDN);
  mpfr_set_d(frd, 1.0, GMP_RNDN);
  mpfr_mul(frpi, frpi, frd, GMP_RNDN);
  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_mul(frd, frpi, frd, GMP_RNDN);
  mpfr_sin(ret, frd, GMP_RNDN);

  mpfr_clears(frpi, frd, NULL);
}

void cospifr(mpfr_t ret, double d) {
  mpfr_t frpi, frd;
  mpfr_inits(frpi, frd, NULL);

  mpfr_const_pi(frpi, GMP_RNDN);
  mpfr_set_d(frd, 1.0, GMP_RNDN);
  mpfr_mul(frpi, frpi, frd, GMP_RNDN);
  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_mul(frd, frpi, frd, GMP_RNDN);
  mpfr_cos(ret, frd, GMP_RNDN);

  mpfr_clears(frpi, frd, NULL);
}

vfloat vset(vfloat v, int idx, float d) {
  float a[VECTLENSP];
  vstoreu_v_p_vf(a, v);
  a[idx] = d;
  return vloadu_vf_p(a);
}

float vget(vfloat v, int idx) {
  float a[VECTLENSP];
  vstoreu_v_p_vf(a, v);
  return a[idx];
}

int main(int argc,char **argv)
{
  mpfr_set_default_prec(256);
  mpfr_inits(fra, frb, frc, frd, frw, frx, fry, frz, NULL);

  conv32_t cd;
  float d, t;
  float d2, zo;
  vfloat vd = vcast_vf_f(0);
  vfloat vd2 = vcast_vf_f(0);
  vfloat vzo = vcast_vf_f(0);
  vfloat vad = vcast_vf_f(0);
  vfloat2 sc, sc2;
  int cnt;
  
  srandom(time(NULL));

#if 0
  cd.f = M_PI;
  mpfr_set_d(frx, cd.f, GMP_RNDN);
  cd.i32+=3;
  printf("%g\n", countULP(cd.f, frx));
#endif

  const float rangemax = 39000;
  
  for(cnt = 0;;cnt++) {
    int e = cnt % VECTLENSP;
    switch(cnt & 7) {
    case 0:
      d = rnd();
      d2 = rnd();
      zo = rnd();
      break;
    case 1:
      cd.f = rint((2 * (double)random() / RAND_MAX - 1) * 1e+10) * M_PI_4;
      cd.i32 += (random() & 0xff) - 0x7f;
      d = cd.f;
      d2 = rnd();
      zo = rnd();
      break;
    default:
      d = rnd_fr();
      d2 = rnd_fr();
      zo = rnd_zo();
      break;
    }

    vd = vset(vd, e, d);
    vd2 = vset(vd2, e, d2);
    vzo = vset(vzo, e, zo);
    vad = vset(vad, e, fabs(d));

    sc  = xsincospif_u05(vd);
    sc2 = xsincospif_u35(vd);

    {
      const double rangemax2 = 1e+7/4;
      
      sinpifr(frx, d);

      double u0 = countULP2(t = vget(sc.x, e), frx);

      if (u0 != 0 && ((fabs(d) <= rangemax2 && u0 > 0.505) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sincospif_u05 sin arg=%.20g ulp=%.20g\n", d, u0);
      }

      double u1 = countULP2(t = vget(sc2.x, e), frx);

      if (u1 != 0 && ((fabs(d) <= rangemax2 && u1 > 1.6) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sincospif_u35 sin arg=%.20g ulp=%.20g\n", d, u1);
      }
    }

    {
      const double rangemax2 = 1e+7/4;
      
      cospifr(frx, d);

      double u0 = countULP2(t = vget(sc.y, e), frx);

      if (u0 != 0 && ((fabs(d) <= rangemax2 && u0 > 0.505) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sincospif_u05 cos arg=%.20g ulp=%.20g\n", d, u0);
      }

      double u1 = countULP2(t = vget(sc.y, e), frx);

      if (u1 != 0 && ((fabs(d) <= rangemax2 && u1 > 1.5) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sincospif_u35 cos arg=%.20g ulp=%.20g\n", d, u1);
      }
    }
    
    sc = xsincosf(vd);
    sc2 = xsincosf_u1(vd);
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sin(frx, frx, GMP_RNDN);

      float u0 = countULP(t = vget(xsinf(vd), e), frx);
      
      if (u0 != 0 && ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sinf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      float u1 = countULP(t = vget(sc.x, e), frx);
      
      if (u1 != 0 && ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sincosf sin arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      float u2 = countULP(t = vget(xsinf_u1(vd), e), frx);
      
      if (u2 != 0 && ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sinf_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      float u3 = countULP(t = vget(sc2.x, e), frx);
      
      if (u3 != 0 && ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sincosf_u1 sin arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cos(frx, frx, GMP_RNDN);

      float u0 = countULP(t = vget(xcosf(vd), e), frx);
      
      if (u0 != 0 && ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " cosf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      float u1 = countULP(t = vget(sc.y, e), frx);
      
      if (u1 != 0 && ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sincosf cos arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      float u2 = countULP(t = vget(xcosf_u1(vd), e), frx);
      
      if (u2 != 0 && ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " cosf_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      float u3 = countULP(t = vget(sc2.y, e), frx);
      
      if (u3 != 0 && ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isnumber(t))) {
	printf(ISANAME " sincosf_u1 cos arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tan(frx, frx, GMP_RNDN);

      float u0 = countULP(t = vget(xtanf(vd), e), frx);
      
      if (u0 != 0 && ((fabs(d) < rangemax && u0 > 3.5) || isnan(t))) {
	printf(ISANAME " tanf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      float u1 = countULP(t = vget(xtanf_u1(vd), e), frx);
      
      if (u1 != 0 && ((fabs(d) <= rangemax && u1 > 1) || isnan(t))) {
	printf(ISANAME " tanf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, fabsf(d), GMP_RNDN);
      mpfr_log(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xlogf(vad), e), frx);
      
      if (u0 > 3.5) {
	printf(ISANAME " logf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = vget(xlogf_u1(vad), e), frx);
      
      if (u1 > 1) {
	printf(ISANAME " logf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, fabsf(d), GMP_RNDN);
      mpfr_log10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xlog10f(vad), e), frx);
      
      if (u0 > 1) {
	printf(ISANAME " log10f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_log1p(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xlog1pf(vd), e), frx);
      
      if ((-1 <= d && d <= 1e+38 && u0 > 1) ||
	  (d < -1 && !isnan(t)) ||
	  (d > 1e+38 && !(u0 <= 1 || isinf(t)))) {
	printf(ISANAME " log1pf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xexpf(vd), e), frx);
      
      if (u0 > 1) {
	printf(ISANAME " expf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp2(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xexp2f(vd), e), frx);
      
      if (u0 > 1) {
	printf(ISANAME " exp2f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xexp10f(vd), e), frx);
      
      if (u0 > 1) {
	printf(ISANAME " exp10f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_expm1(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xexpm1f(vd), e), frx);
      
      if (u0 > 1) {
	printf(ISANAME " expm1f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_pow(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xpowf(vd2, vd), e), frx);
      
      if (u0 > 1) {
	printf(ISANAME " powf arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	printf("correct = %g, test = %g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cbrt(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xcbrtf(vd), e), frx);
      
      if (u0 > 3.5) {
	printf(ISANAME " cbrtf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = vget(xcbrtf_u1(vd), e), frx);
      
      if (u1 > 1) {
	printf(ISANAME " cbrtf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_asin(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xasinf(vzo), e), frx);
      
      if (u0 > 3.5) {
	printf(ISANAME " asinf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = vget(xasinf_u1(vzo), e), frx);
      
      if (u1 > 1) {
	printf(ISANAME " asinf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_acos(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xacosf(vzo), e), frx);
      
      if (u0 > 3.5) {
	printf(ISANAME " acosf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = vget(xacosf_u1(vzo), e), frx);
      
      if (u1 > 1) {
	printf(ISANAME " acosf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atan(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xatanf(vd), e), frx);
      
      if (u0 > 3.5) {
	printf(ISANAME " atanf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = vget(xatanf_u1(vd), e), frx);
      
      if (u1 > 1) {
	printf(ISANAME " atanf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_atan2(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xatan2f(vd2, vd), e), frx);
      
      if (u0 > 3.5) {
	printf(ISANAME " atan2f arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout);
      }

      double u1 = countULP2(t = vget(xatan2f_u1(vd2, vd), e), frx);
      
      if (u1 > 1) {
	printf(ISANAME " atan2f_u1 arg=%.20g, %.20g ulp=%.20g\n", d2, d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xsinhf(vd), e), frx);
      
      if ((fabs(d) <= 88.5 && u0 > 1) ||
	  (d >  88.5 && !(u0 <= 1 || (isinf(t) && t > 0))) ||
	  (d < -88.5 && !(u0 <= 1 || (isinf(t) && t < 0)))) {
	printf(ISANAME " sinhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xcoshf(vd), e), frx);
      
      if ((fabs(d) <= 88.5 && u0 > 1) || !(u0 <= 1 || (isinf(t) && t > 0))) {
	printf(ISANAME " coshf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xtanhf(vd), e), frx);
      
      if (u0 > 1.0001) {
	printf(ISANAME " tanhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_asinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xasinhf(vd), e), frx);
      
      if ((fabs(d) < sqrt(FLT_MAX) && u0 > 1.0001) ||
	  (d >=  sqrt(FLT_MAX) && !(u0 <= 1.0001 || (isinf(t) && t > 0))) ||
	  (d <= -sqrt(FLT_MAX) && !(u0 <= 1.0001 || (isinf(t) && t < 0)))) {
	printf(ISANAME " asinhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_acosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xacoshf(vd), e), frx);
      
      if ((fabs(d) < sqrt(FLT_MAX) && u0 > 1.0001) ||
	  (d >=  sqrt(FLT_MAX) && !(u0 <= 1.0001 || (isinff(t) && t > 0))) ||
	  (d <= -sqrt(FLT_MAX) && !isnan(t))) {
	printf(ISANAME " acoshf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = vget(xatanhf(vd), e), frx);
      
      if (u0 > 1.0001) {
	printf(ISANAME " atanhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
  }
}
