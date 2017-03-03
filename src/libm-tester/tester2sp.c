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

#define DORENAME
#include "rename.h"

#if defined(__APPLE__)
static int isinff(float x) { return x == __builtin_inff() || x == -__builtin_inff(); }
#endif

#define DENORMAL_FLT_MIN (1.4012984643248170709e-45f)
#define POSITIVE_INFINITYf ((float)INFINITY)
#define NEGATIVE_INFINITYf (-(float)INFINITY)

int isnumber(double x) { return !isinf(x) && !isnan(x); }
int isPlusZero(double x) { return x == 0 && copysign(1, x) == 1; }
int isMinusZero(double x) { return x == 0 && copysign(1, x) == -1; }

mpfr_t fra, frb, frc, frd;

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
  mpfr_set_ld(frb, fmaxl(ldexpl(1.0, e-24), DENORMAL_FLT_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fra, frd, c, GMP_RNDN);
  mpfr_div(fra, fra, frb, GMP_RNDN);
  double u = fabs(mpfr_get_d(fra, GMP_RNDN));

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
  mpfr_set_ld(frb, fmaxl(ldexpl(1.0, e-24), FLT_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fra, frd, c, GMP_RNDN);
  mpfr_div(fra, fra, frb, GMP_RNDN);
  double u = fabs(mpfr_get_d(fra, GMP_RNDN));

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

static float nexttoward0f(float x, int n) {
  union {
    float f;
    int32_t u;
  } cx;
  cx.f = x;
  cx.u -= n;
  return x == 0 ? 0 : cx.f;
}

float rnd() {
  conv32_t c;
  switch(random() & 63) {
  case 0: return nexttoward0f( 0.0, -(random() & ((1 << (random() & 31)) - 1)));
  case 1: return nexttoward0f(-0.0, -(random() & ((1 << (random() & 31)) - 1)));
  case 2: return nexttoward0f( INFINITY, (random() & ((1 << (random() & 31)) - 1)));
  case 3: return nexttoward0f(-INFINITY, (random() & ((1 << (random() & 31)) - 1)));
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

int main(int argc,char **argv)
{
  mpfr_t frw, frx, fry, frz;

  mpfr_set_default_prec(256);
  mpfr_inits(fra, frb, frc, frd, frw, frx, fry, frz, NULL);

  conv32_t cd;
  float d, t;
  float d2, d3, zo;

  int cnt, ecnt = 0;
  
  srandom(time(NULL));

#if 0
  cd.f = M_PI;
  mpfr_set_d(frx, cd.f, GMP_RNDN);
  cd.i32+=3;
  printf("%g\n", countULP(cd.f, frx));
#endif

  const float rangemax = 39000;
  
  for(cnt = 0;ecnt < 1000;cnt++) {
    switch(cnt & 7) {
    case 0:
      d = rnd();
      d2 = rnd();
      d3 = rnd();
      zo = rnd();
      break;
    case 1:
      cd.f = rint((2 * (double)random() / RAND_MAX - 1) * 1e+10) * M_PI_4;
      cd.i32 += (random() & 0xff) - 0x7f;
      d = cd.f;
      d2 = rnd();
      d3 = rnd();
      zo = rnd();
      break;
    default:
      d = rnd_fr();
      d2 = rnd_fr();
      d3 = rnd_fr();
      zo = rnd_zo();
      break;
    }

    Sleef_float2 sc  = xsincospif_u05(d);
    Sleef_float2 sc2 = xsincospif_u35(d);

    {
      const float rangemax2 = 1e+7/4;
      
      sinpifr(frx, d);

      double u0 = countULP2(t = sc.x, frx);

      if (u0 != 0 && ((fabs(d) <= rangemax2 && u0 > 0.505) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sincospif_u05 sin arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      double u1 = countULP2(t = sc2.x, frx);

      if (u1 != 0 && ((fabs(d) <= rangemax2 && u1 > 2.0) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sincospif_u35 sin arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }
    }

    {
      const float rangemax2 = 1e+7/4;
      
      cospifr(frx, d);

      double u0 = countULP2(t = sc.y, frx);

      if (u0 != 0 && ((fabs(d) <= rangemax2 && u0 > 0.505) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sincospif_u05 cos arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      double u1 = countULP2(t = sc.y, frx);

      if (u1 != 0 && ((fabs(d) <= rangemax2 && u1 > 2.0) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sincospif_u35 cos arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }
    }

    sc = xsincosf(d);
    sc2 = xsincosf_u1(d);
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sin(frx, frx, GMP_RNDN);

      float u0 = countULP(t = xsinf(d), frx);
      
      if (u0 != 0 && ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sinf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      float u1 = countULP(t = sc.x, frx);
      
      if (u1 != 0 && ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sincosf sin arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }

      float u2 = countULP(t = xsinf_u1(d), frx);
      
      if (u2 != 0 && ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sinf_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout); ecnt++;
      }

      float u3 = countULP(t = sc2.x, frx);
      
      if (u3 != 0 && ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sincosf_u1 sin arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cos(frx, frx, GMP_RNDN);

      float u0 = countULP(t = xcosf(d), frx);
      
      if (u0 != 0 && ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C cosf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      float u1 = countULP(t = sc.y, frx);
      
      if (u1 != 0 && ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sincosf cos arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }

      float u2 = countULP(t = xcosf_u1(d), frx);
      
      if (u2 != 0 && ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C cosf_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout); ecnt++;
      }

      float u3 = countULP(t = sc2.y, frx);
      
      if (u3 != 0 && ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isnumber(t))) {
	printf("Pure C sincosf_u1 cos arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tan(frx, frx, GMP_RNDN);

      float u0 = countULP(t = xtanf(d), frx);
      
      if (u0 != 0 && ((fabs(d) < rangemax && u0 > 3.5) || isnan(t))) {
	printf("Pure C tanf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      float u1 = countULP(t = xtanf_u1(d), frx);
      
      if (u1 != 0 && ((fabs(d) <= rangemax && u1 > 1) || isnan(t))) {
	printf("Pure C tanf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, fabsf(d), GMP_RNDN);
      mpfr_log(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlogf(fabsf(d)), frx);
      
      if (u0 > 3.5) {
	printf("Pure C logf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      double u1 = countULP(t = xlogf_u1(fabsf(d)), frx);
      
      if (u1 > 1) {
	printf("Pure C logf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, fabsf(d), GMP_RNDN);
      mpfr_log10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog10f(fabsf(d)), frx);
      
      if (u0 > 1) {
	printf("Pure C log10f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_log1p(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog1pf(d), frx);
      
      if ((-1 <= d && d <= 1e+38 && u0 > 1) ||
	  (d < -1 && !isnan(t)) ||
	  (d > 1e+38 && !(u0 <= 1 || isinf(t)))) {
	printf("Pure C log1pf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexpf(d), frx);
      
      if (u0 > 1) {
	printf("Pure C expf arg=%.20g ulp=%.20g\n", d, u0);
	printf("correct = %g, test = %g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp2(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp2f(d), frx);
      
      if (u0 > 1) {
	printf("Pure C exp2f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp10f(d), frx);
      
      if (u0 > 1) {
	printf("Pure C exp10f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_expm1(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexpm1f(d), frx);
      
      if (u0 > 1) {
	printf("Pure C expm1f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_pow(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xpowf(d2, d), frx);
      
      if (u0 > 1) {
	printf("Pure C powf arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cbrt(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcbrtf(d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C cbrtf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      double u1 = countULP(t = xcbrtf_u1(d), frx);
      
      if (u1 > 1) {
	printf("Pure C cbrtf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_asin(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xasinf(zo), frx);
      
      if (u0 > 3.5) {
	printf("Pure C asinf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      double u1 = countULP(t = xasinf_u1(zo), frx);
      
      if (u1 > 1) {
	printf("Pure C asinf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_acos(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xacosf(zo), frx);
      
      if (u0 > 3.5) {
	printf("Pure C acosf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      double u1 = countULP(t = xacosf_u1(zo), frx);
      
      if (u1 > 1) {
	printf("Pure C acosf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atan(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatanf(d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C atanf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }

      double u1 = countULP(t = xatanf_u1(d), frx);
      
      if (u1 > 1) {
	printf("Pure C atanf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_atan2(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xatan2f(d2, d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C atan2f arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout); ecnt++;
      }

      double u1 = countULP2(t = xatan2f_u1(d2, d), frx);
      
      if (u1 > 1) {
	printf("Pure C atan2f_u1 arg=%.20g, %.20g ulp=%.20g\n", d2, d, u1);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xsinhf(d), frx);
      
      if ((fabs(d) <= 88.5 && u0 > 1) ||
	  (d >  88.5 && !(u0 <= 1 || (isinf(t) && t > 0))) ||
	  (d < -88.5 && !(u0 <= 1 || (isinf(t) && t < 0)))) {
	printf("Pure C sinhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcoshf(d), frx);
      
      if ((fabs(d) <= 88.5 && u0 > 1) || !(u0 <= 1 || (isinf(t) && t > 0))) {
	printf("Pure C coshf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xtanhf(d), frx);
      
      if (u0 > 1.0001) {
	printf("Pure C tanhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_asinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xasinhf(d), frx);
      
      if ((fabs(d) < sqrt(FLT_MAX) && u0 > 1.0001) ||
	  (d >=  sqrt(FLT_MAX) && !(u0 <= 1.0001 || (isinf(t) && t > 0))) ||
	  (d <= -sqrt(FLT_MAX) && !(u0 <= 1.0001 || (isinf(t) && t < 0)))) {
	printf("Pure C asinhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_acosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xacoshf(d), frx);
      
      if ((fabs(d) < sqrt(FLT_MAX) && u0 > 1.0001) ||
	  (d >=  sqrt(FLT_MAX) && !(u0 <= 1.0001 || (isinff(t) && t > 0))) ||
	  (d <= -sqrt(FLT_MAX) && !isnan(t))) {
	printf("Pure C acoshf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatanhf(d), frx);
      
      if (u0 > 1.0001) {
	printf("Pure C atanhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }

    //

    {
      int exp = (random() & 8191) - 4096;
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_exp(frx, mpfr_get_exp(frx) + exp);

      double u0 = countULP(t = xldexpf(d, exp), frx);

      if (u0 > 0.5001) {
	printf("Pure C ldexpf arg=%.20g %d ulp=%.20g\n", d, exp, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }
  
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_abs(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xfabsf(d), frx);
      
      if (u0 != 0) {
	printf("Pure C fabsf arg=%.20g ulp=%.20g\n", d, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_copysign(frx, frx, fry, GMP_RNDN);

      double u0 = countULP(t = xcopysignf(d, d2), frx);
      
      if (u0 != 0 && !isnan(d2)) {
	printf("Pure C copysignf arg=%.20g, %.20g ulp=%.20g\n", d, d2, u0);
	printf("correct = %g, test = %g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_max(frx, frx, fry, GMP_RNDN);

      double u0 = countULP(t = xfmaxf(d, d2), frx);
      
      if (u0 != 0) {
	printf("Pure C fmaxf arg=%.20g, %.20g ulp=%.20g\n", d, d2, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_min(frx, frx, fry, GMP_RNDN);

      double u0 = countULP(t = xfminf(d, d2), frx);
      
      if (u0 != 0) {
	printf("Pure C fminf arg=%.20g, %.20g ulp=%.20g\n", d, d2, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_dim(frx, frx, fry, GMP_RNDN);

      double u0 = countULP(t = xfdimf(d, d2), frx);
      
      if (u0 > 0.5) {
	printf("Pure C fdimf arg=%.20g, %.20g ulp=%.20g\n", d, d2, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_trunc(frx, frx);

      double u0 = countULP(t = xtruncf(d), frx);
      
      if (u0 != 0) {
	printf("Pure C truncf arg=%.20g ulp=%.20g\n", d, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_floor(frx, frx);

      double u0 = countULP(t = xfloorf(d), frx);
      
      if (u0 != 0) {
	printf("Pure C floorf arg=%.20g ulp=%.20g\n", d, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_ceil(frx, frx);

      double u0 = countULP(t = xceilf(d), frx);
      
      if (u0 != 0) {
	printf("Pure C ceilf arg=%.20g ulp=%.20g\n", d, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_round(frx, frx);

      double u0 = countULP(t = xroundf(d), frx);
      
      if (u0 != 0) {
	printf("Pure C roundf arg=%.24g ulp=%.20g\n", d, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	printf("%.20g\n", xrint(d));
	fflush(stdout); ecnt++;
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_rint(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xrintf(d), frx);
      
      if (u0 != 0) {
	printf("Pure C rintf arg=%.24g ulp=%.20g\n", d, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	double debug = xround(d);
	printf("%.20g\n", debug);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_modf(fry, frz, frx, GMP_RNDN);

      Sleef_float2 t2 = xmodff(d);
      double u0 = countULP(t2.x, frz);
      double u1 = countULP(t2.y, fry);

      if (u0 != 0 || u1 != 0) {
	printf("Pure C modff arg=%.20g ulp=%.20g %.20g\n", d, u0, u1);
	printf("correct = %.20g, %.20g\n", mpfr_get_d(frz, GMP_RNDN), mpfr_get_d(fry, GMP_RNDN));
	printf("test    = %.20g, %.20g\n", t2.x, t2.y);
	fflush(stdout); ecnt++;
      }
    }


    {
      t = xnextafterf(d, d2);
      double c = nextafterf(d, d2);

      if (!(isnan(t) && isnan(c)) && t != c) {
	printf("Pure C nextafterf arg=%.20g, %.20g\n", d, d2);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_exp(frx, 0);

      double u0 = countULP(t = xfrfrexpf(d), frx);

      if (d != 0 && isnumber(d) && u0 != 0) {
	printf("Pure C frfrexpf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      int cexp = mpfr_get_exp(frx);

      int texp = xexpfrexpf(d);
      
      if (d != 0 && isnumber(d) && cexp != texp) {
	printf("Pure C expfrexpf arg=%.20g\n", d);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_hypot(frx, frx, fry, GMP_RNDN);

      double u0 = countULP2(t = xhypotf_u05(d, d2), frx);
      double c = mpfr_get_d(frx, GMP_RNDN);

      if (u0 > 0.5001) {
	printf("Pure C hypotf_u05 arg=%.20g, %.20g  ulp=%.20g\n", d, d2, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_hypot(frx, frx, fry, GMP_RNDN);

      double u0 = countULP2(t = xhypotf_u35(d, d2), frx);
      double c = mpfr_get_d(frx, GMP_RNDN);

      if (u0 >= 3.5) {
	printf("Pure C hypotf_u35 arg=%.20g, %.20g  ulp=%.20g\n", d, d2, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_fmod(frx, frx, fry, GMP_RNDN);

      double u0 = countULP(t = xfmodf(d, d2), frx);
      long double c = mpfr_get_ld(frx, GMP_RNDN);

      if (fabs((double)d / d2) < 1e+38 && u0 > 0.5) {
	printf("Pure C fmodf arg=%.20g, %.20g  ulp=%.20g\n", d, d2, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_set_d(frz, d3, GMP_RNDN);
      mpfr_fma(frx, frx, fry, frz, GMP_RNDN);

      double u0 = countULP2(t = xfmaf(d, d2, d3), frx);
      double c = mpfr_get_d(frx, GMP_RNDN);

      if ((-1e+36 < c && c < 1e+36 && u0 > 0.5001) ||
	  !(u0 <= 0.5001 || isinf(t))) {
	printf("Pure C fmaf arg=%.20g, %.20g, %.20g  ulp=%.20g\n", d, d2, d3, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sqrt(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xsqrtf_u05(d), frx);
      
      if (u0 > 0.5001) {
	printf("Pure C sqrtf_u05 arg=%.20g ulp=%.20g\n", d, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sqrt(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xsqrtf_u35(d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C sqrtf_u35 arg=%.20g ulp=%.20g\n", d, u0);
	printf("correct = %.20g, test = %.20g\n", mpfr_get_d(frx, GMP_RNDN), t);
	fflush(stdout); ecnt++;
      }
    }

  }
}
