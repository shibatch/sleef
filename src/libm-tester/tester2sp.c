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

mpfr_t fra, frb, frc, frd, frw, frx, fry, frz;

#define DENORMAL_FLT_MIN (1.40130e-45f)

double countULP(float d, mpfr_t c) {
  float c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) return 10000;
  if (!isfinite(c2) && !isfinite(d)) return 0;

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
  if (!isfinite(c2) && !isfinite(d)) return 0;

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

float rnd_fr() {
  conv32_t c;
  do {
#ifdef ENABLE_SYS_getrandom
    syscall(SYS_getrandom, &c.u32, sizeof(c.u32), 0);
#else
    c.u32 = (uint32_t)random() | ((uint32_t)random() << 31);
#endif
  } while(!isfinite(c.f));
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
  } while(!isfinite(c.f) || c.f < -1 || 1 < c.f);
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
  mpfr_set_default_prec(256);
  mpfr_inits(fra, frb, frc, frd, frw, frx, fry, frz, NULL);

  conv32_t cd;
  float d, t;
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
    switch(cnt & 7) {
    case 0:
      d = (2 * (float)random() / RAND_MAX - 1) * rangemax;
      break;
    case 1:
      cd.f = rint((2 * (float)random() / RAND_MAX - 1) * rangemax) * M_PI_4;
      cd.i32 += (random() & 31) - 15;
      d = cd.f;
      break;
    case 2:
      d = (2 * (float)random() / RAND_MAX - 1) * rangemax;
      break;
    case 3:
      cd.f = rint((2 * (float)random() / RAND_MAX - 1) * rangemax) * M_PI_4;
      cd.i32 += (random() & 31) - 15;
      d = cd.f;
      break;
    case 4:
      d = (2 * (float)random() / RAND_MAX - 1) * 10000;
      break;
    case 5:
      cd.f = rint((2 * (float)random() / RAND_MAX - 1) * 10000) * M_PI_4;
      cd.i32 += (random() & 31) - 15;
      d = cd.f;
      break;
    default:
      d = rnd_fr();
      break;
    }

    if (!isfinite(d)) continue;

    Sleef_float2 sc  = xsincospif_u05(d);
    Sleef_float2 sc2 = xsincospif_u35(d);

    {
      const float rangemax2 = 1e+7/4;
      
      sinpifr(frx, d);

      double u0 = countULP2(t = sc.x, frx);

      if ((fabs(d) <= rangemax2 && u0 > 0.505) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincospif_u05 sin arg=%.20g ulp=%.20g\n", d, u0);
      }

      double u1 = countULP2(t = sc2.x, frx);

      if ((fabs(d) <= rangemax2 && u1 > 1.6) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincospif_u35 sin arg=%.20g ulp=%.20g\n", d, u1);
      }
    }

    {
      const float rangemax2 = 1e+7/4;
      
      cospifr(frx, d);

      double u0 = countULP2(t = sc.y, frx);

      if ((fabs(d) <= rangemax2 && u0 > 0.505) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincospif_u05 cos arg=%.20g ulp=%.20g\n", d, u0);
      }

      double u1 = countULP2(t = sc.y, frx);

      if ((fabs(d) <= rangemax2 && u1 > 1.5) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincospif_u35 cos arg=%.20g ulp=%.20g\n", d, u1);
      }
    }

    sc = xsincosf(d);
    sc2 = xsincosf_u1(d);
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sin(frx, frx, GMP_RNDN);

      float u0 = countULP(t = xsinf(d), frx);
      
      if ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sinf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      float u1 = countULP(t = sc.x, frx);
      
      if ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincosf sin arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      float u2 = countULP(t = xsinf_u1(d), frx);
      
      if ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sinf_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      float u3 = countULP(t = sc2.x, frx);
      
      if ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincosf_u1 sin arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cos(frx, frx, GMP_RNDN);

      float u0 = countULP(t = xcosf(d), frx);
      
      if ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C cosf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      float u1 = countULP(t = sc.y, frx);
      
      if ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincosf cos arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      float u2 = countULP(t = xcosf_u1(d), frx);
      
      if ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C cosf_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      float u3 = countULP(t = sc2.y, frx);
      
      if ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincosf_u1 cos arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tan(frx, frx, GMP_RNDN);

      float u0 = countULP(t = xtanf(d), frx);
      
      if ((fabs(d) < rangemax && u0 > 3.5) || isnan(t)) {
	printf("Pure C tanf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      float u1 = countULP(t = xtanf_u1(d), frx);
      
      if ((fabs(d) <= rangemax && u1 > 1) || isnan(t)) {
	printf("Pure C tanf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }

    d = rnd_fr();
    float d2 = rnd_fr(), zo = rnd_zo();

    {
      mpfr_set_d(frx, fabsf(d), GMP_RNDN);
      mpfr_log(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlogf(fabsf(d)), frx);
      
      if (u0 > 3.5) {
	printf("Pure C logf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xlogf_u1(fabsf(d)), frx);
      
      if (u1 > 1) {
	printf("Pure C logf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, fabsf(d), GMP_RNDN);
      mpfr_log10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog10f(fabsf(d)), frx);
      
      if (u0 > 1) {
	printf("Pure C log10f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
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
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexpf(d), frx);
      
      if (u0 > 1) {
	printf("Pure C expf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp2(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp2f(d), frx);
      
      if (u0 > 1) {
	printf("Pure C exp2f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp10f(d), frx);
      
      if (u0 > 1) {
	printf("Pure C exp10f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_expm1(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexpm1f(d), frx);
      
      if (u0 > 1) {
	printf("Pure C expm1f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_pow(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xpowf(d2, d), frx);
      
      if (u0 > 1) {
	printf("Pure C powf arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cbrt(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcbrtf(d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C cbrtf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xcbrtf_u1(d), frx);
      
      if (u1 > 1) {
	printf("Pure C cbrtf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_asin(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xasinf(zo), frx);
      
      if (u0 > 3.5) {
	printf("Pure C asinf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xasinf_u1(zo), frx);
      
      if (u1 > 1) {
	printf("Pure C asinf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_acos(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xacosf(zo), frx);
      
      if (u0 > 3.5) {
	printf("Pure C acosf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xacosf_u1(zo), frx);
      
      if (u1 > 1) {
	printf("Pure C acosf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atan(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatanf(d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C atanf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xatanf_u1(d), frx);
      
      if (u1 > 1) {
	printf("Pure C atanf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_atan2(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xatan2f(d2, d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C atan2f arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout);
      }

      double u1 = countULP2(t = xatan2f_u1(d2, d), frx);
      
      if (u1 > 1) {
	printf("Pure C atan2f_u1 arg=%.20g, %.20g ulp=%.20g\n", d2, d, u1);
	fflush(stdout);
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
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcoshf(d), frx);
      
      if ((fabs(d) <= 88.5 && u0 > 1) || !(u0 <= 1 || (isinf(t) && t > 0))) {
	printf("Pure C coshf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xtanhf(d), frx);
      
      if (u0 > 1.0001) {
	printf("Pure C tanhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
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
	fflush(stdout);
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
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatanhf(d), frx);
      
      if (u0 > 1.0001) {
	printf("Pure C atanhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
  }
}
