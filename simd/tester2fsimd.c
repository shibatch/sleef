//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#include "sleefsimd.h"

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
#if 1
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
#if 1
    syscall(SYS_getrandom, &c.u32, sizeof(c.u32), 0);
#else
    c.u32 = (uint32_t)random() | ((uint32_t)random() << 31);
#endif
  } while(!isfinite(c.f) || c.f < -1 || 1 < c.f);
  return c.f;
}

int main(int argc,char **argv)
{
  mpfr_set_default_prec(256);
  mpfr_inits(fra, frb, frc, frd, frw, frx, fry, frz, NULL);

  conv32_t cd;
  float d, t;
  vfloat vd, vd2, vzo, vad;
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

    vd[e] = d;
    sc = xsincosf(vd);
    sc2 = xsincosf_u1(vd);
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sin(frx, frx, GMP_RNDN);

      float u0 = countULP(t = xsinf(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sinf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      float u1 = countULP(t = sc.x[e], frx);
      
      if ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sincosf sin arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      float u2 = countULP(t = xsinf_u1(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sinf_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      float u3 = countULP(t = sc2.x[e], frx);
      
      if ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sincosf_u1 sin arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cos(frx, frx, GMP_RNDN);

      float u0 = countULP(t = xcosf(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " cosf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      float u1 = countULP(t = sc.y[e], frx);
      
      if ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sincosf cos arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      float u2 = countULP(t = xcosf_u1(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " cosf_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      float u3 = countULP(t = sc2.y[e], frx);
      
      if ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sincosf_u1 cos arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tan(frx, frx, GMP_RNDN);

      float u0 = countULP(t = xtanf(vd)[e], frx);
      
      if ((fabs(d) < rangemax && u0 > 3.5) || isnan(t)) {
	printf(SLEEF_ARCH " tanf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      float u1 = countULP(t = xtanf_u1(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u1 > 1) || isnan(t)) {
	printf(SLEEF_ARCH " tanf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }

    d = rnd_fr();
    float d2 = rnd_fr(), zo = rnd_zo();
    vd[e] = d;
    vd2[e] = d2;
    vzo[e] = zo;
    vad[e] = fabs(d);

    {
      mpfr_set_d(frx, fabsf(d), GMP_RNDN);
      mpfr_log(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlogf(vad)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " logf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xlogf_u1(vad)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " logf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, fabsf(d), GMP_RNDN);
      mpfr_log10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog10f(vad)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " log10f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_log1p(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog1pf(vd)[e], frx);
      
      if ((-1 <= d && d <= 1e+38 && u0 > 1) ||
	  (d < -1 && !isnan(t)) ||
	  (d > 1e+38 && !(u0 <= 1 || isinf(t)))) {
	printf(SLEEF_ARCH " log1pf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexpf(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " expf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp2(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp2f(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " exp2f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp10f(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " exp10f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_expm1(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexpm1f(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " expm1f arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_pow(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xpowf(vd2, vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " powf arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cbrt(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcbrtf(vd)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " cbrtf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xcbrtf_u1(vd)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " cbrtf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_asin(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xasinf(vzo)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " asinf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xasinf_u1(vzo)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " asinf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_acos(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xacosf(vzo)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " acosf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xacosf_u1(vzo)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " acosf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atan(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatanf(vd)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " atanf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xatanf_u1(vd)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " atanf_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_atan2(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xatan2f(vd2, vd)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " atan2f arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout);
      }

      double u1 = countULP2(t = xatan2f_u1(vd2, vd)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " atan2f_u1 arg=%.20g, %.20g ulp=%.20g\n", d2, d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xsinhf(vd)[e], frx);
      
      if ((fabs(d) <= 88.5 && u0 > 1) ||
	  (d >  88.5 && !(u0 <= 1 || (isinf(t) && t > 0))) ||
	  (d < -88.5 && !(u0 <= 1 || (isinf(t) && t < 0)))) {
	printf(SLEEF_ARCH " sinhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcoshf(vd)[e], frx);
      
      if ((fabs(d) <= 88.5 && u0 > 1) || !(u0 <= 1 || (isinf(t) && t > 0))) {
	printf(SLEEF_ARCH " coshf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xtanhf(vd)[e], frx);
      
      if (u0 > 1.0001) {
	printf(SLEEF_ARCH " tanhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_asinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xasinhf(vd)[e], frx);
      
      if ((fabs(d) < sqrt(FLT_MAX) && u0 > 1.0001) ||
	  (d >=  sqrt(FLT_MAX) && !(u0 <= 1.0001 || (isinf(t) && t > 0))) ||
	  (d <= -sqrt(FLT_MAX) && !(u0 <= 1.0001 || (isinf(t) && t < 0)))) {
	printf(SLEEF_ARCH " asinhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_acosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xacoshf(vd)[e], frx);
      
      if ((fabs(d) < sqrt(FLT_MAX) && u0 > 1.0001) ||
	  (d >=  sqrt(FLT_MAX) && !(u0 <= 1.0001 || (isinff(t) && t > 0))) ||
	  (d <= -sqrt(FLT_MAX) && !isnan(t))) {
	printf(SLEEF_ARCH " acoshf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatanhf(vd)[e], frx);
      
      if (u0 > 1.0001) {
	printf(SLEEF_ARCH " atanhf arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
  }
}
