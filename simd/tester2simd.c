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

#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#include "sleefsimd.h"

mpfr_t fra, frb, frc, frd, frw, frx, fry, frz;

#define DENORMAL_DBL_MIN (4.94066e-324)

double countULP(double d, mpfr_t c) {
  double c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) return 10000;
  if (!isfinite(c2) && !isfinite(d)) return 0;

  int e;
  frexpl(mpfr_get_d(c, GMP_RNDN), &e);
  mpfr_set_ld(frw, fmaxl(ldexpl(1.0, e-53), DENORMAL_DBL_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fry, frd, c, GMP_RNDN);
  mpfr_div(fry, fry, frw, GMP_RNDN);
  double u = fabs(mpfr_get_d(fry, GMP_RNDN));

  return u;
}

double countULP2(double d, mpfr_t c) {
  double c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) return 10000;
  if (!isfinite(c2) && !isfinite(d)) return 0;

  int e;
  frexpl(mpfr_get_d(c, GMP_RNDN), &e);
  mpfr_set_ld(frw, fmaxl(ldexpl(1.0, e-53), DBL_MIN), GMP_RNDN);

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
} conv_t;

double rnd_fr() {
  conv_t c;
  do {
#if 1
    syscall(SYS_getrandom, &c.u64, sizeof(c.u64), 0);
#else
    c.u64 = random() | ((uint64_t)random() << 31) | ((uint64_t)random() << 62);
#endif
  } while(!isfinite(c.d));
  return c.d;
}

double rnd_zo() {
  conv_t c;
  do {
#if 1
    syscall(SYS_getrandom, &c.u64, sizeof(c.u64), 0);
#else
    c.u64 = random() | ((uint64_t)random() << 31) | ((uint64_t)random() << 62);
#endif
  } while(!isfinite(c.d) || c.d < -1 || 1 < c.d);
  return c.d;
}

int main(int argc,char **argv)
{
  mpfr_set_default_prec(256);
  mpfr_inits(fra, frb, frc, frd, frw, frx, fry, frz, NULL);

  conv_t cd;
  double d, t;
  vdouble vd, vd2, vzo, vad;
  vdouble2 sc, sc2;
  int cnt;
  
  srandom(time(NULL));

#if 0
  cd.d = M_PI;
  mpfr_set_d(frx, cd.d, GMP_RNDN);
  cd.i64+=3;
  printf("%g\n", countULP(cd.d, frx));
#endif

  const double rangemax = 1e+14; // 2^(24*2-1)
  
  for(cnt = 0;;cnt++) {
    int e = cnt % VECTLENDP;
    switch(cnt & 7) {
    case 0:
      d = (2 * (double)random() / RAND_MAX - 1) * rangemax;
      break;
    case 1:
      cd.d = rint((2 * (double)random() / RAND_MAX - 1) * rangemax) * M_PI_4;
      cd.i64 += (random() & 31) - 15;
      d = cd.d;
      break;
    case 2:
      d = (2 * (double)random() / RAND_MAX - 1) * 1e+7;
      break;
    case 3:
      cd.d = rint((2 * (double)random() / RAND_MAX - 1) * 1e+7) * M_PI_4;
      cd.i64 += (random() & 31) - 15;
      d = cd.d;
      break;
    case 4:
      d = (2 * (double)random() / RAND_MAX - 1) * 10000;
      break;
    case 5:
      cd.d = rint((2 * (double)random() / RAND_MAX - 1) * 10000) * M_PI_4;
      cd.i64 += (random() & 31) - 15;
      d = cd.d;
      break;
    default:
      d = rnd_fr();
      break;
    }

    if (!isfinite(d)) continue;

    vd[e] = d;
    sc = xsincos(vd);
    sc2 = xsincos_u1(vd);
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sin(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xsin(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sin arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = sc.x[e], frx);
      
      if ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sincos sin arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      double u2 = countULP(t = xsin_u1(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sin_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      double u3 = countULP(t = sc2.x[e], frx);
      
      if ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sincos_u1 sin arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cos(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcos(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " cos arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = sc.y[e], frx);
      
      if ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sincos cos arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      double u2 = countULP(t = xcos_u1(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " cos_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      double u3 = countULP(t = sc2.y[e], frx);
      
      if ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf(SLEEF_ARCH " sincos_u1 cos arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tan(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xtan(vd)[e], frx);
      
      if ((fabs(d) < 1e+7 && u0 > 3.5) || (fabs(d) <= rangemax && u0 > 5) || isnan(t)) {
	printf(SLEEF_ARCH " tan arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xtan_u1(vd)[e], frx);
      
      if ((fabs(d) <= rangemax && u1 > 1) || isnan(t)) {
	printf(SLEEF_ARCH " tan_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }

    d = rnd_fr();
    double d2 = rnd_fr(), zo = rnd_zo();
    vd[e] = d;
    vd2[e] = d2;
    vzo[e] = zo;
    vad[e] = fabs(d);
    
    {
      mpfr_set_d(frx, fabs(d), GMP_RNDN);
      mpfr_log(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog(vad)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " log arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xlog_u1(vad)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " log_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, fabs(d), GMP_RNDN);
      mpfr_log10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog10(vad)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " log10 arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_log1p(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog1p(vd)[e], frx);
      
      if ((-1 <= d && d <= 1e+307 && u0 > 1) ||
	  (d < -1 && !isnan(t)) ||
	  (d > 1e+307 && !(u0 <= 1 || isinf(t)))) {
	printf(SLEEF_ARCH " log1p arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " exp arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp2(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp2(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " exp2 arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp10(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " exp10 arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_expm1(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexpm1(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " expm1 arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_pow(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xpow(vd2, vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " pow arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cbrt(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcbrt(vd)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " cbrt arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xcbrt_u1(vd)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " cbrt_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_asin(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xasin(vzo)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " asin arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xasin_u1(vzo)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " asin_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_acos(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xacos(vzo)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " acos arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xacos_u1(vzo)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " acos_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atan(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatan(vd)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " atan arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xatan_u1(vd)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " atan_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_atan2(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xatan2(vd2, vd)[e], frx);
      
      if (u0 > 3.5) {
	printf(SLEEF_ARCH " atan2 arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout);
      }

      double u1 = countULP2(t = xatan2_u1(vd2, vd)[e], frx);
      
      if (u1 > 1) {
	printf(SLEEF_ARCH " atan2_u1 arg=%.20g, %.20g ulp=%.20g\n", d2, d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xsinh(vd)[e], frx);
      
      if ((fabs(d) <= 709 && u0 > 1) ||
	  (d >  709 && !(u0 <= 1 || (isinf(t) && t > 0))) ||
	  (d < -709 && !(u0 <= 1 || (isinf(t) && t < 0)))) {
	printf(SLEEF_ARCH " sinh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcosh(vd)[e], frx);
      
      if ((fabs(d) <= 709 && u0 > 1) || !(u0 <= 1 || (isinf(t) && t > 0))) {
	printf(SLEEF_ARCH " cosh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xtanh(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " tanh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_asinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xasinh(vd)[e], frx);
      
      if ((fabs(d) < sqrt(DBL_MAX) && u0 > 1) ||
	  (d >=  sqrt(DBL_MAX) && !(u0 <= 1 || (isinf(t) && t > 0))) ||
	  (d <= -sqrt(DBL_MAX) && !(u0 <= 1 || (isinf(t) && t < 0)))) {
	printf(SLEEF_ARCH " asinh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_acosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xacosh(vd)[e], frx);
      
      if ((fabs(d) < sqrt(DBL_MAX) && u0 > 1) ||
	  (d >=  sqrt(DBL_MAX) && !(u0 <= 1 || (isinf(t) && t > 0))) ||
	  (d <= -sqrt(DBL_MAX) && !isnan(t))) {
	printf(SLEEF_ARCH " acosh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatanh(vd)[e], frx);
      
      if (u0 > 1) {
	printf(SLEEF_ARCH " atanh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
  }
}
