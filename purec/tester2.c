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
#include <linux/random.h>

#include "sleef.h"

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

    
    double2 sc = xsincos(d);
    double2 sc2 = xsincos_u1(d);
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sin(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xsin(d), frx);
      
      if ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sin arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(sc.x, frx);
      
      if ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincos sin arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      double u2 = countULP(t = xsin_u1(d), frx);
      
      if ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sin_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      double u3 = countULP(t = sc2.x, frx);
      
      if ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincos_u1 sin arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cos(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcos(d), frx);
      
      if ((fabs(d) <= rangemax && u0 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C cos arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = sc.y, frx);
      
      if ((fabs(d) <= rangemax && u1 > 3.5) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincos cos arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }

      double u2 = countULP(t = xcos_u1(d), frx);
      
      if ((fabs(d) <= rangemax && u2 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C cos_u1 arg=%.20g ulp=%.20g\n", d, u2);
	fflush(stdout);
      }

      double u3 = countULP(t = sc2.y, frx);
      
      if ((fabs(d) <= rangemax && u3 > 1) || fabs(t) > 1 || !isfinite(t)) {
	printf("Pure C sincos_u1 cos arg=%.20g ulp=%.20g\n", d, u3);
	fflush(stdout);
      }
    }

    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tan(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xtan(d), frx);
      
      if ((fabs(d) < 1e+7 && u0 > 3.5) || (fabs(d) <= rangemax && u0 > 5) || isnan(t)) {
	printf("Pure C tan arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xtan_u1(d), frx);
      
      if ((fabs(d) <= rangemax && u1 > 1) || isnan(t)) {
	printf("Pure C tan_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }

    d = rnd_fr();
    double d2 = rnd_fr(), zo = rnd_zo();
    
    {
      mpfr_set_d(frx, fabs(d), GMP_RNDN);
      mpfr_log(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog(fabs(d)), frx);
      
      if (u0 > 3.5) {
	printf("Pure C log arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xlog_u1(fabs(d)), frx);
      
      if (u1 > 1) {
	printf("Pure C log_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, fabs(d), GMP_RNDN);
      mpfr_log10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog10(fabs(d)), frx);
      
      if (u0 > 1) {
	printf("Pure C log10 arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_log1p(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xlog1p(d), frx);
      
      if ((-1 <= d && d <= 1e+307 && u0 > 1) ||
	  (d < -1 && !isnan(t)) ||
	  (d > 1e+307 && !(u0 <= 1 || isinf(t)))) {
	printf("Pure C log1p arg=%.20g ulp=%.20g\n", d, u0);
	printf("%g\n", t);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp(d), frx);
      
      if (u0 > 1) {
	printf("Pure C exp arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp2(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp2(d), frx);
      
      if (u0 > 1) {
	printf("Pure C exp2 arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_exp10(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexp10(d), frx);
      
      if (u0 > 1) {
	printf("Pure C exp10 arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_expm1(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xexpm1(d), frx);
      
      if (u0 > 1) {
	printf("Pure C expm1 arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_pow(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xpow(d2, d), frx);
      
      if (u0 > 1) {
	printf("Pure C pow arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cbrt(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcbrt(d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C cbrt arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xcbrt_u1(d), frx);
      
      if (u1 > 1) {
	printf("Pure C cbrt_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_asin(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xasin(zo), frx);
      
      if (u0 > 3.5) {
	printf("Pure C asin arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xasin_u1(zo), frx);
      
      if (u1 > 1) {
	printf("Pure C asin_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, zo, GMP_RNDN);
      mpfr_acos(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xacos(zo), frx);
      
      if (u0 > 3.5) {
	printf("Pure C acos arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xacos_u1(zo), frx);
      
      if (u1 > 1) {
	printf("Pure C acos_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atan(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatan(d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C atan arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }

      double u1 = countULP(t = xatan_u1(d), frx);
      
      if (u1 > 1) {
	printf("Pure C atan_u1 arg=%.20g ulp=%.20g\n", d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_set_d(fry, d2, GMP_RNDN);
      mpfr_atan2(frx, fry, frx, GMP_RNDN);

      double u0 = countULP(t = xatan2(d2, d), frx);
      
      if (u0 > 3.5) {
	printf("Pure C atan2 arg=%.20g, %.20g ulp=%.20g\n", d2, d, u0);
	fflush(stdout);
      }

      double u1 = countULP2(t = xatan2_u1(d2, d), frx);
      
      if (u1 > 1) {
	printf("Pure C atan2_u1 arg=%.20g, %.20g ulp=%.20g\n", d2, d, u1);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_sinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xsinh(d), frx);
      
      if ((fabs(d) <= 709 && u0 > 1) ||
	  (d >  709 && !(u0 <= 1 || (isinf(t) && t > 0))) ||
	  (d < -709 && !(u0 <= 1 || (isinf(t) && t < 0)))) {
	printf("Pure C sinh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_cosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xcosh(d), frx);
      
      if ((fabs(d) <= 709 && u0 > 1) || !(u0 <= 1 || (isinf(t) && t > 0))) {
	printf("Pure C cosh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_tanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xtanh(d), frx);
      
      if (u0 > 1) {
	printf("Pure C tanh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_asinh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xasinh(d), frx);
      
      if ((fabs(d) < sqrt(DBL_MAX) && u0 > 1) ||
	  (d >=  sqrt(DBL_MAX) && !(u0 <= 1 || (isinf(t) && t > 0))) ||
	  (d <= -sqrt(DBL_MAX) && !(u0 <= 1 || (isinf(t) && t < 0)))) {
	printf("Pure C asinh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_acosh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xacosh(d), frx);
      
      if ((fabs(d) < sqrt(DBL_MAX) && u0 > 1) ||
	  (d >=  sqrt(DBL_MAX) && !(u0 <= 1 || (isinf(t) && t > 0))) ||
	  (d <= -sqrt(DBL_MAX) && !isnan(t))) {
	printf("Pure C acosh arg=%.20g ulp=%.20g\n", d, u0);
	printf("%.20g\n", t);
	fflush(stdout);
      }
    }
    
    {
      mpfr_set_d(frx, d, GMP_RNDN);
      mpfr_atanh(frx, frx, GMP_RNDN);

      double u0 = countULP(t = xatanh(d), frx);
      
      if (u0 > 1) {
	printf("Pure C atanh arg=%.20g ulp=%.20g\n", d, u0);
	fflush(stdout);
      }
    }
  }
}
