#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpfr.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#define DENORMAL_DBL_MIN (4.9406564584124654418e-324)
#define POSITIVE_INFINITY INFINITY
#define NEGATIVE_INFINITY (-INFINITY)

#define DENORMAL_FLT_MIN (1.4012984643248170709e-45f)
#define POSITIVE_INFINITYf ((float)INFINITY)
#define NEGATIVE_INFINITYf (-(float)INFINITY)

int isnumber(double x) { return !isinf(x) && !isnan(x); }
int isPlusZero(double x) { return x == 0 && copysign(1, x) == 1; }
int isMinusZero(double x) { return x == 0 && copysign(1, x) == -1; }
double sign(double d) { return d < 0 ? -1 : 1; }
int xisnan(double x) { return x != x; }

int isnumberf(float x) { return !isinff(x) && !isnanf(x); }
int isPlusZerof(float x) { return x == 0 && copysignf(1, x) == 1; }
int isMinusZerof(float x) { return x == 0 && copysignf(1, x) == -1; }
float signf(float d) { return d < 0 ? -1 : 1; }
int xisnanf(float x) { return x != x; }

int enableFlushToZero = 0;

double flushToZero(double y) {
  if (enableFlushToZero && fabs(y) < FLT_MIN) y = copysign(0.0, y);
  return y;
}

int cmpDenormsp(float x, mpfr_t fry) {
  float y = mpfr_get_d(fry, GMP_RNDN);
  x = flushToZero(x);
  y = flushToZero(y);
  if (xisnanf(x) && xisnanf(y)) return 1;
  if (xisnanf(x) || xisnanf(y)) return 0;
  if (isinf(x) != isinf(y)) return 0;
  if (x == POSITIVE_INFINITYf && y == POSITIVE_INFINITYf) return 1;
  if (x == NEGATIVE_INFINITYf && y == NEGATIVE_INFINITYf) return 1;
  if (y == 0) {
    if (isPlusZerof(x) && isPlusZerof(y)) return 1;
    if (isMinusZerof(x) && isMinusZerof(y)) return 1;
    return 0;
  }
  if (!xisnanf(x) && !xisnanf(y) && !isinf(x) && !isinf(y)) return signf(x) == signf(y);
  return 0;
}

int cmpDenormdp(double x, mpfr_t fry) {
  double y = mpfr_get_d(fry, GMP_RNDN);
  if (xisnan(x) && xisnan(y)) return 1;
  if (xisnan(x) || xisnan(y)) return 0;
  if (isinf(x) != isinf(y)) return 0;
  if (x == POSITIVE_INFINITY && y == POSITIVE_INFINITY) return 1;
  if (x == NEGATIVE_INFINITY && y == NEGATIVE_INFINITY) return 1;
  if (y == 0) {
    if (isPlusZero(x) && isPlusZero(y)) return 1;
    if (isMinusZero(x) && isMinusZero(y)) return 1;
    return 0;
  }
  if (!xisnan(x) && !xisnan(y) && !isinf(x) && !isinf(y)) return sign(x) == sign(y);
  return 0;
}

int firstTime = 1;
mpfr_t fra, frb, frc, frd;

double countULPdp(double d, mpfr_t c) {
  if (firstTime) {
    mpfr_inits(fra, frb, frc, frd, NULL);
    firstTime = 0;
  }
  double c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) return 10000;
  if (isnan(c2) && isnan(d)) return 0;
  if (isnan(c2) || isnan(d)) return 10001;
  if (c2 == POSITIVE_INFINITY && d == POSITIVE_INFINITY) return 0;
  if (c2 == NEGATIVE_INFINITY && d == NEGATIVE_INFINITY) return 0;

  double v = 0;
  if (isinf(d) && !isinfl(mpfr_get_ld(c, GMP_RNDN))) {
    d = copysign(DBL_MAX, c2);
    v = 1;
  }

  //
  
  int e;
  frexpl(mpfr_get_ld(c, GMP_RNDN), &e);
  mpfr_set_ld(frb, fmaxl(ldexpl(1.0, e-53), DENORMAL_DBL_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fra, frd, c, GMP_RNDN);
  mpfr_div(fra, fra, frb, GMP_RNDN);
  double u = fabs(mpfr_get_d(fra, GMP_RNDN));

  return u + v;
}

double countULP2dp(double d, mpfr_t c) {
  if (firstTime) {
    mpfr_inits(fra, frb, frc, frd, NULL);
    firstTime = 0;
  }
  double c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) return 10000;
  if (isnan(c2) && isnan(d)) return 0;
  if (isnan(c2) || isnan(d)) return 10001;
  if (c2 == POSITIVE_INFINITY && d == POSITIVE_INFINITY) return 0;
  if (c2 == NEGATIVE_INFINITY && d == NEGATIVE_INFINITY) return 0;

  double v = 0;
  if (isinf(d) && !isinfl(mpfr_get_ld(c, GMP_RNDN))) {
    d = copysign(DBL_MAX, c2);
    v = 1;
  }

  //

  int e;
  frexpl(mpfr_get_ld(c, GMP_RNDN), &e);
  mpfr_set_ld(frb, fmaxl(ldexpl(1.0, e-53), DBL_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fra, frd, c, GMP_RNDN);
  mpfr_div(fra, fra, frb, GMP_RNDN);
  double u = fabs(mpfr_get_d(fra, GMP_RNDN));

  return u + v;
}

double countULPsp(float d, mpfr_t c) {
  if (firstTime) {
    mpfr_inits(fra, frb, frc, frd, NULL);
    firstTime = 0;
  }
  d = flushToZero(d);
  float c2 = flushToZero(mpfr_get_d(c, GMP_RNDN));
  if (c2 == 0 && d != 0) return 10000;
  if (isnan(c2) && isnan(d)) return 0;
  if (isnan(c2) || isnan(d)) return 10001;
  if (c2 == POSITIVE_INFINITYf && d == POSITIVE_INFINITYf) return 0;
  if (c2 == NEGATIVE_INFINITYf && d == NEGATIVE_INFINITYf) return 0;

  double v = 0;
  if (isinf(d) && !isinfl(mpfr_get_ld(c, GMP_RNDN))) {
    d = copysign(FLT_MAX, c2);
    v = 1;
  }

  //

  int e;
  frexpl(mpfr_get_ld(c, GMP_RNDN), &e);
  mpfr_set_ld(frb, fmaxl(ldexpl(1.0, e-24), DENORMAL_FLT_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fra, frd, c, GMP_RNDN);
  mpfr_div(fra, fra, frb, GMP_RNDN);
  double u = fabs(mpfr_get_d(fra, GMP_RNDN));

  return u + v;
}

double countULP2sp(float d, mpfr_t c) {
  if (firstTime) {
    mpfr_inits(fra, frb, frc, frd, NULL);
    firstTime = 0;
  }
  d = flushToZero(d);
  float c2 = flushToZero(mpfr_get_d(c, GMP_RNDN));
  if (c2 == 0 && d != 0) return 10000;
  if (isnan(c2) && isnan(d)) return 0;
  if (isnan(c2) || isnan(d)) return 10001;
  if (c2 == POSITIVE_INFINITYf && d == POSITIVE_INFINITYf) return 0;
  if (c2 == NEGATIVE_INFINITYf && d == NEGATIVE_INFINITYf) return 0;

  double v = 0;
  if (isinf(d) && !isinfl(mpfr_get_ld(c, GMP_RNDN))) {
    d = copysign(FLT_MAX, c2);
    v = 1;
  }

  //
  
  int e;
  frexpl(mpfr_get_ld(c, GMP_RNDN), &e);
  mpfr_set_ld(frb, fmaxl(ldexpl(1.0, e-24), FLT_MIN), GMP_RNDN);

  mpfr_set_d(frd, d, GMP_RNDN);
  mpfr_sub(fra, frd, c, GMP_RNDN);
  mpfr_div(fra, fra, frb, GMP_RNDN);
  double u = fabs(mpfr_get_d(fra, GMP_RNDN));

  return u + v;
}

//

void mpfr_sinpi(mpfr_t ret, mpfr_t arg, mpfr_rnd_t rnd) {
  mpfr_t frpi, frd;
  mpfr_inits(frpi, frd, NULL);

  mpfr_const_pi(frpi, GMP_RNDN);
  mpfr_set_d(frd, 1.0, GMP_RNDN);
  mpfr_mul(frpi, frpi, frd, GMP_RNDN);
  mpfr_mul(frd, frpi, arg, GMP_RNDN);
  mpfr_sin(ret, frd, GMP_RNDN);

  mpfr_clears(frpi, frd, NULL);
}

void mpfr_cospi(mpfr_t ret, mpfr_t arg, mpfr_rnd_t rnd) {
  mpfr_t frpi, frd;
  mpfr_inits(frpi, frd, NULL);

  mpfr_const_pi(frpi, GMP_RNDN);
  mpfr_set_d(frd, 1.0, GMP_RNDN);
  mpfr_mul(frpi, frpi, frd, GMP_RNDN);
  mpfr_mul(frd, frpi, arg, GMP_RNDN);
  mpfr_cos(ret, frd, GMP_RNDN);

  mpfr_clears(frpi, frd, NULL);
}
