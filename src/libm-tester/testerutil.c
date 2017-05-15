//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#if defined(__MINGW32__) || defined(__MINGW64__) || defined(_MSC_VER)
#define STDIN_FILENO 0
#else
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#endif

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

//

double u2d(uint64_t u) {
  union {
    double f;
    uint64_t i;
  } tmp;
  tmp.i = u;
  return tmp.f;
}

uint64_t d2u(double d) {
  union {
    double f;
    uint64_t i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

float u2f(uint32_t u) {
  union {
    float f;
    uint32_t i;
  } tmp;
  tmp.i = u;
  return tmp.f;
}

uint32_t f2u(float d) {
  union {
    float f;
    uint32_t i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

//

int readln(int fd, char *buf, int cnt) {
  int i, rcnt = 0;

  if (cnt < 1) return -1;

  while(cnt >= 2) {
    i = read(fd, buf, 1);
    if (i != 1) return i;

    if (*buf == '\n') break;

    rcnt++;
    buf++;
    cnt--;
  }

  *++buf = '\0';
  rcnt++;
  return rcnt;
}

int startsWith(char *str, char *prefix) {
  return strncmp(str, prefix, strlen(prefix)) == 0;
}

//

#ifdef USEMPFR
#include <mpfr.h>

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

double countULPdp(double d, mpfr_t c) {
  mpfr_t fra, frb, frc, frd;
  mpfr_inits(fra, frb, frc, frd, NULL);

  double c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 10000;
  }
  if (isnan(c2) && isnan(d)) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }
  if (isnan(c2) || isnan(d)) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 10001;
  }
  if (c2 == POSITIVE_INFINITY && d == POSITIVE_INFINITY) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }
  if (c2 == NEGATIVE_INFINITY && d == NEGATIVE_INFINITY) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }

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

  mpfr_clears(fra, frb, frc, frd, NULL);
  
  return u + v;
}

double countULP2dp(double d, mpfr_t c) {
  mpfr_t fra, frb, frc, frd;
  mpfr_inits(fra, frb, frc, frd, NULL);

  double c2 = mpfr_get_d(c, GMP_RNDN);
  if (c2 == 0 && d != 0) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 10000;
  }
  if (isnan(c2) && isnan(d)) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }
  if (isnan(c2) || isnan(d)) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 10001;
  }
  if (c2 == POSITIVE_INFINITY && d == POSITIVE_INFINITY) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }
  if (c2 == NEGATIVE_INFINITY && d == NEGATIVE_INFINITY) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }

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

  mpfr_clears(fra, frb, frc, frd, NULL);
  
  return u + v;
}

double countULPsp(float d, mpfr_t c) {
  mpfr_t fra, frb, frc, frd;
  mpfr_inits(fra, frb, frc, frd, NULL);

  d = flushToZero(d);
  float c2 = flushToZero(mpfr_get_d(c, GMP_RNDN));
  if (c2 == 0 && d != 0) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 10000;
  }
  if (isnan(c2) && isnan(d)) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }
  if (isnan(c2) || isnan(d)) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 10001;
  }
  if (c2 == POSITIVE_INFINITYf && d == POSITIVE_INFINITYf) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }
  if (c2 == NEGATIVE_INFINITYf && d == NEGATIVE_INFINITYf) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }

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

  mpfr_clears(fra, frb, frc, frd, NULL);
  
  return u + v;
}

double countULP2sp(float d, mpfr_t c) {
  mpfr_t fra, frb, frc, frd;
  mpfr_inits(fra, frb, frc, frd, NULL);

  d = flushToZero(d);
  float c2 = flushToZero(mpfr_get_d(c, GMP_RNDN));
  if (c2 == 0 && d != 0) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 10000;
  }
  if (isnan(c2) && isnan(d)) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }
  if (isnan(c2) || isnan(d)) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 10001;
  }
  if (c2 == POSITIVE_INFINITYf && d == POSITIVE_INFINITYf) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }
  if (c2 == NEGATIVE_INFINITYf && d == NEGATIVE_INFINITYf) {
    mpfr_clears(fra, frb, frc, frd, NULL);
    return 0;
  }

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

  mpfr_clears(fra, frb, frc, frd, NULL);
  
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

void mpfr_lgamma_nosign(mpfr_t ret, mpfr_t arg, mpfr_rnd_t rnd) {
  int s;
  mpfr_lgamma(ret, &s, arg, rnd);
}
#endif // #define USEMPFR
