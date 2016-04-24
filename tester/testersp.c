#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>

#include <mpfr.h>

#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <signal.h>

#include "nonnumber.h"

#define POSITIVE_INFINITYf ((float)INFINITY)
#define NEGATIVE_INFINITYf (-(float)INFINITY)
#define M_PIf ((float)M_PI)

#define POSITIVE_INFINITY (INFINITY)
#define NEGATIVE_INFINITY (-INFINITY)

typedef int boolean;

#define true 1
#define false 0

int enableFlushToZero = 0;

void stop(char *mes) {
  fprintf(stderr, "%s\n", mes);
  abort();
}

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

int ptoc[2], ctop[2];
int pid;

void startChild(const char *path, char *const argv[]) {
  pipe(ptoc);
  pipe(ctop);

  pid = fork();

  assert(pid != -1);

  if (pid == 0) {
    // child process
    char buf0[1], buf1[1];
    int i;

    close(ptoc[1]);
    close(ctop[0]);

    i = dup2(ptoc[0], fileno(stdin));
    assert(i != -1);

    i = dup2(ctop[1], fileno(stdout));
    assert(i != -1);

    setvbuf(stdin, buf0, _IONBF,0);
    setvbuf(stdout, buf1, _IONBF,0);

    fflush(stdin);
    fflush(stdout);

    execvp(path, argv);

    assert(0);
  }

  // parent process

  close(ptoc[0]);
  close(ctop[1]);
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

boolean isPlusZerof(float x) { return x == 0 && copysignf(1, x) == 1; }
boolean isMinusZerof(float x) { return x == 0 && copysignf(1, x) == -1; }
boolean xisnanf(float x) { return x != x; }
float signf(float d) { return d < 0 ? -1 : 1; }

boolean isPlusZero(double x) { return x == 0 && copysign(1, x) == 1; }
boolean isMinusZero(double x) { return x == 0 && copysign(1, x) == -1; }
boolean xisnan(double x) { return x != x; }

double flushToZero(double y) {
  if (enableFlushToZero && fabs(y) < 1.2e-38) y = copysign(0.0, y);
  return y;
}

boolean cmpDenorm(float x, float y) {
  y = flushToZero(y);
  if (xisnanf(x) && xisnanf(y)) return true;
  if (xisnanf(x) || xisnanf(y)) return false;
  if (isinf(x) != isinf(y)) return false;
  if (x == POSITIVE_INFINITYf && y == POSITIVE_INFINITYf) return true;
  if (x == NEGATIVE_INFINITYf && y == NEGATIVE_INFINITYf) return true;
  if (y == 0) {
    if (isPlusZerof(x) && isPlusZerof(y)) return true;
    if (isMinusZerof(x) && isMinusZerof(y)) return true;
    return false;
  }
  if (!xisnanf(x) && !xisnanf(y) && !isinf(x) && !isinf(y)) return signf(x) == signf(y);
  return false;
}

double ulp(double x) {
  x = fabsf(x);
  int exp;

  if (x == 0) {
    return FLT_MIN;
  } else {
    frexpf(x, &exp);
  }

  return fmaxf(ldexpf(1.0, exp-24), FLT_MIN);
}

double countULP(double x, double y) {
  x = flushToZero(x);
  y = flushToZero(y);
  float fx = (float)x;
  float fy = (float)y;
  if (xisnan(fx) && xisnan(fy)) return 0;
  if (xisnan(fx) || xisnan(fy)) return 10000;
  if (isinf(fx)) {
    if (signf(fx) == signf(fy) && fabs(fy) > 1e+37) return 0; // Relaxed infinity handling
    return 10001;
  }
  if (fx == POSITIVE_INFINITY && fy == POSITIVE_INFINITY) return 0;
  if (fx == NEGATIVE_INFINITY && fy == NEGATIVE_INFINITY) return 0;
  if (fy == 0) {
    if (fx == 0) return 0;
    return 10002;
  }
  if (!xisnan(fx) && !xisnan(fy) && !isinf(fx) && !isinf(fy)) {
    return fabs((x - y) / ulp(y));
  }
  return 10003;
}

//

double sinfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_sin(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double sinlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_sin(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double cosfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_cos(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double coslfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_cos(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double tanfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_tan(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double tanlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_tan(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double asinfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_asin(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double asinlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_asin(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double acosfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_acos(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double acoslfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_acos(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double atanfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_atan(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double atanlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_atan(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double atan2fr(double y, double x) {
  mpfr_t frx, fry;
  mpfr_inits(frx, fry, NULL);
  mpfr_set_d(fry, y, GMP_RNDN);
  mpfr_set_d(frx, x, GMP_RNDN);
  mpfr_atan2(frx, fry, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, fry, NULL);
  return ret;
}

long double atan2lfr(long double y, long double x) {
  mpfr_t frx, fry;
  mpfr_inits(frx, fry, NULL);
  mpfr_set_ld(fry, y, GMP_RNDN);
  mpfr_set_ld(frx, x, GMP_RNDN);
  mpfr_atan2(frx, fry, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, fry, NULL);
  return ret;
}

double logfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_log(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double loglfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_log(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double expfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_exp(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double explfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_exp(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double powfr(double x, double y) {
  mpfr_t frx, fry;
  mpfr_inits(frx, fry, NULL);
  mpfr_set_d(frx, x, GMP_RNDN);
  mpfr_set_d(fry, y, GMP_RNDN);
  mpfr_pow(frx, frx, fry, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, fry, NULL);
  return ret;
}

long double powlfr(long double x, long double y) {
  mpfr_t frx, fry;
  mpfr_inits(frx, fry, NULL);
  mpfr_set_ld(frx, x, GMP_RNDN);
  mpfr_set_ld(fry, y, GMP_RNDN);
  mpfr_pow(frx, frx, fry, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, fry, NULL);
  return ret;
}

double sinhfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_sinh(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double coshfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_cosh(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double tanhfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_tanh(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double asinhfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_asinh(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double acoshfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_acosh(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double atanhfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_atanh(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double sinhlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_sinh(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double coshlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_cosh(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double tanhlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_tanh(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double asinhlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_asinh(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double acoshlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_acosh(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double atanhlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_atanh(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double sqrtlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_sqrt(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double sqrtfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_sqrt(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double cbrtfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_cbrt(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double cbrtlfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_cbrt(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double exp2fr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_exp2(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double exp2lfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_exp2(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double exp10fr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_exp10(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double exp10lfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_exp10(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double expm1fr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_expm1(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double expm1lfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_expm1(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double log10fr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_log10(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double log10lfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_log10(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

double log1pfr(double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_d(frx, d, GMP_RNDN);
  mpfr_log1p(frx, frx, GMP_RNDN);
  double ret = mpfr_get_d(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

long double log1plfr(long double d) {
  mpfr_t frx;
  mpfr_inits(frx, NULL);
  mpfr_set_ld(frx, d, GMP_RNDN);
  mpfr_log1p(frx, frx, GMP_RNDN);
  long double ret = mpfr_get_ld(frx, GMP_RNDN);
  mpfr_clears(frx, NULL);
  return ret;
}

//

typedef struct {
  float x, y;
} float2;

float child_sinf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "sinf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_sinf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_cosf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "cosf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_cosf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float2 child_sincosf(float x) {
  char str[256];
  uint32_t u, v;

  sprintf(str, "sincosf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_sincosf");
  sscanf(str, "%x %x", &u, &v);

  float2 ret;
  ret.x = u2f(u);
  ret.y = u2f(v);
  return ret;
}

float child_tanf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "tanf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_tanf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_asinf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "asinf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_asinf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_acosf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "acosf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_acosf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_atanf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "atanf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_atanf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_atan2f(float y, float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "atan2f %x %x\n", f2u(y), f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_atan2f");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_logf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "logf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_logf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_expf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "expf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_expf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_powf(float x, float y) {
  char str[256];
  uint32_t u;

  sprintf(str, "powf %x %x\n", f2u(x), f2u(y));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_powf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_sinhf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "sinhf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_sinhf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_coshf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "coshf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_coshf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_tanhf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "tanhf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_tanhf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_asinhf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "asinhf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_asinhf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_acoshf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "acoshf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_acoshf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_atanhf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "atanhf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_atanhf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_sqrtf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "sqrtf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_sqrtf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_cbrtf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "cbrtf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_cbrtf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_exp2f(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "exp2f %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_exp2f");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_exp10f(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "exp10f %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_exp10f");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_expm1f(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "expm1f %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_expm1f");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_log10f(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "log10f %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_log10f");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_log1pf(float x) {
  char str[256];
  uint32_t u;

  sprintf(str, "log1pf %x\n", f2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_log1pf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

float child_ldexpf(float x, int q) {
  char str[256];
  uint32_t u;

  sprintf(str, "ldexpf %x %x\n", f2u(x), f2u(q));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_powf");
  sscanf(str, "%x", &u);
  return u2f(u);
}

int allTestsPassed = 1;

void showResult(int success) {
  if (!success) allTestsPassed = 0;
  fprintf(stderr, "%s\n", success ? " OK" : " NG **************");
}

void do_test() {
  int i, j;

  fprintf(stderr, "Denormal/nonnumber test atan2f(y, x)\n\n");

  fprintf(stderr, "If y is +0 and x is -0, +pi is returned ... ");
  showResult(child_atan2f(+0.0, -0.0) == M_PIf);

  fprintf(stderr, "If y is -0 and x is -0, -pi is returned ... ");
  showResult(child_atan2f(-0.0, -0.0) == -M_PIf);

  fprintf(stderr, "If y is +0 and x is +0, +0 is returned ... ");
  showResult(isPlusZerof(child_atan2f(+0.0, +0.0)));

  fprintf(stderr, "If y is -0 and x is +0, -0 is returned ... ");
  showResult(isMinusZerof(child_atan2f(-0.0, +0.0)));

  fprintf(stderr, "If y is positive infinity and x is negative infinity, +3*pi/4 is returned ... ");
  showResult(child_atan2f(POSITIVE_INFINITYf, NEGATIVE_INFINITYf) == 3*M_PIf/4);

  fprintf(stderr, "If y is negative infinity and x is negative infinity, -3*pi/4 is returned ... ");
  showResult(child_atan2f(NEGATIVE_INFINITYf, NEGATIVE_INFINITYf) == -3*M_PIf/4);

  fprintf(stderr, "If y is positive infinity and x is positive infinity, +pi/4 is returned ... ");
  showResult(child_atan2f(POSITIVE_INFINITYf, POSITIVE_INFINITYf) == M_PIf/4);

  fprintf(stderr, "If y is negative infinity and x is positive infinity, -pi/4 is returned ... ");
  showResult(child_atan2f(NEGATIVE_INFINITYf, POSITIVE_INFINITYf) == -M_PIf/4);

  {
    fprintf(stderr, "If y is +0 and x is less than 0, +pi is returned ... ");

    float ya[] = { +0.0 };
    float xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };

    boolean success = true;

    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_atan2f(ya[j], xa[i]) != M_PIf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is -0 and x is less than 0, -pi is returned ... ");

    float ya[] = { -0.0 };
    float xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_atan2f(ya[j], xa[i]) != -M_PIf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is less than 0 and x is 0, -pi/2 is returned ... ");

    float ya[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };
    float xa[] = { +0.0, -0.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_atan2f(ya[j], xa[i]) != -M_PIf/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is greater than 0 and x is 0, pi/2 is returned ... ");


    float ya[] = { 100000.5, 100000, 3, 2.5, 2, 1.5, 1.0, 0.5 };
    float xa[] = { +0.0, -0.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_atan2f(ya[j], xa[i]) != M_PIf/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is greater than 0 and x is -0, pi/2 is returned ... ");

    float ya[] = { 100000.5, 100000, 3, 2.5, 2, 1.5, 1.0, 0.5 };
    float xa[] = { -0.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_atan2f(ya[j], xa[i]) != M_PIf/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is positive infinity, and x is finite, pi/2 is returned ... ");

    float ya[] = { POSITIVE_INFINITYf };
    float xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_atan2f(ya[j], xa[i]) != M_PIf/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is negative infinity, and x is finite, -pi/2 is returned ... ");

    float ya[] = { NEGATIVE_INFINITYf };
    float xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_atan2f(ya[j], xa[i]) != -M_PIf/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a finite value greater than 0, and x is negative infinity, +pi is returned ... ");

    float ya[] = { 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    float xa[] = { NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_atan2f(ya[j], xa[i]) != M_PIf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a finite value less than 0, and x is negative infinity, -pi is returned ... ");

    float ya[] = { -0.5, -1.5, -2.0, -2.5, -3.0, -100000, -100000.5 };
    float xa[] = { NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_atan2f(ya[j], xa[i]) != -M_PIf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a finite value greater than 0, and x is positive infinity, +0 is returned ... ");

    float ya[] = { 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    float xa[] = { POSITIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isPlusZerof(child_atan2f(ya[j], xa[i]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a finite value less than 0, and x is positive infinity, -0 is returned ... ");

    float ya[] = { -0.5, -1.5, -2.0, -2.5, -3.0, -100000, -100000.5 };
    float xa[] = { POSITIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isMinusZerof(child_atan2f(ya[j], xa[i]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is NaN, a NaN is returned ... ");

    float ya[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5, NANf };
    float xa[] = { NANf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!xisnanf(child_atan2f(ya[j], xa[i]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a NaN, the result is a NaN ... ");

    float ya[] = { NANf };
    float xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5, NANf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!xisnanf(child_atan2f(ya[j], xa[i]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  fprintf(stderr, "\nend of atan2f denormal/nonnumber test\n\n");

  //

  fprintf(stderr, "\nDenormal/nonnumber test pow(x, y)\n\n");

  fprintf(stderr, "If x is +1 and y is a NaN, the result is 1.0 ... ");
  showResult(child_powf(1, NANf) == 1.0);

  fprintf(stderr, "If y is 0 and x is a NaN, the result is 1.0 ... ");
  showResult(child_powf(NANf, 0) == 1.0);

  fprintf(stderr, "If x is -1, and y is positive infinity, the result is 1.0 ... ");
  showResult(child_powf(-1, POSITIVE_INFINITYf) == 1.0);

  fprintf(stderr, "If x is -1, and y is negative infinity, the result is 1.0 ... ");
  showResult(child_powf(-1, NEGATIVE_INFINITYf) == 1.0);

  {
    fprintf(stderr, "If x is a finite value less than 0, and y is a finite non-integer, a NaN is returned ... ");

    float xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };
    float ya[] = { -100000.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!xisnanf(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is a NaN, the result is a NaN ... ");

    float xa[] = { NANf };
    float ya[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!xisnanf(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a NaN, the result is a NaN ... ");

    float xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    float ya[] = { NANf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!xisnanf(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is +0, and y is an odd integer greater than 0, the result is +0 ... ");

    float xa[] = { +0.0 };
    float ya[] = { 1, 3, 5, 7, 100001 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isPlusZerof(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is -0, and y is an odd integer greater than 0, the result is -0 ... ");

    float xa[] = { -0.0 };
    float ya[] = { 1, 3, 5, 7, 100001 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isMinusZerof(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is 0, and y greater than 0 and not an odd integer, the result is +0 ... ");

    float xa[] = { +0.0, -0.0 };
    float ya[] = { 0.5, 1.5, 2.0, 2.5, 4.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isPlusZerof(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the absolute value of x is less than 1, and y is negative infinity, the result is positive infinity ... ");

    float xa[] = { -0.999, -0.5, -0.0, +0.0, +0.5, +0.999 };
    float ya[] = { NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_powf(xa[i], ya[j]) != POSITIVE_INFINITYf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the absolute value of x is greater than 1, and y is negative infinity, the result is +0 ... ");

    float xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    float ya[] = { NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isPlusZerof(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the absolute value of x is less than 1, and y is positive infinity, the result is +0 ... ");

    float xa[] = { -0.999, -0.5, -0.0, +0.0, +0.5, +0.999 };
    float ya[] = { POSITIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isPlusZerof(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the absolute value of x is greater than 1, and y is positive infinity, the result is positive infinity ... ");

    float xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    float ya[] = { POSITIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_powf(xa[i], ya[j]) != POSITIVE_INFINITYf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is negative infinity, and y is an odd integer less than 0, the result is -0 ... ");

    float xa[] = { NEGATIVE_INFINITYf };
    float ya[] = { -100001, -5, -3, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isMinusZerof(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is negative infinity, and y less than 0 and not an odd integer, the result is +0 ... ");

    float xa[] = { NEGATIVE_INFINITYf };
    float ya[] = { -100000.5, -100000, -4, -2.5, -2, -1.5, -0.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isPlusZerof(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is negative infinity, and y is an odd integer greater than 0, the result is negative infinity ... ");

    float xa[] = { NEGATIVE_INFINITYf };
    float ya[] = { 1, 3, 5, 7, 100001 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_powf(xa[i], ya[j]) != NEGATIVE_INFINITYf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is negative infinity, and y greater than 0 and not an odd integer, the result is positive infinity ... ");

    float xa[] = { NEGATIVE_INFINITYf };
    float ya[] = { 0.5, 1.5, 2, 2.5, 3.5, 4, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_powf(xa[i], ya[j]) != POSITIVE_INFINITYf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is positive infinity, and y less than 0, the result is +0 ... ");

    float xa[] = { POSITIVE_INFINITYf };
    float ya[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!isPlusZerof(child_powf(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is positive infinity, and y greater than 0, the result is positive infinity ... ");

    float xa[] = { POSITIVE_INFINITYf };
    float ya[] = { 0.5, 1, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_powf(xa[i], ya[j]) != POSITIVE_INFINITYf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is +0, and y is an odd integer less than 0, +HUGE_VAL is returned ... ");

    float xa[] = { +0.0 };
    float ya[] = { -100001, -5, -3, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_powf(xa[i], ya[j]) != POSITIVE_INFINITYf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is -0, and y is an odd integer less than 0, -HUGE_VAL is returned ... ");

    float xa[] = { -0.0 };
    float ya[] = { -100001, -5, -3, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_powf(xa[i], ya[j]) != NEGATIVE_INFINITYf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is 0, and y is less than 0 and not an odd integer, +HUGE_VAL is returned ... ");

    float xa[] = { +0.0, -0.0 };
    float ya[] = { -100000.5, -100000, -4, -2.5, -2, -1.5, -0.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (child_powf(xa[i], ya[j]) != POSITIVE_INFINITYf) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the result overflows, the functions return HUGE_VAL with the mathematically correct sign ... ");

    float xa[] = { 1000, -1000 };
    float ya[] = { 1000, 1000.5, 1001 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(float) && success;j++) {
	if (!cmpDenorm(child_powf(xa[i], ya[j]), powfr(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  fprintf(stderr, "\nEnd of pow denormal/nonnumber test\n\n");
	
  //

  {
    fprintf(stderr, "sinf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_sinf(xa[i]), sinfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_sinf(xa[i]), sinfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "sin in sincosf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      float2 q = child_sincosf(xa[i]);
      if (!cmpDenorm(q.x, sinfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], q.x, sinfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "cosf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_cosf(xa[i]), cosfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_cosf(xa[i]), cosfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "cos in sincosf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      float2 q = child_sincosf(xa[i]);
      if (!cmpDenorm(q.y, cosfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], q.y, cosfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "tanf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, M_PIf/2, -M_PIf/2 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_tanf(xa[i]), tanfr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "asinf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 2, -2, 1, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_asinf(xa[i]), asinfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_asinf(xa[i]), asinfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "acosf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 2, -2, 1, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_acosf(xa[i]), acosfr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "atanf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_atanf(xa[i]), atanfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_atanf(xa[i]), atanfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "logf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 0, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_logf(xa[i]), logfr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "expf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, -2000, 2000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_expf(xa[i]), expfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_expf(xa[i]), expfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "sinhf denormal/nonnumber test ... ");

    float xa[] = { NANf, +0.0, -0.0, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_sinhf(xa[i]), sinhfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_sinhf(xa[i]), sinhfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "coshf denormal/nonnumber test ... ");

    float xa[] = { NANf, +0.0, -0.0, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_coshf(xa[i]), coshfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_coshf(xa[i]), coshfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "tanhf denormal/nonnumber test ... ");

    float xa[] = { NANf, +0.0, -0.0, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_tanhf(xa[i]), tanhfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_tanhf(xa[i]), tanhfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "asinhf denormal/nonnumber test ... ");

    float xa[] = { NANf, +0.0, -0.0, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_asinhf(xa[i]), asinhfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_asinhf(xa[i]), asinhfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "acoshf denormal/nonnumber test ... ");

    float xa[] = { NANf, +0.0, -0.0, 1.0, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_acoshf(xa[i]), acoshfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_acoshf(xa[i]), acoshfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "atanhf denormal/nonnumber test ... ");

    float xa[] = { NANf, +0.0, -0.0, 1.0, -1.0, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_atanhf(xa[i]), atanhfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_atanhf(xa[i]), atanhfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "sqrtf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, +0.0, -0.0, -1.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_sqrtf(xa[i]), sqrtfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_sqrtf(xa[i]), sqrtfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "cbrtf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, +0.0, -0.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_cbrtf(xa[i]), cbrtfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_cbrtf(xa[i]), cbrtfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "exp2f denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_exp2f(xa[i]), exp2fr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_exp2f(xa[i]), exp2fr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "exp10f denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_exp10f(xa[i]), exp10fr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_exp10f(xa[i]), exp10fr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "expm1f denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_expm1f(xa[i]), expm1fr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_expm1f(xa[i]), expm1fr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "log10f denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 0, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_log10f(xa[i]), log10fr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "log1pf denormal/nonnumber test ... ");

    float xa[] = { NANf, POSITIVE_INFINITYf, NEGATIVE_INFINITYf, 0, -1, -2 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(float) && success;i++) {
      if (!cmpDenorm(child_log1pf(xa[i]), log1pfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_log1pf(xa[i]), log1pfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  fprintf(stderr, "ldexpf denormal/nonnumber test ... ");

  {
    boolean success = true;
    for(i=-10000;i<=10000 && success;i++) {
      float d = child_ldexpf(1.0f, i);
      float c = ldexpf(1.0f, i);

      boolean pass = (isfinite(c) && d == c) || cmpDenorm(d, c);
      if (!pass) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", (double)i, d, c);
	success = false;
	break;
      }
    }

    showResult(success);
  }

  //

  fprintf(stderr, "\nAccuracy test (max error in ulp)\n");

  //

  //

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_sinf(d);
      double c = sinlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SIN;
      }
    }

    for(d = -10000;d < 10000;d += 0.201) {
      float q = child_sinf(d);
      double c = sinlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SIN;
      }
    }

    int i;

    for(i=1;i<10000;i++) {
      float start = u2f(f2u(M_PI_4 * i)-20);
      float end = u2f(f2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2f(f2u(d)+1)) {
	float q = child_sinf(d);
	double c = sinlfr(flushToZero(d));
	double u = countULP(q, c);
	max = fmax(max, u);
	if (u > 1000) {
	  fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	  goto STOP_SIN;
	}
      }
    }

  STOP_SIN:

    fprintf(stderr, "sinf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_cosf(d);
      double c = coslfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = -10000;d < 10000;d += 0.201) {
      float q = child_cosf(d);
      double c = coslfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    int i;

    for(i=1;i<10000;i++) {
      float start = u2f(f2u(M_PI_4 * i)-20);
      float end = u2f(f2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2f(f2u(d)+1)) {
	float q = child_cosf(d);
	double c = coslfr(flushToZero(d));
	double u = countULP(q, c);
	max = fmax(max, u);
      }
    }

    fprintf(stderr, "cosf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float2 q = child_sincosf(d);
      long double c = sinlfr(d);
      double u = fabs((q.x - c) / ulp(c));
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q.x, (double)c, d, (double)ulp(c));
	goto STOP_SIN2;
      }
    }

    for(d = -10000;d < 10000;d += 0.201) {
      float2 q = child_sincosf(d);
      long double c = sinlfr(d);
      double u = fabs((q.x - c) / ulp(c));
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q.x, (double)c, d, (double)ulp(c));
	goto STOP_SIN2;
      }
    }

    int i;

    for(i=1;i<10000;i++) {
      float start = u2f(f2u(M_PI_4 * i)-20);
      float end = u2f(f2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2f(f2u(d)+1)) {
	float2 q = child_sincosf(d);
	double c = sinlfr(d);
	double u = fabs((q.x - c) / ulp(c));
	max = fmax(max, u);
	if (u > 1000) {
	  fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q.x, (double)c, d, (double)ulp(c));
	  goto STOP_SIN2;
	}
      }
    }

  STOP_SIN2:

    fprintf(stderr, "sin in sincosf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float2 q = child_sincosf(d);
      double c = coslfr(d);
      double u = fabs((q.y - c) / ulp(c));
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q.y, (double)c, d, (double)ulp(c));
	goto STOP_COS2;
      }
    }

    for(d = -10000;d < 10000;d += 0.201) {
      float2 q = child_sincosf(d);
      double c = coslfr(d);
      double u = fabs((q.y - c) / ulp(c));
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q.y, (double)c, d, (double)ulp(c));
	goto STOP_COS2;
      }
    }

    int i;

    for(i=1;i<10000;i++) {
      float start = u2f(f2u(M_PI_4 * i)-20);
      float end = u2f(f2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2f(f2u(d)+1)) {
	float2 q = child_sincosf(d);
	double c = coslfr(d);
	double u = fabs((q.y - c) / ulp(c));
	max = fmax(max, u);
      }
    }

  STOP_COS2:

    fprintf(stderr, "cos in sincosf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_tanf(d);
      double c = tanlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = -10000;d < 10000;d += 0.201) {
      float q = child_tanf(d);
      double c = tanlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    int i;

    for(i=1;i<10000;i++) {
      float start = u2f(f2u(M_PI_4 * i)-20);
      float end = u2f(f2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2f(f2u(d)+1)) {
	float q = child_tanf(d);
	double c = tanlfr(flushToZero(d));
	double u = countULP(q, c);
	max = fmax(max, u);
      }
    }

    fprintf(stderr, "tanf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -1;d < 1;d += 0.00002) {
      float q = child_asinf(d);
      double c = asinlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ASIN;
      }
    }

  STOP_ASIN:

    fprintf(stderr, "asinf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -1;d < 1;d += 0.00002) {
      float q = child_acosf(d);
      double c = acoslfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ACOS;
      }
    }

  STOP_ACOS:

    fprintf(stderr, "acosf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_atanf(d);
      double c = atanlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %g, d = %g\n", q, d);
	goto STOP_ATAN;
      }
    }

    for(d = -10000;d < 10000;d += 0.201) {
      float q = child_atanf(d);
      double c = atanlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %g, d = %g\n", q, d);
	goto STOP_ATAN;
      }
    }

  STOP_ATAN:

    fprintf(stderr, "atanf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float x, y, max = 0;

    for(y = -10;y < 10;y += 0.051) {
      for(x = -10;x < 10;x += 0.052) {
	float q = child_atan2f(y, x);
	double c = atan2lfr(flushToZero(y), flushToZero(x));
	double u = countULP(q, c);
	max = fmax(max, u);
	if (u > 1000) {
	  fprintf(stderr, "q = %g, y = %g, x = %g\n", q, y, x);
	  goto STOP_ATAN2;
	}
      }
    }

    for(y = -100;y < 100;y += 0.51) {
      for(x = -100;x < 100;x += 0.52) {
	float q = child_atan2f(y, x);
	double c = atan2lfr(flushToZero(y), flushToZero(x));
	double u = countULP(q, c);
	max = fmax(max, u);
	if (u > 1000) {
	  fprintf(stderr, "q = %g, y = %g, x = %g\n", q, y, x);
	  goto STOP_ATAN2;
	}
      }
    }

  STOP_ATAN2:

    fprintf(stderr, "atan2f : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = 0.0001;d < 10;d += 0.0001) {
      float q = child_logf(d);
      double c = loglfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_LOG;
      }
    }

    for(d = 0.0001;d < 10000;d += 0.1) {
      float q = child_logf(d);
      double c = loglfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_LOG;
      }
    }

    int i;
    for(i = -1000;i <= 1000;i++) {
      d = pow(1.1, i);
      float q = child_logf(d);
      double c = loglfr(flushToZero(d));
      double u = countULP(q, c);
      if (flushToZero(d * 0.1) == 0.0 && q == NEGATIVE_INFINITYf) u = 0;
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_LOG;
      }
    }

  STOP_LOG:

    fprintf(stderr, "logf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_expf(d);
      double c = explfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = -1000;d < 1000;d += 0.1) {
      float q = child_expf(d);
      double c = explfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    fprintf(stderr, "expf : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float x, y, max = 0;

    for(y = 0.1;y < 100;y += 0.21) {
      for(x = -100;x < 100;x += 0.22) {
	float q = child_powf(x, y);
	double c = powlfr(flushToZero(x), flushToZero(y));
	double u = countULP(q, c);
	max = fmax(max, u);
	if (u > 10) {
	  fprintf(stderr, "q = %g, c = %g, x = %g, y = %g\n", q, c, x, y);
	  goto STOP_POW;
	}
      }
    }

    float d;
    for(d = -1000;d < 1000;d += 0.1) {
      float q = child_powf(2.1f, d);
      double c = powlfr(2.1f, flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);

      if (u > 10) {
	fprintf(stderr, "q = %g, c = %g, d = %g\n", q, c, d);
	goto STOP_POW;
      }
    }

  STOP_POW:

    fprintf(stderr, "powf : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = 0;d < 20000;d += 0.2) {
      float q = child_sqrtf(d);
      double c = sqrtlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SQRT;
      }
    }

    int i;
    for(i = -1000;i <= 1000;i++) {
      d = pow(1.1, i);
      float q = child_sqrtf(d);
      double c = sqrtlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SQRT;
      }
    }

  STOP_SQRT:

    fprintf(stderr, "sqrtf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -10000;d < 10000;d += 0.2) {
      float q = child_cbrtf(d);
      double c = cbrtlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_CBRT;
      }
    }

    int i;
    for(i = -1000;i <= 1000;i++) {
      d = pow(1.1, i);
      float q = child_cbrtf(d);
      double c = cbrtlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_CBRT;
      }
    }

  STOP_CBRT:

    fprintf(stderr, "cbrtf : %lf ... ", max);

    showResult(max < 5);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_sinhf(d);
      double c = sinhlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SINH;
      }
    }

    for(d = -100;d < 100;d += 0.02) {
      float q = child_sinhf(d);
      double c = sinhlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SINH;
      }
    }

  STOP_SINH:

    fprintf(stderr, "sinhf : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_coshf(d);
      double c = coshlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_COSH;
      }
    }

    for(d = -100;d < 100;d += 0.02) {
      float q = child_coshf(d);
      double c = coshlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_COSH;
      }
    }

  STOP_COSH:

    fprintf(stderr, "coshf : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_tanhf(d);
      double c = tanhlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_TANH;
      }
    }

  STOP_TANH:

    fprintf(stderr, "tanhf : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_asinhf(d);
      double c = asinhlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ASINH;
      }
    }

    for(d = -1000;d < 1000;d += 0.02) {
      float q = child_asinhf(d);
      double c = asinhlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ASINH;
      }
    }

  STOP_ASINH:

    fprintf(stderr, "asinhf : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = 1;d < 10;d += 0.0002) {
      float q = child_acoshf(d);
      double c = acoshlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ACOSH;
      }
    }

    for(d = 1;d < 1000;d += 0.02) {
      float q = child_acoshf(d);
      double c = acoshlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ACOSH;
      }
    }

  STOP_ACOSH:

    fprintf(stderr, "acoshf : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_atanhf(d);
      double c = atanhlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ATANH;
      }
    }

    for(d = -1000;d < 1000;d += 0.023) {
      float q = child_atanhf(d);
      double c = atanhlfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ATANH;
      }
    }

  STOP_ATANH:

    fprintf(stderr, "atanhf : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = 0.0001;d < 10;d += 0.0001) {
      float q = child_log10f(d);
      double c = log10lfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = 0.0001;d < 10000;d += 0.1) {
      float q = child_log10f(d);
      double c = log10lfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    fprintf(stderr, "log10f : %lf ... ", max);

    showResult(max < 1);
  }


  {
    float d, max = 0;

    for(d = 0.0001;d < 10;d += 0.0001) {
      float q = child_log1pf(d);
      double c = log1plfr(flushToZero(d));
      double u = countULP(q, c);
      if (u > 10) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_LOG1P;
      }
      max = fmax(max, u);
    }

    for(d = 0.0001;d < 10000;d += 0.1) {
      float q = child_log1pf(d);
      double c = log1plfr(flushToZero(d));
      double u = countULP(q, c);
      if (u > 10) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_LOG1P;
      }
      max = fmax(max, u);
    }

    for(d = 0;d < 300;d += 0.02) {
      float d2 = pow(10, -d);
      float q = child_log1pf(d2);
      double c = log1plfr(flushToZero(d2));
      double u = countULP(q, c);
      if (flushToZero(d2 * 0.1) == 0.0 && q == 0.0) u = 0;
      if (u > 10) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d2, (double)ulp(c));
	goto STOP_LOG1P;
      }
      max = fmax(max, u);
    }

    for(d = 0;d < 300;d += 0.02) {
      float d2 = -pow(10, -d);
      float q = child_log1pf(d2);
      double c = log1plfr(flushToZero(d2));
      double u = countULP(q, c);
      if (flushToZero(d2 * 0.1) == 0.0 && q == 0.0) u = 0;
      if (u > 10) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d2, (double)ulp(c));
	goto STOP_LOG1P;
      }
      max = fmax(max, u);
    }

  STOP_LOG1P:

    fprintf(stderr, "log1pf : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_exp2f(d);
      double c = exp2lfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_EXP2;
      }
    }

    for(d = -120;d < 1000;d += 0.023) {
      float q = child_exp2f(d);
      double c = exp2lfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_EXP2;
      }
    }

  STOP_EXP2:

    fprintf(stderr, "exp2f : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_exp10f(d);
      double c = exp10lfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_EXP10;
      }
    }

    for(d = -35;d < 1000;d += 0.023) {
      float q = child_exp10f(d);
      double c = exp10lfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_EXP10;
      }
    }

  STOP_EXP10:

    fprintf(stderr, "exp10f : %lf ... ", max);

    showResult(max < 1);
  }

  {
    float d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      float q = child_expm1f(d);
      double c = expm1lfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_EXPM1;
      }
    }

    for(d = -1000;d < 1000;d += 0.023) {
      float q = child_expm1f(d);
      double c = expm1lfr(flushToZero(d));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_EXPM1;
      }
    }

    for(d = 0;d < 300;d += 0.02) {
      float d2 = pow(10, -d);
      float q = child_expm1f(d2);
      double c = expm1lfr(flushToZero(d2));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_EXPM1;
      }
    }

    for(d = 0;d < 300;d += 0.02) {
      float d2 = -pow(10, -d);
      float q = child_expm1f(d2);
      double c = expm1lfr(flushToZero(d2));
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_EXPM1;
      }
    }

  STOP_EXPM1:

    fprintf(stderr, "expm1f : %lf ... ", max);

    showResult(max < 5);
  }
}

int main(int argc, char **argv) {
  char *argv2[argc];
  int i, a2s;

  for(a2s=1;a2s<argc;a2s++) {
    if (strcmp(argv[a2s], "--flushtozero") == 0) {
      enableFlushToZero = 1;
    } else {
      break;
    }
  }

  for(i=a2s;i<argc;i++) argv2[i-a2s] = argv[i];
  argv2[argc-a2s] = NULL;

  mpfr_set_default_prec(128);

  startChild(argv2[0], argv2);

  do_test();

  if (allTestsPassed) {
    fprintf(stderr, "\n\n*** All tests passed");
    if (enableFlushToZero) fprintf(stderr, " (flush to zero)");
    fprintf(stderr, "\n");
  } else {
    fprintf(stderr, "\n\n*** There were errors in some tests\n");
  }

  if (allTestsPassed) return 0;

  return -1;
}
