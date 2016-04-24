#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <inttypes.h>

#include <mpfr.h>

#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <signal.h>

#include "nonnumber.h"

#define POSITIVE_INFINITY INFINITY
#define NEGATIVE_INFINITY (-INFINITY)

typedef int boolean;

#define true 1
#define false 0

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

    fprintf(stderr, "execvp in startChild : %s\n", strerror(errno));

    assert(0);
  }

  // parent process

  close(ptoc[0]);
  close(ctop[1]);
}

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

//

boolean isPlusZero(double x) { return x == 0 && copysign(1, x) == 1; }
boolean isMinusZero(double x) { return x == 0 && copysign(1, x) == -1; }
boolean xisnan(double x) { return x != x; }
double sign(double d) { return d < 0 ? -1 : 1; }

boolean cmpDenorm(double x, double y) {
  if (xisnan(x) && xisnan(y)) return true;
  if (xisnan(x) || xisnan(y)) return false;
  if (isinf(x) != isinf(y)) return false;
  if (x == POSITIVE_INFINITY && y == POSITIVE_INFINITY) return true;
  if (x == NEGATIVE_INFINITY && y == NEGATIVE_INFINITY) return true;
  if (y == 0) {
    if (isPlusZero(x) && isPlusZero(y)) return true;
    if (isMinusZero(x) && isMinusZero(y)) return true;
    return false;
  }
  if (!xisnan(x) && !xisnan(y) && !isinf(x) && !isinf(y)) return sign(x) == sign(y);
  return false;
}

long double ulp(long double x) {
  x = fabsl(x);
  int exp;

  if (x == 0) {
    return DBL_MIN;
  } else {
    frexpl(x, &exp);
  }

  return fmax(ldexp(1.0, exp-53), DBL_MIN);
}

double countULP(long double x, long double y) {
  double fx = x;
  double fy = y;
  if (xisnan(fx) && xisnan(fy)) return 0;
  if (xisnan(fx) || xisnan(fy)) return 10000;
  if (isinf(fx)) {
    if (sign(fx) == sign(fy) && fabs(fy) > 1e+300) return 0; // Relaxed infinity handling
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
  double x, y;
} double2;

double child_sin(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "sin_u1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_sin");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_cos(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "cos_u1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_cos");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double2 child_sincos(double x) {
  char str[256];
  uint64_t u, v;

  sprintf(str, "sincos_u1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_sincos");
  sscanf(str, "%" PRIx64 " %" PRIx64, &u, &v);

  double2 ret;
  ret.x = u2d(u);
  ret.y = u2d(v);
  return ret;
}

double child_tan(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "tan_u1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_tan");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_asin(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "asin_u1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_asin_");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_acos(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "acos_u1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_acos");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_atan(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "atan_u1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_atan");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_atan2(double y, double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "atan2_u1 %" PRIx64 " %" PRIx64 "\n", d2u(y), d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_atan2");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_log(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "log_u1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_log");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_exp(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "exp %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_exp");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_pow(double x, double y) {
  char str[256];
  uint64_t u;

  sprintf(str, "pow %" PRIx64 " %" PRIx64 "\n", d2u(x), d2u(y));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_pow");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_sinh(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "sinh %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_sinh");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_cosh(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "cosh %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_cosh");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_tanh(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "tanh %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_tanh");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_asinh(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "asinh %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_asinh");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_acosh(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "acosh %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_acosh");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_atanh(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "atanh %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_atanh");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_sqrt(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "sqrt %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_sqrt");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_cbrt(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "cbrt_u1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_cbrt");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_exp2(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "exp2 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_exp2");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_exp10(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "exp10 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_exp10");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_expm1(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "expm1 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_expm1");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_log10(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "log10 %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_log10");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_log1p(double x) {
  char str[256];
  uint64_t u;

  sprintf(str, "log1p %" PRIx64 "\n", d2u(x));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_log1p");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

double child_ldexp(double x, int q) {
  char str[256];
  uint64_t u;

  sprintf(str, "ldexp %" PRIx64 " %" PRIx64 "\n", d2u(x), d2u(q));
  write(ptoc[1], str, strlen(str));
  if (readln(ctop[0], str, 255) < 1) stop("child_ldexp");
  sscanf(str, "%" PRIx64, &u);
  return u2d(u);
}

int allTestsPassed = 1;

void showResult(int success) {
  if (!success) allTestsPassed = 0;
  fprintf(stderr, "%s\n", success ? " OK" : " NG **************");
}

void do_test() {
  int i, j;

  fprintf(stderr, "Denormal/nonnumber test atan2_u1(y, x)\n\n");

  fprintf(stderr, "If y is +0 and x is -0, +pi is returned ... ");
  showResult(child_atan2(+0.0, -0.0) == M_PI);

  fprintf(stderr, "If y is -0 and x is -0, -pi is returned ... ");
  showResult(child_atan2(-0.0, -0.0) == -M_PI);

  fprintf(stderr, "If y is +0 and x is +0, +0 is returned ... ");
  showResult(isPlusZero(child_atan2(+0.0, +0.0)));

  fprintf(stderr, "If y is -0 and x is +0, -0 is returned ... ");
  showResult(isMinusZero(child_atan2(-0.0, +0.0)));

  fprintf(stderr, "If y is positive infinity and x is negative infinity, +3*pi/4 is returned ... ");
  showResult(child_atan2(POSITIVE_INFINITY, NEGATIVE_INFINITY) == 3*M_PI/4);

  fprintf(stderr, "If y is negative infinity and x is negative infinity, -3*pi/4 is returned ... ");
  showResult(child_atan2(NEGATIVE_INFINITY, NEGATIVE_INFINITY) == -3*M_PI/4);

  fprintf(stderr, "If y is positive infinity and x is positive infinity, +pi/4 is returned ... ");
  showResult(child_atan2(POSITIVE_INFINITY, POSITIVE_INFINITY) == M_PI/4);

  fprintf(stderr, "If y is negative infinity and x is positive infinity, -pi/4 is returned ... ");
  showResult(child_atan2(NEGATIVE_INFINITY, POSITIVE_INFINITY) == -M_PI/4);

  {
    fprintf(stderr, "If y is +0 and x is less than 0, +pi is returned ... ");

    double ya[] = { +0.0 };
    double xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };

    boolean success = true;

    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_atan2(ya[j], xa[i]) != M_PI) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is -0 and x is less than 0, -pi is returned ... ");

    double ya[] = { -0.0 };
    double xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_atan2(ya[j], xa[i]) != -M_PI) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is less than 0 and x is 0, -pi/2 is returned ... ");

    double ya[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };
    double xa[] = { +0.0, -0.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_atan2(ya[j], xa[i]) != -M_PI/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is greater than 0 and x is 0, pi/2 is returned ... ");


    double ya[] = { 100000.5, 100000, 3, 2.5, 2, 1.5, 1.0, 0.5 };
    double xa[] = { +0.0, -0.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_atan2(ya[j], xa[i]) != M_PI/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is greater than 0 and x is -0, pi/2 is returned ... ");

    double ya[] = { 100000.5, 100000, 3, 2.5, 2, 1.5, 1.0, 0.5 };
    double xa[] = { -0.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_atan2(ya[j], xa[i]) != M_PI/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is positive infinity, and x is finite, pi/2 is returned ... ");

    double ya[] = { POSITIVE_INFINITY };
    double xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_atan2(ya[j], xa[i]) != M_PI/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is negative infinity, and x is finite, -pi/2 is returned ... ");

    double ya[] = { NEGATIVE_INFINITY };
    double xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_atan2(ya[j], xa[i]) != -M_PI/2) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a finite value greater than 0, and x is negative infinity, +pi is returned ... ");

    double ya[] = { 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    double xa[] = { NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_atan2(ya[j], xa[i]) != M_PI) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a finite value less than 0, and x is negative infinity, -pi is returned ... ");

    double ya[] = { -0.5, -1.5, -2.0, -2.5, -3.0, -100000, -100000.5 };
    double xa[] = { NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_atan2(ya[j], xa[i]) != -M_PI) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a finite value greater than 0, and x is positive infinity, +0 is returned ... ");

    double ya[] = { 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    double xa[] = { POSITIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isPlusZero(child_atan2(ya[j], xa[i]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a finite value less than 0, and x is positive infinity, -0 is returned ... ");

    double ya[] = { -0.5, -1.5, -2.0, -2.5, -3.0, -100000, -100000.5 };
    double xa[] = { POSITIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isMinusZero(child_atan2(ya[j], xa[i]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is NaN, a NaN is returned ... ");

    double ya[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5, NAN };
    double xa[] = { NAN };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!xisnan(child_atan2(ya[j], xa[i]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a NaN, the result is a NaN ... ");

    double ya[] = { NAN };
    double xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5, NAN };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!xisnan(child_atan2(ya[j], xa[i]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  fprintf(stderr, "\nend of atan2_u1 denormal/nonnumber test\n");

  //

#if 0
  fprintf(stderr, "\nDenormal/nonnumber test pow(x, y)\n\n");

  fprintf(stderr, "If x is +1 and y is a NaN, the result is 1.0 ... ");
  showResult(child_pow(1, NAN) == 1.0);

  fprintf(stderr, "If y is 0 and x is a NaN, the result is 1.0 ... ");
  showResult(child_pow(NAN, 0) == 1.0);

  fprintf(stderr, "If x is -1, and y is positive infinity, the result is 1.0 ... ");
  showResult(child_pow(-1, POSITIVE_INFINITY) == 1.0);

  fprintf(stderr, "If x is -1, and y is negative infinity, the result is 1.0 ... ");
  showResult(child_pow(-1, NEGATIVE_INFINITY) == 1.0);

  {
    fprintf(stderr, "If x is a finite value less than 0, and y is a finite non-integer, a NaN is returned ... ");

    double xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };
    double ya[] = { -100000.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!xisnan(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is a NaN, the result is a NaN ... ");

    double xa[] = { NAN };
    double ya[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!xisnan(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If y is a NaN, the result is a NaN ... ");

    double xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    double ya[] = { NAN };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!xisnan(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is +0, and y is an odd integer greater than 0, the result is +0 ... ");

    double xa[] = { +0.0 };
    double ya[] = { 1, 3, 5, 7, 100001 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isPlusZero(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is -0, and y is an odd integer greater than 0, the result is -0 ... ");

    double xa[] = { -0.0 };
    double ya[] = { 1, 3, 5, 7, 100001 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isMinusZero(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is 0, and y greater than 0 and not an odd integer, the result is +0 ... ");

    double xa[] = { +0.0, -0.0 };
    double ya[] = { 0.5, 1.5, 2.0, 2.5, 4.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isPlusZero(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the absolute value of x is less than 1, and y is negative infinity, the result is positive infinity ... ");

    double xa[] = { -0.999, -0.5, -0.0, +0.0, +0.5, +0.999 };
    double ya[] = { NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_pow(xa[i], ya[j]) != POSITIVE_INFINITY) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the absolute value of x is greater than 1, and y is negative infinity, the result is +0 ... ");

    double xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    double ya[] = { NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isPlusZero(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the absolute value of x is less than 1, and y is positive infinity, the result is +0 ... ");

    double xa[] = { -0.999, -0.5, -0.0, +0.0, +0.5, +0.999 };
    double ya[] = { POSITIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isPlusZero(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the absolute value of x is greater than 1, and y is positive infinity, the result is positive infinity ... ");

    double xa[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
    double ya[] = { POSITIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_pow(xa[i], ya[j]) != POSITIVE_INFINITY) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is negative infinity, and y is an odd integer less than 0, the result is -0 ... ");

    double xa[] = { NEGATIVE_INFINITY };
    double ya[] = { -100001, -5, -3, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isMinusZero(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is negative infinity, and y less than 0 and not an odd integer, the result is +0 ... ");

    double xa[] = { NEGATIVE_INFINITY };
    double ya[] = { -100000.5, -100000, -4, -2.5, -2, -1.5, -0.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isPlusZero(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is negative infinity, and y is an odd integer greater than 0, the result is negative infinity ... ");

    double xa[] = { NEGATIVE_INFINITY };
    double ya[] = { 1, 3, 5, 7, 100001 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_pow(xa[i], ya[j]) != NEGATIVE_INFINITY) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is negative infinity, and y greater than 0 and not an odd integer, the result is positive infinity ... ");

    double xa[] = { NEGATIVE_INFINITY };
    double ya[] = { 0.5, 1.5, 2, 2.5, 3.5, 4, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_pow(xa[i], ya[j]) != POSITIVE_INFINITY) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is positive infinity, and y less than 0, the result is +0 ... ");

    double xa[] = { POSITIVE_INFINITY };
    double ya[] = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!isPlusZero(child_pow(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is positive infinity, and y greater than 0, the result is positive infinity ... ");

    double xa[] = { POSITIVE_INFINITY };
    double ya[] = { 0.5, 1, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_pow(xa[i], ya[j]) != POSITIVE_INFINITY) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is +0, and y is an odd integer less than 0, +HUGE_VAL is returned ... ");

    double xa[] = { +0.0 };
    double ya[] = { -100001, -5, -3, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_pow(xa[i], ya[j]) != POSITIVE_INFINITY) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is -0, and y is an odd integer less than 0, -HUGE_VAL is returned ... ");

    double xa[] = { -0.0 };
    double ya[] = { -100001, -5, -3, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_pow(xa[i], ya[j]) != NEGATIVE_INFINITY) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If x is 0, and y is less than 0 and not an odd integer, +HUGE_VAL is returned ... ");

    double xa[] = { +0.0, -0.0 };
    double ya[] = { -100000.5, -100000, -4, -2.5, -2, -1.5, -0.5 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (child_pow(xa[i], ya[j]) != POSITIVE_INFINITY) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "If the result overflows, the functions return HUGE_VAL with the mathematically correct sign ... ");

    double xa[] = { 1000, -1000 };
    double ya[] = { 1000, 1000.5, 1001 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      for(j=0;j<sizeof(ya)/sizeof(double) && success;j++) {
	if (!cmpDenorm(child_pow(xa[i], ya[j]), powfr(xa[i], ya[j]))) {
	  success = false;
	  break;
	}
      }
    }

    showResult(success);
  }

  fprintf(stderr, "\nEnd of pow denormal/nonnumber test\n\n");
#endif

  //

  {
    fprintf(stderr, "sin_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_sin(xa[i]), sinfr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "sin in sincos_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      double2 q = child_sincos(xa[i]);
      if (!cmpDenorm(q.x, sinfr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "cos_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_cos(xa[i]), cosfr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "cos in sincos_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      double2 q = child_sincos(xa[i]);
      if (!cmpDenorm(q.y, cosfr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "tan_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY, M_PI/2, -M_PI/2 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_tan(xa[i]), tanfr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "asin_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY, 2, -2, 1, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_asin(xa[i]), asinfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_asin(xa[i]), asinfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "acos_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY, 2, -2, 1, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_acos(xa[i]), acosfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_acos(xa[i]), acosfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "atan_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_atan(xa[i]), atanfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_atan(xa[i]), atanfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "log_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY, 0, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_log(xa[i]), logfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_log(xa[i]), logfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

#if 0
  {
    fprintf(stderr, "exp denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_exp(xa[i]), expfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_exp(xa[i]), expfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "sinh denormal/nonnumber test ... ");

    double xa[] = { NAN, +0.0, -0.0, POSITIVE_INFINITY, NEGATIVE_INFINITY, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_sinh(xa[i]), sinhfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_sinh(xa[i]), sinhfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "cosh denormal/nonnumber test ... ");

    double xa[] = { NAN, +0.0, -0.0, POSITIVE_INFINITY, NEGATIVE_INFINITY, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_cosh(xa[i]), coshfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_cosh(xa[i]), coshfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "tanh denormal/nonnumber test ... ");

    double xa[] = { NAN, +0.0, -0.0, POSITIVE_INFINITY, NEGATIVE_INFINITY, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_tanh(xa[i]), tanhfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_tanh(xa[i]), tanhfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "asinh denormal/nonnumber test ... ");

    double xa[] = { NAN, +0.0, -0.0, POSITIVE_INFINITY, NEGATIVE_INFINITY, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_asinh(xa[i]), asinhfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_asinh(xa[i]), asinhfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "acosh denormal/nonnumber test ... ");

    double xa[] = { NAN, +0.0, -0.0, 1.0, POSITIVE_INFINITY, NEGATIVE_INFINITY, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_acosh(xa[i]), acoshfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_acosh(xa[i]), acoshfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "atanh denormal/nonnumber test ... ");

    double xa[] = { NAN, +0.0, -0.0, 1.0, -1.0, POSITIVE_INFINITY, NEGATIVE_INFINITY, 10000, -10000 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_atanh(xa[i]), atanhfr(xa[i]))) {
	fprintf(stderr, "\nxa = %.20g, d = %.20g, c = %.20g", xa[i], child_atanh(xa[i]), atanhfr(xa[i]));
	success = false;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "sqrt denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY, +0.0, -0.0, -1.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_sqrt(xa[i]), sqrtfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_sqrt(xa[i]), sqrtfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }
#endif

  {
    fprintf(stderr, "cbrt_u1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY, +0.0, -0.0 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_cbrt(xa[i]), cbrtfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_cbrt(xa[i]), cbrtfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

#if 0
  {
    fprintf(stderr, "exp2 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_exp2(xa[i]), exp2fr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_exp2(xa[i]), exp2fr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "exp10 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_exp10(xa[i]), exp10fr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_exp10(xa[i]), exp10fr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "expm1 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_expm1(xa[i]), expm1fr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_expm1(xa[i]), expm1fr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "log10 denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY, 0, -1 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_log10(xa[i]), log10fr(xa[i]))) {
	success = false;
	break;
      }
    }

    showResult(success);
  }

  {
    fprintf(stderr, "log1p denormal/nonnumber test ... ");

    double xa[] = { NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY, 0, -1, -2 };

    boolean success = true;
    for(i=0;i<sizeof(xa)/sizeof(double) && success;i++) {
      if (!cmpDenorm(child_log1p(xa[i]), log1pfr(xa[i]))) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", xa[i], child_log1p(xa[i]), log1pfr(xa[i]));
	success = false;
	break;
      }
    }

    showResult(success);
  }

  fprintf(stderr, "ldexp denormal/nonnumber test ... ");

  {
    boolean success = true;
    for(i=-10000;i<=10000 && success;i++) {
      double d = child_ldexp(1.0, i);
      double c = ldexp(1.0, i);

      boolean pass = (isfinite(c) && d == c) || cmpDenorm(c, d);
      if (!pass) {
	fprintf(stderr, "xa = %.20g, d = %.20g, c = %.20g\n", (double)i, d, c);
	success = false;
	break;
      }
    }

    showResult(success);
  }
#endif

  //

  fprintf(stderr, "\nAccuracy test (max error in ulp)\n");

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_sin(d);
      long double c = sinlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SIN;
      }
    }

    for(d = -10000000;d < 10000000;d += 200.1) {
      double q = child_sin(d);
      long double c = sinlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SIN;
      }
    }

    int i;

    for(i=1;i<10000;i++) {
      double start = u2d(d2u(M_PI_4 * i)-20);
      double end = u2d(d2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2d(d2u(d)+1)) {
	double q = child_sin(d);
	long double c = sinlfr(d);
	double u = countULP(q, c);
	max = fmax(max, u);
	if (u > 1000) {
	  fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	  goto STOP_SIN;
	}
      }
    }

  STOP_SIN:

    fprintf(stderr, "sin_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_cos(d);
      long double c = coslfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = -10000000;d < 10000000;d += 200.1) {
      double q = child_cos(d);
      long double c = coslfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    int i;

    for(i=1;i<10000;i++) {
      double start = u2d(d2u(M_PI_4 * i)-20);
      double end = u2d(d2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2d(d2u(d)+1)) {
	double q = child_cos(d);
	long double c = coslfr(d);
	double u = countULP(q, c);
	max = fmax(max, u);
      }
    }

    fprintf(stderr, "cos_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double2 q = child_sincos(d);
      long double c = sinlfr(d);
      double u = fabs((q.x - c) / ulp(c));
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q.x, (double)c, d, (double)ulp(c));
	goto STOP_SIN2;
      }
    }

    for(d = -10000000;d < 10000000;d += 200.1) {
      double2 q = child_sincos(d);
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
      double start = u2d(d2u(M_PI_4 * i)-20);
      double end = u2d(d2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2d(d2u(d)+1)) {
	double2 q = child_sincos(d);
	long double c = sinlfr(d);
	double u = fabs((q.x - c) / ulp(c));
	max = fmax(max, u);
	if (u > 1000) {
	  fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q.x, (double)c, d, (double)ulp(c));
	  goto STOP_SIN2;
	}
      }
    }

  STOP_SIN2:

    fprintf(stderr, "sin in sincos_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double2 q = child_sincos(d);
      long double c = coslfr(d);
      double u = fabs((q.y - c) / ulp(c));
      max = fmax(max, u);
    }

    for(d = -10000000;d < 10000000;d += 200.1) {
      double2 q = child_sincos(d);
      long double c = coslfr(d);
      double u = fabs((q.y - c) / ulp(c));
      max = fmax(max, u);
    }

    int i;

    for(i=1;i<10000;i++) {
      double start = u2d(d2u(M_PI_4 * i)-20);
      double end = u2d(d2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2d(d2u(d)+1)) {
	double2 q = child_sincos(d);
	long double c = coslfr(d);
	double u = fabs((q.y - c) / ulp(c));
	max = fmax(max, u);
      }
    }

    fprintf(stderr, "cos in sincos_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_tan(d);
      long double c = tanlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = -10000000;d < 10000000;d += 200.1) {
      double q = child_tan(d);
      long double c = tanlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    int i;

    for(i=1;i<10000;i++) {
      double start = u2d(d2u(M_PI_4 * i)-20);
      double end = u2d(d2u(M_PI_4 * i)+20);

      for(d = start;d <= end;d = u2d(d2u(d)+1)) {
	double q = child_tan(d);
	long double c = tanlfr(d);
	double u = countULP(q, c);
	max = fmax(max, u);
      }
    }

    fprintf(stderr, "tan_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -1;d < 1;d += 0.00002) {
      double q = child_asin(d);
      long double c = asinlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ASIN;
      }
    }

  STOP_ASIN:

    fprintf(stderr, "asin_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -1;d < 1;d += 0.00002) {
      double q = child_acos(d);
      long double c = acoslfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ACOS;
      }
    }

  STOP_ACOS:

    fprintf(stderr, "acos_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_atan(d);
      long double c = atanlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %g, d = %g\n", q, d);
	goto STOP_ATAN;
      }
    }

    for(d = -10000;d < 10000;d += 0.2) {
      double q = child_atan(d);
      long double c = atanlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %g, d = %g\n", q, d);
	goto STOP_ATAN;
      }
    }

  STOP_ATAN:

    fprintf(stderr, "atan_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double x, y, max = 0;

    for(y = -10;y < 10;y += 0.05) {
      for(x = -10;x < 10;x += 0.05) {
	double q = child_atan2(y, x);
	long double c = atan2lfr(y, x);
	double u = countULP(q, c);
	max = fmax(max, u);
	if (u > 1000) {
	  fprintf(stderr, "q = %g, y = %g, x = %g\n", q, y, x);
	  goto STOP_ATAN2;
	}
      }
    }

    for(y = -100;y < 100;y += 0.51) {
      for(x = -100;x < 100;x += 0.51) {
	double q = child_atan2(y, x);
	long double c = atan2lfr(y, x);
	double u = countULP(q, c);
	max = fmax(max, u);
	if (u > 1000) {
	  fprintf(stderr, "q = %g, y = %g, x = %g\n", q, y, x);
	  goto STOP_ATAN2;
	}
      }
    }

  STOP_ATAN2:

    fprintf(stderr, "atan2_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = 0.0001;d < 10;d += 0.0001) {
      double q = child_log(d);
      long double c = loglfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = 0.0001;d < 10000;d += 0.1) {
      double q = child_log(d);
      long double c = loglfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    int i;
    for(i = -1000;i <= 1000;i++) {
      d = pow(2.1, i);
      double q = child_log(d);
      long double c = loglfr(d);
       double u = countULP(q, c);
      max = fmax(max, u);
    }

    fprintf(stderr, "log_u1 : %lf ... ", max);

    showResult(max < 1);
  }

#if 0
  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_exp(d);
      long double c = explfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = -1000;d < 1000;d += 0.02) {
      double q = child_exp(d);
      long double c = explfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    fprintf(stderr, "exp : %lf ... ", max);

    showResult(max < 1);
  }
#endif

#if 0
  {
    double x, y, max = 0;

    for(y = 0.1;y < 100;y += 0.2) {
      for(x = -100;x < 100;x += 0.2) {
	double q = child_pow(x, y);
	long double c = powlfr(x, y);
	double u = countULP(q, c);
	max = fmax(max, u);
	if (u > 1000) {
	  fprintf(stderr, "q = %g, x = %g, y = %g\n", q, x, y);
	  goto STOP_POW;
	}
      }
    }

  STOP_POW:

    fprintf(stderr, "pow : %lf ... ", max);

    showResult(max < 1);
  }

#if 0
  {
    double d, max = 0;

    for(d = 0;d < 20000;d += 0.2) {
      double q = child_sqrt(d);
      long double c = sqrtlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SQRT;
      }
    }

  STOP_SQRT:

    fprintf(stderr, "sqrt : %lf ... ", max);

    showResult(max < 1);
  }
#endif

  {
    double d, max = 0;

    for(d = -10000;d < 10000;d += 0.2) {
      double q = child_cbrt(d);
      long double c = cbrtlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_CBRT;
      }
    }

    int i;
    for(i = -1000;i <= 1000;i++) {
      d = pow(2.1, i);
      double q = child_cbrt(d);
      long double c = cbrtlfr(d);
       double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_CBRT;
      }
    }

  STOP_CBRT:

    fprintf(stderr, "cbrt_u1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_sinh(d);
      long double c = sinhlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SINH;
      }
    }

    for(d = -1000;d < 1000;d += 0.02) {
      double q = child_sinh(d);
      long double c = sinhlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_SINH;
      }
    }

  STOP_SINH:

    fprintf(stderr, "sinh : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_cosh(d);
      long double c = coshlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_COSH;
      }
    }

    for(d = -1000;d < 1000;d += 0.02) {
      double q = child_cosh(d);
      long double c = coshlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_COSH;
      }
    }

  STOP_COSH:

    fprintf(stderr, "cosh : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_tanh(d);
      long double c = tanhlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_TANH;
      }
    }

    for(d = -1000;d < 1000;d += 0.02) {
      double q = child_tanh(d);
      long double c = tanhlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_TANH;
      }
    }

  STOP_TANH:

    fprintf(stderr, "tanh : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_asinh(d);
      long double c = asinhlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ASINH;
      }
    }

    for(d = -1000;d < 1000;d += 0.02) {
      double q = child_asinh(d);
      long double c = asinhlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ASINH;
      }
    }

  STOP_ASINH:

    fprintf(stderr, "asinh : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = 1;d < 10;d += 0.0002) {
      double q = child_acosh(d);
      long double c = acoshlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ACOSH;
      }
    }

    for(d = 1;d < 1000;d += 0.02) {
      double q = child_acosh(d);
      long double c = acoshlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ACOSH;
      }
    }

  STOP_ACOSH:

    fprintf(stderr, "acosh : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_atanh(d);
      long double c = atanhlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ATANH;
      }
    }

    for(d = -1000;d < 1000;d += 0.02) {
      double q = child_atanh(d);
      long double c = atanhlfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
      if (u > 1000) {
	fprintf(stderr, "q = %.20g\nc = %.20g\nd = %.20g\nulp = %g\n", q, (double)c, d, (double)ulp(c));
	goto STOP_ATANH;
      }
    }

  STOP_ATANH:

    fprintf(stderr, "atanh : %lf ... ", max);

    showResult(max < 1);
  }
#endif

#if 0
  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_exp2(d);
      long double c = exp2lfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = -1000;d < 1000;d += 0.02) {
      double q = child_exp2(d);
      long double c = exp2lfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    fprintf(stderr, "exp2 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_exp10(d);
      long double c = exp10lfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = -300;d < 300;d += 0.01) {
      double q = child_exp10(d);
      long double c = exp10lfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    fprintf(stderr, "exp10 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = -10;d < 10;d += 0.0002) {
      double q = child_expm1(d);
      long double c = expm1lfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = -1000;d < 1000;d += 0.02) {
      double q = child_expm1(d);
      long double c = expm1lfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = 0;d < 300;d += 0.02) {
      double d2 = pow(10, -d);
      double q = child_expm1(d2);
      long double c = expm1lfr(d2);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = 0;d < 300;d += 0.02) {
      double d2 = -pow(10, -d);
      double q = child_expm1(d2);
      long double c = expm1lfr(d2);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    fprintf(stderr, "expm1 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = 0.0001;d < 10;d += 0.0001) {
      double q = child_log10(d);
      long double c = log10lfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = 0.0001;d < 10000;d += 0.1) {
      double q = child_log10(d);
      long double c = log10lfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    fprintf(stderr, "log10 : %lf ... ", max);

    showResult(max < 1);
  }

  {
    double d, max = 0;

    for(d = 0.0001;d < 10;d += 0.0001) {
      double q = child_log1p(d);
      long double c = log1plfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = 0.0001;d < 10000;d += 0.1) {
      double q = child_log1p(d);
      long double c = log1plfr(d);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = 0;d < 300;d += 0.02) {
      double d2 = pow(10, -d);
      double q = child_log1p(d2);
      long double c = log1plfr(d2);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    for(d = 0;d < 300;d += 0.02) {
      double d2 = -pow(10, -d);
      double q = child_log1p(d2);
      long double c = log1plfr(d2);
      double u = countULP(q, c);
      max = fmax(max, u);
    }

    fprintf(stderr, "log1p : %lf ... ", max);

    showResult(max < 1);
  }
#endif
}

int main(int argc, char **argv) {
  char *argv2[argc];
  int i;

  for(i=1;i<argc;i++) argv2[i-1] = argv[i];
  argv2[argc-1] = NULL;

  mpfr_set_default_prec(128);

  startChild(argv2[0], argv2);

  do_test();

  if (allTestsPassed) {
    fprintf(stderr, "\n\n*** All tests passed\n");
  } else {
    fprintf(stderr, "\n\n*** There were errors in some tests\n");
  }

  if (allTestsPassed) return 0;

  return -1;
}
