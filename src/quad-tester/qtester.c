//          Copyright Naoki Shibata 2010 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// This define is needed to prevent the `execvpe` function to raise a
// warning at compile time. For more information, see
// https://linux.die.net/man/3/execvp.
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <errno.h>
#include <inttypes.h>

#if defined(POWER64_UNDEF_USE_EXTERN_INLINES)
// This is a workaround required to cross compile for PPC64 binaries
#include <features.h>
#ifdef __USE_EXTERN_INLINES
#undef __USE_EXTERN_INLINES
#endif
#endif

#include <math.h>
#include <mpfr.h>

#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <signal.h>

#include "misc.h"
#include "qtesterutil.h"

void stop(char *mes) {
  fprintf(stderr, "%s\n", mes);
  exit(-1);
}

int ptoc[2], ctop[2];
int pid;
FILE *fpctop;

extern char **environ;

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

    fflush(stdin);
    fflush(stdout);
    
    i = dup2(ptoc[0], fileno(stdin));
    assert(i != -1);

    i = dup2(ctop[1], fileno(stdout));
    assert(i != -1);

    setvbuf(stdin, buf0, _IONBF,0);
    setvbuf(stdout, buf1, _IONBF,0);

    fflush(stdin);
    fflush(stdout);

#if !defined(__APPLE__) && !defined(__FreeBSD__)
    execvpe(path, argv, environ);
#else
    execvp(path, argv);
#endif

    fprintf(stderr, "execvp in startChild : %s\n", strerror(errno));

    exit(-1);
  }

  // parent process

  close(ptoc[0]);
  close(ctop[1]);
}

//

typedef union {
  Sleef_quad q;
  struct {
    uint64_t l, h;
  };
} cnv128;

#define child_q_q(funcStr, arg) do {					\
    char str[256];							\
    cnv128 c;								\
    c.q = arg;								\
    sprintf(str, funcStr " %" PRIx64 ":%" PRIx64 "\n", c.h, c.l);	\
    write(ptoc[1], str, strlen(str));					\
    if (fgets(str, 255, fpctop) == NULL) stop("child " funcStr);	\
    sscanf(str, "%" PRIx64 ":%" PRIx64, &c.h, &c.l);			\
    return c.q;								\
  } while(0)

#define child_q2_q(funcStr, arg) do {					\
    char str[256];							\
    cnv128 c0, c1;							\
    c0.q = arg;								\
    sprintf(str, funcStr " %" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l);	\
    write(ptoc[1], str, strlen(str));					\
    if (fgets(str, 255, fpctop) == NULL) stop("child " funcStr);	\
    sscanf(str, "%" PRIx64 ":%" PRIx64 " %" PRIx64 ":%" PRIx64 , &c0.h, &c0.l, &c1.h, &c1.l); \
    Sleef_quad2 ret = { c0.q, c1.q };					\
    return ret;								\
  } while(0)

#define child_q_q_q(funcStr, arg0, arg1) do {				\
    char str[256];							\
    cnv128 c0, c1;							\
    c0.q = arg0;							\
    c1.q = arg1;							\
    sprintf(str, funcStr " %" PRIx64 ":%" PRIx64 " %" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l, c1.h, c1.l); \
    write(ptoc[1], str, strlen(str));					\
    if (fgets(str, 255, fpctop) == NULL) stop("child " funcStr);	\
    sscanf(str, "%" PRIx64 ":%" PRIx64, &c0.h, &c0.l);			\
    return c0.q;							\
  } while(0)

#define child_i_q_q(funcStr, arg0, arg1) do {				\
    char str[256];							\
    cnv128 c0, c1;							\
    c0.q = arg0;							\
    c1.q = arg1;							\
    sprintf(str, funcStr " %" PRIx64 ":%" PRIx64 " %" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l, c1.h, c1.l); \
    write(ptoc[1], str, strlen(str));					\
    if (fgets(str, 255, fpctop) == NULL) stop("child " funcStr);	\
    int i;								\
    sscanf(str, "%d", &i);						\
    return i;								\
  } while(0)

#define child_d_q(funcStr, arg) do {					\
    char str[256];							\
    cnv128 c;								\
    c.q = arg;								\
    sprintf(str, funcStr " %" PRIx64 ":%" PRIx64 "\n", c.h, c.l);	\
    write(ptoc[1], str, strlen(str));					\
    if (fgets(str, 255, fpctop) == NULL) stop("child " funcStr);	\
    uint64_t u;								\
    sscanf(str, "%" PRIx64, &u);					\
    return u2d(u);							\
  } while(0)

#define child_q_d(funcStr, arg) do {					\
    char str[256];							\
    uint64_t u;								\
    sprintf(str, funcStr " %" PRIx64 "\n", d2u(arg));			\
    write(ptoc[1], str, strlen(str));					\
    if (fgets(str, 255, fpctop) == NULL) stop("child " funcStr);	\
    cnv128 c;								\
    sscanf(str, "%" PRIx64 ":%" PRIx64, &c.h, &c.l);			\
    return c.q;								\
  } while(0)

#define child_q_str(funcStr, arg) do {					\
    char str[256];							\
    sprintf(str, funcStr " %s\n", arg);					\
    write(ptoc[1], str, strlen(str));					\
    if (fgets(str, 255, fpctop) == NULL) stop("child " funcStr);	\
    cnv128 c;								\
    sscanf(str, "%" PRIx64 ":%" PRIx64, &c.h, &c.l);			\
    return c.q;								\
  } while(0)

#define child_str_q(funcStr, ret, arg) do {				\
    char str[256];							\
    cnv128 c;								\
    c.q = arg;								\
    sprintf(str, funcStr " %" PRIx64 ":%" PRIx64 "\n", c.h, c.l);	\
    write(ptoc[1], str, strlen(str));					\
    if (fgets(str, 255, fpctop) == NULL) stop("child " funcStr);	\
    sscanf(str, "%63s", ret);						\
  } while(0)

Sleef_quad child_addq_u05(Sleef_quad x, Sleef_quad y) { child_q_q_q("addq_u05", x, y); }
Sleef_quad child_subq_u05(Sleef_quad x, Sleef_quad y) { child_q_q_q("subq_u05", x, y); }
Sleef_quad child_mulq_u05(Sleef_quad x, Sleef_quad y) { child_q_q_q("mulq_u05", x, y); }
Sleef_quad child_divq_u05(Sleef_quad x, Sleef_quad y) { child_q_q_q("divq_u05", x, y); }
Sleef_quad child_negq(Sleef_quad x) { child_q_q("negq", x); }

int child_cmpltq(Sleef_quad x, Sleef_quad y) { child_i_q_q("cmpltq", x, y); }
int child_cmpgtq(Sleef_quad x, Sleef_quad y) { child_i_q_q("cmpgtq", x, y); }
int child_cmpleq(Sleef_quad x, Sleef_quad y) { child_i_q_q("cmpleq", x, y); }
int child_cmpgeq(Sleef_quad x, Sleef_quad y) { child_i_q_q("cmpgeq", x, y); }
int child_cmpeqq(Sleef_quad x, Sleef_quad y) { child_i_q_q("cmpeqq", x, y); }
int child_cmpneqq(Sleef_quad x, Sleef_quad y) { child_i_q_q("cmpneqq", x, y); }
int child_unordq(Sleef_quad x, Sleef_quad y) { child_i_q_q("unordq", x, y); }

Sleef_quad child_cast_from_doubleq(double x) { child_q_d("cast_from_doubleq", x); }
double child_cast_to_doubleq(Sleef_quad x) { child_d_q("cast_to_doubleq", x); }

Sleef_quad child_strtoq(const char *s) { child_q_str("strtoq", s); }
void child_qtostr(char *ret, Sleef_quad x) { child_str_q("qtostr", ret, x); }

Sleef_quad child_sqrtq_u05(Sleef_quad x) { child_q_q("sqrtq_u05", x); }
Sleef_quad child_sinq_u10(Sleef_quad x) { child_q_q("sinq_u10", x); }
Sleef_quad child_cosq_u10(Sleef_quad x) { child_q_q("cosq_u10", x); }
Sleef_quad child_tanq_u10(Sleef_quad x) { child_q_q("tanq_u10", x); }
Sleef_quad child_asinq_u10(Sleef_quad x) { child_q_q("asinq_u10", x); }
Sleef_quad child_acosq_u10(Sleef_quad x) { child_q_q("acosq_u10", x); }
Sleef_quad child_atanq_u10(Sleef_quad x) { child_q_q("atanq_u10", x); }
Sleef_quad child_expq_u10(Sleef_quad x) { child_q_q("expq_u10", x); }
Sleef_quad child_exp2q_u10(Sleef_quad x) { child_q_q("exp2q_u10", x); }
Sleef_quad child_exp10q_u10(Sleef_quad x) { child_q_q("exp10q_u10", x); }
Sleef_quad child_expm1q_u10(Sleef_quad x) { child_q_q("expm1q_u10", x); }
Sleef_quad child_logq_u10(Sleef_quad x) { child_q_q("logq_u10", x); }
Sleef_quad child_log2q_u10(Sleef_quad x) { child_q_q("log2q_u10", x); }
Sleef_quad child_log10q_u10(Sleef_quad x) { child_q_q("log10q_u10", x); }
Sleef_quad child_log1pq_u10(Sleef_quad x) { child_q_q("log1pq_u10", x); }

Sleef_quad child_copysignq(Sleef_quad x, Sleef_quad y) { child_q_q_q("copysignq", x, y); }
Sleef_quad child_fabsq(Sleef_quad x) { child_q_q("fabsq", x); }
Sleef_quad child_fmaxq(Sleef_quad x, Sleef_quad y) { child_q_q_q("fmaxq", x, y); }
Sleef_quad child_fminq(Sleef_quad x, Sleef_quad y) { child_q_q_q("fminq", x, y); }

//

#define cmpDenorm_q(mpfrFunc, childFunc, argx) do {			\
    mpfr_set_f128(frx, argx, GMP_RNDN);					\
    mpfrFunc(frz, frx, GMP_RNDN);					\
    Sleef_quad t = childFunc(argx);					\
    double u = countULPf128(t, frz, 1);					\
    if (u >= 10) {							\
      fprintf(stderr, "\narg     = %s\ntest    = %s\ncorrect = %s\nulp = %g\n",	\
	      sprintf128(argx), sprintf128(t), sprintfr(frz), u);	\
      success = 0;							\
      break;								\
    }									\
  } while(0)

#define cmpDenorm_q_q(mpfrFunc, childFunc, argx, argy) do {		\
    mpfr_set_f128(frx, argx, GMP_RNDN);					\
    mpfr_set_f128(fry, argy, GMP_RNDN);					\
    mpfrFunc(frz, frx, fry, GMP_RNDN);					\
    Sleef_quad t = childFunc(argx, argy);				\
    double u = countULPf128(t, frz, 1);					\
    if (u >= 10) {							\
      Sleef_quad qz = mpfr_get_f128(frz, GMP_RNDN);			\
      fprintf(stderr, "\narg     = %s,\n          %s\ntest    = %s\ncorrect = %s\nulp = %g\n", \
	      sprintf128(argx), sprintf128(argy), sprintf128(t), sprintf128(qz), u); \
      success = 0;							\
      break;								\
    }									\
  } while(0)

#define checkAccuracy_q(mpfrFunc, childFunc, argx, bound) do {		\
    mpfr_set_f128(frx, argx, GMP_RNDN);					\
    mpfrFunc(frz, frx, GMP_RNDN);					\
    Sleef_quad t = childFunc(argx);					\
    double e = countULPf128(t, frz, 0);					\
    maxError = fmax(maxError, e);					\
    if (e > bound) {							\
      fprintf(stderr, "\narg = %s, test = %s, correct = %s, ULP = %lf\n", \
	      sprintf128(argx), sprintf128(childFunc(argx)), sprintfr(frz), countULPf128(t, frz, 0)); \
      success = 0;							\
      break;								\
    }									\
  } while(0)

#define checkAccuracy_q_q(mpfrFunc, childFunc, argx, argy, bound) do {	\
    mpfr_set_f128(frx, argx, GMP_RNDN);					\
    mpfr_set_f128(fry, argy, GMP_RNDN);					\
    mpfrFunc(frz, frx, fry, GMP_RNDN);					\
    Sleef_quad t = childFunc(argx, argy);				\
    double e = countULPf128(t, frz, 0);					\
    maxError = fmax(maxError, e);					\
    if (e > bound) {							\
      fprintf(stderr, "\narg = %s, %s, test = %s, correct = %s, ULP = %lf\n", \
	      sprintf128(argx), sprintf128(argy), sprintf128(childFunc(argx, argy)), sprintfr(frz), countULPf128(t, frz, 0)); \
      success = 0;							\
      break;								\
    }									\
  } while(0)

#define testComparison(mpfrFunc, childFunc, argx, argy) do {		\
    mpfr_set_f128(frx, argx, GMP_RNDN);					\
    mpfr_set_f128(fry, argy, GMP_RNDN);					\
    int c = mpfrFunc(frx, fry);						\
    int t = childFunc(argx, argy);					\
    if ((c != 0) != (t != 0)) {						\
      fprintf(stderr, "\narg = %s, %s, test = %d, correct = %d\n",	\
	      sprintf128(argx), sprintf128(argy), t, c);		\
      success = 0;							\
      break;								\
    }									\
  } while(0)

//

#define cmpDenormOuterLoop_q_q(mpfrFunc, childFunc, checkVals) do {	\
    for(int i=0;i<sizeof(checkVals)/sizeof(char *);i++) {		\
      Sleef_quad a0 = cast_q_str(checkVals[i]);				\
      for(int j=0;j<sizeof(checkVals)/sizeof(char *) && success;j++) {	\
	Sleef_quad a1 = cast_q_str(checkVals[j]);			\
	cmpDenorm_q_q(mpfrFunc, childFunc, a0, a1);			\
      }									\
    }									\
  } while(0)

#define cmpDenormOuterLoop_q(mpfrFunc, childFunc, checkVals) do {	\
    for(int i=0;i<sizeof(checkVals)/sizeof(char *);i++) {		\
      Sleef_quad a0 = cast_q_str(checkVals[i]);				\
      cmpDenorm_q(mpfrFunc, childFunc, a0);				\
    }									\
  } while(0)

//

#define checkAccuracyOuterLoop_q_q(mpfrFunc, childFunc, minStr, maxStr, nLoop, bound, seed) do { \
    xsrand(seed);							\
    Sleef_quad min = cast_q_str(minStr), max = cast_q_str(maxStr);	\
    for(int i=0;i<nLoop && success;i++) {				\
      Sleef_quad x = rndf128(min, max), y = rndf128(min, max);		\
      checkAccuracy_q_q(mpfrFunc, childFunc, x, y, bound);		\
    }									\
  } while(0)

#define checkAccuracyOuterLoop_q(mpfrFunc, childFunc, minStr, maxStr, nLoop, bound, seed) do { \
    xsrand(seed);							\
    Sleef_quad min = cast_q_str(minStr), max = cast_q_str(maxStr);	\
    for(int i=0;i<nLoop && success;i++) {				\
      Sleef_quad x = rndf128(min, max);					\
      checkAccuracy_q(mpfrFunc, childFunc, x, bound);			\
    }									\
  } while(0)

#define testComparisonOuterLoop(mpfrFunc, childFunc, checkVals) do {	\
    for(int i=0;i<sizeof(checkVals)/sizeof(char *);i++) {		\
      Sleef_quad a0 = cast_q_str(checkVals[i]);				\
      for(int j=0;j<sizeof(checkVals)/sizeof(char *) && success;j++) {	\
	Sleef_quad a1 = cast_q_str(checkVals[j]);			\
	testComparison(mpfrFunc, childFunc, a0, a1);			\
      }									\
    }									\
  } while(0)

//

void checkResult(int success, double e) {
  if (!success) {
    fprintf(stderr, "\n\n*** Test failed\n");
    exit(-1);
  }

  if (e != -1) {
    fprintf(stderr, "OK (%g ulp)\n", e);
  } else {
    fprintf(stderr, "OK\n");
  }
}

#define STR_QUAD_MIN "3.36210314311209350626267781732175260e-4932"
#define STR_QUAD_MAX "1.18973149535723176508575932662800702e+4932"
#define STR_QUAD_DENORM_MIN "6.475175119438025110924438958227646552e-4966"

void do_test(int options) {
  mpfr_set_default_prec(256);
  mpfr_t frx, fry, frz;
  mpfr_inits(frx, fry, frz, NULL);

  int success = 1;

  static const char *stdCheckVals[] = {
    "0.0", "-0.0", "+0.5", "-0.5", "+1.0", "-1.0", "+1.5", "-1.5", "+2.0", "-2.0", "+2.5", "-2.5",
    "1.234", "-1.234", "+1.234e+100", "-1.234e+100", "+1.234e-100", "-1.234e-100",
    "+1.234e+3000", "-1.234e+3000", "+1.234e-3000", "-1.234e-3000",
    "3.1415926535897932384626433832795028841971693993751058209749445923078164",
    "+" STR_QUAD_MIN, "-" STR_QUAD_MIN,
    "+" STR_QUAD_DENORM_MIN, "-" STR_QUAD_DENORM_MIN,
    "NaN", "Inf", "-Inf"
  };

#define NTEST 1000

  double errorBound = 0.5000000001;
  double maxError;

  fprintf(stderr, "addq_u05 : ");
  maxError = 0;
  cmpDenormOuterLoop_q_q(mpfr_add, child_addq_u05, stdCheckVals);
  checkAccuracyOuterLoop_q_q(mpfr_add, child_addq_u05, "-1e-100", "-1e+100", 5 * NTEST, errorBound, 0);
  checkAccuracyOuterLoop_q_q(mpfr_add, child_addq_u05, "0", "Inf", 5 * NTEST, errorBound, 1);
  checkResult(success, maxError);

  fprintf(stderr, "subq_u05 : ");
  maxError = 0;
  cmpDenormOuterLoop_q_q(mpfr_sub, child_subq_u05, stdCheckVals);
  checkAccuracyOuterLoop_q_q(mpfr_sub, child_subq_u05, "-1e-100", "-1e+100", 5 * NTEST, errorBound, 0);
  checkAccuracyOuterLoop_q_q(mpfr_sub, child_subq_u05, "0", "Inf", 5 * NTEST, errorBound, 1);
  checkResult(success, maxError);

  fprintf(stderr, "mulq_u05 : ");
  maxError = 0;
  cmpDenormOuterLoop_q_q(mpfr_mul, child_mulq_u05, stdCheckVals);
  checkAccuracyOuterLoop_q_q(mpfr_mul, child_mulq_u05, "-1e-100", "-1e+100", 5 * NTEST, errorBound, 0);
  checkAccuracyOuterLoop_q_q(mpfr_mul, child_mulq_u05, "0", "Inf", 5 * NTEST, errorBound, 1);
  checkResult(success, maxError);

  fprintf(stderr, "divq_u05 : ");
  maxError = 0;
  cmpDenormOuterLoop_q_q(mpfr_div, child_divq_u05, stdCheckVals);
  checkAccuracyOuterLoop_q_q(mpfr_div, child_divq_u05, "-1e-100", "-1e+100", 5 * NTEST, errorBound, 0);
  checkAccuracyOuterLoop_q_q(mpfr_div, child_divq_u05, "0", "Inf", 5 * NTEST, errorBound, 1);
  checkResult(success, maxError);

  fprintf(stderr, "negq : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_neg, child_negq, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_neg, child_negq, "-1e-100", "-1e+100", 5 * NTEST, errorBound, 0);
  checkAccuracyOuterLoop_q(mpfr_neg, child_negq, "0", "Inf", 5 * NTEST, errorBound, 1);
  checkResult(success, maxError);

  //

  fprintf(stderr, "cmpltq : ");
  testComparisonOuterLoop(mpfr_less_p, child_cmpltq, stdCheckVals);
  checkResult(success, -1);

  fprintf(stderr, "cmpgtq : ");
  testComparisonOuterLoop(mpfr_greater_p, child_cmpgtq, stdCheckVals);
  checkResult(success, -1);

  fprintf(stderr, "cmpleq : ");
  testComparisonOuterLoop(mpfr_lessequal_p, child_cmpleq, stdCheckVals);
  checkResult(success, -1);

  fprintf(stderr, "cmpgeq : ");
  testComparisonOuterLoop(mpfr_greaterequal_p, child_cmpgeq, stdCheckVals);
  checkResult(success, -1);

  fprintf(stderr, "cmpeq : ");
  testComparisonOuterLoop(mpfr_equal_p, child_cmpeqq, stdCheckVals);
  checkResult(success, -1);

  fprintf(stderr, "cmpneq : ");
  testComparisonOuterLoop(mpfr_lessgreater_p, child_cmpneqq, stdCheckVals);
  checkResult(success, -1);

  fprintf(stderr, "unordq : ");
  testComparisonOuterLoop(mpfr_unordered_p, child_unordq, stdCheckVals);
  checkResult(success, -1);

  //

  fprintf(stderr, "cast_from_doubleq : ");
  {
    xsrand(0);
    for(int i=0;i<10 * NTEST;i++) {
      double d;
      switch(i) {
	case 0: d = +0.0; break;
	case 1: d = -0.0; break;
	case 2: d = +SLEEF_INFINITY; break;
	case 3: d = -SLEEF_INFINITY; break;
	case 4: d = SLEEF_NAN; break;
	default : memrand(&d, sizeof(d));
      }
      Sleef_quad qt = child_cast_from_doubleq(d);
      mpfr_set_d(frz, d, GMP_RNDN);
      Sleef_quad qc = mpfr_get_f128(frz, GMP_RNDN);
      if (memcmp(&qt, &qc, sizeof(Sleef_quad)) == 0) continue;
      if (isnanf128(qt) && isnanf128(qc)) continue;
      fprintf(stderr, "\narg     = %.20g\ntest    = %s\ncorrect = %s\n",
	      d, sprintf128(qt), sprintf128(qc));
      success = 0;
      break;
    }
    checkResult(success, -1);
  }

  fprintf(stderr, "cast_to_doubleq : ");
  {
    xsrand(0);
    Sleef_quad min = cast_q_str("0"), max = cast_q_str("Inf");
    for(int i=0;i<10 * NTEST;i++) {
      Sleef_quad x;
      if (i < sizeof(stdCheckVals)/sizeof(char *)) {
	x = cast_q_str(stdCheckVals[i]);
      } else {
	x = rndf128(min, max);
      }
      double dt = child_cast_to_doubleq(x);
      mpfr_set_f128(frz, x, GMP_RNDN);
      double dc = mpfr_get_d(frz, GMP_RNDN);
      if (dt == dc) continue;
      if (isnan(dt) && isnan(dc)) continue;
      fprintf(stderr, "\narg     = %s\ntest    = %.20g\ncorrect = %.20g\n",
	      sprintf128(x), dt, dc);
      success = 0;
      break;
    }
    checkResult(success, -1);
  }

  //

  fprintf(stderr, "sqrtq_u05 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_sqrt, child_sqrtq_u05, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_sqrt, child_sqrtq_u05, "1e-100", "1e+100", 5 * NTEST, errorBound, 0);
  checkAccuracyOuterLoop_q(mpfr_sqrt, child_sqrtq_u05, "0", "Inf", 5 * NTEST, errorBound, 1);
  checkResult(success, maxError);

  fprintf(stderr, "sinq_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_sin, child_sinq_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_sin, child_sinq_u10, "-1e-100", "-1e+100", 1 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_sin, child_sinq_u10, "-0", "-Inf", 1 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "cosq_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_cos, child_cosq_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_cos, child_cosq_u10, "-1e-100", "-1e+100", 1 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_cos, child_cosq_u10, "-0", "-Inf", 1 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "tanq_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_tan, child_tanq_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_tan, child_tanq_u10, "-1e-100", "-1e+100", 1 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_tan, child_tanq_u10, "-0", "-Inf", 1 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "asinq_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_asin, child_asinq_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_asin, child_asinq_u10, "-1e-100", "-1", 5 * NTEST, 1.0, 0);
  checkResult(success, maxError);

  fprintf(stderr, "acosq_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_acos, child_acosq_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_acos, child_acosq_u10, "-1e-100", "-1", 5 * NTEST, 1.0, 0);
  checkResult(success, maxError);

  fprintf(stderr, "atanq_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_atan, child_atanq_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_atan, child_atanq_u10, "-1e-100", "-1e+100", 3 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_atan, child_atanq_u10, "-0", "-Inf", 3 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "expq_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_exp, child_expq_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_exp, child_expq_u10, "-1e-100", "-1e+100", 3 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_exp, child_expq_u10, "-0", "-Inf", 3 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "exp2q_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_exp2, child_exp2q_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_exp2, child_exp2q_u10, "-1e-100", "-1e+100", 3 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_exp2, child_exp2q_u10, "-0", "-Inf", 3 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "exp10q_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_exp10, child_exp10q_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_exp10, child_exp10q_u10, "-1e-100", "-1e+100", 3 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_exp10, child_exp10q_u10, "-0", "-Inf", 3 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "expm1q_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_expm1, child_expm1q_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_expm1, child_expm1q_u10, "-1e-100", "-1e+100", 3 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_expm1, child_expm1q_u10, "-0", "-Inf", 3 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "logq_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_log, child_logq_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_log, child_logq_u10, "1e-100", "1e+100", 3 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_log, child_logq_u10, "0", "Inf", 3 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "log2q_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_log2, child_log2q_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_log2, child_log2q_u10, "1e-100", "1e+100", 3 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_log2, child_log2q_u10, "0", "Inf", 3 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "log10q_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_log10, child_log10q_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_log10, child_log10q_u10, "1e-100", "1e+100", 3 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_log10, child_log10q_u10, "0", "Inf", 3 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  fprintf(stderr, "log1pq_u10 : ");
  maxError = 0;
  cmpDenormOuterLoop_q(mpfr_log1p, child_log1pq_u10, stdCheckVals);
  checkAccuracyOuterLoop_q(mpfr_log1p, child_log1pq_u10, "1e-100", "1e+100", 3 * NTEST, 1.0, 0);
  checkAccuracyOuterLoop_q(mpfr_log1p, child_log1pq_u10, "0", "Inf", 3 * NTEST, 1.0, 1);
  checkResult(success, maxError);

  if ((options & 2) != 0) {
    fprintf(stderr, "strtoq : ");
    for(int i=0;i<sizeof(stdCheckVals)/sizeof(char *);i++) {
      Sleef_quad a0 = cast_q_str(stdCheckVals[i]);
      Sleef_quad a1 = child_strtoq(stdCheckVals[i]);
      if (memcmp(&a0, &a1, sizeof(Sleef_quad)) == 0) continue;
      if (isnanf128(a0) && isnanf128(a1)) continue;

      fprintf(stderr, "\narg     = %s\ntest    = %s\ncorrect = %s\n",
	      stdCheckVals[i], sprintf128(a1), sprintf128(a0));
      success = 0;
      break;
    }
    checkResult(success, maxError);

    fprintf(stderr, "qtostr : ");
    for(int i=0;i<sizeof(stdCheckVals)/sizeof(char *);i++) {
      Sleef_quad a0 = cast_q_str(stdCheckVals[i]);
      char s[100];
      child_qtostr(s, a0);
      Sleef_quad a1 = cast_q_str(s);
      if (memcmp(&a0, &a1, sizeof(Sleef_quad)) == 0) continue;
      if (isnanf128(a0) && isnanf128(a1)) continue;

      fprintf(stderr, "\narg     = %s\nteststr = %s\ntest    = %s\ncorrect = %s\n",
	      stdCheckVals[i], s, sprintf128(a0), sprintf128(a1));
      success = 0;
      break;
    }
    checkResult(success, maxError);
  }
}

int main(int argc, char **argv) {
  char *argv2[argc+2], *commandSde = NULL;
  int i, a2s, options;

  // BUGFIX: this flush is to prevent incorrect syncing with the
  // `iut*` executable that causes failures in the CPU detection on
  // some CI systems.
  fflush(stdout);

  for(a2s=1;a2s<argc;a2s++) {
    if (a2s+1 < argc && strcmp(argv[a2s], "--sde") == 0) {
      commandSde = argv[a2s+1];
      a2s++;
    } else {
      break;
    }
  }

  for(i=a2s;i<argc;i++) argv2[i-a2s] = argv[i];
  argv2[argc-a2s] = NULL;
  
  startChild(argv2[0], argv2);
  fflush(stdin);
  // BUGFIX: this flush is to prevent incorrect syncing with the
  // `iut*` executable that causes failures in the CPU detection on
  // some CI systems.
  fflush(stdin);

  {
    char str[256];

    if (readln(ctop[0], str, 255) < 1 ||
	sscanf(str, "%d", &options) != 1 ||
	(options & 1) == 0) {
      if (commandSde != NULL) {
	close(ctop[0]);
	close(ptoc[1]);

	argv2[0] = commandSde;
	argv2[1] = "--";
	for(i=a2s;i<argc;i++) argv2[i-a2s+2] = argv[i];
	argv2[argc-a2s+2] = NULL;
	
	startChild(argv2[0], argv2);

	if (readln(ctop[0], str, 255) < 1) stop("Feature detection(sde, readln)");
	if (sscanf(str, "%d", &options) != 1) stop("Feature detection(sde, sscanf)");
	if ((options & 1) == 0) {
	  fprintf(stderr, "\n\nTester : *** CPU does not support the necessary feature(SDE)\n");
	  return 0;
	}

	fprintf(stderr, "*** Using SDE\n");
      } else {
	fprintf(stderr, "\n\nTester : *** CPU does not support the necessary feature\n");
	return 0;
      }
    }
  }

  fprintf(stderr, "\n\n*** qtester : now testing %s\n", argv2[0]);

  fpctop = fdopen(ctop[0], "r");
  
  do_test(options);

  fprintf(stderr, "\n\n*** All tests passed\n");

  exit(0);
}
