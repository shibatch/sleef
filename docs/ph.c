// Explanatory source code for the modified Payne Hanek reduction
// http://dx.doi.org/10.1109/TPDS.2019.2960333

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpfr.h>

typedef struct { double x, y; } double2;
double2 dd(double d) { double2 r = { d, 0 }; return r; }
int64_t d2i(double d) { union { double f; int64_t i; } tmp = {.f = d }; return tmp.i; }
double i2d(int64_t i) { union { double f; int64_t i; } tmp = {.i = i }; return tmp.f; }
double upper(double d) { return i2d(d2i(d) & 0xfffffffff8000000LL); }
double clearlsb(double d) { return i2d(d2i(d) & 0xfffffffffffffffeLL); }

double2 ddrenormalize(double2 t) {
  double2 s = dd(t.x + t.y);
  s.y = t.x - s.x + t.y;
  return s;
}

double2 ddadd(double2 x, double2 y) {
  double2 r = dd(x.x + y.x);
  double v = r.x - x.x;
  r.y = (x.x - (r.x - v)) + (y.x - v) + (x.y + y.y);
  return r;
}

double2 ddmul(double x, double y) {
  double2 r = dd(x * y);
  r.y = fma(x, y, -r.x);
  return r;
}

double2 ddmul2(double2 x, double2 y) {
  double2 r = ddmul(x.x, y.x);
  r.y += x.x * y.y + x.y * y.x;
  return r;
}

// This function computes remainder(a, PI/2)
double2 modifiedPayneHanek(double a) {
  double table[4];
  int scale = fabs(a) > 1e+200 ? -128 : 0;
  a = ldexp(a, scale);

  // Table genration

  mpfr_set_default_prec(2048);
  mpfr_t pi, m;
  mpfr_inits(pi, m, NULL);
  mpfr_const_pi(pi, GMP_RNDN);

  mpfr_d_div(m, 2, pi, GMP_RNDN);
  mpfr_set_exp(m, mpfr_get_exp(m) + (ilogb(a) - 53 - scale));
  mpfr_frac(m, m, GMP_RNDN);
  mpfr_set_exp(m, mpfr_get_exp(m) - (ilogb(a) - 53));

  for(int i=0;i<4;i++) {
    table[i] = clearlsb(mpfr_get_d(m, GMP_RNDN));
    mpfr_sub_d(m, m, table[i], GMP_RNDN);
  }

  mpfr_clears(pi, m, NULL);

  // Main computation

  double2 x = dd(0);
  for(int i=0;i<4;i++) {
    x = ddadd(x, ddmul(a, table[i]));
    x.x = x.x - round(x.x);
    x = ddrenormalize(x);
  }

  double2 pio2 = { 3.141592653589793*0.5, 1.2246467991473532e-16*0.5 };
  x = ddmul2(x, pio2);
  return fabs(a) < 0.785398163397448279 ? dd(a) : x;
}

int main(int argc, char **argv) {
  double a = ldexp(6381956970095103.0, 797);
  if (argc > 1) a = atof(argv[1]);
  printf("a = %.20g\n", a);

  //

  mpfr_set_default_prec(2048);
  mpfr_t pi, pio2, x, y, r;
  mpfr_inits(pi, pio2, x, y, r, NULL);

  mpfr_const_pi(pi, GMP_RNDN);
  mpfr_mul_d(pio2, pi, 0.5, GMP_RNDN);

  //

  mpfr_set_d(x, a, GMP_RNDN);
  mpfr_remainder(r, x, pio2, GMP_RNDN);

  mpfr_printf("mpfr = %.64RNf\n", r);

  //

  double2 dd = modifiedPayneHanek(a);
  mpfr_set_d(x, dd.x, GMP_RNDN);
  mpfr_add_d(x, x, dd.y, GMP_RNDN);

  mpfr_printf("dd   = %.64RNf\n", x);

  mpfr_sub(x, x, r, GMP_RNDN);
  mpfr_abs(x, x, GMP_RNDN);
  mpfr_div(x, x, r, GMP_RNDN);

  double err = mpfr_get_d(x, GMP_RNDN);
  printf("error  = %g\n", err);
}
