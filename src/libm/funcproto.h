//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

typedef struct {
  char *name;
  int ulp;
  int ulpSuffix;
  int funcType;
} funcSpec;

/*
  ulp : (error bound in ulp) * 10

  ulpSuffix:
  0 : ""
  1 : "_u1"
  2 : "_u05"
  3 : "_u35"
  4 : "_u15"

  funcType:
  0 : double func(double);
  1 : double func(double, double);
  2 : double2 func(double);   GNUABI : void func(double, double *, double *);
  3 : double func(double, int);
  4 : int func(double);
  5 : double func(double, double, double);
  6 : double2 func(double);   GNUABI : double func(double, double *);
 */

funcSpec funcList[] = {
  { "sin", 35, 0, 0 },
  { "cos", 35, 0, 0 },
  { "sincos", 35, 0, 2 },
  { "tan", 35, 0, 0 },
  { "asin", 35, 0, 0 },
  { "acos", 35, 0, 0 },
  { "atan", 35, 0, 0 },
  { "atan2", 35, 0, 1 },
  { "log", 35, 0, 0 },
  { "cbrt", 35, 0, 0 },
  { "sin", 10, 1, 0 },
  { "cos", 10, 1, 0 },
  { "sincos", 10, 1, 2 },
  { "tan", 10, 1, 0 },
  { "asin", 10, 1, 0 },
  { "acos", 10, 1, 0 },
  { "atan", 10, 1, 0 },
  { "atan2", 10, 1, 1 },
  { "log", 10, 1, 0 },
  { "cbrt", 10, 1, 0 },
  { "exp", 10, 0, 0 },
  { "pow", 10, 0, 1 },
  { "sinh", 10, 0, 0 },
  { "cosh", 10, 0, 0 },
  { "tanh", 10, 0, 0 },
  { "asinh", 10, 0, 0 },
  { "acosh", 10, 0, 0 },
  { "atanh", 10, 0, 0 },
  { "exp2", 10, 0, 0 },
  { "exp10", 10, 0, 0 },
  { "expm1", 10, 0, 0 },
  { "log10", 10, 0, 0 },
  { "log1p", 10, 0, 0 },
  { "sincospi", 5, 2, 2 },
  { "sincospi", 35, 3, 2 },
  { "sinpi", 5, 2, 0 },
  { "cospi", 5, 2, 0 },
  { "ldexp", -1, 0, 3 },
  { "ilogb", -1, 0, 4 },

  { "fma", -1, 0, 5 },
  { "sqrt", 5, 2, 0 },
  { "sqrt", 35, 3, 0 },
  { "hypot", 5, 2, 1 },
  { "hypot", 35, 3, 1 },
  { "fabs", -1, 0, 0 },
  { "copysign", -1, 0, 1 },
  { "fmax", -1, 0, 1 },
  { "fmin", -1, 0, 1 },
  { "fdim", -1, 0, 1 },
  { "trunc", -1, 0, 0 },
  { "floor", -1, 0, 0 },
  { "ceil", -1, 0, 0 },
  { "round", -1, 0, 0 },
  { "rint", -1, 0, 0 },
  { "nextafter", -1, 0, 1 },
  { "frfrexp", -1, 0, 0 },
  { "expfrexp", -1, 0, 4 },
  { "fmod", -1, 0, 1 },
  { "modf", -1, 0, 6 },

  { "lgamma", 10, 1, 0 },
  { "tgamma", 10, 1, 0 },
  { "erf", 10, 1, 0 },
  { "erfc", 15, 4, 0 },
  
  { NULL, -1, 0, 0 },
};
