//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

typedef struct {
  char *name;
  int ulp;
  int ulpSuffix;
  int funcType;
  int flags;
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
  0 : vdouble func(vdouble);
  1 : vdouble func(vdouble, vdouble);
  2 : vdouble2 func(vdouble);   GNUABI : void func(vdouble, double *, double *);
  3 : vdouble func(vdouble, vint);
  4 : vint func(vdouble);
  5 : vdouble func(vdouble, vdouble, vdouble);
  6 : vdouble2 func(vdouble);   GNUABI : vdouble func(vdouble, double *);
  7 : int func(int);
  8 : void *func(int);

  flags:
  (1 << 0) : No GNUABI
  (1 << 1) : Corresponding function is common between deterministic and non-deterministic versions
  (1 << 2) : Deterministic implementation with "y" prefix
 */

#define NOGNUABI (1 << 0)
#define DET_COMMON (1 << 1)
#define DET (1 << 2)

funcSpec funcList[] = {
  { "sin", 35, 0, 0, 0 },
  { "cos", 35, 0, 0, 0 },
  { "sincos", 35, 0, 2, 0 },
  { "tan", 35, 0, 0, 0 },
  { "asin", 35, 0, 0, 0 },
  { "asin", 35, 0, 0, DET },
  { "acos", 35, 0, 0, 0 },
  { "atan", 35, 0, 0, 0 },
  { "atan2", 35, 0, 1, DET_COMMON },
  { "log", 35, 0, 0, DET_COMMON },
  { "cbrt", 35, 0, 0, DET_COMMON },
  { "sin", 10, 1, 0, 0 },
  { "cos", 10, 1, 0, 0 },
  { "sincos", 10, 1, 2, 0 },
  { "tan", 10, 1, 0, 0 },
  { "asin", 10, 1, 0, 0 },
  { "acos", 10, 1, 0, DET_COMMON },
  { "atan", 10, 1, 0, DET_COMMON },
  { "atan2", 10, 1, 1, DET_COMMON },
  { "log", 10, 1, 0, DET_COMMON },
  { "cbrt", 10, 1, 0, DET_COMMON },
  { "exp", 10, 0, 0, DET_COMMON },
  { "pow", 10, 0, 1, DET_COMMON },
  { "sinh", 10, 0, 0, DET_COMMON },
  { "cosh", 10, 0, 0, DET_COMMON },
  { "tanh", 10, 0, 0, DET_COMMON },
  { "sinh", 35, 3, 0, DET_COMMON },
  { "cosh", 35, 3, 0, DET_COMMON },
  { "tanh", 35, 3, 0, DET_COMMON },

  { "asinh", 10, 0, 0, DET_COMMON },
  { "acosh", 10, 0, 0, DET_COMMON },
  { "atanh", 10, 0, 0, DET_COMMON },
  { "exp2", 10, 0, 0, DET_COMMON },
  { "exp10", 10, 0, 0, DET_COMMON },
  { "expm1", 10, 0, 0, DET_COMMON },
  { "log10", 10, 0, 0, DET_COMMON },
  { "log2", 10, 0, 0, DET_COMMON },
  { "log1p", 10, 0, 0, DET_COMMON },
  { "sincospi", 5, 2, 2, DET_COMMON },
  { "sincospi", 35, 3, 2, DET_COMMON },
  { "sinpi", 5, 2, 0, DET_COMMON },
  { "cospi", 5, 2, 0, DET_COMMON },
  { "ldexp", -1, 0, 3, DET_COMMON },
  { "ilogb", -1, 0, 4, 0 },

  { "fma", -1, 0, 5, 0 },
  { "sqrt", -1, 0, 0, 0 },
  { "sqrt", 5, 2, 0, NOGNUABI },
  { "sqrt", 35, 3, 0, 0 },
  { "hypot", 5, 2, 1, DET_COMMON },
  { "hypot", 35, 3, 1, DET_COMMON },
  { "fabs", -1, 0, 0, DET_COMMON },
  { "copysign", -1, 0, 1, DET_COMMON },
  { "fmax", -1, 0, 1, DET_COMMON },
  { "fmin", -1, 0, 1, DET_COMMON },
  { "fdim", -1, 0, 1, DET_COMMON },
  { "trunc", -1, 0, 0, DET_COMMON },
  { "floor", -1, 0, 0, DET_COMMON },
  { "ceil", -1, 0, 0, DET_COMMON },
  { "round", -1, 0, 0, DET_COMMON },
  { "rint", -1, 0, 0, DET_COMMON },
  { "nextafter", -1, 0, 1, DET_COMMON },
  { "frfrexp", -1, 0, 0, DET_COMMON },
  { "expfrexp", -1, 0, 4, DET_COMMON },
  { "fmod", -1, 0, 1, DET_COMMON },
  { "modf", -1, 0, 6, DET_COMMON },

  { "lgamma", 10, 1, 0, DET_COMMON },
  { "tgamma", 10, 1, 0, DET_COMMON },
  { "erf", 10, 1, 0, DET_COMMON },
  { "erfc", 15, 4, 0, DET_COMMON },
  
  { "getInt", -1, 0, 7, NOGNUABI },
  { "getPtr", -1, 0, 8, NOGNUABI },
  
  { NULL, -1, 0, 0, 0 },
};
