//   Copyright Naoki Shibata and contributors 2010 - 2020.
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
  1 : "_u10"
  2 : "_u05"

  funcType:
  0 : vargquad func(vargquad);
  1 : vargquad func(vargquad, vargquad);
  2 : vargquad2 func(vargquad);
  3 : vargquad func(vargquad, vint);
  4 : vint func(vargquad);
  5 : vargquad func(vargquad, vargquad, vargquad);
  6 : vargquad2 func(vargquad);
  7 : int func(int);
  8 : void *func(int);
  9 : vint func(vargquad, vargquad);
  10 : vdouble func(vargquad);
  11 : vargquad func(vdouble);
  12 : vmask func(vargquad);
  13 : vargquad func(vmask);
  14 : vargquad func(Sleef_quad *);
  15 : void func(Sleef_quad *, vargquad);
  16 : Sleef_quad func(vargquad, int);
  17 : vargquad func(vargquad, int, Sleef_quad);
 */

funcSpec funcList[] = {
  { "add", 5, 2, 1, 0 },
  { "sub", 5, 2, 1, 0 },
  { "mul", 5, 2, 1, 0 },
  { "div", 5, 2, 1, 0 },
  { "neg", -1, 0, 0, 0 },
  { "sqrt", 5, 2, 0, 0 },

  { "cmplt", -1, 0, 9, 0 },
  { "cmple", -1, 0, 9, 0 },
  { "cmpgt", -1, 0, 9, 0 },
  { "cmpge", -1, 0, 9, 0 },
  { "cmpeq", -1, 0, 9, 0 },
  { "cmpneq", -1, 0, 9, 0 },
  { "unord", -1, 0, 9, 0 },

  { "cast_to_double", -1, 0, 10, 0 },
  { "cast_from_double", -1, 0, 11, 0 },
  { "cast_to_int64", -1, 0, 12, 0 },
  { "cast_from_int64", -1, 0, 13, 0 },
  { "cast_to_uint64", -1, 0, 12, 0 },
  { "cast_from_uint64", -1, 0, 13, 0 },
  { "load", -1, 0, 14, 0 },
  { "store", -1, 0, 15, 0 },
  { "get", -1, 0, 16, 0 },
  { "set", -1, 0, 17, 0 },

  { "sin", 10, 1, 0, 0 },
  { "cos", 10, 1, 0, 0 },
  { "tan", 10, 1, 0, 0 },

  { "asin", 10, 1, 0, 0 },
  { "acos", 10, 1, 0, 0 },
  { "atan", 10, 1, 0, 0 },

  { "exp", 10, 1, 0, 0 },
  { "exp2", 10, 1, 0, 0 },
  { "exp10", 10, 1, 0, 0 },
  { "expm1", 10, 1, 0, 0 },

  { "log", 10, 1, 0, 0 },
  { "log2", 10, 1, 0, 0 },
  { "log10", 10, 1, 0, 0 },
  { "log1p", 10, 1, 0, 0 },

  //{ "sincos", 10, 1, 2, 0 },
  //{ "ldexp", -1, 0, 3, 0 },
  //{ "ilogb", -1, 0, 4, 0 },
  //{ "fma", -1, 0, 5, 0 },
  
  { NULL, -1, 0, 0, 0 },
};
