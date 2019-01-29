//          Copyright Naoki Shibata 2010 - 2019.
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
 */

funcSpec funcList[] = {
  { "add", 5, 2, 1, 0 },
  { "sub", 5, 2, 1, 0 },
  { "mul", 5, 2, 1, 0 },
  { "div", 5, 2, 1, 0 },
  { "neg", -1, 0, 0, 0 },
  { "sqrt", 5, 2, 0, 0 },
  //{ "sincos", 10, 1, 2, 0 },
  //{ "ldexp", -1, 0, 3, 0 },
  //{ "ilogb", -1, 0, 4, 0 },
  //{ "fma", -1, 0, 5, 0 },
  
  { NULL, -1, 0, 0, 0 },
};
