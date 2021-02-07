//   Copyright Naoki Shibata and contributors 2010 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define SLEEF_ENABLE_OMP_SIMD
#include "sleef.h"

#define N 1024
double a[N], b[N], c[N], d[N];
float e[N], f[N], g[N], h[N];

void testsin() {
  for(int i=0;i<N;i++) a[i] = Sleef_sin(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_sin_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_sin_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_sin_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_sin_u35
}

void testsind1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_sind1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_sind1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_sind1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_sind1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_sind1_u10
}

void testsind1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_sind1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_sind1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_sind1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_sind1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_sind1_u35
}

void testsinf() {
  for(int i=0;i<N;i++) e[i] = Sleef_sinf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_sinf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_sinf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_sinf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_sinf_u35
}

void testsinf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_sinf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_sinf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_sinf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_sinf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_sinf1_u10
}

void testsinf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_sinf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_sinf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_sinf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_sinf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_sinf1_u35
}

void testcos() {
  for(int i=0;i<N;i++) a[i] = Sleef_cos(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_cos_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_cos_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_cos_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_cos_u35
}

void testcosd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_cosd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_cosd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_cosd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_cosd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_cosd1_u10
}

void testcosd1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_cosd1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_cosd1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_cosd1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_cosd1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_cosd1_u35
}

void testcosf() {
  for(int i=0;i<N;i++) e[i] = Sleef_cosf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_cosf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_cosf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_cosf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_cosf_u35
}

void testcosf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_cosf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_cosf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_cosf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_cosf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_cosf1_u10
}

void testcosf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_cosf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_cosf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_cosf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_cosf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_cosf1_u35
}

void testtan() {
  for(int i=0;i<N;i++) a[i] = Sleef_tan(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_tan_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_tan_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_tan_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_tan_u35
}

void testtand1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_tand1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_tand1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_tand1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_tand1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_tand1_u10
}

void testtand1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_tand1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_tand1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_tand1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_tand1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_tand1_u35
}

void testtanf() {
  for(int i=0;i<N;i++) e[i] = Sleef_tanf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_tanf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_tanf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_tanf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_tanf_u35
}

void testtanf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_tanf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_tanf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_tanf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_tanf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_tanf1_u10
}

void testtanf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_tanf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_tanf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_tanf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_tanf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_tanf1_u35
}

void testasin() {
  for(int i=0;i<N;i++) a[i] = Sleef_asin(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_asin_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_asin_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_asin_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_asin_u35
}

void testasind1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_asind1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_asind1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_asind1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_asind1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_asind1_u10
}

void testasind1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_asind1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_asind1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_asind1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_asind1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_asind1_u35
}

void testasinf() {
  for(int i=0;i<N;i++) e[i] = Sleef_asinf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_asinf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_asinf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_asinf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_asinf_u35
}

void testasinf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_asinf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_asinf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_asinf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_asinf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_asinf1_u10
}

void testasinf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_asinf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_asinf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_asinf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_asinf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_asinf1_u35
}

void testacos() {
  for(int i=0;i<N;i++) a[i] = Sleef_acos(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_acos_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_acos_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_acos_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_acos_u35
}

void testacosd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_acosd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_acosd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_acosd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_acosd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_acosd1_u10
}

void testacosd1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_acosd1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_acosd1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_acosd1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_acosd1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_acosd1_u35
}

void testacosf() {
  for(int i=0;i<N;i++) e[i] = Sleef_acosf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_acosf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_acosf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_acosf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_acosf_u35
}

void testacosf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_acosf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_acosf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_acosf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_acosf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_acosf1_u10
}

void testacosf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_acosf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_acosf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_acosf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_acosf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_acosf1_u35
}

void testatan() {
  for(int i=0;i<N;i++) a[i] = Sleef_atan(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_atan_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_atan_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_atan_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_atan_u35
}

void testatand1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_atand1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_atand1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_atand1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_atand1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_atand1_u10
}

void testatand1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_atand1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_atand1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_atand1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_atand1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_atand1_u35
}

void testatanf() {
  for(int i=0;i<N;i++) e[i] = Sleef_atanf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_atanf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_atanf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_atanf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_atanf_u35
}

void testatanf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_atanf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_atanf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_atanf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_atanf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_atanf1_u10
}

void testatanf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_atanf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_atanf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_atanf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_atanf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_atanf1_u35
}

void testatan2() {
  for(int i=0;i<N;i++) a[i] = Sleef_atan2(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_atan2_u10
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_atan2_u35
// CHECK-AVX2: _ZGVdN4vv_Sleef_atan2_u10
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_atan2_u35
}

void testatan2d1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_atan2d1_u10(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_atan2d1_u10
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_atan2d1_u10
// CHECK-AVX2: _ZGVdN4vv_Sleef_atan2d1_u10
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_atan2d1_u10
}

void testatan2d1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_atan2d1_u35(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_atan2d1_u35
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_atan2d1_u35
// CHECK-AVX2: _ZGVdN4vv_Sleef_atan2d1_u35
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_atan2d1_u35
}

void testatan2f() {
  for(int i=0;i<N;i++) e[i] = Sleef_atan2f(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_atan2f_u10
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_atan2f_u35
// CHECK-AVX2: _ZGVdN8vv_Sleef_atan2f_u10
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_atan2f_u35
}

void testatan2f1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_atan2f1_u10(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_atan2f1_u10
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_atan2f1_u10
// CHECK-AVX2: _ZGVdN8vv_Sleef_atan2f1_u10
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_atan2f1_u10
}

void testatan2f1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_atan2f1_u35(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_atan2f1_u35
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_atan2f1_u35
// CHECK-AVX2: _ZGVdN8vv_Sleef_atan2f1_u35
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_atan2f1_u35
}

void testsinh() {
  for(int i=0;i<N;i++) a[i] = Sleef_sinh(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_sinh_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_sinh_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_sinh_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_sinh_u35
}

void testsinhd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_sinhd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_sinhd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_sinhd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_sinhd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_sinhd1_u10
}

void testsinhd1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_sinhd1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_sinhd1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_sinhd1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_sinhd1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_sinhd1_u35
}

void testsinhf() {
  for(int i=0;i<N;i++) e[i] = Sleef_sinhf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_sinhf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_sinhf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_sinhf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_sinhf_u35
}

void testsinhf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_sinhf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_sinhf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_sinhf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_sinhf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_sinhf1_u10
}

void testsinhf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_sinhf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_sinhf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_sinhf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_sinhf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_sinhf1_u35
}

void testcosh() {
  for(int i=0;i<N;i++) a[i] = Sleef_cosh(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_cosh_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_cosh_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_cosh_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_cosh_u35
}

void testcoshd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_coshd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_coshd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_coshd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_coshd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_coshd1_u10
}

void testcoshd1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_coshd1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_coshd1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_coshd1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_coshd1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_coshd1_u35
}

void testcoshf() {
  for(int i=0;i<N;i++) e[i] = Sleef_coshf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_coshf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_coshf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_coshf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_coshf_u35
}

void testcoshf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_coshf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_coshf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_coshf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_coshf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_coshf1_u10
}

void testcoshf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_coshf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_coshf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_coshf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_coshf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_coshf1_u35
}

void testtanh() {
  for(int i=0;i<N;i++) a[i] = Sleef_tanh(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_tanh_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_tanh_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_tanh_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_tanh_u35
}

void testtanhd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_tanhd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_tanhd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_tanhd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_tanhd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_tanhd1_u10
}

void testtanhd1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_tanhd1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_tanhd1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_tanhd1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_tanhd1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_tanhd1_u35
}

void testtanhf() {
  for(int i=0;i<N;i++) e[i] = Sleef_tanhf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_tanhf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_tanhf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_tanhf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_tanhf_u35
}

void testtanhf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_tanhf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_tanhf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_tanhf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_tanhf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_tanhf1_u10
}

void testtanhf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_tanhf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_tanhf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_tanhf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_tanhf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_tanhf1_u35
}

void testasinh() {
  for(int i=0;i<N;i++) a[i] = Sleef_asinh(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_asinh_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_asinh_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_asinh_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_asinh_u10
}

void testasinhd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_asinhd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_asinhd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_asinhd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_asinhd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_asinhd1_u10
}

void testasinhf() {
  for(int i=0;i<N;i++) e[i] = Sleef_asinhf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_asinhf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_asinhf_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_asinhf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_asinhf_u10
}

void testasinhf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_asinhf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_asinhf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_asinhf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_asinhf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_asinhf1_u10
}

void testacosh() {
  for(int i=0;i<N;i++) a[i] = Sleef_acosh_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_acosh_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_acosh_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_acosh_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_acosh_u10
}

void testacoshd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_acoshd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_acoshd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_acoshd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_acoshd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_acoshd1_u10
}

void testacoshf() {
  for(int i=0;i<N;i++) e[i] = Sleef_acoshf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_acoshf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_acoshf_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_acoshf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_acoshf_u10
}

void testacoshf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_acoshf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_acoshf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_acoshf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_acoshf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_acoshf1_u10
}

void testatanh() {
  for(int i=0;i<N;i++) a[i] = Sleef_atanh_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_atanh_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_atanh_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_atanh_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_atanh_u10
}

void testatanhd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_atanhd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_atanhd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_atanhd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_atanhd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_atanhd1_u10
}

void testatanhf() {
  for(int i=0;i<N;i++) e[i] = Sleef_atanhf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_atanhf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_atanhf_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_atanhf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_atanhf_u10
}

void testatanhf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_atanhf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_atanhf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_atanhf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_atanhf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_atanhf1_u10
}

void testlog() {
  for(int i=0;i<N;i++) a[i] = Sleef_log(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_log_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_log_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_log_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_log_u35
}

void testlogd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_logd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_logd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_logd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_logd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_logd1_u10
}

void testlogd1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_logd1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_logd1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_logd1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_logd1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_logd1_u35
}

void testlogf() {
  for(int i=0;i<N;i++) e[i] = Sleef_logf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_logf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_logf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_logf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_logf_u35
}

void testlogf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_logf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_logf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_logf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_logf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_logf1_u10
}

void testlogf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_logf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_logf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_logf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_logf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_logf1_u35
}

void testlog2() {
  for(int i=0;i<N;i++) a[i] = Sleef_log2_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_log2_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_log2_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_log2_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_log2_u10
}

void testlog2d1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_log2d1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_log2d1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_log2d1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_log2d1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_log2d1_u10
}

void testlog2f() {
  for(int i=0;i<N;i++) e[i] = Sleef_log2f(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_log2f_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_log2f_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_log2f_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_log2f_u10
}

void testlog2f1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_log2f1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_log2f1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_log2f1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_log2f1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_log2f1_u10
}

void testlog10() {
  for(int i=0;i<N;i++) a[i] = Sleef_log10_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_log10_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_log10_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_log10_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_log10_u10
}

void testlog10d1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_log10d1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_log10d1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_log10d1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_log10d1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_log10d1_u10
}

void testlog10f() {
  for(int i=0;i<N;i++) e[i] = Sleef_log10f(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_log10f_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_log10f_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_log10f_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_log10f_u10
}

void testlog10f1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_log10f1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_log10f1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_log10f1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_log10f1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_log10f1_u10
}

void testlog1p() {
  for(int i=0;i<N;i++) a[i] = Sleef_log1p_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_log1p_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_log1p_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_log1p_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_log1p_u10
}

void testlog1pd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_log1pd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_log1pd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_log1pd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_log1pd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_log1pd1_u10
}

void testlog1pf() {
  for(int i=0;i<N;i++) e[i] = Sleef_log1pf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_log1pf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_log1pf_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_log1pf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_log1pf_u10
}

void testlog1pf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_log1pf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_log1pf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_log1pf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_log1pf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_log1pf1_u10
}

void testexp() {
  for(int i=0;i<N;i++) a[i] = Sleef_exp_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_exp_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_exp_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_exp_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_exp_u10
}

void testexpd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_expd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_expd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_expd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_expd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_expd1_u10
}

void testexpf() {
  for(int i=0;i<N;i++) e[i] = Sleef_expf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_expf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_expf_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_expf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_expf_u10
}

void testexpf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_expf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_expf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_expf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_expf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_expf1_u10
}

void testexp2() {
  for(int i=0;i<N;i++) a[i] = Sleef_exp2_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_exp2_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_exp2_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_exp2_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_exp2_u10
}

void testexp2d1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_exp2d1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_exp2d1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_exp2d1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_exp2d1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_exp2d1_u10
}

void testexp2f() {
  for(int i=0;i<N;i++) e[i] = Sleef_exp2f(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_exp2f_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_exp2f_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_exp2f_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_exp2f_u10
}

void testexp2f1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_exp2f1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_exp2f1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_exp2f1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_exp2f1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_exp2f1_u10
}

void testexp10() {
  for(int i=0;i<N;i++) a[i] = Sleef_exp10_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_exp10_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_exp10_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_exp10_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_exp10_u10
}

void testexp10d1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_exp10d1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_exp10d1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_exp10d1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_exp10d1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_exp10d1_u10
}

void testexp10f() {
  for(int i=0;i<N;i++) e[i] = Sleef_exp10f(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_exp10f_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_exp10f_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_exp10f_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_exp10f_u10
}

void testexp10f1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_exp10f1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_exp10f1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_exp10f1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_exp10f1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_exp10f1_u10
}

void testexpm1() {
  for(int i=0;i<N;i++) a[i] = Sleef_expm1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_expm1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_expm1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_expm1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_expm1_u10
}

void testexpm1d1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_expm1d1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_expm1d1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_expm1d1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_expm1d1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_expm1d1_u10
}

void testexpm1f() {
  for(int i=0;i<N;i++) e[i] = Sleef_expm1f(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_expm1f_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_expm1f_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_expm1f_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_expm1f_u10
}

void testexpm1f1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_expm1f1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_expm1f1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_expm1f1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_expm1f1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_expm1f1_u10
}

void testpow() {
  for(int i=0;i<N;i++) a[i] = Sleef_pow_u10(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_pow_u10
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_pow_u10
// CHECK-AVX2: _ZGVdN4vv_Sleef_pow_u10
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_pow_u10
}

void testpowd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_powd1_u10(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_powd1_u10
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_powd1_u10
// CHECK-AVX2: _ZGVdN4vv_Sleef_powd1_u10
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_powd1_u10
}

void testpowf() {
  for(int i=0;i<N;i++) e[i] = Sleef_powf(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_powf_u10
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_powf_u10
// CHECK-AVX2: _ZGVdN8vv_Sleef_powf_u10
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_powf_u10
}

void testpowf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_powf1_u10(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_powf1_u10
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_powf1_u10
// CHECK-AVX2: _ZGVdN8vv_Sleef_powf1_u10
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_powf1_u10
}

void testcbrt() {
  for(int i=0;i<N;i++) a[i] = Sleef_cbrt(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_cbrt_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_cbrt_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_cbrt_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_cbrt_u35
}

void testcbrtd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_cbrtd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_cbrtd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_cbrtd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_cbrtd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_cbrtd1_u10
}

void testcbrtd1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_cbrtd1_u35(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_cbrtd1_u35
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_cbrtd1_u35
// CHECK-AVX2: _ZGVdN4v_Sleef_cbrtd1_u35
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_cbrtd1_u35
}

void testcbrtf() {
  for(int i=0;i<N;i++) e[i] = Sleef_cbrtf(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_cbrtf_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_cbrtf_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_cbrtf_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_cbrtf_u35
}

void testcbrtf1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_cbrtf1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_cbrtf1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_cbrtf1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_cbrtf1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_cbrtf1_u10
}

void testcbrtf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_cbrtf1_u35(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_cbrtf1_u35
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_cbrtf1_u35
// CHECK-AVX2: _ZGVdN8v_Sleef_cbrtf1_u35
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_cbrtf1_u35
}

void testhypot() {
  for(int i=0;i<N;i++) a[i] = Sleef_hypot(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_hypot_u05
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_hypot_u35
// CHECK-AVX2: _ZGVdN4vv_Sleef_hypot_u05
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_hypot_u35
}

void testhypotd1_u05() {
  for(int i=0;i<N;i++) a[i] = Sleef_hypotd1_u05(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_hypotd1_u05
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_hypotd1_u05
// CHECK-AVX2: _ZGVdN4vv_Sleef_hypotd1_u05
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_hypotd1_u05
}

void testhypotd1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_hypotd1_u35(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_hypotd1_u35
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_hypotd1_u35
// CHECK-AVX2: _ZGVdN4vv_Sleef_hypotd1_u35
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_hypotd1_u35
}

void testhypotf() {
  for(int i=0;i<N;i++) e[i] = Sleef_hypotf(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_hypotf_u05
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_hypotf_u35
// CHECK-AVX2: _ZGVdN8vv_Sleef_hypotf_u05
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_hypotf_u35
}

void testhypotf1_u05() {
  for(int i=0;i<N;i++) e[i] = Sleef_hypotf1_u05(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_hypotf1_u05
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_hypotf1_u05
// CHECK-AVX2: _ZGVdN8vv_Sleef_hypotf1_u05
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_hypotf1_u05
}

void testhypotf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_hypotf1_u35(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_hypotf1_u35
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_hypotf1_u35
// CHECK-AVX2: _ZGVdN8vv_Sleef_hypotf1_u35
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_hypotf1_u35
}

void testerf() {
  for(int i=0;i<N;i++) a[i] = Sleef_erf(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_erf_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_erf_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_erf_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_erf_u10
}

void testerfd1_u10() {
  for(int i=0;i<N;i++) a[i] = Sleef_erfd1_u10(b[i]);
// CHECK-SSE2: _ZGVbN2v_Sleef_erfd1_u10
// CHECK-SSE2-FAST: _ZGVbN2v_Sleef_erfd1_u10
// CHECK-AVX2: _ZGVdN4v_Sleef_erfd1_u10
// CHECK-AVX2-FAST: _ZGVdN4v_Sleef_erfd1_u10
}

void testerff() {
  for(int i=0;i<N;i++) e[i] = Sleef_erff(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_erff_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_erff_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_erff_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_erff_u10
}

void testerff1_u10() {
  for(int i=0;i<N;i++) e[i] = Sleef_erff1_u10(f[i]);
// CHECK-SSE2: _ZGVbN4v_Sleef_erff1_u10
// CHECK-SSE2-FAST: _ZGVbN4v_Sleef_erff1_u10
// CHECK-AVX2: _ZGVdN8v_Sleef_erff1_u10
// CHECK-AVX2-FAST: _ZGVdN8v_Sleef_erff1_u10
}

void testfmod() {
  for(int i=0;i<N;i++) a[i] = Sleef_fmod(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_fmod
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_fmod
// CHECK-AVX2: _ZGVdN4vv_Sleef_fmod
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_fmod
}

void testfmodd1() {
  for(int i=0;i<N;i++) a[i] = Sleef_fmodd1(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_fmodd1
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_fmodd1
// CHECK-AVX2: _ZGVdN4vv_Sleef_fmodd1
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_fmodd1
}

void testfmodf() {
  for(int i=0;i<N;i++) e[i] = Sleef_fmodf(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_fmodf
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_fmodf
// CHECK-AVX2: _ZGVdN8vv_Sleef_fmodf
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_fmodf
}

void testfmodf1() {
  for(int i=0;i<N;i++) e[i] = Sleef_fmodf1(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_fmodf1
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_fmodf1
// CHECK-AVX2: _ZGVdN8vv_Sleef_fmodf1
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_fmodf1
}

void testremainder() {
  for(int i=0;i<N;i++) a[i] = Sleef_remainder(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_remainder
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_remainder
// CHECK-AVX2: _ZGVdN4vv_Sleef_remainder
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_remainder
}

void testremainderd1_u35() {
  for(int i=0;i<N;i++) a[i] = Sleef_remainderd1(b[i], c[i]);
// CHECK-SSE2: _ZGVbN2vv_Sleef_remainderd1
// CHECK-SSE2-FAST: _ZGVbN2vv_Sleef_remainderd1
// CHECK-AVX2: _ZGVdN4vv_Sleef_remainderd1
// CHECK-AVX2-FAST: _ZGVdN4vv_Sleef_remainderd1
}

void testremainderf() {
  for(int i=0;i<N;i++) e[i] = Sleef_remainderf(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_remainderf
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_remainderf
// CHECK-AVX2: _ZGVdN8vv_Sleef_remainderf
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_remainderf
}

void testremainderf1_u35() {
  for(int i=0;i<N;i++) e[i] = Sleef_remainderf1(f[i], g[i]);
// CHECK-SSE2: _ZGVbN4vv_Sleef_remainderf1
// CHECK-SSE2-FAST: _ZGVbN4vv_Sleef_remainderf1
// CHECK-AVX2: _ZGVdN8vv_Sleef_remainderf1
// CHECK-AVX2-FAST: _ZGVdN8vv_Sleef_remainderf1
}
