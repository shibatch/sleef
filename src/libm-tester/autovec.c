//   Copyright Naoki Shibata and contributors 2010 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef __cplusplus

#define SLEEF_REPLACE_LIBM_FUNCS
#define SLEEF_ENABLE_OMP_SIMD
#include "sleef.h"

#else

#define SLEEF_ENABLE_OMP_SIMD
#include "sleef.h"
using namespace sleef;

#endif

#define N 1024
double a[N], b[N], c[N], d[N];
float e[N], f[N], g[N], h[N];

void testsin() {
// CHECK_LABEL: testsin
  for(int i=0;i<N;i++) a[i] = sin(b[i]);
// CHECK: _ZGVdN4v_Sleef_sin_u10
}

void testsinf() {
// CHECK_LABEL: testsinf
  for(int i=0;i<N;i++) e[i] = sinf(f[i]);
// CHECK: _ZGVdN8v_Sleef_sinf_u10
}

void testcos() {
// CHECK_LABEL: testcos
  for(int i=0;i<N;i++) a[i] = cos(b[i]);
// CHECK: _ZGVdN4v_Sleef_cos_u10
}

void testcosf() {
// CHECK_LABEL: testcosf
  for(int i=0;i<N;i++) e[i] = cosf(f[i]);
// CHECK: _ZGVdN8v_Sleef_cosf_u10
}

void testtan() {
// CHECK_LABEL: testtan
  for(int i=0;i<N;i++) a[i] = tan(b[i]);
// CHECK: _ZGVdN4v_Sleef_tan_u10
}

void testtanf() {
// CHECK_LABEL: testtanf
  for(int i=0;i<N;i++) e[i] = tanf(f[i]);
// CHECK: _ZGVdN8v_Sleef_tanf_u10
}

void testasin() {
// CHECK_LABEL: testasin
  for(int i=0;i<N;i++) a[i] = asin(b[i]);
// CHECK: _ZGVdN4v_Sleef_asin_u10
}

void testasinf() {
// CHECK_LABEL: testasinf
  for(int i=0;i<N;i++) e[i] = asinf(f[i]);
// CHECK: _ZGVdN8v_Sleef_asinf_u10
}

void testacos() {
// CHECK_LABEL: testacos
  for(int i=0;i<N;i++) a[i] = acos(b[i]);
// CHECK: _ZGVdN4v_Sleef_acos_u10
}

void testacosf() {
// CHECK_LABEL: testacosf
  for(int i=0;i<N;i++) e[i] = acosf(f[i]);
// CHECK: _ZGVdN8v_Sleef_acosf_u10
}

void testatan() {
// CHECK_LABEL: testatan
  for(int i=0;i<N;i++) a[i] = atan(b[i]);
// CHECK: _ZGVdN4v_Sleef_atan_u10
}

void testatanf() {
// CHECK_LABEL: testatanf
  for(int i=0;i<N;i++) e[i] = atanf(f[i]);
// CHECK: _ZGVdN8v_Sleef_atanf_u10
}

void testatan2() {
// CHECK_LABEL: testatan2
  for(int i=0;i<N;i++) a[i] = atan2(b[i], c[i]);
// CHECK: _ZGVdN4vv_Sleef_atan2_u10
}

void testatan2f() {
// CHECK_LABEL: testatan2f
  for(int i=0;i<N;i++) e[i] = atan2f(f[i], g[i]);
// CHECK: _ZGVdN8vv_Sleef_atan2f_u10
}

void testsinh() {
// CHECK_LABEL: testsinh
  for(int i=0;i<N;i++) a[i] = sinh(b[i]);
// CHECK: _ZGVdN4v_Sleef_sinh_u10
}

void testsinhf() {
// CHECK_LABEL: testsinhf
  for(int i=0;i<N;i++) e[i] = sinhf(f[i]);
// CHECK: _ZGVdN8v_Sleef_sinhf_u10
}

void testcosh() {
// CHECK_LABEL: testcosh
  for(int i=0;i<N;i++) a[i] = cosh(b[i]);
// CHECK: _ZGVdN4v_Sleef_cosh_u10
}

void testcoshf() {
// CHECK_LABEL: testcoshf
  for(int i=0;i<N;i++) e[i] = coshf(f[i]);
// CHECK: _ZGVdN8v_Sleef_coshf_u10
}

void testtanh() {
// CHECK_LABEL: testtanh
  for(int i=0;i<N;i++) a[i] = tanh(b[i]);
// CHECK: _ZGVdN4v_Sleef_tanh_u10
}

void testtanhf() {
// CHECK_LABEL: testtanhf
  for(int i=0;i<N;i++) e[i] = tanhf(f[i]);
// CHECK: _ZGVdN8v_Sleef_tanhf_u10
}

void testasinh() {
// CHECK_LABEL: testasinh
  for(int i=0;i<N;i++) a[i] = asinh(b[i]);
// CHECK: _ZGVdN4v_Sleef_asinh_u10
}

void testasinhf() {
// CHECK_LABEL: testasinhf
  for(int i=0;i<N;i++) e[i] = asinhf(f[i]);
// CHECK: _ZGVdN8v_Sleef_asinhf_u10
}

void testacosh() {
// CHECK_LABEL: testacosh
  for(int i=0;i<N;i++) a[i] = acosh(b[i]);
// CHECK: _ZGVdN4v_Sleef_acosh_u10
}

void testacoshf() {
// CHECK_LABEL: testacoshf
  for(int i=0;i<N;i++) e[i] = acoshf(f[i]);
// CHECK: _ZGVdN8v_Sleef_acoshf_u10
}

void testatanh() {
// CHECK_LABEL: testatanh
  for(int i=0;i<N;i++) a[i] = atanh(b[i]);
// CHECK: _ZGVdN4v_Sleef_atanh_u10
}

void testatanhf() {
// CHECK_LABEL: testatanhf
  for(int i=0;i<N;i++) e[i] = atanhf(f[i]);
// CHECK: _ZGVdN8v_Sleef_atanhf_u10
}

void testlog() {
// CHECK_LABEL: testlog
  for(int i=0;i<N;i++) a[i] = log(b[i]);
// CHECK: _ZGVdN4v_Sleef_log_u10
}

void testlogf() {
// CHECK_LABEL: testlogf
  for(int i=0;i<N;i++) e[i] = logf(f[i]);
// CHECK: _ZGVdN8v_Sleef_logf_u10
}

void testlog2() {
// CHECK_LABEL: testlog2
  for(int i=0;i<N;i++) a[i] = log2(b[i]);
// CHECK: _ZGVdN4v_Sleef_log2_u10
}

void testlog2f() {
// CHECK_LABEL: testlog2f
  for(int i=0;i<N;i++) e[i] = log2f(f[i]);
// CHECK: _ZGVdN8v_Sleef_log2f_u10
}

void testlog10() {
// CHECK_LABEL: testlog10
  for(int i=0;i<N;i++) a[i] = log10(b[i]);
// CHECK: _ZGVdN4v_Sleef_log10_u10
}

void testlog10f() {
// CHECK_LABEL: testlog10f
  for(int i=0;i<N;i++) e[i] = log10f(f[i]);
// CHECK: _ZGVdN8v_Sleef_log10f_u10
}

void testlog1p() {
// CHECK_LABEL: testlog1p
  for(int i=0;i<N;i++) a[i] = log1p(b[i]);
// CHECK: _ZGVdN4v_Sleef_log1p_u10
}

void testlog1pf() {
// CHECK_LABEL: testlog1pf
  for(int i=0;i<N;i++) e[i] = log1pf(f[i]);
// CHECK: _ZGVdN8v_Sleef_log1pf_u10
}

void testexp() {
// CHECK_LABEL: testexp
  for(int i=0;i<N;i++) a[i] = exp(b[i]);
// CHECK: _ZGVdN4v_Sleef_exp_u10
}

void testexpf() {
// CHECK_LABEL: testexpf
  for(int i=0;i<N;i++) e[i] = expf(f[i]);
// CHECK: _ZGVdN8v_Sleef_expf_u10
}

void testexp2() {
// CHECK_LABEL: testexp2
  for(int i=0;i<N;i++) a[i] = exp2(b[i]);
// CHECK: _ZGVdN4v_Sleef_exp2_u10
}

void testexp2f() {
// CHECK_LABEL: testexp2f
  for(int i=0;i<N;i++) e[i] = exp2f(f[i]);
// CHECK: _ZGVdN8v_Sleef_exp2f_u10
}

void testexp10() {
// CHECK_LABEL: testexp10
  for(int i=0;i<N;i++) a[i] = exp10(b[i]);
// CHECK: _ZGVdN4v_Sleef_exp10_u10
}

void testexp10f() {
// CHECK_LABEL: testexp10f
  for(int i=0;i<N;i++) e[i] = exp10f(f[i]);
// CHECK: _ZGVdN8v_Sleef_exp10f_u10
}

void testexpm1() {
// CHECK_LABEL: testexpm1
  for(int i=0;i<N;i++) a[i] = expm1(b[i]);
// CHECK: _ZGVdN4v_Sleef_expm1_u10
}

void testexpm1f() {
// CHECK_LABEL: testexpm1f
  for(int i=0;i<N;i++) e[i] = expm1f(f[i]);
// CHECK: _ZGVdN8v_Sleef_expm1f_u10
}

void testpow() {
// CHECK_LABEL: testpow
  for(int i=0;i<N;i++) a[i] = pow(b[i], c[i]);
// CHECK: _ZGVdN4vv_Sleef_pow_u10
}

void testpowf() {
// CHECK_LABEL: testpowf
  for(int i=0;i<N;i++) e[i] = powf(f[i], g[i]);
// CHECK: _ZGVdN8vv_Sleef_powf_u10
}

void testcbrt() {
// CHECK_LABEL: testcbrt
  for(int i=0;i<N;i++) a[i] = cbrt(b[i]);
// CHECK: _ZGVdN4v_Sleef_cbrt_u10
}

void testcbrtf() {
// CHECK_LABEL: testcbrtf
  for(int i=0;i<N;i++) e[i] = cbrtf(f[i]);
// CHECK: _ZGVdN8v_Sleef_cbrtf_u10
}

void testhypot() {
// CHECK_LABEL: testhypot
  for(int i=0;i<N;i++) a[i] = hypot(b[i], c[i]);
// CHECK: _ZGVdN4vv_Sleef_hypot_u05
}

void testhypotf() {
// CHECK_LABEL: testhypotf
  for(int i=0;i<N;i++) e[i] = hypotf(f[i], g[i]);
// CHECK: _ZGVdN8vv_Sleef_hypotf_u05
}

void testerf() {
// CHECK_LABEL: testerf
  for(int i=0;i<N;i++) a[i] = erf(b[i]);
// CHECK: _ZGVdN4v_Sleef_erf_u10
}

void testerff() {
// CHECK_LABEL: testerff
  for(int i=0;i<N;i++) e[i] = erff(f[i]);
// CHECK: _ZGVdN8v_Sleef_erff_u10
}

void testfmod() {
// CHECK_LABEL: testfmod
  for(int i=0;i<N;i++) a[i] = fmod(b[i], c[i]);
// CHECK: _ZGVdN4vv_Sleef_fmod
}

void testfmodf() {
// CHECK_LABEL: testfmodf
  for(int i=0;i<N;i++) e[i] = fmodf(f[i], g[i]);
// CHECK: _ZGVdN8vv_Sleef_fmodf
}

void testremainder() {
// CHECK_LABEL: testremainder
  for(int i=0;i<N;i++) a[i] = remainder(b[i], c[i]);
// CHECK: _ZGVdN4vv_Sleef_remainder
}

void testremainderf() {
// CHECK_LABEL: testremainderf
  for(int i=0;i<N;i++) e[i] = remainderf(f[i], g[i]);
// CHECK: _ZGVdN8vv_Sleef_remainderf
}

void passed() {
// CHECK_LABEL: passed
}
