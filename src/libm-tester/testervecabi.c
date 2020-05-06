#include <stdio.h>
#include <stdlib.h>
#include <sleef.h>

#define N (65536 - 1)
#define THRES 1e-10
#define THRESF 0.02

double a[N], b[N], c[N], d[N];
float e[N], f[N], g[N], h[N];

static void check(char *mes, double thres) {
  double err = 0;
  for (int i = 0; i < N; i++) err += d[i] >= 0 ? d[i] : -d[i];
  if (err > thres) {
    printf("%s, error=%g\n", mes, err);
    exit(-1);
  }
}

static void checkf(char *mes, double thres) {
  double err = 0;
  for (int i = 0; i < N; i++) err += h[i] >= 0 ? h[i] : -h[i];
  if (err > thres) {
    printf("%s, error=%g\n", mes, err);
    exit(-1);
  }
}

int main() {
  int i;

  for (i = 0; i < N; i++) {
    a[i] = 1.5 * rand() / (double)RAND_MAX + 1e-100;
    b[i] = 1.5 * rand() / (double)RAND_MAX + 1e-100;
    c[i] = 1.5 * rand() / (double)RAND_MAX + 1e-100;
  }

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_asin_u10(Sleef_sin_u10(a[i])) - a[i];
  check("sin_u10, asin_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_asin_u35(Sleef_sin_u35(a[i])) - a[i];
  check("sin_u35, asin_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_acos_u10(Sleef_cos_u10(a[i])) - a[i];
  check("cos_u10, acos_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_acos_u35(Sleef_cos_u35(a[i])) - a[i];
  check("cos_u35, acos_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_atan_u10(Sleef_tan_u10(a[i])) - a[i];
  check("tan_u10, atan_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_atan_u35(Sleef_tan_u35(a[i])) - a[i];
  check("tan_u35, atan_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_atan2_u10(b[i] * Sleef_sinpi_u05(a[i]*0.1), b[i] * Sleef_cospi_u05(a[i]*0.1)) - a[i]*0.3141592653589793;
  check("sinpi_u05, cospi_u05, atan2_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_atan2_u35(b[i] * Sleef_sinpi_u05(a[i]*0.1), b[i] * Sleef_cospi_u05(a[i]*0.1)) - a[i]*0.3141592653589793;
  check("sinpi_u05, cospi_u05, atan2_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_log2_u10(Sleef_exp2_u10(a[i])) - a[i];
  check("log2_u10, exp2_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_log2_u35(Sleef_exp2_u35(a[i])) - a[i];
  check("log2_u35, exp2_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_log10_u10(Sleef_exp10_u35(a[i])) - a[i];
  check("log10_u10, exp10_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_log10_u10(Sleef_exp10_u10(a[i])) - a[i];
  check("log10_u10, exp10_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_log1p_u10(Sleef_expm1_u10(a[i])) - a[i];
  check("log1p_u10, expm1_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_pow_u10(a[i], b[i]) - Sleef_exp_u10(Sleef_log_u10(a[i]) * b[i]);
  check("pow_u10, exp_u10, log_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_pow_u10(a[i], b[i]) - Sleef_exp_u10(Sleef_log_u35(a[i]) * b[i]);
  check("pow_u10, exp_u10, log_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_cbrt_u10(a[i] * a[i] * a[i]) - a[i];
  check("cbrt_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_cbrt_u35(a[i] * a[i] * a[i]) - a[i];
  check("cbrt_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_asinh_u10(Sleef_sinh_u10(a[i])) - a[i];
  check("asinh_u10, sinh_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_asinh_u10(Sleef_sinh_u35(a[i])) - a[i];
  check("asinh_u10, sinh_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_acosh_u10(Sleef_cosh_u10(a[i])) - a[i];
  check("acosh_u10, cosh_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_acosh_u10(Sleef_cosh_u35(a[i])) - a[i];
  check("acosh_u10, cosh_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_atanh_u10(Sleef_tanh_u10(a[i])) - a[i];
  check("atanh_u10, tanh_u10", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_atanh_u10(Sleef_tanh_u35(a[i])) - a[i];
  check("atanh_u10, tanh_u35", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_fma(a[i], b[i], c[i]) - (a[i] * b[i] + c[i]);
  check("fma", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_hypot_u05(a[i], b[i]) - Sleef_sqrt_u05(a[i] * a[i] + b[i] * b[i]);
  check("hypot_u05, sqrt_u05", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_hypot_u35(a[i], b[i]) - Sleef_sqrt_u05(a[i] * a[i] + b[i] * b[i]);
  check("hypot_u35, sqrt_u05", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_fmod(a[i], b[i]) - (a[i] - Sleef_floor(a[i] / b[i]) * b[i]);
  check("fmod, floor", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_remainder(a[i], b[i]) - (a[i] - Sleef_rint(a[i] / b[i]) * b[i]);
  check("remainder, rint", THRES);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) d[i] = Sleef_nextafter(Sleef_nextafter(a[i], b[i]), -b[i]) - a[i];
  check("nextafter", THRES);

  //

  for (i = 0; i < N; i++) {
    e[i] = 1.5 * rand() / (double)RAND_MAX + 1e-100;
    f[i] = 1.5 * rand() / (double)RAND_MAX + 1e-100;
    g[i] = 1.5 * rand() / (double)RAND_MAX + 1e-100;
  }

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_asinf_u10(Sleef_sinf_u10(e[i])) - e[i];
  checkf("sinf_u10, asinf_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_asinf_u35(Sleef_sinf_u35(e[i])) - e[i];
  checkf("sinf_u35, asinf_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_acosf_u10(Sleef_cosf_u10(e[i])) - e[i];
  checkf("cosf_u10, acosf_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_acosf_u35(Sleef_cosf_u35(e[i])) - e[i];
  checkf("cosf_u35, acosf_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_atanf_u10(Sleef_tanf_u10(e[i])) - e[i];
  checkf("tanf_u10, atanf_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_atanf_u35(Sleef_tanf_u35(e[i])) - e[i];
  checkf("tanf_u35, atanf_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_atan2f_u10(f[i] * Sleef_sinpif_u05(e[i]*0.1), f[i] * Sleef_cospif_u05(e[i]*0.1)) - e[i]*0.3141592653589793;
  checkf("sinpif_u05, cospif_u05, atan2f_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_atan2f_u35(f[i] * Sleef_sinpif_u05(e[i]*0.1), f[i] * Sleef_cospif_u05(e[i]*0.1)) - e[i]*0.3141592653589793;
  checkf("sinpif_u05, cospif_u05, atan2f_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_log2f_u10(Sleef_exp2f_u10(e[i])) - e[i];
  checkf("log2f_u10, exp2f_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_log2f_u35(Sleef_exp2f_u35(e[i])) - e[i];
  checkf("log2f_u35, exp2f_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_log10f_u10(Sleef_exp10f_u35(e[i])) - e[i];
  checkf("log10f_u10, exp10f_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_log10f_u10(Sleef_exp10f_u10(e[i])) - e[i];
  checkf("log10f_u10, exp10f_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_log1pf_u10(Sleef_expm1f_u10(e[i])) - e[i];
  checkf("log1pf_u10, expm1f_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_powf_u10(e[i], f[i]) - Sleef_expf_u10(Sleef_logf_u10(e[i]) * f[i]);
  checkf("powf_u10, expf_u10, logf_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_powf_u10(e[i], f[i]) - Sleef_expf_u10(Sleef_logf_u35(e[i]) * f[i]);
  checkf("powf_u10, expf_u10, logf_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_cbrtf_u10(e[i] * e[i] * e[i]) - e[i];
  checkf("cbrtf_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_cbrtf_u35(e[i] * e[i] * e[i]) - e[i];
  checkf("cbrtf_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_asinhf_u10(Sleef_sinhf_u10(e[i])) - e[i];
  checkf("asinhf_u10, sinhf_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_asinhf_u10(Sleef_sinhf_u35(e[i])) - e[i];
  checkf("asinhf_u10, sinhf_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_acoshf_u10(Sleef_coshf_u10(e[i])) - e[i];
  checkf("acoshf_u10, coshf_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_acoshf_u10(Sleef_coshf_u35(e[i])) - e[i];
  checkf("acoshf_u10, coshf_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_atanhf_u10(Sleef_tanhf_u10(e[i])) - e[i];
  checkf("atanhf_u10, tanhf_u10", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_atanhf_u10(Sleef_tanhf_u35(e[i])) - e[i];
  checkf("atanhf_u10, tanhf_u35", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_fmaf(e[i], f[i], g[i]) - (e[i] * f[i] + g[i]);
  checkf("fmaf", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_hypotf_u05(e[i], f[i]) - Sleef_sqrtf_u05(e[i] * e[i] + f[i] * f[i]);
  checkf("hypotf_u05, sqrtf_u05", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_hypotf_u35(e[i], f[i]) - Sleef_sqrtf_u05(e[i] * e[i] + f[i] * f[i]);
  checkf("hypotf_u35, sqrtf_u05", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_fmodf(e[i], f[i]) - (e[i] - Sleef_floorf(e[i] / f[i]) * f[i]);
  checkf("fmodf, floorf", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_remainderf(e[i], f[i]) - (e[i] - Sleef_rintf(e[i] / f[i]) * f[i]);
  checkf("remainderf, rintf", THRESF);

#pragma omp parallel for simd
  for (i = 0; i < N; i++) h[i] = Sleef_nextafterf(Sleef_nextafter(e[i], f[i]), -f[i]) - e[i];
  checkf("nextafterf", THRESF);

  //

  exit(0);
}
