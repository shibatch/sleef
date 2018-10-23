/// This program makes sure that all the symbols that a
/// GNUABI-compatible compiler (clang or gcc) can generate when
/// vectorizing functions call from `#include <math.h>` are present in
/// `libsleefgnuabi.so`.
///
/// The header `math.h` is not the same on all systems, and different
/// macros can activate different sets of functions. The list provide
/// here shoudl cover the union of all possible systems that we want
/// to support. In particular, the test is checking that the "finite"
/// symmbols from `#include <bits/math-finite.h>` are present for
/// those systems supporting them.

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>

#if defined(ENABLE_SSE4) || defined(ENABLE_SSE2)
#define ISA_TOKEN b
#define VLEN_SP 4
#define VLEN_DP 2
#define VECTOR_CC
#endif /* defined(ENABLE_SSE4) || defined(ENABLE_SSE2) */

#ifdef ENABLE_AVX
#define ISA_TOKEN c
#define VLEN_SP 8
#define VLEN_DP 4
#define VECTOR_CC
#endif /* ENABLE_AVX */

#ifdef ENABLE_AVX2
#define ISA_TOKEN d
#define VLEN_SP 8
#define VLEN_DP 4
#define VECTOR_CC
#endif /* ENABLE_AVX2 */

#ifdef ENABLE_AVX512F
#define ISA_TOKEN e
#define VLEN_SP 16
#define VLEN_DP 8
#define VECTOR_CC
#endif /* ENABLE_AVX512F */

#ifdef ENABLE_ADVSIMD
#define ISA_TOKEN n
#define VLEN_SP 4
#ifdef ENABLE_AAVPCS
#define VECTOR_CC __attribute__((aarch64_vector_pcs))
#else
#define VECTOR_CC
#endif
#define VLEN_DP 2
#endif /* ENABLE_ADVSIMDF */

#ifdef ENABLE_SVE
#include <arm_sve.h>
#define ISA_TOKEN s
#define VLEN_SP (svcntw())
#define VLEN_DP (svcntd())
#define VLA_TOKEN x
#define VECTOR_CC
#endif /* ENABLE_SVE */

// GNUABI name mangling macro.
#ifndef MASKED_GNUABI
#define __MAKE_FN_NAME(name, t, vl, p) _ZGV##t##N##vl##p##_##name
#else /* MASKED_GNUABI */
#define __MAKE_FN_NAME(name, t, vl, p) _ZGV##t##M##vl##p##_##name
#endif /* MASKED_GNUABI */
// Level-1 expansion macros for declaration and call. The signature of
// each function has three input paramters to avoid segfaults of
// sincos-like functions that are effectively loading data from
// memory.
#define __DECLARE(name, t, vl, p)                                              \
  extern void VECTOR_CC __MAKE_FN_NAME(name, t, vl, p)(int *, int *, int *)
#define __CALL(name, t, vl, p) __MAKE_FN_NAME(name, t, vl, p)(b0, b1, b2)

// Make sure that the architectural macros are defined for each vector
// extension.
#ifndef ISA_TOKEN
#error "Missing ISA token"
#endif

#ifndef VLEN_DP
#error "Missing VLEN_DP"
#endif

#ifndef VLEN_DP
#error "Missing VLEN_SP"
#endif

#if defined(ENABLE_SVE) && !defined(VLA_TOKEN)
#error "Missing VLA_TOKEN"
#endif /* defined(ENABLE_SVE) && !defined(VLA_TOKEN) */

// Declaration and call, first level expantion to pick up the
// ISA_TOKEN and VLEN_* architectural macros.
#ifndef ENABLE_SVE
#define DECLARE_DP(name, p) __DECLARE(name, ISA_TOKEN, VLEN_DP, p)
#define CALL_DP(name, p) __CALL(name, ISA_TOKEN, VLEN_DP, p)
#else /* ENABLE_SVE */
#define DECLARE_DP(name, p) __DECLARE(name, ISA_TOKEN, VLA_TOKEN, p)
#define CALL_DP(name, p) __CALL(name, ISA_TOKEN, VLA_TOKEN, p)
#endif /* ENABLE_SVE */

// Douple precision function declarations.
DECLARE_DP(__acos_finite, v);
DECLARE_DP(__acosh_finite, v);
DECLARE_DP(__asin_finite, v);
DECLARE_DP(__atan2_finite, vv);
DECLARE_DP(__atanh_finite, v);
DECLARE_DP(__cosh_finite, v);
DECLARE_DP(__exp10_finite, v);
DECLARE_DP(__exp2_finite, v);
DECLARE_DP(__exp_finite, v);
DECLARE_DP(__fmod_finite, vv);
DECLARE_DP(__modf_finite, vl8);
DECLARE_DP(__hypot_finite, vv);
DECLARE_DP(__log10_finite, v);
// DECLARE_DP(__log2_finite,v);
DECLARE_DP(__log_finite, v);
DECLARE_DP(__pow_finite, vv);
DECLARE_DP(__sinh_finite, v);
DECLARE_DP(__sqrt_finite, v);
DECLARE_DP(acos, v);
DECLARE_DP(acosh, v);
DECLARE_DP(asin, v);
DECLARE_DP(asinh, v);
DECLARE_DP(atan, v);
DECLARE_DP(atan2, vv);
DECLARE_DP(__atan2_finite, vv);
DECLARE_DP(atanh, v);
DECLARE_DP(cbrt, v);
DECLARE_DP(ceil, v);
DECLARE_DP(copysign, vv);
DECLARE_DP(cos, v);
DECLARE_DP(cosh, v);
DECLARE_DP(cospi, v);
DECLARE_DP(erf, v);
DECLARE_DP(erfc, v);
DECLARE_DP(exp, v);
DECLARE_DP(exp10, v);
DECLARE_DP(exp2, v);
DECLARE_DP(expfrexp, v);
DECLARE_DP(expm1, v);
DECLARE_DP(fabs, v);
DECLARE_DP(fdim, vv);
DECLARE_DP(floor, v);
DECLARE_DP(fma, vvv);
DECLARE_DP(fmax, vv);
DECLARE_DP(fmin, vv);
DECLARE_DP(fmod, vv);
DECLARE_DP(frfrexp, v);
DECLARE_DP(hypot, vv);
DECLARE_DP(ilogb, v);
DECLARE_DP(ldexp, vv);
DECLARE_DP(lgamma, v);
DECLARE_DP(log, v);
DECLARE_DP(log10, v);
DECLARE_DP(log1p, v);
DECLARE_DP(log2, v);
DECLARE_DP(modf, vl8);
DECLARE_DP(nextafter, vv);
DECLARE_DP(pow, vv);
DECLARE_DP(rint, v);
DECLARE_DP(round, v);
DECLARE_DP(sin, v);
DECLARE_DP(sincos, vl8l8);
DECLARE_DP(sincospi, vl8l8);
DECLARE_DP(sinh, v);
DECLARE_DP(sinpi, v);
DECLARE_DP(sqrt, v);
DECLARE_DP(tan, v);
DECLARE_DP(tanh, v);
DECLARE_DP(tgamma, v);
DECLARE_DP(trunc, v);

#ifndef ENABLE_SVE
#define DECLARE_SP(name, p) __DECLARE(name, ISA_TOKEN, VLEN_SP, p)
#define CALL_SP(name, p) __CALL(name, ISA_TOKEN, VLEN_SP, p)
#else /* ENABLE_SVE */
#define DECLARE_SP(name, p) __DECLARE(name, ISA_TOKEN, VLA_TOKEN, p)
#define CALL_SP(name, p) __CALL(name, ISA_TOKEN, VLA_TOKEN, p)
#endif /* ENABLE_SVE */

// Single precision function declarations.
DECLARE_SP(__acosf_finite, v);
DECLARE_SP(__acoshf_finite, v);
DECLARE_SP(__asinf_finite, v);
DECLARE_SP(__atan2f_finite, vv);
DECLARE_SP(__atanhf_finite, v);
DECLARE_SP(__coshf_finite, v);
DECLARE_SP(__exp10f_finite, v);
DECLARE_SP(__exp2f_finite, v);
DECLARE_SP(__expf_finite, v);
DECLARE_SP(__fmodf_finite, vv);
DECLARE_SP(__modff_finite, vl4);
DECLARE_SP(__hypotf_finite, vv);
DECLARE_SP(__log10f_finite, v);
// DECLARE_SP(__log2f_finite,v);
DECLARE_SP(__logf_finite, v);
DECLARE_SP(__powf_finite, vv);
DECLARE_SP(__sinhf_finite, v);
DECLARE_SP(__sqrtf_finite, v);
DECLARE_SP(acosf, v);
DECLARE_SP(acoshf, v);
DECLARE_SP(asinf, v);
DECLARE_SP(asinhf, v);
DECLARE_SP(atanf, v);
DECLARE_SP(atan2f, vv);
DECLARE_SP(atanhf, v);
DECLARE_SP(cbrtf, v);
DECLARE_SP(ceilf, v);
DECLARE_SP(copysignf, vv);
DECLARE_SP(cosf, v);
DECLARE_SP(coshf, v);
DECLARE_SP(cospif, v);
DECLARE_SP(erff, v);
DECLARE_SP(erfcf, v);
DECLARE_SP(expf, v);
DECLARE_SP(exp10f, v);
DECLARE_SP(exp2f, v);
DECLARE_SP(expm1f, v);
DECLARE_SP(fabsf, v);
DECLARE_SP(fdimf, vv);
DECLARE_SP(floorf, v);
DECLARE_SP(fmaf, vvv);
DECLARE_SP(fmaxf, vv);
DECLARE_SP(fminf, vv);
DECLARE_SP(fmodf, vv);
DECLARE_SP(frfrexpf, v);
DECLARE_SP(hypotf, vv);
#ifndef ENABLE_AVX
// These two functions are not checked in some configurations due to
// the issue in https://github.com/shibatch/sleef/issues/221
DECLARE_SP(expfrexpf, v);
DECLARE_SP(ilogbf, v);
#endif
DECLARE_SP(ldexpf, vv);
DECLARE_SP(lgammaf, v);
DECLARE_SP(logf, v);
DECLARE_SP(log10f, v);
DECLARE_SP(log1pf, v);
DECLARE_SP(log2f, v);
DECLARE_SP(modff, vl4);
DECLARE_SP(nextafterf, vv);
DECLARE_SP(powf, vv);
DECLARE_SP(rintf, v);
DECLARE_SP(roundf, v);
DECLARE_SP(sinf, v);
DECLARE_SP(sincosf, vl4l4);
DECLARE_SP(sincospif, vl4l4);
DECLARE_SP(sinhf, v);
DECLARE_SP(sinpif, v);
DECLARE_SP(sqrtf, v);
DECLARE_SP(tanf, v);
DECLARE_SP(tanhf, v);
DECLARE_SP(tgammaf, v);
DECLARE_SP(truncf, v);

static jmp_buf sigjmp;

static void sighandler(int signum) { longjmp(sigjmp, 1); }

int detectFeature() {
  signal(SIGILL, sighandler);

  if (setjmp(sigjmp) == 0) {
    int b0[VLEN_SP];
    int b1[VLEN_SP];
    int b2[VLEN_SP];
    CALL_DP(__acos_finite, v);
    signal(SIGILL, SIG_DFL);
    return 1;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}

int main(void) {

  if (!detectFeature()) {
    return 0;
  }

  // Allocate enough memory to make sure that sincos-like functions can
  // load a full vector when invoked. All functions must operate on
  // these variables, which are printed at the end of the execution to
  // make sure that the compiler doesn't optimize out the calls.
  int b0[VLEN_SP];
  int b1[VLEN_SP];
  int b2[VLEN_SP];

  // Double precision function call.
  CALL_DP(__acos_finite, v);
  CALL_DP(__acosh_finite, v);
  CALL_DP(__asin_finite, v);
  CALL_DP(__atan2_finite, vv);
  CALL_DP(__atanh_finite, v);
  CALL_DP(__cosh_finite, v);
  CALL_DP(__exp10_finite, v);
  CALL_DP(__exp2_finite, v);
  CALL_DP(__exp_finite, v);
  CALL_DP(__fmod_finite, vv);
  CALL_DP(__modf_finite, vl8);
  CALL_DP(__hypot_finite, vv);
  CALL_DP(__log10_finite, v);
  // CALL_DP(__log2_finite,v);
  CALL_DP(__log_finite, v);
  CALL_DP(__pow_finite, vv);
  CALL_DP(__sinh_finite, v);
  CALL_DP(__sqrt_finite, v);
  CALL_DP(acos, v);
  CALL_DP(acosh, v);
  CALL_DP(asin, v);
  CALL_DP(asinh, v);
  CALL_DP(atan, v);
  CALL_DP(atan2, vv);
  CALL_DP(atanh, v);
  CALL_DP(cbrt, v);
  CALL_DP(ceil, v);
  CALL_DP(copysign, vv);
  CALL_DP(cos, v);
  CALL_DP(cosh, v);
  CALL_DP(cospi, v);
  CALL_DP(erf, v);
  CALL_DP(erfc, v);
  CALL_DP(exp, v);
  CALL_DP(exp10, v);
  CALL_DP(exp2, v);
  CALL_DP(expfrexp, v);
  CALL_DP(expm1, v);
  CALL_DP(fabs, v);
  CALL_DP(fdim, vv);
  CALL_DP(floor, v);
  CALL_DP(fma, vvv);
  CALL_DP(fmax, vv);
  CALL_DP(fmin, vv);
  CALL_DP(fmod, vv);
  CALL_DP(frfrexp, v);
  CALL_DP(hypot, vv);
  CALL_DP(ilogb, v);
  CALL_DP(ldexp, vv);
  CALL_DP(lgamma, v);
  CALL_DP(log, v);
  CALL_DP(log10, v);
  CALL_DP(log1p, v);
  CALL_DP(log2, v);
  CALL_DP(modf, vl8);
  CALL_DP(nextafter, vv);
  CALL_DP(pow, vv);
  CALL_DP(rint, v);
  CALL_DP(round, v);
  CALL_DP(sin, v);
  CALL_DP(sincos, vl8l8);
  CALL_DP(sincospi, vl8l8);
  CALL_DP(sinh, v);
  CALL_DP(sinpi, v);
  CALL_DP(sqrt, v);
  CALL_DP(tan, v);
  CALL_DP(tanh, v);
  CALL_DP(tgamma, v);
  CALL_DP(trunc, v);

  // Single precision function call.
  CALL_SP(__acosf_finite, v);
  CALL_SP(__acoshf_finite, v);
  CALL_SP(__asinf_finite, v);
  CALL_SP(__atan2f_finite, vv);
  CALL_SP(__atanhf_finite, v);
  CALL_SP(__coshf_finite, v);
  CALL_SP(__exp10f_finite, v);
  CALL_SP(__exp2f_finite, v);
  CALL_SP(__expf_finite, v);
  CALL_SP(__fmodf_finite, vv);
  CALL_SP(__modff_finite, vl4);
  CALL_SP(__hypotf_finite, vv);
  CALL_SP(__log10f_finite, v);
  // CALL_SP(__log2f_finite,v);
  CALL_SP(__logf_finite, v);
  CALL_SP(__powf_finite, vv);
  CALL_SP(__sinhf_finite, v);
  CALL_SP(__sqrtf_finite, v);
  CALL_SP(acosf, v);
  CALL_SP(acoshf, v);
  CALL_SP(asinf, v);
  CALL_SP(asinhf, v);
  CALL_SP(atanf, v);
  CALL_SP(atan2f, vv);
  CALL_SP(atanhf, v);
  CALL_SP(cbrtf, v);
  CALL_SP(ceilf, v);
  CALL_SP(copysignf, vv);
  CALL_SP(cosf, v);
  CALL_SP(coshf, v);
  CALL_SP(cospif, v);
  CALL_SP(erff, v);
  CALL_SP(erfcf, v);
  CALL_SP(expf, v);
  CALL_SP(exp10f, v);
  CALL_SP(exp2f, v);
  CALL_SP(expm1f, v);
  CALL_SP(fabsf, v);
  CALL_SP(fdimf, vv);
  CALL_SP(floorf, v);
  CALL_SP(fmaf, vvv);
  CALL_SP(fmaxf, vv);
  CALL_SP(fminf, vv);
  CALL_SP(fmodf, vv);
  CALL_SP(frfrexpf, v);
  CALL_SP(hypotf, vv);
#ifndef ENABLE_AVX
// These two functions are not checked in some configurations due to
// the issue in https://github.com/shibatch/sleef/issues/221
  CALL_SP(expfrexpf, v);
  CALL_SP(ilogbf, v);
#endif
  CALL_SP(ldexpf, vv);
  CALL_SP(lgammaf, v);
  CALL_SP(logf, v);
  CALL_SP(log10f, v);
  CALL_SP(log1pf, v);
  CALL_SP(log2f, v);
  CALL_SP(modff, vl4);
  CALL_SP(nextafterf, vv);
  CALL_SP(powf, vv);
  CALL_SP(rintf, v);
  CALL_SP(roundf, v);
  CALL_SP(sinf, v);
  CALL_SP(sincosf, vl4l4);
  CALL_SP(sincospif, vl4l4);
  CALL_SP(sinhf, v);
  CALL_SP(sinpif, v);
  CALL_SP(sqrtf, v);
  CALL_SP(tanf, v);
  CALL_SP(tanhf, v);
  CALL_SP(tgammaf, v);
  CALL_SP(truncf, v);

  // Print the vars to make sure the compiler does not remove the
  // calls.
  for (int i = 0; i < VLEN_SP; ++i)
    printf("%d %d %d\n", b0[i], b1[i], b2[i]);

  return 0;
}
