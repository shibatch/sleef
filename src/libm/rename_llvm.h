//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define xsin __llvm_sin_f64
#define xcos __llvm_cos_f64
#define xsincos __llvm_sincos_f64
#define xtan __llvm_tan_f64
#define xasin __llvm_asin_f64
#define xacos __llvm_acos_f64
#define xatan __llvm_atan_f64
#define xatan2 __llvm_atan2_f64
#define xlog __llvm_log_f64
#define xcbrt __llvm_cbrt_f64

#define xsin_u1 __llvm_sin_u1_f64
#define xcos_u1 __llvm_cos_u1_f64
#define xsincos_u1 __llvm_sincos_u1_f64
#define xtan_u1 __llvm_tan_u1_f64
#define xasin_u1 __llvm_asin_u1_f64
#define xacos_u1 __llvm_acos_u1_f64
#define xatan_u1 __llvm_atan_u1_f64
#define xatan2_u1 __llvm_atan2_u1_f64
#define xlog_u1 __llvm_log_u1_f64
#define xcbrt_u1 __llvm_cbrt_u1_f64

#define xexp __llvm_exp_u1_f64
#define xpow __llvm_pow_u1_f64
#define xsinh __llvm_sinh_u1_f64
#define xcosh __llvm_cosh_u1_f64
#define xtanh __llvm_tanh_u1_f64
#define xasinh __llvm_asinh_u1_f64
#define xacosh __llvm_acosh_u1_f64
#define xatanh __llvm_atanh_u1_f64

#define xexp2 __llvm_exp2_u1_f64
#define xexp10 __llvm_exp10_u1_f64
#define xexpm1 __llvm_expm1_u1_f64
#define xlog10 __llvm_log10_u1_f64
#define xlog1p __llvm_log1p_u1_f64

#define xsincospi_u05 __llvm_sincospi_u05_f64
#define xsincospi __llvm_sincospi_f64
#define xsinpi_u05 __llvm_sinpi_u05_f64

#define xldexp __llvm_ldexp_f64
#define xilogb __llvm_ilogb_f64

#define xfma __llvm_fma_f64
#define xsqrt_u05 __llvm_sqrt_u05_f64
#define xhypot_u05 __llvm_hypot_u05_f64
#define xhypot __llvm_hypot_f64

#define xfabs __llvm_fabs_f64
#define xcopysign __llvm_copysign_f64
#define xfmax __llvm_fmax_f64
#define xfmin __llvm_fmin_f64
#define xfdim __llvm_fdim_f64
#define xtrunc __llvm_trunc_f64
#define xfloor __llvm_floor_f64
#define xceil __llvm_ceil_f64
#define xround __llvm_round_f64
#define xrint __llvm_rint_f64
#define xnextafter __llvm_nextafter_f64
#define xfrfrexp __llvm_frfrexp_f64
#define xexpfrexp __llvm_expfrexp_f64
#define xfmod __llvm_fmod_f64
#define xmodf __llvm_modf_f64

#define xlgamma_u1 __llvm_lgamma_u1_f64
#define xtgamma_u1 __llvm_tgamma_u1_f64
#define xerf_u1 __llvm_erf_u1_f64
#define xerfc_u15 __llvm_erfc_u15_f64

//

#define xsinf __llvm_sin_f32
#define xcosf __llvm_cos_f32
#define xsincosf __llvm_sincos_f32
#define xtanf __llvm_tan_f32
#define xasinf __llvm_asin_f32
#define xacosf __llvm_acos_f32
#define xatanf __llvm_atan_f32
#define xatan2f __llvm_atan2_f32
#define xlogf __llvm_log_f32
#define xcbrtf __llvm_cbrt_f32

#define xsinf_u1 __llvm_sin_u1_f32
#define xcosf_u1 __llvm_cos_u1_f32
#define xsincosf_u1 __llvm_sincos_u1_f32
#define xtanf_u1 __llvm_tan_u1_f32
#define xasinf_u1 __llvm_asin_u1_f32
#define xacosf_u1 __llvm_acos_u1_f32
#define xatanf_u1 __llvm_atan_u1_f32
#define xatan2f_u1 __llvm_atan2_u1_f32
#define xlogf_u1 __llvm_log_u1_f32
#define xcbrtf_u1 __llvm_cbrt_u1_f32

#define xexpf __llvm_exp_u1_f32
#define xpowf __llvm_pow_u1_f32
#define xsinhf __llvm_sinh_u1_f32
#define xcoshf __llvm_cosh_u1_f32
#define xtanhf __llvm_tanh_u1_f32
#define xasinhf __llvm_asinh_u1_f32
#define xacoshf __llvm_acosh_u1_f32
#define xatanhf __llvm_atanh_u1_f32

#define xexp2f __llvm_exp2_u1_f32
#define xexp10f __llvm_exp10_u1_f32
#define xexpm1f __llvm_expm1_u1_f32
#define xlog10f __llvm_log10_u1_f32
#define xlog1pf __llvm_log1p_u1_f32

#define xsincospif_u05 __llvm_sincospi_u05_f32
#define xsincospif __llvm_sincospi_f32

#define xldexpf __llvm_ldexp_f32
#define xilogbf __llvm_ilogb_f32

#define xfmaf __llvm_fma_f32
#define xsqrtf_u05 __llvm_sqrt_u05_f32
#define xsqrtf __llvm_sqrt_f32
#define xhypotf_u05 __llvm_hypot_u05_f32
#define xhypotf __llvm_hypot_f32

#define xfabsf __llvm_fabs_f32
#define xcopysignf __llvm_copysign_f32
#define xfmaxf __llvm_fmax_f32
#define xfminf __llvm_fmin_f32
#define xfdimf __llvm_fdim_f32
#define xtruncf __llvm_trunc_f32
#define xfloorf __llvm_floor_f32
#define xceilf __llvm_ceil_f32
#define xroundf __llvm_round_f32
#define xrintf __llvm_rint_f32
#define xnextafterf __llvm_nextafter_f32
#define xfrfrexpf __llvm_frfrexp_f32
#define xexpfrexpf __llvm_expfrexp_f32
#define xfmodf __llvm_fmod_f32
#define xmodff __llvm_modf_f32

//

#define xsincospil_u05 __llvm_sincospi_u05_f80
#define xsincospil __llvm_sincospi_f80

#define xsincospiq_u05 __llvm_sincospi_u05_f128
#define xsincospiq __llvm_sincospi_f128
