//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>
#include <assert.h>

#if defined(POWER64_UNDEF_USE_EXTERN_INLINES)
// This is a workaround required to cross compile for PPC64 binaries
#include <features.h>
#ifdef __USE_EXTERN_INLINES
#undef __USE_EXTERN_INLINES
#endif
#endif

#include <math.h>

#if defined(_MSC_VER)
#define STDIN_FILENO 0
#else
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#endif

#include "misc.h"
#include "sleef.h"
#include "testerutil.h"

#define DORENAME

#ifdef ENABLE_SSE2
#define CONFIG 2
#include "helpersse2.h"
#include "renamesse2.h"
typedef Sleef___m128d_2 vdouble2;
typedef Sleef___m128_2 vfloat2;
#endif

#ifdef ENABLE_SSE4
#define CONFIG 4
#include "helpersse2.h"
#include "renamesse4.h"
typedef Sleef___m128d_2 vdouble2;
typedef Sleef___m128_2 vfloat2;
#endif

#ifdef ENABLE_AVX
#define CONFIG 1
#include "helperavx.h"
#include "renameavx.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_FMA4
#define CONFIG 4
#include "helperavx.h"
#include "renamefma4.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_AVX2
#define CONFIG 1
#include "helperavx2.h"
#include "renameavx2.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_AVX2128
#define CONFIG 1
#include "helperavx2_128.h"
#include "renameavx2128.h"
typedef Sleef___m128d_2 vdouble2;
typedef Sleef___m128_2 vfloat2;
#endif

#ifdef ENABLE_AVX512F
#define CONFIG 1
#include "helperavx512f.h"
#include "renameavx512f.h"
typedef Sleef___m512d_2 vdouble2;
typedef Sleef___m512_2 vfloat2;
#endif

#ifdef ENABLE_AVX512FNOFMA
#define CONFIG 2
#include "helperavx512f.h"
#include "renameavx512fnofma.h"
typedef Sleef___m512d_2 vdouble2;
typedef Sleef___m512_2 vfloat2;
#endif

#ifdef ENABLE_NEON32
#define CONFIG 1
#include "helperneon32.h"
#include "renameneon32.h"
typedef Sleef_float32x4_t_2 vfloat2;
#endif

#ifdef ENABLE_NEON32VFPV4
#define CONFIG 4
#include "helperneon32.h"
#include "renameneon32vfpv4.h"
typedef Sleef_float32x4_t_2 vfloat2;
#endif

#ifdef ENABLE_ADVSIMD
#define CONFIG 1
#include "helperadvsimd.h"
#include "renameadvsimd.h"
typedef Sleef_float64x2_t_2 vdouble2;
typedef Sleef_float32x4_t_2 vfloat2;
#endif

#ifdef ENABLE_ADVSIMDNOFMA
#define CONFIG 2
#include "helperadvsimd.h"
#include "renameadvsimdnofma.h"
typedef Sleef_float64x2_t_2 vdouble2;
typedef Sleef_float32x4_t_2 vfloat2;
#endif

#ifdef ENABLE_DSP128
#define CONFIG 2
#include "helpersse2.h"
#include "renamedsp128.h"
typedef Sleef___m128d_2 vdouble2;
typedef Sleef___m128_2 vfloat2;
#endif

#ifdef ENABLE_SVE
#define CONFIG 1
#include "helpersve.h"
#include "renamesve.h"
typedef Sleef_svfloat64_t_2 vdouble2;
typedef Sleef_svfloat32_t_2 vfloat2;
#endif

#ifdef ENABLE_SVENOFMA
#define CONFIG 2
#include "helpersve.h"
#include "renamesvenofma.h"
typedef Sleef_svfloat64_t_2 vdouble2;
typedef Sleef_svfloat32_t_2 vfloat2;
#endif

#ifdef ENABLE_DSP256
#define CONFIG 1
#include "helperavx.h"
#include "renamedsp256.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_VSX
#define CONFIG 1
#include "helperpower_128.h"
#include "renamevsx.h"
typedef Sleef_vector_double_2 vdouble2;
typedef Sleef_vector_float_2 vfloat2;
#endif

#ifdef ENABLE_VSXNOFMA
#define CONFIG 2
#include "helperpower_128.h"
#include "renamevsxnofma.h"
typedef Sleef_vector_double_2 vdouble2;
typedef Sleef_vector_float_2 vfloat2;
#endif

#ifdef ENABLE_VECEXT
#define CONFIG 1
#include "helpervecext.h"
#include "norename.h"
#endif

#ifdef ENABLE_PUREC
#define CONFIG 1
#include "helperpurec.h"
#include "norename.h"
#endif

#ifdef ENABLE_PUREC_SCALAR
#define CONFIG 1
#include "helperpurec_scalar.h"
#include "renamepurec_scalar.h"
typedef Sleef_double_2 vdouble2;
typedef Sleef_float_2 vfloat2;
#endif

#ifdef ENABLE_PURECFMA_SCALAR
#define CONFIG 2
#include "helperpurec_scalar.h"
#include "renamepurecfma_scalar.h"
typedef Sleef_double_2 vdouble2;
typedef Sleef_float_2 vfloat2;
#endif

//

#ifdef ENABLE_DP
int check_featureDP() {
  if (vavailability_i(1) == 0) return 0;
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = 1.0;
  }
  vdouble a = vloadu_vd_p(s);
  a = xpow(a, a);
  vstoreu_v_p_vd(s, a);
  return 1;
}
#else
int check_featureDP() {
}
#endif

#ifdef ENABLE_SP
int check_featureSP() {
  if (vavailability_i(2) == 0) return 0;
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = 1.0;
  }
  vfloat a = vloadu_vf_p(s);
  a = xpowf(a, a);
  vstoreu_v_p_vf(s, a);
  return 1;
}
#else
int check_featureSP() {
}
#endif

//

#define func_d_d(funcStr, funcName) {		\
    if (startsWith(buf, funcStr " ")) {		\
      uint64_t u;					\
      sscanf(buf, funcStr " %" PRIx64, &u);		\
      double s[VECTLENDP];				\
      int i;						\
      for(i=0;i<VECTLENDP;i++) {			\
	s[i] = rand()/(double)RAND_MAX*20000-10000;	\
      }							\
      int idx = rand() & (VECTLENDP-1);			\
      s[idx] = u2d(u);					\
      vdouble a = vloadu_vd_p(s);			\
      a = funcName(a);					\
      vstoreu_v_p_vd(s, a);				\
      u = d2u(s[idx]);					\
      printf("%" PRIx64 "\n", u);			\
      fflush(stdout);					\
      continue;						\
    }							\
  }

#define func_d2_d(funcStr, funcName) {		\
    if (startsWith(buf, funcStr " ")) {		\
      uint64_t u;				\
      sscanf(buf, funcStr " %" PRIx64, &u);	\
      double s[VECTLENDP], t[VECTLENDP];			\
      int i;							\
      for(i=0;i<VECTLENDP;i++) {				\
	s[i] = rand()/(double)RAND_MAX*20000-10000;		\
	t[i] = rand()/(double)RAND_MAX*20000-10000;		\
      }								\
      int idx = rand() & (VECTLENDP-1);				\
      s[idx] = u2d(u);						\
      vdouble2 v;						\
      vdouble a = vloadu_vd_p(s);				\
      v = funcName(a);						\
      vstoreu_v_p_vd(s, v.x);					\
      vstoreu_v_p_vd(t, v.y);					\
      Sleef_double2 d2;						\
      d2.x = s[idx];						\
      d2.y = t[idx];						\
      printf("%" PRIx64 " %" PRIx64 "\n", d2u(d2.x), d2u(d2.y));	\
      fflush(stdout);					\
      continue;						\
    }							\
  }

#define func_d_d_d(funcStr, funcName) {	\
    if (startsWith(buf, funcStr " ")) {	\
      uint64_t u, v;					\
      sscanf(buf, funcStr " %" PRIx64 " %" PRIx64, &u, &v);	\
      double s[VECTLENDP], t[VECTLENDP];			\
      int i;							\
      for(i=0;i<VECTLENDP;i++) {				\
	s[i] = rand()/(double)RAND_MAX*20000-10000;		\
	t[i] = rand()/(double)RAND_MAX*20000-10000;		\
      }								\
      int idx = rand() & (VECTLENDP-1);				\
      s[idx] = u2d(u);						\
      t[idx] = u2d(v);						\
      vdouble a, b;						\
      a = vloadu_vd_p(s);					\
      b = vloadu_vd_p(t);					\
      a = funcName(a, b);						\
      vstoreu_v_p_vd(s, a);					\
      u = d2u(s[idx]);						\
      printf("%" PRIx64 "\n", u);				\
      fflush(stdout);						\
      continue;							\
    }								\
  }

#define func_d_d_i(funcStr, funcName) {	\
    if (startsWith(buf, funcStr " ")) {	\
      uint64_t u, v;							\
      sscanf(buf, funcStr " %" PRIx64 " %" PRIx64, &u, &v);		\
      double s[VECTLENDP];						\
      int t[VECTLENDP*2];						\
      int i;								\
      for(i=0;i<VECTLENDP;i++) {					\
	s[i] = rand()/(double)RAND_MAX*20000-10000;			\
	t[i] = (int)(rand()/(double)RAND_MAX*20000-10000);		\
      }									\
      int idx = rand() & (VECTLENDP-1);					\
      s[idx] = u2d(u);							\
      t[idx] = (int)u2d(v);						\
      vstoreu_v_p_vd(s, funcName(vloadu_vd_p(s), vloadu_vi_p(t)));	\
      u = d2u(s[idx]);							\
      printf("%" PRIx64 "\n", u);  					\
      fflush(stdout);							\
      continue;								\
    }									\
  }
    
#define func_i_d(funcStr, funcName) {				\
    if (startsWith(buf, funcStr " ")) {				\
      uint64_t u;						\
      int i;							\
      sscanf(buf, funcStr " %" PRIx64, &u);			\
      double s[VECTLENDP];					\
      int t[VECTLENDP*2];					\
      for(i=0;i<VECTLENDP;i++) {				\
	s[i] = rand()/(double)RAND_MAX*20000-10000;		\
      }								\
      int idx = rand() & (VECTLENDP-1);				\
      s[idx] = u2d(u);						\
      vdouble a = vloadu_vd_p(s);				\
      vint vi = funcName(a);					\
      vstoreu_v_p_vi(t, vi);					\
      i = t[idx];						\
      printf("%d\n", i);					\
      fflush(stdout);						\
      continue;							\
    }								\
  }

//

#define func_f_f(funcStr, funcName) {		\
    if (startsWith(buf, funcStr " ")) {		\
      uint32_t u;				\
      sscanf(buf, funcStr " %x", &u);		\
      float s[VECTLENSP];			\
      int i;					\
      for(i=0;i<VECTLENSP;i++) {			\
	s[i] = rand()/(float)RAND_MAX*20000-10000;	\
      }							\
      int idx = rand() & (VECTLENSP-1);			\
      s[idx] = u2f(u);					\
      vfloat a = vloadu_vf_p(s);			\
      a = funcName(a);					\
      vstoreu_v_p_vf(s, a);				\
      u = f2u(s[idx]);						\
      printf("%x\n", u);					\
      fflush(stdout);						\
      continue;							\
    }								\
  }

#define func_f2_f(funcStr, funcName) {		\
    if (startsWith(buf, funcStr " ")) {		\
      uint32_t u;				\
      sscanf(buf, funcStr " %x", &u);		\
      float s[VECTLENSP], t[VECTLENSP];			\
      int i;						\
      for(i=0;i<VECTLENSP;i++) {			\
	s[i] = rand()/(float)RAND_MAX*20000-10000;	\
	t[i] = rand()/(float)RAND_MAX*20000-10000;	\
      }							\
      int idx = rand() & (VECTLENSP-1);			\
      s[idx] = u2f(u);					\
      vfloat2 v;					\
      vfloat a = vloadu_vf_p(s);			\
      v = funcName(a);					\
      vstoreu_v_p_vf(s, v.x);				\
      vstoreu_v_p_vf(t, v.y);				\
      Sleef_float2 d2;					\
      d2.x = s[idx];					\
      d2.y = t[idx];					\
      printf("%x %x\n", f2u(d2.x), f2u(d2.y));		\
      fflush(stdout);						\
      continue;							\
    }								\
  }

#define func_f_f_f(funcStr, funcName) {		\
    if (startsWith(buf, funcStr " ")) {		\
      uint32_t u, v;				\
      sscanf(buf, funcStr " %x %x", &u, &v);	\
      float s[VECTLENSP], t[VECTLENSP];		\
      int i;					\
      for(i=0;i<VECTLENSP;i++) {			\
	s[i] = rand()/(float)RAND_MAX*20000-10000;	\
	t[i] = rand()/(float)RAND_MAX*20000-10000;	\
      }							\
      int idx = rand() & (VECTLENSP-1);			\
      s[idx] = u2f(u);					\
      t[idx] = u2f(v);					\
      vfloat a, b;					\
      a = vloadu_vf_p(s);				\
      b = vloadu_vf_p(t);				\
      a = funcName(a, b);				\
      vstoreu_v_p_vf(s, a);				\
      u = f2u(s[idx]);					\
      printf("%x\n", u);				\
      fflush(stdout);						\
      continue;							\
    }								\
  }

//

#define BUFSIZE 1024

int do_test(int argc, char **argv) {
  srand(time(NULL));

  {
    int k = 0;
#ifdef ENABLE_DP
    k += 1;
#endif
#ifdef ENABLE_SP
    k += 2;
#endif
#if defined(ENABLE_NEON32) || defined(ENABLE_NEON32VFPV4)
    k += 4; // flush to zero
#elif defined(ENABLE_VECEXT)
    if (vcast_f_vf(xpowf(vcast_vf_f(0.5f), vcast_vf_f(140))) == 0) k += 4;
#endif
#if defined(ENABLE_DSP128) || defined(ENABLE_DSP256)
    k += 8; // dispatcher
#endif

    printf("%d\n", k);
    fflush(stdout);
  }

  fprintf(stderr, "IUT : %s\n", (const char *)xgetPtrf(0));
  fflush(stderr);
  
  char buf[BUFSIZE];

  for(;;) {
    if (readln(STDIN_FILENO, buf, BUFSIZE-1) < 1) break;

#ifdef ENABLE_DP
    func_d_d("sin", xsin);
    func_d_d("cos", xcos);
    func_d_d("tan", xtan);
    func_d_d("asin", xasin);
    func_d_d("acos", xacos);
    func_d_d("atan", xatan);
    func_d_d("log", xlog);
    func_d_d("exp", xexp);

    func_d_d("sqrt", xsqrt);
    func_d_d("sqrt_u05", xsqrt_u05);
    func_d_d("sqrt_u35", xsqrt_u35);
    func_d_d("cbrt", xcbrt);
    func_d_d("cbrt_u1", xcbrt_u1);

    func_d_d("sinh", xsinh);
    func_d_d("cosh", xcosh);
    func_d_d("tanh", xtanh);
    func_d_d("sinh_u35", xsinh_u35);
    func_d_d("cosh_u35", xcosh_u35);
    func_d_d("tanh_u35", xtanh_u35);
    func_d_d("asinh", xasinh);
    func_d_d("acosh", xacosh);
    func_d_d("atanh", xatanh);

    func_d_d("sin_u1", xsin_u1);
    func_d_d("cos_u1", xcos_u1);
    func_d_d("tan_u1", xtan_u1);
    func_d_d("sinpi_u05", xsinpi_u05);
    func_d_d("cospi_u05", xcospi_u05);
    func_d_d("asin_u1", xasin_u1);
    func_d_d("acos_u1", xacos_u1);
    func_d_d("atan_u1", xatan_u1);
    func_d_d("log_u1", xlog_u1);

    func_d_d("exp2", xexp2);
    func_d_d("exp10", xexp10);
    func_d_d("expm1", xexpm1);
    func_d_d("log10", xlog10);
    func_d_d("log2", xlog2);
    func_d_d("log1p", xlog1p);

    func_d2_d("sincos", xsincos);
    func_d2_d("sincos_u1", xsincos_u1);
    func_d2_d("sincospi_u35", xsincospi_u35);
    func_d2_d("sincospi_u05", xsincospi_u05);

    func_d_d_d("pow", xpow);
    func_d_d_d("atan2", xatan2);
    func_d_d_d("atan2_u1", xatan2_u1);

    func_d_d_i("ldexp", xldexp);

    func_i_d("ilogb", xilogb);

    func_d_d("fabs", xfabs);
    func_d_d("trunc", xtrunc);
    func_d_d("floor", xfloor);
    func_d_d("ceil", xceil);
    func_d_d("round", xround);
    func_d_d("rint", xrint);
    func_d_d("frfrexp", xfrfrexp);
    func_i_d("expfrexp", xexpfrexp);

    func_d_d_d("hypot_u05", xhypot_u05);
    func_d_d_d("hypot_u35", xhypot_u35);
    func_d_d_d("copysign", xcopysign);
    func_d_d_d("fmax", xfmax);
    func_d_d_d("fmin", xfmin);
    func_d_d_d("fdim", xfdim);
    func_d_d_d("nextafter", xnextafter);
    func_d_d_d("fmod", xfmod);

    func_d2_d("modf", xmodf);

    func_d_d("tgamma_u1", xtgamma_u1);
    func_d_d("lgamma_u1", xlgamma_u1);
    func_d_d("erf_u1", xerf_u1);
    func_d_d("erfc_u15", xerfc_u15);

#if !defined(ENABLE_DSP128) && !defined(ENABLE_DSP256)
    func_d_d("ysin", ysin);
    func_d_d("ycos", ycos);
    func_d_d("ytan", ytan);
    func_d_d("yasin", yasin);
    func_d_d("yacos", yacos);
    func_d_d("yatan", yatan);
    func_d_d("ylog", ylog);
    func_d_d("yexp", yexp);

    func_d_d("ycbrt", ycbrt);
    func_d_d("ycbrt_u1", ycbrt_u1);

    func_d_d("ysinh", ysinh);
    func_d_d("ycosh", ycosh);
    func_d_d("ytanh", ytanh);
    func_d_d("ysinh_u35", ysinh_u35);
    func_d_d("ycosh_u35", ycosh_u35);
    func_d_d("ytanh_u35", ytanh_u35);
    func_d_d("yasinh", yasinh);
    func_d_d("yacosh", yacosh);
    func_d_d("yatanh", yatanh);

    func_d_d("ysin_u1", ysin_u1);
    func_d_d("ycos_u1", ycos_u1);
    func_d_d("ytan_u1", ytan_u1);
    func_d_d("ysinpi_u05", ysinpi_u05);
    func_d_d("ycospi_u05", ycospi_u05);
    func_d_d("yasin_u1", yasin_u1);
    func_d_d("yacos_u1", yacos_u1);
    func_d_d("yatan_u1", yatan_u1);
    func_d_d("ylog_u1", ylog_u1);

    func_d_d("yexp2", yexp2);
    func_d_d("yexp10", yexp10);
    func_d_d("yexpm1", yexpm1);
    func_d_d("ylog10", ylog10);
    func_d_d("ylog2", ylog2);
    func_d_d("ylog1p", ylog1p);

    func_d2_d("ysincos", ysincos);
    func_d2_d("ysincos_u1", ysincos_u1);
    func_d2_d("ysincospi_u35", ysincospi_u35);
    func_d2_d("ysincospi_u05", ysincospi_u05);

    func_d_d_d("ypow", ypow);
    func_d_d_d("yatan2", yatan2);
    func_d_d_d("yatan2_u1", yatan2_u1);

    func_d_d_i("yldexp", yldexp);

    func_i_d("yilogb", yilogb);

    func_d_d("yfabs", yfabs);
    func_d_d("ytrunc", ytrunc);
    func_d_d("yfloor", yfloor);
    func_d_d("yceil", yceil);
    func_d_d("yround", yround);
    func_d_d("yrint", yrint);
    func_d_d("yfrfrexp", yfrfrexp);
    func_i_d("yexpfrexp", yexpfrexp);

    func_d_d_d("yhypot_u05", yhypot_u05);
    func_d_d_d("yhypot_u35", yhypot_u35);
    func_d_d_d("ycopysign", ycopysign);
    func_d_d_d("yfmax", yfmax);
    func_d_d_d("yfmin", yfmin);
    func_d_d_d("yfdim", yfdim);
    func_d_d_d("ynextafter", ynextafter);
    func_d_d_d("yfmod", yfmod);

    func_d2_d("ymodf", ymodf);

    func_d_d("ytgamma_u1", ytgamma_u1);
    func_d_d("ylgamma_u1", ylgamma_u1);
    func_d_d("yerf_u1", yerf_u1);
    func_d_d("yerfc_u15", yerfc_u15);
#endif
#endif

#ifdef ENABLE_SP
    func_f_f("sinf", xsinf);
    func_f_f("cosf", xcosf);
    func_f_f("tanf", xtanf);
    func_f_f("asinf", xasinf);
    func_f_f("acosf", xacosf);
    func_f_f("atanf", xatanf);
    func_f_f("logf", xlogf);
    func_f_f("expf", xexpf);

    func_f_f("sqrtf", xsqrtf);
    func_f_f("sqrtf_u05", xsqrtf_u05);
    func_f_f("sqrtf_u35", xsqrtf_u35);
    func_f_f("cbrtf", xcbrtf);
    func_f_f("cbrtf_u1", xcbrtf_u1);

    func_f_f("sinhf", xsinhf);
    func_f_f("coshf", xcoshf);
    func_f_f("tanhf", xtanhf);
    func_f_f("sinhf_u35", xsinhf_u35);
    func_f_f("coshf_u35", xcoshf_u35);
    func_f_f("tanhf_u35", xtanhf_u35);
    func_f_f("asinhf", xasinhf);
    func_f_f("acoshf", xacoshf);
    func_f_f("atanhf", xatanhf);

    func_f_f("sinf_u1", xsinf_u1);
    func_f_f("cosf_u1", xcosf_u1);
    func_f_f("tanf_u1", xtanf_u1);
    func_f_f("sinpif_u05", xsinpif_u05);
    func_f_f("cospif_u05", xcospif_u05);
    func_f_f("asinf_u1", xasinf_u1);
    func_f_f("acosf_u1", xacosf_u1);
    func_f_f("atanf_u1", xatanf_u1);
    func_f_f("logf_u1", xlogf_u1);

    func_f_f("exp2f", xexp2f);
    func_f_f("exp10f", xexp10f);
    func_f_f("expm1f", xexpm1f);
    func_f_f("log10f", xlog10f);
    func_f_f("log2f", xlog2f);
    func_f_f("log1pf", xlog1pf);

    func_f2_f("sincosf", xsincosf);
    func_f2_f("sincosf_u1", xsincosf_u1);
    func_f2_f("sincospif_u35", xsincospif_u35);
    func_f2_f("sincospif_u05", xsincospif_u05);

    func_f_f_f("powf", xpowf);
    func_f_f_f("atan2f", xatan2f);
    func_f_f_f("atan2f_u1", xatan2f_u1);

    func_f_f("fabsf", xfabsf);
    func_f_f("truncf", xtruncf);
    func_f_f("floorf", xfloorf);
    func_f_f("ceilf", xceilf);
    func_f_f("roundf", xroundf);
    func_f_f("rintf", xrintf);
    func_f_f("frfrexpf", xfrfrexpf);

    func_f_f_f("hypotf_u05", xhypotf_u05);
    func_f_f_f("hypotf_u35", xhypotf_u35);
    func_f_f_f("copysignf", xcopysignf);
    func_f_f_f("fmaxf", xfmaxf);
    func_f_f_f("fminf", xfminf);
    func_f_f_f("fdimf", xfdimf);
    func_f_f_f("nextafterf", xnextafterf);
    func_f_f_f("fmodf", xfmodf);

    func_f2_f("modff", xmodff);

    func_f_f("tgammaf_u1", xtgammaf_u1);
    func_f_f("lgammaf_u1", xlgammaf_u1);
    func_f_f("erff_u1", xerff_u1);
    func_f_f("erfcf_u15", xerfcf_u15);

#if !defined(ENABLE_DSP128) && !defined(ENABLE_DSP256)
    func_f_f("ysinf", ysinf);
    func_f_f("ycosf", ycosf);
    func_f_f("ytanf", ytanf);
    func_f_f("yasinf", yasinf);
    func_f_f("yacosf", yacosf);
    func_f_f("yatanf", yatanf);
    func_f_f("ylogf", ylogf);
    func_f_f("yexpf", yexpf);

    func_f_f("ycbrtf", ycbrtf);
    func_f_f("ycbrtf_u1", ycbrtf_u1);

    func_f_f("ysinhf", ysinhf);
    func_f_f("ycoshf", ycoshf);
    func_f_f("ytanhf", ytanhf);
    func_f_f("ysinhf_u35", ysinhf_u35);
    func_f_f("ycoshf_u35", ycoshf_u35);
    func_f_f("ytanhf_u35", ytanhf_u35);
    func_f_f("yasinhf", yasinhf);
    func_f_f("yacoshf", yacoshf);
    func_f_f("yatanhf", yatanhf);

    func_f_f("ysinf_u1", ysinf_u1);
    func_f_f("ycosf_u1", ycosf_u1);
    func_f_f("ytanf_u1", ytanf_u1);
    func_f_f("ysinpif_u05", ysinpif_u05);
    func_f_f("ycospif_u05", ycospif_u05);
    func_f_f("yasinf_u1", yasinf_u1);
    func_f_f("yacosf_u1", yacosf_u1);
    func_f_f("yatanf_u1", yatanf_u1);
    func_f_f("ylogf_u1", ylogf_u1);

    func_f_f("yexp2f", yexp2f);
    func_f_f("yexp10f", yexp10f);
    func_f_f("yexpm1f", yexpm1f);
    func_f_f("ylog10f", ylog10f);
    func_f_f("ylog2f", ylog2f);
    func_f_f("ylog1pf", ylog1pf);

    func_f2_f("ysincosf", ysincosf);
    func_f2_f("ysincosf_u1", ysincosf_u1);
    func_f2_f("ysincospif_u35", ysincospif_u35);
    func_f2_f("ysincospif_u05", ysincospif_u05);

    func_f_f_f("ypowf", ypowf);
    func_f_f_f("yatan2f", yatan2f);
    func_f_f_f("yatan2f_u1", yatan2f_u1);

    func_f_f("yfabsf", yfabsf);
    func_f_f("ytruncf", ytruncf);
    func_f_f("yfloorf", yfloorf);
    func_f_f("yceilf", yceilf);
    func_f_f("yroundf", yroundf);
    func_f_f("yrintf", yrintf);
    func_f_f("yfrfrexpf", yfrfrexpf);

    func_f_f_f("yhypotf_u05", yhypotf_u05);
    func_f_f_f("yhypotf_u35", yhypotf_u35);
    func_f_f_f("ycopysignf", ycopysignf);
    func_f_f_f("yfmaxf", yfmaxf);
    func_f_f_f("yfminf", yfminf);
    func_f_f_f("yfdimf", yfdimf);
    func_f_f_f("ynextafterf", ynextafterf);
    func_f_f_f("yfmodf", yfmodf);

    func_f2_f("ymodff", ymodff);

    func_f_f("ytgammaf_u1", ytgammaf_u1);
    func_f_f("ylgammaf_u1", ylgammaf_u1);
    func_f_f("yerff_u1", yerff_u1);
    func_f_f("yerfcf_u15", yerfcf_u15);
#endif
#endif
  }

  return 0;
}
