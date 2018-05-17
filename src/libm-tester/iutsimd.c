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
#include <math.h>
#include <assert.h>

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
#ifdef DORENAME
#include "renamesve.h"
typedef Sleef_svfloat64_t_2 vdouble2;
typedef Sleef_svfloat32_t_2 vfloat2;
#endif
#endif

#ifdef ENABLE_DSP256
#define CONFIG 1
#include "helperavx.h"
#include "renamedsp256.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
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
#endif
  }

  return 0;
}
