#include "check_feature.h"
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>

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

#ifdef ENABLE_DSP256
#define CONFIG 1
#include "helperavx.h"
#include "renamedsp256.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif



#ifdef ENABLE_DP
void check_featureDP() {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = 1.0;
  }
  vdouble a = vloadu_vd_p(s);
  a = xpow(a, a);
  vstoreu_v_p_vd(s, a);
}
#else
void check_featureDP() {
}
#endif

#ifdef ENABLE_SP
void check_featureSP() {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = 1.0;
  }
  vfloat a = vloadu_vf_p(s);
  a = xpowf(a, a);
  vstoreu_v_p_vf(s, a);
}
#else
void check_featureSP() {
}
#endif

static jmp_buf sigjmp;

static void sighandler(int signum) {
  longjmp(sigjmp, 1);
}

int detectFeatureDP() {
  signal(SIGILL, sighandler);

  if (setjmp(sigjmp) == 0) {
    check_featureDP();
    signal(SIGILL, SIG_DFL);
    return 1;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}

int detectFeatureSP() {
  signal(SIGILL, sighandler);

  if (setjmp(sigjmp) == 0) {
    check_featureSP();
    signal(SIGILL, SIG_DFL);
    return 1;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}
