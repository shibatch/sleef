#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(ENABLE_SSE2) || defined(ENABLE_SSE4)

#ifdef ENABLE_SSE2
#define NAME "SSE2"
#define CONFIG 2
#endif

#ifdef ENABLE_SSE4
#define NAME "SSE4"
#define CONFIG 4
#endif

#include "helpersse2.h"
#define vacos_finite _ZGVbN2v___acos_finite
#define vasin_finite _ZGVbN2v___asin_finite

#define vacosf_finite _ZGVbN4v___acosf_finite
#define vasinf_finite _ZGVbN4v___asinf_finite

#endif /* defined(ENABLE_SSE2) || defined(ENABLE_SSE4) */

#ifdef ENABLE_AVX
#define NAME "AVX"
#define CONFIG 1
#include "helperavx.h"
#define vacos_finite _ZGVcN4v___acos_finite
#define vasin_finite _ZGVcN4v___asin_finite

#define vacosf_finite _ZGVcN8v___acosf_finite
#define vasinf_finite _ZGVcN8v___asinf_finite
#endif /* ENABLE_AVX */

#ifdef ENABLE_AVX2
#define NAME "AVX2"
#define CONFIG 1
#include "helperavx2.h"
#define vacos_finite _ZGVdN4v___acos_finite
#define vasin_finite _ZGVdN4v___asin_finite

#define vacosf_finite _ZGVdN8v___acosf_finite
#define vasinf_finite _ZGVdN8v___asinf_finite
#endif /* ENABLE_AVX2 */

#ifdef ENABLE_AVX512F
#define NAME "AVX512F"
#define CONFIG 1
#include "helperavx512f.h"
#define vacos_finite _ZGVeN8v___acos_finite
#define vasin_finite _ZGVeN8v___asin_finite

#define vacosf_finite _ZGVeN16v___acosf_finite
#define vasinf_finite _ZGVeN16v___asinf_finite
#endif /* ENABLE_AVX512F */

#ifdef NAME

extern vdouble vacos_finite(vdouble);
extern vdouble vasin_finite(vdouble);

extern vfloat vacosf_finite(vfloat);
extern vfloat vasinf_finite(vfloat);

void check_featureDP() {
#ifdef ENABLE_DP
  double s[VECTLENDP];
  int i;
  for (i = 0; i < VECTLENDP; i++) {
    s[i] = 1.0;
  }
  vdouble a = vloadu_vd_p(s);
  a = vacos_finite(a);
  vstoreu_v_p_vd(s, a);
#endif
}

void check_featureSP() {
#ifdef ENABLE_SP
  float s[VECTLENSP];
  int i;
  for (i = 0; i < VECTLENSP; i++) {
    s[i] = 1.0;
  }
  vfloat a = vloadu_vf_p(s);
  a = vacosf_finite(a);
  vstoreu_v_p_vf(s, a);
#endif
}

static jmp_buf sigjmp;

static void sighandler(int signum) { longjmp(sigjmp, 1); }

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

int main(void) {
  if (!detectFeatureDP() || !detectFeatureSP()) {
    printf("%s is not available on this machine.\n", NAME);
    return 0;
  }

#ifdef ENABLE_DP
  {
    double in[VECTLENDP], out[VECTLENDP];
    vdouble vin, vout;
    int i;

    for (i = 0; i < VECTLENDP; ++i)
      in[i] = i;

    vin = vload_vd_p(in);
    vout = vacos_finite(vin);
    vout = vasin_finite(vout);
    vstore_v_p_vd(out, vout);

    printf("DP: %s... ", NAME);
    for (i = 0; i < VECTLENDP; ++i)
      printf("%g ", out[i]);
    printf(" ...passed. \n");
  }
#endif

#ifdef ENABLE_SP
  {
    float in[VECTLENSP], out[VECTLENSP];
    vfloat vin, vout;
    int i;

    for (i = 0; i < VECTLENSP; ++i)
      in[i] = i;

    vin = vload_vf_p(in);
    vout = vacosf_finite(vin);
    vout = vasinf_finite(vout);
    vstore_v_p_vf(out, vout);

    printf("SP: %s... ", NAME);
    for (i = 0; i < VECTLENSP; ++i)
      printf("%g ", out[i]);
    printf(" ...passed. \n");
  }
#endif

  return 0;
}

#else /* #ifdef NAME */

int main(void) {
  printf("Nothing here.\n");
  return 0;
}

#endif /* #ifdef NAME */
