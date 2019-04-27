//          Copyright Naoki Shibata 2010 - 2019.
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
#include "sleefquad.h"
#include "qtesterutil.h"

//

#ifdef ENABLE_PUREC_SCALAR
#define CONFIG 1
#include "helperpurec_scalar.h"
#include "qrenamepurec_scalar.h"
#endif

#ifdef ENABLE_PURECFMA_SCALAR
#define CONFIG 2
#include "helperpurec_scalar.h"
#include "qrenamepurecfma_scalar.h"
#endif

#ifdef ENABLE_SSE2
#define CONFIG 2
#include "helpersse2.h"
#include "qrenamesse2.h"
#endif

#ifdef ENABLE_AVX2128
#define CONFIG 1
#include "helperavx2_128.h"
#include "qrenameavx2128.h"
#endif

#ifdef ENABLE_AVX
#define CONFIG 1
#include "helperavx.h"
#include "qrenameavx.h"
#endif

#ifdef ENABLE_FMA4
#define CONFIG 4
#include "helperavx.h"
#include "qrenamefma4.h"
#endif

#ifdef ENABLE_AVX2
#define CONFIG 1
#include "helperavx2.h"
#include "qrenameavx2.h"
#endif

#ifdef ENABLE_AVX512F
#define CONFIG 1
#include "helperavx512f.h"
#include "qrenameavx512f.h"
#endif

#ifdef ENABLE_ADVSIMD
#define CONFIG 1
#include "helperadvsimd.h"
#include "qrenameadvsimd.h"
#endif

#ifdef ENABLE_SVE
#define CONFIG 1
#include "helpersve.h"
#include "qrenamesve.h"
#endif

#ifdef ENABLE_VSX
#define CONFIG 1
#include "helperpower_128.h"
#include "qrenamevsx.h"
#endif

#ifdef ENABLE_DSP128
#define CONFIG 2
#include "helpersse2.h"
#include "qrenamedsp128.h"
#endif

#ifdef ENABLE_DSP256
#define CONFIG 1
#include "helperavx.h"
#include "qrenamedsp256.h"
#endif

//

int check_featureQP() {
  if (vavailability_i(1) == 0) return 0;
  vargquad a;
  memrand(&a, sizeof(vargquad));
  a = xsqrtq_u05(a);
  return 1;
}

//

typedef union {
  Sleef_quad q;
  struct {
    uint64_t l, h;
  };
} cnv128;

#define BUFSIZE 1024

#define func_q_q(funcStr, funcName) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      int lane = xrand() % VECTLENDP;					\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      vargquad a0;							\
      memrand(&a0, sizeof(vargquad));					\
      a0.s[lane] = c0.q;						\
      a0 = funcName(a0);						\
      c0.q = a0.s[lane];						\
      printf("%" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l);			\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_q_q_q(funcStr, funcName) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      int lane = xrand() % VECTLENDP;					\
      cnv128 c0, c1;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64 " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l, &c1.h, &c1.l); \
      vargquad a0, a1;							\
      memrand(&a0, sizeof(vargquad));					\
      memrand(&a1, sizeof(vargquad));					\
      a0.s[lane] = c0.q;						\
      a1.s[lane] = c1.q;						\
      a0 = funcName(a0, a1);						\
      c0.q = a0.s[lane];						\
      printf("%" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l);			\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_i_q_q(funcStr, funcName) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      int lane = xrand() % VECTLENDP;					\
      cnv128 c0, c1;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64 " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l, &c1.h, &c1.l); \
      vargquad a0, a1;							\
      memrand(&a0, sizeof(vargquad));					\
      memrand(&a1, sizeof(vargquad));					\
      a0.s[lane] = c0.q;						\
      a1.s[lane] = c1.q;						\
      vint vi = funcName(a0, a1);					\
      int t[VECTLENDP*2];						\
      vstoreu_v_p_vi(t, vi);						\
      printf("%d\n", t[lane]);						\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_d_q(funcStr, funcName) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      int lane = xrand() % VECTLENDP;					\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      vargquad a0;							\
      memrand(&a0, sizeof(vargquad));					\
      a0.s[lane] = c0.q;						\
      double d[VECTLENDP];						\
      vstoreu_v_p_vd(d, funcName(a0));					\
      printf("%" PRIx64 "\n", d2u(d[lane]));				\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_q_d(funcStr, funcName) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      int lane = xrand() % VECTLENDP;					\
      uint64_t u;							\
      sscanf(buf, funcStr " %" PRIx64, &u);				\
      double s[VECTLENDP];						\
      memrand(s, sizeof(s));						\
      s[lane] = u2d(u);							\
      vargquad a0 = funcName(vloadu_vd_p(s));				\
      cnv128 c0;							\
      c0.q = a0.s[lane];						\
      printf("%" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l);			\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_strtoq(funcStr) {						\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      char s[64];							\
      sscanf(buf, funcStr " %63s", s);					\
      Sleef_quad1 a0;							\
      a0 = Sleef_strtoq(s, NULL, 10);					\
      cnv128 c0;							\
      c0.q = a0.s[0];							\
      printf("%" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l);			\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_qtostr(funcStr) {						\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      Sleef_quad1 a0;							\
      a0.s[0] = c0.q;							\
      char s[64];							\
      Sleef_qtostr(s, 63, a0, 10);					\
      printf("%s\n", s);						\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

int do_test(int argc, char **argv) {
  xsrand(time(NULL));

  {
    int k = 0;
    k += 1;
#ifdef ENABLE_PUREC_SCALAR
    k += 2; // Enable string testing
#endif
    printf("%d\n", k);
    fflush(stdout);
  }
  
  char buf[BUFSIZE];
  fgets(buf, BUFSIZE-1, stdin);
  int sentinel = 0;

  while(!feof(stdin) && sentinel < 2) {
    func_q_q_q("addq_u05", xaddq_u05);
    func_q_q_q("subq_u05", xsubq_u05);
    func_q_q_q("mulq_u05", xmulq_u05);
    func_q_q_q("divq_u05", xdivq_u05);
    func_q_q("sqrtq_u05", xsqrtq_u05);
    func_q_q("sinq_u10", xsinq_u10);
    func_q_q("cosq_u10", xcosq_u10);
    func_q_q("tanq_u10", xtanq_u10);
    func_q_q("asinq_u10", xasinq_u10);
    func_q_q("acosq_u10", xacosq_u10);
    func_q_q("atanq_u10", xatanq_u10);
    func_q_q("expq_u10", xexpq_u10);
    func_q_q("exp2q_u10", xexp2q_u10);
    func_q_q("exp10q_u10", xexp10q_u10);
    func_q_q("expm1q_u10", xexpm1q_u10);
    func_q_q("logq_u10", xlogq_u10);
    func_q_q("log2q_u10", xlog2q_u10);
    func_q_q("log10q_u10", xlog10q_u10);
    func_q_q("log1pq_u10", xlog1pq_u10);
    func_q_q("negq", xnegq);
    func_q_d("cast_from_doubleq", xcast_from_doubleq);
    func_d_q("cast_to_doubleq", xcast_to_doubleq);
    func_i_q_q("cmpltq", xcmpltq);
    func_i_q_q("cmpgtq", xcmpgtq);
    func_i_q_q("cmpleq", xcmpleq);
    func_i_q_q("cmpgeq", xcmpgeq);
    func_i_q_q("cmpeqq", xcmpeqq);
    func_i_q_q("cmpneqq", xcmpneqq);
    func_i_q_q("unordq", xunordq);
    func_strtoq("strtoq");
    func_qtostr("qtostr");
    sentinel++;
  }

  return 0;
}
