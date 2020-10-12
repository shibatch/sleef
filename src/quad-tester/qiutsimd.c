//   Copyright Naoki Shibata and contributors 2010 - 2020.
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
#define VARGQUAD Sleef_quad
#endif

#ifdef ENABLE_PURECFMA_SCALAR
#define CONFIG 2
#include "helperpurec_scalar.h"
#include "qrenamepurecfma_scalar.h"
#define VARGQUAD Sleef_quad
#endif

#ifdef ENABLE_SSE2
#define CONFIG 2
#include "helpersse2.h"
#include "qrenamesse2.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifdef ENABLE_AVX2128
#define CONFIG 1
#include "helperavx2_128.h"
#include "qrenameavx2128.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifdef ENABLE_AVX
#define CONFIG 1
#include "helperavx.h"
#include "qrenameavx.h"
#define VARGQUAD Sleef_quadx4
#endif

#ifdef ENABLE_FMA4
#define CONFIG 4
#include "helperavx.h"
#include "qrenamefma4.h"
#define VARGQUAD Sleef_quadx4
#endif

#ifdef ENABLE_AVX2
#define CONFIG 1
#include "helperavx2.h"
#include "qrenameavx2.h"
#define VARGQUAD Sleef_quadx4
#endif

#ifdef ENABLE_AVX512F
#define CONFIG 1
#include "helperavx512f.h"
#include "qrenameavx512f.h"
#define VARGQUAD Sleef_quadx8
#endif

#ifdef ENABLE_ADVSIMD
#define CONFIG 1
#include "helperadvsimd.h"
#include "qrenameadvsimd.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifdef ENABLE_SVE
#define CONFIG 1
#include "helpersve.h"
#include "qrenamesve.h"
#define VARGQUAD Sleef_svquad
#define SIZEOF_VARGQUAD (svcntd()*8)
#endif

#ifdef ENABLE_VSX
#define CONFIG 1
#include "helperpower_128.h"
#include "qrenamevsx.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifdef ENABLE_ZVECTOR2
#define CONFIG 140
#include "helpers390x_128.h"
#include "qrenamezvector2.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifdef ENABLE_DSP128
#define CONFIG 2
#include "helpersse2.h"
#include "qrenamedsp128.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifdef ENABLE_DSP256
#define CONFIG 1
#include "helperavx.h"
#include "qrenamedsp256.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifndef SIZEOF_VARGQUAD
#define SIZEOF_VARGQUAD sizeof(VARGQUAD)
#endif

//

int check_featureQP() {
  if (vavailability_i(1) == 0) return 0;
  VARGQUAD a;
  memrand(&a, SIZEOF_VARGQUAD);
  a = xsqrtq_u05(a);
  return 1;
}

//

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
typedef union {
  Sleef_quad q;
  struct {
    uint64_t h, l;
  };
} cnv128;
#else
typedef union {
  Sleef_quad q;
  struct {
    uint64_t l, h;
  };
} cnv128;
#endif

#define BUFSIZE 1024

#define func_q_q(funcStr, funcName) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      int lane = xrand() % VECTLENDP;					\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      VARGQUAD a0;							\
      memrand(&a0, SIZEOF_VARGQUAD);					\
      a0 = xsetq(a0, lane, c0.q);					\
      a0 = funcName(a0);						\
      c0.q = xgetq(a0, lane);						\
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
      VARGQUAD a0, a1;							\
      memrand(&a0, SIZEOF_VARGQUAD);					\
      memrand(&a1, SIZEOF_VARGQUAD);					\
      a0 = xsetq(a0, lane, c0.q);					\
      a1 = xsetq(a1, lane, c1.q);					\
      a0 = funcName(a0, a1);						\
      c0.q = xgetq(a0, lane);						\
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
      VARGQUAD a0, a1;							\
      memrand(&a0, SIZEOF_VARGQUAD);					\
      memrand(&a1, SIZEOF_VARGQUAD);					\
      a0 = xsetq(a0, lane, c0.q);					\
      a1 = xsetq(a1, lane, c1.q);					\
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
      VARGQUAD a0;							\
      memrand(&a0, SIZEOF_VARGQUAD);					\
      a0 = xsetq(a0, lane, c0.q);					\
      double d[VECTLENDP];						\
      vstoreu_v_p_vd(d, funcName(a0));					\
      printf("%" PRIx64 "\n", d2u(d[lane]));				\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_q_d(funcStr, funcName) {			\
    while (startsWith(buf, funcStr " ")) {		\
      sentinel = 0;					\
      int lane = xrand() % VECTLENDP;			\
      uint64_t u;					\
      sscanf(buf, funcStr " %" PRIx64, &u);		\
      double s[VECTLENDP];				\
      memrand(s, sizeof(s));				\
      s[lane] = u2d(u);					\
      VARGQUAD a0 = funcName(vloadu_vd_p(s));		\
      cnv128 c0;					\
      c0.q = xgetq(a0, lane);				\
      printf("%" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l);	\
      fflush(stdout);					\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;	\
    }							\
  }

#define func_i64_q(funcStr, funcName) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      int lane = xrand() % VECTLENDP;					\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      VARGQUAD a0;							\
      memrand(&a0, SIZEOF_VARGQUAD);					\
      a0 = xsetq(a0, lane, c0.q);					\
      double d[VECTLENDP];						\
      vstoreu_v_p_vd(d, vreinterpret_vd_vm(vreinterpret_vm_vi64(funcName(a0)))); \
      printf("%" PRIx64 "\n", d2u(d[lane]));				\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_q_i64(funcStr, funcName) {			\
    while (startsWith(buf, funcStr " ")) {		\
      sentinel = 0;					\
      int lane = xrand() % VECTLENDP;			\
      uint64_t u;					\
      sscanf(buf, funcStr " %" PRIx64, &u);		\
      double s[VECTLENDP];				\
      memrand(s, sizeof(s));				\
      s[lane] = u2d(u);					\
      VARGQUAD a0 = funcName(vreinterpret_vi64_vm(vreinterpret_vm_vd(vloadu_vd_p(s))));	\
      cnv128 c0;					\
      c0.q = xgetq(a0, lane);				\
      printf("%" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l);	\
      fflush(stdout);					\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;	\
    }							\
  }

#define func_u64_q(funcStr, funcName) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      int lane = xrand() % VECTLENDP;					\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      VARGQUAD a0;							\
      memrand(&a0, SIZEOF_VARGQUAD);					\
      a0 = xsetq(a0, lane, c0.q);					\
      double d[VECTLENDP];						\
      vstoreu_v_p_vd(d, vreinterpret_vd_vm(vreinterpret_vm_vu64(funcName(a0)))); \
      printf("%" PRIx64 "\n", d2u(d[lane]));				\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_q_u64(funcStr, funcName) {			\
    while (startsWith(buf, funcStr " ")) {		\
      sentinel = 0;					\
      int lane = xrand() % VECTLENDP;			\
      uint64_t u;					\
      sscanf(buf, funcStr " %" PRIx64, &u);		\
      double s[VECTLENDP];				\
      memrand(s, sizeof(s));				\
      s[lane] = u2d(u);					\
      VARGQUAD a0 = funcName(vreinterpret_vu64_vm(vreinterpret_vm_vd(vloadu_vd_p(s))));	\
      cnv128 c0;					\
      c0.q = xgetq(a0, lane);				\
      printf("%" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l);	\
      fflush(stdout);					\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;	\
    }							\
  }

#define func_strtoq(funcStr) {						\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      char s[64];							\
      sscanf(buf, funcStr " %63s", s);					\
      Sleef_quad a0;							\
      a0 = Sleef_strtoq(s, NULL);					\
      cnv128 c0;							\
      c0.q = a0;							\
      printf("%" PRIx64 ":%" PRIx64 "\n", c0.h, c0.l);			\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#if !(defined(ENABLEFLOAT128) && defined(__clang__))
#define func_snprintf_40Qg(funcStr) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      Sleef_quad a0;							\
      a0 = c0.q;							\
      char s[64];							\
      Sleef_snprintf(s, 63, "%.40Qg", a0);				\
      printf("%s\n", s);						\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_snprintf_Qa(funcStr) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      Sleef_quad a0;							\
      a0 = c0.q;							\
      char s[64];							\
      Sleef_snprintf(s, 63, "%Qa", a0);					\
      printf("%s\n", s);						\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }
#else
#define func_snprintf_40Qg(funcStr) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      Sleef_quad a0;							\
      a0 = c0.q;							\
      char s[64];							\
      Sleef_snprintf(s, 63, "%.40Pg", &a0);				\
      printf("%s\n", s);						\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }

#define func_snprintf_Qa(funcStr) {					\
    while (startsWith(buf, funcStr " ")) {				\
      sentinel = 0;							\
      cnv128 c0;							\
      sscanf(buf, funcStr " %" PRIx64 ":%" PRIx64, &c0.h, &c0.l);	\
      Sleef_quad a0;							\
      a0 = c0.q;							\
      char s[64];							\
      Sleef_snprintf(s, 63, "%Pa", &a0);				\
      printf("%s\n", s);						\
      fflush(stdout);							\
      if (fgets(buf, BUFSIZE-1, stdin) == NULL) break;			\
    }									\
  }
#endif

int do_test(int argc, char **argv) {
  xsrand(time(NULL));

  {
    int k = 0;
    k += 1;
#if defined(ENABLE_PUREC_SCALAR)
    k += 2; // Enable string testing
#endif
    printf("%d\n", k);
    fflush(stdout);
  }
  
  // Do simple testing on splat, select and SLEEF_Q
  {
    VARGQUAD v0 = xsplatq(SLEEF_Q(+0x1921fb54442d1LL, 0x8469898cc51701b8ULL, 1));
    VARGQUAD v1;
    memset(&v1, 0, sizeof(v1));
    v1 = xsetq(v1, 1, SLEEF_Q(+0x15bf0a8b14576LL, 0x95355fb8ac404e7aULL, 1));
    v1 = xmulq_u05(v0, v1);

    vint vi = xicmpeqq(v1, xsplatq(SLEEF_Q(+0x1114580b45d47LL, 0x49e6108579a2d0caULL, 3)));
    int t[VECTLENDP*2];
    memset(t, 0, sizeof(t));
    vstoreu_v_p_vi(t, vi);

    if (!(t[0] == 0 && t[1] == 1)) {
      fprintf(stderr, "Basic testing on splat and select failed\n");
      exit(-1);
    }
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
    func_q_i64("cast_from_int64q", xcast_from_int64q);
    func_i64_q("cast_to_int64q", xcast_to_int64q);
    func_q_u64("cast_from_uint64q", xcast_from_uint64q);
    func_u64_q("cast_to_uint64q", xcast_to_uint64q);
    func_i_q_q("icmpltq", xicmpltq);
    func_i_q_q("icmpgtq", xicmpgtq);
    func_i_q_q("icmpleq", xicmpleq);
    func_i_q_q("icmpgeq", xicmpgeq);
    func_i_q_q("icmpeqq", xicmpeqq);
    func_i_q_q("icmpneq", xicmpneq);
    func_i_q_q("icmpq", xicmpq);
    func_i_q_q("iunordq", xiunordq);
    func_strtoq("strtoq");
    func_snprintf_40Qg("snprintf_40Qg");
    func_snprintf_Qa("snprintf_Qa");
    sentinel++;
  }

  //

  return 0;
}
