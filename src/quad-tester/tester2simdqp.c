//   Copyright Naoki Shibata and contributors 2010 - 2020.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpfr.h>
#include <time.h>
#include <float.h>
#include <limits.h>

#include <unistd.h>

#include <math.h>

#include "sleef.h"
#include "sleefquad.h"

#include "misc.h"
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
#endif

#ifdef ENABLE_VSX
#define CONFIG 1
#include "helperpower_128.h"
#include "qrenamevsx.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifdef ENABLE_VSX3
#define CONFIG 3
#include "helperpower_128.h"
#include "qrenamevsx3.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifdef ENABLE_VXE
#define CONFIG 140
#include "helpers390x_128.h"
#include "qrenamevxe.h"
#define VARGQUAD Sleef_quadx2
#endif

#ifdef ENABLE_VXE2
#define CONFIG 150
#include "helpers390x_128.h"
#include "qrenamevxe2.h"
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
#define VARGQUAD Sleef_quadx4
#endif

//

#define DENORMAL_DBL_MIN (4.9406564584124654418e-324)

#define POSITIVE_INFINITY INFINITY
#define NEGATIVE_INFINITY (-INFINITY)

typedef union {
  Sleef_quad q;
  xuint128 x;
  struct {
    uint64_t l, h;
  };
} cnv_t;

Sleef_quad nexttoward0q(Sleef_quad x, int n) {
  cnv_t cx;
  cx.q = x;
  cx.x = add128(cx.x, xu(n < 0 ? 0 : -1, -(int64_t)n));
  return cx.q;
}

static VARGQUAD vset(VARGQUAD v, int idx, Sleef_quad d) { return xsetq(v, idx, d); }
static Sleef_quad vget(VARGQUAD v, int idx) { return xgetq(v, idx); }

vdouble vsetd(vdouble v, int idx, double d) {
  double a[VECTLENDP];
  vstoreu_v_p_vd(a, v);
  a[idx] = d;
  return vloadu_vd_p(a);
}

double vgetd(vdouble v, int idx) {
  double a[VECTLENDP];
  vstoreu_v_p_vd(a, v);
  return a[idx];
}

vmask vsetm(vmask v, int idx, uint64_t d) {
  uint64_t a[VECTLENDP];
  vstoreu_v_p_vd((double *)a, vreinterpret_vd_vm(v));
  a[idx] = d;
  return vreinterpret_vm_vd(vloadu_vd_p((double *)a));
}

int64_t vgeti64(vint64 v, int idx) {
  int64_t a[VECTLENDP];
  vstoreu_v_p_vd((double *)a, vreinterpret_vd_vm(vreinterpret_vm_vi64(v)));
  return a[idx];
}

uint64_t vgetu64(vuint64 v, int idx) {
  uint64_t a[VECTLENDP];
  vstoreu_v_p_vd((double *)a, vreinterpret_vd_vm(vreinterpret_vm_vu64(v)));
  return a[idx];
}

static int vgeti(vint v, int idx) {
  int a[VECTLENDP*2];
  vstoreu_v_p_vi(a, v);
  return a[idx];
}

int main(int argc,char **argv)
{
  mpfr_set_default_prec(1024);
  xsrand(time(NULL) + (((int)getpid()) << 12));
  srandom(time(NULL) + (((int)getpid()) << 12));

  //

  const Sleef_quad oneQ = cast_q_str("1");
  const Sleef_quad oneEMinus300Q = cast_q_str("1e-300");
  const Sleef_quad oneEMinus10Q  = cast_q_str("1e-10");
  const Sleef_quad oneEPlus10Q   = cast_q_str("1e+10");
  const Sleef_quad oneEMinus100Q = cast_q_str("1e-100");
  const Sleef_quad oneEPlus100Q  = cast_q_str("1e+100");
  const Sleef_quad oneEMinus1000Q = cast_q_str("1e-1000");
  const Sleef_quad oneEPlus1000Q  = cast_q_str("1e+1000");
  const Sleef_quad quadMin = cast_q_str("3.36210314311209350626267781732175260e-4932");
  const Sleef_quad quadMax = cast_q_str("1.18973149535723176508575932662800702e+4932");
  const Sleef_quad quadDenormMin = cast_q_str("6.475175119438025110924438958227646552e-4966");
  const Sleef_quad M_PI_2Q  = cast_q_str("1.5707963267948966192313216916397514");

  //

  int cnt, ecnt = 0;
  VARGQUAD a0, a1, a2, a3;
  vdouble vd0 = vcast_vd_d(0), vd1, vd2, vd3;
  Sleef_quad q0, q1, q2, q3, t;
  mpfr_t frw, frx, fry, frz;
  mpfr_inits(frw, frx, fry, frz, NULL);

#ifndef ENABLE_SVE
  memset(&a0, 0, sizeof(a0));
  memset(&a1, 0, sizeof(a1));
  memset(&a2, 0, sizeof(a2));
  memset(&a3, 0, sizeof(a3));
#endif

  for(cnt = 0;ecnt < 1000;cnt++) {
    int e = cnt % VECTLENDP;

    // In the following switch-case statement, I am trying to test
    // with numbers that tends to trigger bugs. Each case is executed
    // once in 128 times of loop execution.
    switch(cnt & 127) {
    case 127:
      q0 = nexttoward0q(quadMin, (xrand() & 63) - 31);
      q1 = rndf128x();
      break;
    case 126:
      q0 = nexttoward0q(quadMax, (xrand() & 31));
      q1 = rndf128x();
      break;
    case 125:
      q0 = nexttoward0q(quadDenormMin, -(int)(xrand() & 31));
      q1 = rndf128x();
      break;
#if defined(ENABLEFLOAT128)
#define SLEEF_QUAD_MIN 3.36210314311209350626267781732175260e-4932Q
#define SLEEF_QUAD_MAX 1.18973149535723176508575932662800702e+4932Q
    case 124:
      q0 = rndf128x();
      q1 = rndf128x();
      q1 += q0;
      break;
    case 123:
      q0 = rndf128x();
      q1 = rndf128x();
      q1 -= q0;
      break;
    case 122:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 += q1;
      break;
    case 121:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 -= q1;
      break;
    case 120:
      q0 = rndf128x();
      q1 = rndf128x();
      q1 += 1;
      break;
    case 119:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 += 1;
      break;
    case 118:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 += 1;
      q1 -= 1;
      break;
    case 117:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 -= 1;
      q1 += 1;
      break;
    case 116:
      q0 = rndf128x();
      q1 = rndf128x();
      q1 += copysign(1, q1) * SLEEF_QUAD_MIN;
      break;
    case 115:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 += copysign(1, q0) * SLEEF_QUAD_MIN;
      break;
    case 114:
      q0 = rndf128x();
      q1 = rndf128x();
      q1 -= copysign(1, q1) * SLEEF_QUAD_MIN;
      break;
    case 113:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 -= copysign(1, q0) * SLEEF_QUAD_MIN;
      break;
#endif
    default:
      // Each case in the following switch-case statement is executed
      // once in 8 loops.
      switch(cnt & 7) {
      case 0:
	q0 = rndf128(oneEMinus10Q, oneEPlus10Q);
	q1 = rndf128(oneEMinus10Q, oneEPlus10Q);
	break;
      case 1:
	q0 = rndf128(oneEMinus100Q, oneEPlus100Q);
	q1 = rndf128(oneEMinus100Q, oneEPlus100Q);
	break;
      case 2:
	q0 = rndf128(oneEMinus1000Q, oneEPlus1000Q);
	q1 = rndf128(oneEMinus1000Q, oneEPlus1000Q);
	break;
      default:
	q0 = rndf128x();
	q1 = rndf128x();
	break;
      }
      break;
    }

    a0 = vset(a0, e, q0);
    a1 = vset(a1, e, q1);
    mpfr_set_f128(frx, q0, GMP_RNDN);
    mpfr_set_f128(fry, q1, GMP_RNDN);

    {
      mpfr_add(frz, frx, fry, GMP_RNDN);

      double u0 = countULPf128(t = vget(xaddq_u05(a0, a1), e), frz, 0);
      
      if (u0 > 0.5000000001) {
	printf(ISANAME " add arg=%s %s ulp=%.20g\n", sprintf128(q0), sprintf128(q1), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_sub(frz, frx, fry, GMP_RNDN);

      double u0 = countULPf128(t = vget(xsubq_u05(a0, a1), e), frz, 0);
      
      if (u0 > 0.5000000001) {
	printf(ISANAME " sub arg=%s %s ulp=%.20g\n", sprintf128(q0), sprintf128(q1), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_mul(frz, frx, fry, GMP_RNDN);

      double u0 = countULPf128(t = vget(xmulq_u05(a0, a1), e), frz, 0);
      
      if (u0 > 0.5000000001) {
	printf(ISANAME " mul arg=%s %s ulp=%.20g\n", sprintf128(q0), sprintf128(q1), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_div(frz, frx, fry, GMP_RNDN);

      double u0 = countULPf128(t = vget(xdivq_u05(a0, a1), e), frz, 0);
      
      if (u0 > 0.5000000001) {
	printf(ISANAME " div arg=%s %s ulp=%.20g\n", sprintf128(q0), sprintf128(q1), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_sqrt(frz, frx, GMP_RNDN);

      double u0 = countULPf128(t = vget(xsqrtq_u05(a0), e), frz, 0);

      if (u0 > 0.5000000001) {
	printf(ISANAME " sqrt arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      double d = mpfr_get_d(frx, GMP_RNDN);
      vd0 = vsetd(vd0, e, d);
      t = vget(xcast_from_doubleq(vd0), e);
      mpfr_set_d(frz, d, GMP_RNDN);
      Sleef_quad q2 = mpfr_get_f128(frz, GMP_RNDN);

      if (memcmp(&t, &q2, sizeof(Sleef_quad)) != 0 && !(isnanf128(t) && isnanf128(q2))) {
	printf(ISANAME " cast_from_double arg=%.20g\n", d);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(q2));
	fflush(stdout); ecnt++;
      }
    }

    {
      double td = vgetd(xcast_to_doubleq(a0), e);
      double cd = mpfr_get_d(frx, GMP_RNDN);

      if (fabs(cd) >= DBL_MIN && cd != td && !(isnan(td) && isnan(cd))) {
	printf(ISANAME " cast_to_double arg=%s\n", sprintf128(q0));
	printf("test = %.20g\n", td);
	printf("corr = %.20g\n", cd);
	fflush(stdout); ecnt++;
      }
    }

    {
      int64_t i64 = mpfr_get_sj(frx, GMP_RNDN);
      vd0 = vreinterpret_vd_vm(vsetm(vreinterpret_vm_vd(vd0), e, i64));
      t = vget(xcast_from_int64q(vreinterpret_vi64_vm(vreinterpret_vm_vd(vd0))), e);
      mpfr_set_sj(frz, i64, GMP_RNDN);
      Sleef_quad q2 = mpfr_get_f128(frz, GMP_RNDN);

      if (memcmp(&t, &q2, sizeof(Sleef_quad)) != 0) {
	printf(ISANAME " cast_from_int64q arg=%lld\n", (long long)i64);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(q2));
	fflush(stdout); ecnt++;
      }
    }

    {
      int64_t td = vgeti64(xcast_to_int64q(a0), e);
      int64_t cd = mpfr_get_sj(frx, GMP_RNDZ);

      if (cd != td && !isnan(mpfr_get_d(frx, GMP_RNDN))) {
	printf(ISANAME " cast_to_int64q arg=%s\n", sprintf128(q0));
	printf("test = %lld\n", (long long)td);
	printf("corr = %lld\n", (long long)cd);
	fflush(stdout); ecnt++;
      }
    }

    {
      uint64_t u64 = mpfr_get_uj(frx, GMP_RNDN);
      vd0 = vreinterpret_vd_vm(vsetm(vreinterpret_vm_vd(vd0), e, u64));
      t = vget(xcast_from_uint64q(vreinterpret_vu64_vm(vreinterpret_vm_vd(vd0))), e);
      mpfr_set_uj(frz, u64, GMP_RNDN);
      Sleef_quad q2 = mpfr_get_f128(frz, GMP_RNDN);

      if (memcmp(&t, &q2, sizeof(Sleef_quad)) != 0) {
	printf(ISANAME " cast_from_uint64q arg=%llu\n", (unsigned long long)u64);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(q2));
	fflush(stdout); ecnt++;
      }
    }

    {
      uint64_t td = vgetu64(xcast_to_uint64q(a0), e);
      uint64_t cd = mpfr_get_uj(frx, GMP_RNDZ);

      if (cd != td && !isnan(mpfr_get_d(frx, GMP_RNDN))) {
	printf(ISANAME " cast_to_uint64q arg=%s\n", sprintf128(q0));
	printf("test = %llu\n", (unsigned long long)td);
	printf("corr = %llu\n", (unsigned long long)cd);
	fflush(stdout); ecnt++;
      }
    }

    {
      int ci = mpfr_less_p(frx, fry);
      int ti = vgeti(xicmpltq(a0, a1), e);

      if (ci != ti) {
	printf(ISANAME " icmpltq arg=%s, %s,  test = %d, corr = %d \n", sprintf128(q0), sprintf128(q1), ti, ci);
	fflush(stdout); ecnt++;
      }
    }

    {
      int ci = mpfr_greater_p(frx, fry);
      int ti = vgeti(xicmpgtq(a0, a1), e);

      if (ci != ti) {
	printf(ISANAME " icmpgtq arg=%s, %s,  test = %d, corr = %d \n", sprintf128(q0), sprintf128(q1), ti, ci);
	fflush(stdout); ecnt++;
      }
    }

    {
      int ci = mpfr_lessequal_p(frx, fry);
      int ti = vgeti(xicmpleq(a0, a1), e);

      if (ci != ti) {
	printf(ISANAME " icmpleq arg=%s, %s,  test = %d, corr = %d \n", sprintf128(q0), sprintf128(q1), ti, ci);
	fflush(stdout); ecnt++;
      }
    }

    {
      int ci = mpfr_greaterequal_p(frx, fry);
      int ti = vgeti(xicmpgeq(a0, a1), e);

      if (ci != ti) {
	printf(ISANAME " icmpgeq arg=%s, %s,  test = %d, corr = %d \n", sprintf128(q0), sprintf128(q1), ti, ci);
	fflush(stdout); ecnt++;
      }
    }

    {
      int ci = mpfr_equal_p(frx, fry);
      int ti = vgeti(xicmpeqq(a0, a1), e);

      if (ci != ti) {
	printf(ISANAME " icmpeq arg=%s, %s,  test = %d, corr = %d \n", sprintf128(q0), sprintf128(q1), ti, ci);
	fflush(stdout); ecnt++;
      }
    }

    {
      int ci = mpfr_lessgreater_p(frx, fry);
      int ti = vgeti(xicmpneq(a0, a1), e);

      if (ci != ti) {
	printf(ISANAME " icmpne arg=%s, %s,  test = %d, corr = %d \n", sprintf128(q0), sprintf128(q1), ti, ci);
	fflush(stdout); ecnt++;
      }
    }

    {
      int ci = mpfr_cmp(frx, fry);
      int ti = vgeti(xicmpq(a0, a1), e);

      if (ci != ti) {
	printf(ISANAME " icmp arg=%s, %s,  test = %d, corr = %d \n", sprintf128(q0), sprintf128(q1), ti, ci);
	fflush(stdout); ecnt++;
      }
    }

    {
      int ci = mpfr_unordered_p(frx, fry);
      int ti = vgeti(xiunordq(a0, a1), e);

      if (ci != ti) {
	printf(ISANAME " iunord arg=%s, %s,  test = %d, corr = %d \n", sprintf128(q0), sprintf128(q1), ti, ci);
	fflush(stdout); ecnt++;
      }
    }

#ifdef ENABLE_PUREC_SCALAR
#if !(defined(ENABLEFLOAT128) && defined(__clang__))
    if ((cnt & 15) == 1) {
      char s[64];
      Sleef_quad q1;

      Sleef_snprintf(s, 63, "%.40Qg", a0);
      q1 = vget(Sleef_strtoq(s, NULL), e);
      if (memcmp(&q0, &q1, sizeof(Sleef_quad)) != 0 && !(isnanf128(q0) && isnanf128(q1))) {
	printf("snprintf(Qg)/strtoq arg=%s str=%s test=%s\n", sprintf128(q0), s, sprintf128(q1));
	fflush(stdout); ecnt++;
      }

      Sleef_snprintf(s, 63, "%Qa", a0);
      q1 = vget(Sleef_strtoq(s, NULL), e);
      if (memcmp(&q0, &q1, sizeof(Sleef_quad)) != 0 && !(isnanf128(q0) && isnanf128(q1))) {
	printf("snprintf(Qa)/strtoq arg=%s str=%s test=%s\n", sprintf128(q0), s, sprintf128(q1));
	fflush(stdout); ecnt++;
      }
    }
#else
    if ((cnt & 15) == 1) {
      char s[64];
      Sleef_quad q1;

      Sleef_snprintf(s, 63, "%.40Pg", &a0);
      q1 = vget(Sleef_strtoq(s, NULL), e);
      if (memcmp(&q0, &q1, sizeof(Sleef_quad)) != 0 && !(isnanf128(q0) && isnanf128(q1))) {
	printf("snprintf(Qg)/strtoq arg=%s str=%s test=%s\n", sprintf128(q0), s, sprintf128(q1));
	fflush(stdout); ecnt++;
      }

      Sleef_snprintf(s, 63, "%Pa", &a0);
      q1 = vget(Sleef_strtoq(s, NULL), e);
      if (memcmp(&q0, &q1, sizeof(Sleef_quad)) != 0 && !(isnanf128(q0) && isnanf128(q1))) {
	printf("snprintf(Qa)/strtoq arg=%s str=%s test=%s\n", sprintf128(q0), s, sprintf128(q1));
	fflush(stdout); ecnt++;
      }
    }
#endif
#endif

    {
      mpfr_exp(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xexpq_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " exp arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_exp2(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xexp2q_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " exp2 arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_exp10(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xexp10q_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " exp10 arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_expm1(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xexpm1q_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " expm1 arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_log(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xlogq_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " log arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_log2(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xlog2q_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " log2 arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_log10(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xlog10q_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " log10 arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_log1p(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xlog1pq_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " log1p arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_atan(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xatanq_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " atan arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    q0 = rndf128(oneEMinus300Q, oneQ);
    a0 = vset(a0, e, q0);
    mpfr_set_f128(frx, q0, GMP_RNDN);

    {
      mpfr_asin(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xasinq_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " asin arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_acos(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xacosq_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " acos arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

#if defined(ENABLEFLOAT128)
    switch(cnt & 31) {
    case 0: {
      memrand(&q0, sizeof(__float128));
      q0 = q0 * M_PI_2Q;
    }
      break;

    case 1: {
      int t;
      memrand(&t, sizeof(int));
      t &= ~((~0UL) << (xrand() & 31));
      q0 = t * M_PI_2Q;
    }
      break;

    case 2:
      q0 = rndf128x();
      break;

    default:
      q0 = rndf128(1e-20, 1e+20);
      break;
    }

    a0 = vset(a0, e, q0);
    mpfr_set_f128(frx, q0, GMP_RNDN);
#endif

    {
      mpfr_sin(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xsinq_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " sin arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_cos(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xcosq_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " cos arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

    {
      mpfr_tan(frz, frx, GMP_RNDN);
      double u0 = countULPf128(t = vget(xtanq_u10(a0), e), frz, 0);

      if (u0 > 0.8) {
	printf(ISANAME " tan arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }
  }
}
