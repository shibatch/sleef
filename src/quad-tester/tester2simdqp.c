//          Copyright Naoki Shibata 2010 - 2018.
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

#if defined(POWER64_UNDEF_USE_EXTERN_INLINES)
// This is a workaround required to cross compile for PPC64 binaries
#include <features.h>
#ifdef __USE_EXTERN_INLINES
#undef __USE_EXTERN_INLINES
#endif
#endif

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

static vargquad vset(vargquad v, int idx, Sleef_quad d) { v.s[idx] = d; return v; }
static Sleef_quad vget(vargquad v, int idx) { return v.s[idx]; }

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

  const Sleef_quad oneEMinus10Q  = cast_q_str("1e-10");
  const Sleef_quad oneEPlus10Q   = cast_q_str("1e+10");
  const Sleef_quad oneEMinus100Q = cast_q_str("1e-100");
  const Sleef_quad oneEPlus100Q  = cast_q_str("1e+100");
  const Sleef_quad oneEMinus1000Q = cast_q_str("1e-1000");
  const Sleef_quad oneEPlus1000Q  = cast_q_str("1e+1000");
  const Sleef_quad quadMin = cast_q_str("3.36210314311209350626267781732175260e-4932");
  const Sleef_quad quadMax = cast_q_str("1.18973149535723176508575932662800702e+4932");
  const Sleef_quad quadDenormMin = cast_q_str("6.475175119438025110924438958227646552e-4966");

  //

  int cnt, ecnt = 0;
  vargquad a0, a1, a2, a3;
  Sleef_quad q0, q1, q2, q3, t;
  mpfr_t frw, frx, fry, frz;
  mpfr_inits(frw, frx, fry, frz, NULL);

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
      q1 += 1;
      break;
    case 121:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 += 1;
      q1 -= 1;
      break;
    case 120:
      q0 = rndf128x();
      q1 = rndf128x();
      q1 += copysign(1, q1) * SLEEF_QUAD_MIN;
      break;
    case 119:
      q0 = rndf128x();
      q1 = rndf128x();
      q1 = copysign(1, q1) * SLEEF_QUAD_MIN;
      break;
    case 118:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 += copysign(1, q0);
      q1  = copysign(1, q1) * SLEEF_QUAD_MIN;
      break;
    case 117:
      q0 = rndf128x();
      q1 = rndf128x();
      q1 = copysign(1, q1) * SLEEF_QUAD_MIN;
      break;
    case 116:
      q0 = rndf128x();
      q1 = rndf128x();
      q0 += copysign(1, q0);
      q1  = copysign(1, q1) * SLEEF_QUAD_MIN;
      break;
    case 115:
      q0 = rndf128x();
      q1 = rndf128x();
      q1 += copysign(1, q1) * SLEEF_QUAD_MAX;
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

    {
      mpfr_set_f128(frx, q0, GMP_RNDN);
      mpfr_set_f128(fry, q1, GMP_RNDN);
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
      mpfr_set_f128(frx, q0, GMP_RNDN);
      mpfr_set_f128(fry, q1, GMP_RNDN);
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
      mpfr_set_f128(frx, q0, GMP_RNDN);
      mpfr_set_f128(fry, q1, GMP_RNDN);
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
      mpfr_set_f128(frx, q0, GMP_RNDN);
      mpfr_set_f128(fry, q1, GMP_RNDN);
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
      mpfr_set_f128(frx, q0, GMP_RNDN);
      mpfr_sqrt(frz, frx, GMP_RNDN);

      double u0 = countULPf128(t = vget(xsqrtq_u05(a0), e), frz, 0);

      if (u0 > 0.5000000001) {
	printf(ISANAME " sqrt arg=%s ulp=%.20g\n", sprintf128(q0), u0);
	printf("test = %s\n", sprintf128(t));
	printf("corr = %s\n\n", sprintf128(mpfr_get_f128(frz, GMP_RNDN)));
	fflush(stdout); ecnt++;
      }
    }

#ifdef ENABLE_PUREC_SCALAR
    if ((cnt & 15) == 1) {
      char s[64];
      Sleef_qtostr(s, 63, a0, 10);
      Sleef_quad q1 = vget(Sleef_strtoq(s, NULL, 10), e);
      if (memcmp(&q0, &q1, sizeof(Sleef_quad)) != 0 && !(isnanf128(q0) && isnanf128(q1))) {
	printf("qtostr/strtoq arg=%s\n", sprintf128(q0));
	fflush(stdout); ecnt++;
      }
    }
#endif
  }
}
