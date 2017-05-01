//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdint.h>

#ifdef ENABLE_DP
typedef struct {
  vdouble x, y;
} vdouble2;

vdouble xldexp(vdouble x, vint q);
vint xilogb(vdouble d);

vdouble xsin(vdouble d);
vdouble xcos(vdouble d);
vdouble2 xsincos(vdouble d);
vdouble xtan(vdouble d);
vdouble xasin(vdouble s);
vdouble xacos(vdouble s);
vdouble xatan(vdouble s);
vdouble xatan2(vdouble y, vdouble x);
vdouble xlog(vdouble d);
vdouble xexp(vdouble d);
vdouble xpow(vdouble x, vdouble y);

vdouble xsinh(vdouble d);
vdouble xcosh(vdouble d);
vdouble xtanh(vdouble d);
vdouble xasinh(vdouble s);
vdouble xacosh(vdouble s);
vdouble xatanh(vdouble s);

vdouble xcbrt(vdouble d);

vdouble xexp2(vdouble a);
vdouble xexp10(vdouble a);
vdouble xexpm1(vdouble a);
vdouble xlog10(vdouble a);
vdouble xlog1p(vdouble a);

vdouble xsin_u1(vdouble d);
vdouble xcos_u1(vdouble d);
vdouble2 xsincos_u1(vdouble d);
vdouble xtan_u1(vdouble d);
vdouble xasin_u1(vdouble s);
vdouble xacos_u1(vdouble s);
vdouble xatan_u1(vdouble s);
vdouble xatan2_u1(vdouble y, vdouble x);
vdouble xlog_u1(vdouble d);
vdouble xcbrt_u1(vdouble d);

vdouble2 xsincospi_u05(vdouble d);
vdouble2 xsincospi_u35(vdouble d);
vdouble xsinpi_u05(vdouble d);

vdouble xldexp(vdouble, vint);
vint xilogb(vdouble);
vdouble xfma(vdouble, vdouble, vdouble);
vdouble xsqrt_u05(vdouble);

vdouble xhypot_u05(vdouble, vdouble);
vdouble xhypot_u35(vdouble, vdouble);

vdouble xfabs(vdouble);
vdouble xcopysign(vdouble, vdouble);
vdouble xfmax(vdouble, vdouble);
vdouble xfmin(vdouble, vdouble);
vdouble xfdim(vdouble, vdouble);
vdouble xtrunc(vdouble);
vdouble xfloor(vdouble);
vdouble xceil(vdouble);
vdouble xround(vdouble);
vdouble xrint(vdouble);
vdouble xnextafter(vdouble, vdouble);
vdouble xfrfrexp(vdouble);
vint xexpfrexp(vdouble);
vdouble xfmod(vdouble, vdouble);
vdouble2 xmodf(vdouble);

vdouble xlgamma_u1(vdouble);
vdouble xtgamma_u1(vdouble);

vdouble xerf_u1(vdouble);
vdouble xerfc_u15(vdouble);
#endif

//

#ifdef ENABLE_SP
typedef struct {
  vfloat x, y;
} vfloat2;

vfloat xldexpf(vfloat x, vint2 q);
vint2 xilogbf(vfloat d);

vfloat xsinf(vfloat d);
vfloat xcosf(vfloat d);
vfloat2 xsincosf(vfloat d);
vfloat xtanf(vfloat d);
vfloat xasinf(vfloat s);
vfloat xacosf(vfloat s);
vfloat xatanf(vfloat s);
vfloat xatan2f(vfloat y, vfloat x);
vfloat xlogf(vfloat d);
vfloat xexpf(vfloat d);
vfloat xcbrtf(vfloat s);

vfloat xpowf(vfloat x, vfloat y);
vfloat xsinhf(vfloat x);
vfloat xcoshf(vfloat x);
vfloat xtanhf(vfloat x);
vfloat xasinhf(vfloat x);
vfloat xacoshf(vfloat x);
vfloat xatanhf(vfloat x);
vfloat xexp2f(vfloat a);
vfloat xexp10f(vfloat a);
vfloat xexpm1f(vfloat a);
vfloat xlog10f(vfloat a);
vfloat xlog1pf(vfloat a);

vfloat xsinf_u1(vfloat d);
vfloat xcosf_u1(vfloat d);
vfloat2 xsincosf_u1(vfloat d);
vfloat xtanf_u1(vfloat d);
vfloat xasinf_u1(vfloat s);
vfloat xacosf_u1(vfloat s);
vfloat xatanf_u1(vfloat s);
vfloat xatan2f_u1(vfloat y, vfloat x);
vfloat xlogf_u1(vfloat d);
vfloat xcbrtf_u1(vfloat s);

vfloat2 xsincospif_u05(vfloat d);
vfloat2 xsincospif_u35(vfloat d);


vfloat xldexpf(vfloat, vint2);
vint2 xilogbf(vfloat);
vfloat xfmaf(vfloat, vfloat, vfloat);
vfloat xsqrtf_u05(vfloat s);
vfloat xsqrtf_u35(vfloat s);

vfloat xhypotf_u05(vfloat, vfloat);
vfloat xhypotf_u35(vfloat, vfloat);

vfloat xfabsf(vfloat);
vfloat xcopysignf(vfloat, vfloat);
vfloat xfmaxf(vfloat, vfloat);
vfloat xfminf(vfloat, vfloat);
vfloat xfdimf(vfloat, vfloat);
vfloat xtruncf(vfloat);
vfloat xfloorf(vfloat);
vfloat xceilf(vfloat);
vfloat xroundf(vfloat);
vfloat xrintf(vfloat);
vfloat xnextafterf(vfloat, vfloat);
vfloat xfrfrexpf(vfloat);
vint2 xexpfrexpf(vfloat);
vfloat xfmodf(vfloat, vfloat);
vfloat2 xmodff(vfloat);
#endif
