//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Always use -ffp-contract=off option to compile SLEEF.

#include <assert.h>
#include <math.h>
#include <limits.h>

#if defined (__GNUC__) || defined (__INTEL_COMPILER) || defined (__clang__)
#define INLINE __attribute__((always_inline))
#else
#define INLINE inline
#endif

#include "nonnumber.h"

#ifdef ENABLE_SSE2
#include "helpersse2.h"
#endif

#ifdef ENABLE_AVX
#include "helperavx.h"
#endif

#ifdef ENABLE_AVX2
#include "helperavx2.h"
#endif

#ifdef ENABLE_AVX512F
#include "helperavx512f.h"
#endif

#ifdef ENABLE_FMA4
#include "helperfma4.h"
#endif

#ifdef ENABLE_NEON32
#include "helperneon32.h"
#endif

#ifdef ENABLE_NEON64
#include "helperneon64.h"
#endif

#ifdef ENABLE_CLANGVEC
#include "helperclangvec.h"
#endif

//

#include "df.h"

//

#define PI4_Af 0.78515625f
#define PI4_Bf 0.00024187564849853515625f
#define PI4_Cf 3.7747668102383613586e-08f
#define PI4_Df 1.2816720341285448015e-12f

#define PI_Af 3.140625f
#define PI_Bf 0.0009670257568359375f
#define PI_Cf 6.2771141529083251953e-07f
#define PI_Df 1.2154201256553420762e-10f

#define PI_XDf 1.2141754268668591976e-10f
#define PI_XEf 1.2446743939339977025e-13f

#define TRIGRANGEMAXf 1e+7 // 39000
#define SQRT_FLT_MAX 18446743523953729536.0

#define L2Uf 0.693145751953125f
#define L2Lf 1.428606765330187045e-06f
#define R_LN2f 1.442695040888963407359924681001892137426645954152985934135449406931f

//

static INLINE vopmask visnegzero_vo_vf(vfloat d) {
  return veq_vo_vi2_vi2(vreinterpret_vi2_vf(d), vreinterpret_vi2_vf(vcast_vf_f(-0.0)));
}

static INLINE vmask vsignbit_vm_vf(vfloat f) {
  return vand_vm_vm_vm(vreinterpret_vm_vf(f), vreinterpret_vm_vf(vcast_vf_f(-0.0f)));
}

static INLINE vfloat vmulsign_vf_vf_vf(vfloat x, vfloat y) {
  return (vfloat)vxor_vm_vm_vm((vmask)x, vsignbit_vm_vf(y));
}

static INLINE vfloat vsign_vf_vf(vfloat f) {
  return (vfloat)vor_vm_vm_vm((vmask)vcast_vf_f(1.0f), vand_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)f));
}

static INLINE vopmask vsignbit_vo_vf(vfloat d) {
  return veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vreinterpret_vi2_vf(d), vcast_vi2_i(0x80000000)), vcast_vi2_i(0x80000000));
}

static INLINE vint2 vsel_vi2_vf_vf_vi2_vi2(vfloat f0, vfloat f1, vint2 x, vint2 y) {
  return vsel_vi2_vo_vi2_vi2(vlt_vo_vf_vf(f0, f1), x, y);
}

static INLINE vint2 vsel_vi2_vf_vi2(vfloat d, vint2 x) {
  return vand_vi2_vo_vi2(vsignbit_vo_vf(d), x);
}

#ifndef ENABLE_GETEXP_SP
static INLINE vint2 vilogbk_vi2_vf(vfloat d) {
  vopmask o = vlt_vo_vf_vf(d, vcast_vf_f(5.421010862427522E-20f));
  d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(vcast_vf_f(1.8446744073709552E19f), d), d);
  vint2 q = vand_vi2_vi2_vi2(vsrl_vi2_vi2_i(vcast_vi2_vm(vreinterpret_vm_vf(d)), 23), vcast_vi2_i(0xff));
  q = vsub_vi2_vi2_vi2(q, vsel_vi2_vo_vi2_vi2(o, vcast_vi2_i(64 + 0x7f), vcast_vi2_i(0x7f)));
  return q;
}
#endif

//

vint2 xilogbf(vfloat d) {
  vint2 e = vilogbk_vi2_vf(vabs_vf_vf(d));
  e = vsel_vi2_vo_vi2_vi2(veq_vo_vf_vf(d, vcast_vf_f(0.0f)), vcast_vi2_i(FP_ILOGB0), e);
  e = vsel_vi2_vo_vi2_vi2(visnan_vo_vf(d), vcast_vi2_i(FP_ILOGBNAN), e);
  e = vsel_vi2_vo_vi2_vi2(visinf_vo_vf(d), vcast_vi2_i(INT_MAX), e);
  return e;
}

static INLINE vfloat vpow2i_vf_vi2(vint2 q) {
  return (vfloat)vcast_vm_vi2(vsll_vi2_vi2_i(vadd_vi2_vi2_vi2(q, vcast_vi2_i(0x7f)), 23));
}

static INLINE vfloat vldexp_vf_vf_vi2(vfloat x, vint2 q) {
  vfloat u;
  vint2 m = vsra_vi2_vi2_i(q, 31);
  m = vsll_vi2_vi2_i(vsub_vi2_vi2_vi2(vsra_vi2_vi2_i(vadd_vi2_vi2_vi2(m, q), 6), m), 4);
  q = vsub_vi2_vi2_vi2(q, vsll_vi2_vi2_i(m, 2));
  m = vadd_vi2_vi2_vi2(m, vcast_vi2_i(0x7f));
  m = vand_vi2_vi2_vi2(vgt_vi2_vi2_vi2(m, vcast_vi2_i(0)), m);
  vint2 n = vgt_vi2_vi2_vi2(m, vcast_vi2_i(0xff));
  m = vor_vi2_vi2_vi2(vandnot_vi2_vi2_vi2(n, m), vand_vi2_vi2_vi2(n, vcast_vi2_i(0xff)));
  u = vreinterpret_vf_vm(vcast_vm_vi2(vsll_vi2_vi2_i(m, 23)));
  x = vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(x, u), u), u), u);
  u = vreinterpret_vf_vm(vcast_vm_vi2(vsll_vi2_vi2_i(vadd_vi2_vi2_vi2(q, vcast_vi2_i(0x7f)), 23)));
  return vmul_vf_vf_vf(x, u);
}

vfloat xldexpf(vfloat x, vint2 q) { return vldexp_vf_vf_vi2(x, q); }

vfloat xsinf(vfloat d) {
  vint2 q;
  vfloat u, s, r = d;

  q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f((float)M_1_PI)));
  u = vcast_vf_vi2(q);

  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Af), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Bf), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Cf), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Df), d);

  s = vmul_vf_vf_vf(d, d);

  d = (vfloat)vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1)), (vmask)vcast_vf_f(-0.0f)), (vmask)d);

  u = vcast_vf_f(2.6083159809786593541503e-06f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.0001981069071916863322258f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833307858556509017944336f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.166666597127914428710938f));

  u = vadd_vf_vf_vf(vmul_vf_vf_vf(s, vmul_vf_vf_vf(u, d)), d);

  u = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visnegzero_vo_vf(r),
				    vgt_vo_vf_vf(vabs_vf_vf(r), vcast_vf_f(TRIGRANGEMAXf))),
		       vcast_vf_f(-0.0), u);

  u = (vfloat)vor_vm_vo32_vm(visinf_vo_vf(d), (vmask)u);

  return u;
}

vfloat xcosf(vfloat d) {
  vint2 q;
  vfloat u, s, r = d;

  q = vrint_vi2_vf(vsub_vf_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f((float)M_1_PI)), vcast_vf_f(0.5f)));
  q = vadd_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, q), vcast_vi2_i(1));

  u = vcast_vf_vi2(q);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Af*0.5f), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Bf*0.5f), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Cf*0.5f), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Df*0.5f), d);

  s = vmul_vf_vf_vf(d, d);

  d = (vfloat)vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(0)), (vmask)vcast_vf_f(-0.0f)), (vmask)d);

  u = vcast_vf_f(2.6083159809786593541503e-06f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.0001981069071916863322258f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833307858556509017944336f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.166666597127914428710938f));

  u = vadd_vf_vf_vf(vmul_vf_vf_vf(s, vmul_vf_vf_vf(u, d)), d);

  u = (vfloat)vandnot_vm_vo32_vm(vgt_vo_vf_vf(vabs_vf_vf(r), vcast_vf_f(TRIGRANGEMAXf)),
				 (vmask)u);

  u = (vfloat)vor_vm_vo32_vm(visinf_vo_vf(d), (vmask)u);

  return u;
}

vfloat2 xsincosf(vfloat d) {
  vint2 q;
  vopmask o;
  vfloat u, s, t, rx, ry;
  vfloat2 r;

  q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f((float)M_2_PI)));

  s = d;

  u = vcast_vf_vi2(q);
  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Af*0.5f), s);
  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Bf*0.5f), s);
  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Cf*0.5f), s);
  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Df*0.5f), s);

  t = s;

  s = vmul_vf_vf_vf(s, s);

  u = vcast_vf_f(-0.000195169282960705459117889f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833215750753879547119141f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.166666537523269653320312f));
  u = vmul_vf_vf_vf(vmul_vf_vf_vf(u, s), t);

  rx = vadd_vf_vf_vf(t, u);
  rx = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0f), rx);

  u = vcast_vf_f(-2.71811842367242206819355e-07f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(2.47990446951007470488548e-05f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.00138888787478208541870117f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0416666641831398010253906f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.5));

  ry = vmla_vf_vf_vf_vf(s, u, vcast_vf_f(1));

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(0));
  r.x = vsel_vf_vo_vf_vf(o, rx, ry);
  r.y = vsel_vf_vo_vf_vf(o, ry, rx);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(2));
  r.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.x)));

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(2)), vcast_vi2_i(2));
  r.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.y)));

  o = vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAXf));
  r.x = (vfloat)vandnot_vm_vo32_vm(o, (vmask)r.x);
  r.y = (vfloat)vandnot_vm_vo32_vm(o, (vmask)r.y);
  
  o = visinf_vo_vf(d);
  r.x = (vfloat)vor_vm_vo32_vm(o, (vmask)r.x);
  r.y = (vfloat)vor_vm_vo32_vm(o, (vmask)r.y);

  return r;
}

vfloat xtanf(vfloat d) {
  vint2 q;
  vopmask o;
  vfloat u, s, x;

  q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f((float)(2 * M_1_PI))));

  x = d;

  u = vcast_vf_vi2(q);
  x = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Af*0.5f), x);
  x = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Bf*0.5f), x);
  x = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Cf*0.5f), x);
  x = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Df*0.5f), x);

  s = vmul_vf_vf_vf(x, x);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1));
  x = (vfloat)vxor_vm_vm_vm(vand_vm_vo32_vm(o, (vmask)vcast_vf_f(-0.0f)), (vmask)x);

  u = vcast_vf_f(0.00927245803177356719970703f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00331984995864331722259521f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0242998078465461730957031f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0534495301544666290283203f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.133383005857467651367188f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.333331853151321411132812f));

  u = vmla_vf_vf_vf_vf(s, vmul_vf_vf_vf(u, x), x);

  u = vsel_vf_vo_vf_vf(o, vrec_vf_vf(u), u);

  u = (vfloat)vor_vm_vo32_vm(visinf_vo_vf(d), (vmask)u);

  return u;
}

vfloat xsinf_u1(vfloat d) {
  vint2 q;
  vfloat u;
  vfloat2 s, t, x;

  q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(M_1_PI)));
  u = vcast_vf_vi2(q);

  s = dfadd2_vf2_vf_vf (d, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Af)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Bf)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Cf)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Df)));

  t = s;
  s = dfsqu_vf2_vf2(s);

  u = vcast_vf_f(2.6083159809786593541503e-06f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.0001981069071916863322258f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.00833307858556509017944336f));

  x = dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf(vcast_vf_f(-0.166666597127914428710938f), vmul_vf_vf_vf(u, s.x)), s));

  x = dfmul_vf2_vf2_vf2(t, x);
  u = vadd_vf_vf_vf(x.x, x.y);

  u = (vfloat)vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1)), (vmask)vcast_vf_f(-0.0)), (vmask)u);
  u = vsel_vf_vo_vf_vf(vandnot_vo_vo_vo(visinf_vo_vf(d), vor_vo_vo_vo(visnegzero_vo_vf(d),
								      vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAXf)))),
		       vcast_vf_f(-0.0), u);

  return u;
}

vfloat xcosf_u1(vfloat d) {
  vint2 q;
  vfloat u;
  vfloat2 s, t, x;

  q = vrint_vi2_vf(vmla_vf_vf_vf_vf(d, vcast_vf_f(M_1_PI), vcast_vf_f(-0.5)));
  q = vadd_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, q), vcast_vi2_i(1));
  u = vcast_vf_vi2(q);

  s = dfadd2_vf2_vf_vf (d, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Af*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Bf*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Cf*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Df*0.5f)));

  t = s;
  s = dfsqu_vf2_vf2(s);

  u = vcast_vf_f(2.6083159809786593541503e-06f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.0001981069071916863322258f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.00833307858556509017944336f));

  x = dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf(vcast_vf_f(-0.166666597127914428710938f), vmul_vf_vf_vf(u, s.x)), s));

  x = dfmul_vf2_vf2_vf2(t, x);
  u = vadd_vf_vf_vf(x.x, x.y);

  u = (vfloat)vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(0)), (vmask)vcast_vf_f(-0.0)), (vmask)u);

  u = (vfloat)vandnot_vm_vo32_vm(vandnot_vo_vo_vo(visinf_vo_vf(d), vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAXf))),
				 (vmask)u);
  
  return u;
}

vfloat2 xsincosf_u1(vfloat d) {
  vint2 q;
  vopmask o;
  vfloat u, rx, ry;
  vfloat2 r, s, t, x;

  q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(2 * M_1_PI)));
  u = vcast_vf_vi2(q);

  s = dfadd2_vf2_vf_vf (d, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Af*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Bf*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Cf*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Df*0.5f)));

  t = s;

  s = dfsqu_vf2_vf2(s);
  s.x = vadd_vf_vf_vf(s.x, s.y);

  u = vcast_vf_f(-0.000195169282960705459117889f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.00833215750753879547119141f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.166666537523269653320312f));

  u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(s.x, t.x));

  x = dfadd_vf2_vf2_vf(t, u);
  rx = vadd_vf_vf_vf(x.x, x.y);

  rx = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0f), rx);

  u = vcast_vf_f(-2.71811842367242206819355e-07f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(2.47990446951007470488548e-05f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.00138888787478208541870117f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0416666641831398010253906f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.5));

  x = dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf_vf(s.x, u));
  ry = vadd_vf_vf_vf(x.x, x.y);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(0));
  r.x = vsel_vf_vo_vf_vf(o, rx, ry);
  r.y = vsel_vf_vo_vf_vf(o, ry, rx);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(2));
  r.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.x)));

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(2)), vcast_vi2_i(2));
  r.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.y)));

  o = vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAXf));
  r.x = (vfloat)vandnot_vm_vo32_vm(o, (vmask)r.x);
  r.y = (vfloat)vandnot_vm_vo32_vm(o, (vmask)r.y);
  
  o = visinf_vo_vf(d);
  r.x = (vfloat)vor_vm_vo32_vm(o, (vmask)r.x);
  r.y = (vfloat)vor_vm_vo32_vm(o, (vmask)r.y);

  return r;
}

vfloat xtanf_u1(vfloat d) {
  vint2 q;
  vfloat u;
  vfloat2 s, t, x;
  vopmask o;

  q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(M_2_PI)));
  u = vcast_vf_vi2(q);

  s = dfadd2_vf2_vf_vf (d, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Af*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Bf*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_Cf*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_XDf*0.5f)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_XEf*0.5f)));

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1));
  vmask n = vand_vm_vo32_vm(o, (vmask)vcast_vf_f(-0.0));
  s.x = (vfloat)vxor_vm_vm_vm((vmask)s.x, n);
  s.y = (vfloat)vxor_vm_vm_vm((vmask)s.y, n);

  t = s;
  s = dfsqu_vf2_vf2(s);
  s = dfnormalize_vf2_vf2(s);

  u = vcast_vf_f(0.00446636462584137916564941f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-8.3920182078145444393158e-05f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0109639242291450500488281f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0212360303848981857299805f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0540687143802642822265625f));

  x = dfadd_vf2_vf_vf(vcast_vf_f(0.133325666189193725585938f), vmul_vf_vf_vf(u, s.x));
  x = dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf2(vcast_vf_f(0.33333361148834228515625f), dfmul_vf2_vf2_vf2(s, x)), s));
  x = dfmul_vf2_vf2_vf2(t, x);

  x = vsel_vf2_vo_vf2_vf2(o, dfrec_vf2_vf2(x), x);

  u = vadd_vf_vf_vf(x.x, x.y);

  u = vsel_vf_vo_vf_vf(vandnot_vo_vo_vo(visinf_vo_vf(d),
					vor_vo_vo_vo(visnegzero_vo_vf(d),
						     vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAXf)))),
		       vcast_vf_f(-0.0f), u);
  
  return u;
}

vfloat xatanf(vfloat d) {
  vfloat s, t, u;
  vint2 q;

  q = vsel_vi2_vf_vi2(d, vcast_vi2_i(2));
  s = vabs_vf_vf(d);

  q = vsel_vi2_vf_vf_vi2_vi2(vcast_vf_f(1.0f), s, vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), q);
  s = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(vcast_vf_f(1.0f), s), vrec_vf_vf(s), s);

  t = vmul_vf_vf_vf(s, s);

  u = vcast_vf_f(0.00282363896258175373077393f);
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.0159569028764963150024414f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.0425049886107444763183594f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.0748900920152664184570312f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.106347933411598205566406f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.142027363181114196777344f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.199926957488059997558594f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.333331018686294555664062f));

  t = vmla_vf_vf_vf_vf(s, vmul_vf_vf_vf(t, u), s);

  t = vsel_vf_vo_vf_vf(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1)), vsub_vf_vf_vf(vcast_vf_f((float)(M_PI/2)), t), t);

  t = (vfloat)vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(2)), (vmask)vcast_vf_f(-0.0f)), (vmask)t);

#ifdef ENABLE_NEON32
  t = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vmulsign_vf_vf_vf(vcast_vf_f(1.5874010519681994747517056f), d), t);
#endif

  return t;
}

static INLINE vfloat atan2kf(vfloat y, vfloat x) {
  vfloat s, t, u;
  vint2 q;
  vopmask p;

  q = vsel_vi2_vf_vi2(x, vcast_vi2_i(-2));
  x = vabs_vf_vf(x);

  q = vsel_vi2_vf_vf_vi2_vi2(x, y, vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), q);
  p = vlt_vo_vf_vf(x, y);
  s = vsel_vf_vo_vf_vf(p, vneg_vf_vf(x), y);
  t = vmax_vf_vf_vf(x, y);

  s = vdiv_vf_vf_vf(s, t);
  t = vmul_vf_vf_vf(s, s);

  u = vcast_vf_f(0.00282363896258175373077393f);
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.0159569028764963150024414f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.0425049886107444763183594f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.0748900920152664184570312f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.106347933411598205566406f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.142027363181114196777344f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.199926957488059997558594f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.333331018686294555664062f));

  t = vmla_vf_vf_vf_vf(s, vmul_vf_vf_vf(t, u), s);
  t = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f((float)(M_PI/2)), t);

  return t;
}

static INLINE vfloat visinf2_vf_vf_vf(vfloat d, vfloat m) {
  return (vfloat)vand_vm_vo32_vm(visinf_vo_vf(d), vor_vm_vm_vm(vsignbit_vm_vf(d), (vmask)m));
}

vfloat xatan2f(vfloat y, vfloat x) {
  vfloat r = atan2kf(vabs_vf_vf(y), x);

  r = vmulsign_vf_vf_vf(r, x);
  r = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0.0f))), vsub_vf_vf_vf(vcast_vf_f((float)(M_PI/2)), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f((float)(M_PI/2)), x))), r);
  r = vsel_vf_vo_vf_vf(visinf_vo_vf(y), vsub_vf_vf_vf(vcast_vf_f((float)(M_PI/2)), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f((float)(M_PI/4)), x))), r);

  r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(y, vcast_vf_f(0.0f)), (vfloat)vand_vm_vo32_vm(vsignbit_vo_vf(x), (vmask)vcast_vf_f((float)M_PI)), r);

  r = (vfloat)vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), (vmask)vmulsign_vf_vf_vf(r, y));
  return r;
}

vfloat xasinf(vfloat d) {
  vfloat x, y;
  x = vadd_vf_vf_vf(vcast_vf_f(1.0f), d);
  y = vsub_vf_vf_vf(vcast_vf_f(1.0f), d);
  x = vmul_vf_vf_vf(x, y);
  x = vsqrt_vf_vf(x);
  x = (vfloat)vor_vm_vo32_vm(visnan_vo_vf(x), (vmask)atan2kf(vabs_vf_vf(d), x));
  return vmulsign_vf_vf_vf(x, d);
}

vfloat xacosf(vfloat d) {
  vfloat x, y;
  x = vadd_vf_vf_vf(vcast_vf_f(1.0f), d);
  y = vsub_vf_vf_vf(vcast_vf_f(1.0f), d);
  x = vmul_vf_vf_vf(x, y);
  x = vsqrt_vf_vf(x);
  x = vmulsign_vf_vf_vf(atan2kf(x, vabs_vf_vf(d)), d);
  y = (vfloat)vand_vm_vo32_vm(vsignbit_vo_vf(d), (vmask)vcast_vf_f((float)M_PI));
  x = vadd_vf_vf_vf(x, y);
  return x;
}

//

static INLINE vfloat2 atan2kf_u1(vfloat2 y, vfloat2 x) {
  vfloat u;
  vfloat2 s, t;
  vint2 q;
  vopmask p;
  vmask r;
  
  q = vsel_vi2_vf_vf_vi2_vi2(x.x, vcast_vf_f(0), vcast_vi2_i(-2), vcast_vi2_i(0));
  p = vlt_vo_vf_vf(x.x, vcast_vf_f(0));
  r = vand_vm_vo32_vm(p, (vmask)vcast_vf_f(-0.0));
  x.x = (vfloat)vxor_vm_vm_vm((vmask)x.x, r);
  x.y = (vfloat)vxor_vm_vm_vm((vmask)x.y, r);

  q = vsel_vi2_vf_vf_vi2_vi2(x.x, y.x, vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), q);
  p = vlt_vo_vf_vf(x.x, y.x);
  s = vsel_vf2_vo_vf2_vf2(p, dfneg_vf2_vf2(x), y);
  t = vsel_vf2_vo_vf2_vf2(p, y, x);

  s = dfdiv_vf2_vf2_vf2(s, t);
  t = dfsqu_vf2_vf2(s);
  t = dfnormalize_vf2_vf2(t);

  u = vcast_vf_f(-0.00176397908944636583328247f);
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(0.0107900900766253471374512f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(-0.0309564601629972457885742f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(0.0577365085482597351074219f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(-0.0838950723409652709960938f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(0.109463557600975036621094f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(-0.142626821994781494140625f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(0.199983194470405578613281f));

  t = dfmul_vf2_vf2_vf2(t, dfadd_vf2_vf_vf(vcast_vf_f(-0.333332866430282592773438f), vmul_vf_vf_vf(u, t.x)));
  t = dfmul_vf2_vf2_vf2(s, dfadd_vf2_vf_vf2(vcast_vf_f(1), t));
  t = dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf(vcast_vf2_f_f(1.5707963705062866211f, -4.3711388286737928865e-08f), vcast_vf_vi2(q)), t);

  return t;
}

vfloat xatan2f_u1(vfloat y, vfloat x) {
  vfloat2 d = atan2kf_u1(vcast_vf2_vf_vf(vabs_vf_vf(y), vcast_vf_f(0)), vcast_vf2_vf_vf(x, vcast_vf_f(0)));
  vfloat r = vadd_vf_vf_vf(d.x, d.y);

  r = vmulsign_vf_vf_vf(r, x);
  r = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0))), vsub_vf_vf_vf(vcast_vf_f(M_PI/2), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f(M_PI/2), x))), r);
  r = vsel_vf_vo_vf_vf(visinf_vo_vf(y), vsub_vf_vf_vf(vcast_vf_f(M_PI/2), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f(M_PI/4), x))), r);
  r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(y, vcast_vf_f(0.0f)), (vfloat)vand_vm_vo32_vm(vsignbit_vo_vf(x), (vmask)vcast_vf_f((float)M_PI)), r);

  r = (vfloat)vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), (vmask)vmulsign_vf_vf_vf(r, y));
  return r;
}

vfloat xasinf_u1(vfloat d) {
  vfloat2 d2 = atan2kf_u1(vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0)), dfsqrt_vf2_vf2(dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf(vcast_vf_f(1), d), dfsub_vf2_vf_vf(vcast_vf_f(1), d))));
  vfloat r = vadd_vf_vf_vf(d2.x, d2.y);
  r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1)), vcast_vf_f(1.570796326794896557998982), r);
  return vmulsign_vf_vf_vf(r, d);
}

vfloat xacosf_u1(vfloat d) {
  vfloat2 d2 = atan2kf_u1(dfsqrt_vf2_vf2(dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf(vcast_vf_f(1), d), dfsub_vf2_vf_vf(vcast_vf_f(1), d))), vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0)));
  d2 = dfscale_vf2_vf2_vf(d2, vmulsign_vf_vf_vf(vcast_vf_f(1), d));

  vopmask o;
  o = vneq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1));
  d2.x = (vfloat)vand_vm_vo32_vm(o, (vmask)d2.x);
  d2.y = (vfloat)vand_vm_vo32_vm(o, (vmask)d2.y);
  o = vsignbit_vo_vf(d);
  d2 = vsel_vf2_vo_vf2_vf2(o, dfadd_vf2_vf2_vf2(vcast_vf2_f_f(3.1415927410125732422f,-8.7422776573475857731e-08f), d2), d2);

  return vadd_vf_vf_vf(d2.x, d2.y);
}

vfloat xatanf_u1(vfloat d) {
  vfloat2 d2 = atan2kf_u1(vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0)), vcast_vf2_f_f(1, 0));
  vfloat r = vadd_vf_vf_vf(d2.x, d2.y);
  r = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vcast_vf_f(1.570796326794896557998982), r);
  return vmulsign_vf_vf_vf(r, d);
}

//

vfloat xlogf(vfloat d) {
  vfloat x, x2, t, m;

#ifndef ENABLE_AVX512F
  vint2 e = vilogbk_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  m = vldexp_vf_vf_vi2(d, vneg_vi2_vi2(e));
#else
  vfloat e = _mm512_getexp_ps(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0f), e);
  m = _mm512_getmant_ps(d, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan);
#endif
  
  x = vdiv_vf_vf_vf(vadd_vf_vf_vf(vcast_vf_f(-1.0f), m), vadd_vf_vf_vf(vcast_vf_f(1.0f), m));
  x2 = vmul_vf_vf_vf(x, x);

  t = vcast_vf_f(0.2392828464508056640625f);
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.28518211841583251953125f));
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.400005877017974853515625f));
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.666666686534881591796875f));
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(2.0f));

#ifndef ENABLE_AVX512F
  x = vmla_vf_vf_vf_vf(x, t, vmul_vf_vf_vf(vcast_vf_f(0.693147180559945286226764f), vcast_vf_vi2(e)));
  x = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(INFINITYf), x);
  x = (vfloat)vor_vm_vo32_vm(vgt_vo_vf_vf(vcast_vf_f(0), d), (vmask)x);
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0)), vcast_vf_f(-INFINITYf), x);
#else
  x = vmla_vf_vf_vf_vf(x, t, vmul_vf_vf_vf(vcast_vf_f(0.693147180559945286226764f), e));
  x = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(INFINITYf), x);
#endif
  
  return x;
}

vfloat xexpf(vfloat d) {
  vint2 q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(R_LN2f)));
  vfloat s, u;

  s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf), d);
  s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf), s);

#if 0
  u = vcast_vf_f(0.00136324646882712841033936f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00836596917361021041870117f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0416710823774337768554688f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.166665524244308471679688f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.499999850988388061523438f));
#else
  u = vcast_vf_f(0.000198527617612853646278381);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00139304355252534151077271));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833336077630519866943359));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0416664853692054748535156));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.166666671633720397949219));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.5));
#endif
  u = vadd_vf_vf_vf(vcast_vf_f(1.0f), vmla_vf_vf_vf_vf(vmul_vf_vf_vf(s, s), u, s));

  u = vldexp_vf_vf_vi2(u, q);

  u = (vfloat)vandnot_vm_vo32_vm(vlt_vo_vf_vf(d, vcast_vf_f(-104)), (vmask)u);

  return u;
}

#ifdef ENABLE_NEON32
vfloat xsqrtf(vfloat d) {
  vfloat e = (vfloat)vadd_vi2_vi2_vi2(vcast_vi2_i(0x20000000), vand_vi2_vi2_vi2(vcast_vi2_i(0x7f000000), vsrl_vi2_vi2_i((vint2)d, 1)));
  vfloat m = (vfloat)vadd_vi2_vi2_vi2(vcast_vi2_i(0x3f000000), vand_vi2_vi2_vi2(vcast_vi2_i(0x01ffffff), (vint2)d));
  float32x4_t x = vrsqrteq_f32(m);
  x = vmulq_f32(x, vrsqrtsq_f32(m, vmulq_f32(x, x)));
  float32x4_t u = vmulq_f32(x, m);
  u = vmlaq_f32(u, vmlsq_f32(m, u, u), vmulq_f32(x, vdupq_n_f32(0.5)));
  e = (vfloat)vandnot_vm_vo32_vm(veq_vo_vf_vf(d, vcast_vf_f(0)), (vmask)e);
  u = vmul_vf_vf_vf(e, u);

  u = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vcast_vf_f(INFINITYf), u);
  u = (vfloat)vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(d), vlt_vo_vf_vf(d, vcast_vf_f(0))), (vmask)u);
  u = vmulsign_vf_vf_vf(u, d);

  return u;
}
#elif defined(ENABLE_CLANGVEC)
vfloat xsqrtf(vfloat d) {
  vfloat q = vsqrt_vf_vf(d);
  q = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0), q);
  return vsel_vf_vo_vf_vf(vispinf_vo_vf(d), INFINITYf, q);
}
#else
vfloat xsqrtf(vfloat d) { return vsqrt_vf_vf(d); }
#endif

vfloat xcbrtf(vfloat d) {
  vfloat x, y, q = vcast_vf_f(1.0), t;
  vint2 e, qu, re;

#ifdef ENABLE_GETEXP_DP
  vfloat s = d;
#endif
  e = vadd_vi2_vi2_vi2(vilogbk_vi2_vf(vabs_vf_vf(d)), vcast_vi2_i(1));
  d = vldexp_vf_vf_vi2(d, vneg_vi2_vi2(e));

  t = vadd_vf_vf_vf(vcast_vf_vi2(e), vcast_vf_f(6144));
  qu = vtruncate_vi2_vf(vmul_vf_vf_vf(t, vcast_vf_f(1.0f/3.0f)));
  re = vtruncate_vi2_vf(vsub_vf_vf_vf(t, vmul_vf_vf_vf(vcast_vf_vi2(qu), vcast_vf_f(3))));

  q = vsel_vf_vo_vf_vf(veq_vo_vi2_vi2(re, vcast_vi2_i(1)), vcast_vf_f(1.2599210498948731647672106f), q);
  q = vsel_vf_vo_vf_vf(veq_vo_vi2_vi2(re, vcast_vi2_i(2)), vcast_vf_f(1.5874010519681994747517056f), q);
  q = vldexp_vf_vf_vi2(q, vsub_vi2_vi2_vi2(qu, vcast_vi2_i(2048)));

  q = vmulsign_vf_vf_vf(q, d);
  d = vabs_vf_vf(d);

  x = vcast_vf_f(-0.601564466953277587890625f);
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.8208892345428466796875f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-5.532182216644287109375f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(5.898262500762939453125f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-3.8095417022705078125f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.2241256237030029296875f));

  y = vmul_vf_vf_vf(vmul_vf_vf_vf(d, x), x);
  y = vmul_vf_vf_vf(vsub_vf_vf_vf(y, vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(2.0f / 3.0f), y), vmla_vf_vf_vf_vf(y, x, vcast_vf_f(-1.0f)))), q);

#ifdef ENABLE_GETEXP_DP
  y = vsel_vf_vo_vf_vf(visinf_vo_vf(s), vmulsign_vf_vf_vf(vcast_vf_f(INFINITYf), s), y);
  y = vsel_vf_vo_vf_vf(veq_vo_vf_vf(s, vcast_vf_f(0)), vmulsign_vf_vf_vf(vcast_vf_f(0), s), y);
#endif
  
  return y;
}

vfloat xcbrtf_u1(vfloat d) {
  vfloat x, y, z, t;
  vfloat2 q2 = vcast_vf2_f_f(1, 0), u, v;
  vint2 e, qu, re;

#ifdef ENABLE_GETEXP_DP
  vfloat s = d;
#endif
  e = vadd_vi2_vi2_vi2(vilogbk_vi2_vf(vabs_vf_vf(d)), vcast_vi2_i(1));
  d = vldexp_vf_vf_vi2(d, vneg_vi2_vi2(e));

  t = vadd_vf_vf_vf(vcast_vf_vi2(e), vcast_vf_f(6144));
  qu = vtruncate_vi2_vf(vmul_vf_vf_vf(t, vcast_vf_f(1.0/3.0)));
  re = vtruncate_vi2_vf(vsub_vf_vf_vf(t, vmul_vf_vf_vf(vcast_vf_vi2(qu), vcast_vf_f(3))));

  q2 = vsel_vf2_vo_vf2_vf2(veq_vo_vi2_vi2(re, vcast_vi2_i(1)), vcast_vf2_f_f(1.2599210739135742188f, -2.4018701694217270415e-08), q2);
  q2 = vsel_vf2_vo_vf2_vf2(veq_vo_vi2_vi2(re, vcast_vi2_i(2)), vcast_vf2_f_f(1.5874010324478149414f,  1.9520385308169352356e-08), q2);

  q2.x = vmulsign_vf_vf_vf(q2.x, d); q2.y = vmulsign_vf_vf_vf(q2.y, d);
  d = vabs_vf_vf(d);

  x = vcast_vf_f(-0.601564466953277587890625f);
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.8208892345428466796875f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-5.532182216644287109375f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(5.898262500762939453125f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-3.8095417022705078125f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.2241256237030029296875f));

  y = vmul_vf_vf_vf(x, x); y = vmul_vf_vf_vf(y, y); x = vsub_vf_vf_vf(x, vmul_vf_vf_vf(vmlanp_vf_vf_vf_vf(d, y, x), vcast_vf_f(-1.0 / 3.0)));

  z = x;

  u = dfmul_vf2_vf_vf(x, x);
  u = dfmul_vf2_vf2_vf2(u, u);
  u = dfmul_vf2_vf2_vf(u, d);
  u = dfadd2_vf2_vf2_vf(u, vneg_vf_vf(x));
  y = vadd_vf_vf_vf(u.x, u.y);

  y = vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(-2.0 / 3.0), y), z);
  v = dfadd2_vf2_vf2_vf(dfmul_vf2_vf_vf(z, z), y);
  v = dfmul_vf2_vf2_vf(v, d);
  v = dfmul_vf2_vf2_vf2(v, q2);
  z = vldexp_vf_vf_vi2(vadd_vf_vf_vf(v.x, v.y), vsub_vi2_vi2_vi2(qu, vcast_vi2_i(2048)));

  z = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vmulsign_vf_vf_vf(vcast_vf_f(INFINITY), q2.x), z);
  z = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0)), (vfloat)vsignbit_vm_vf(q2.x), z);

#ifdef ENABLE_GETEXP_DP
  z = vsel_vf_vo_vf_vf(visinf_vo_vf(s), vmulsign_vf_vf_vf(vcast_vf_f(INFINITYf), s), z);
  z = vsel_vf_vo_vf_vf(veq_vo_vf_vf(s, vcast_vf_f(0)), vmulsign_vf_vf_vf(vcast_vf_f(0), s), z);
#endif

  return z;
}

static INLINE vfloat2 logkf(vfloat d) {
  vfloat2 x, x2;
  vfloat t, m;

#ifndef ENABLE_AVX512F
  vint2 e = vilogbk_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  m = vldexp_vf_vf_vi2(d, vneg_vi2_vi2(e));
#else
  vfloat e = _mm512_getexp_ps(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0f), e);
  m = _mm512_getmant_ps(d, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan);
#endif

  x = dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf_vf(vcast_vf_f(-1), m), dfadd2_vf2_vf_vf(vcast_vf_f(1), m));
  x2 = dfsqu_vf2_vf2(x);

  t = vcast_vf_f(0.240320354700088500976562);
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.285112679004669189453125));
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.400007992982864379882812));
  vfloat2 c = vcast_vf2_f_f(0.66666662693023681640625f, 3.69183861259614332084311e-09f);

#ifndef ENABLE_AVX512F
  return dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf(vcast_vf2_vf_vf(vcast_vf_f(0.69314718246459960938f), vcast_vf_f(-1.904654323148236017e-09f)),
					     vcast_vf_vi2(e)),
			    dfadd2_vf2_vf2_vf2(dfscale_vf2_vf2_vf(x, vcast_vf_f(2)),
					       dfmul_vf2_vf2_vf2(dfmul_vf2_vf2_vf2(x2, x),
								 dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf(x2, t), c))));
#else
  return dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf(vcast_vf2_vf_vf(vcast_vf_f(0.69314718246459960938f), vcast_vf_f(-1.904654323148236017e-09f)), e),
			    dfadd2_vf2_vf2_vf2(dfscale_vf2_vf2_vf(x, vcast_vf_f(2)),
					       dfmul_vf2_vf2_vf2(dfmul_vf2_vf2_vf2(x2, x),
								 dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf(x2, t), c))));
#endif
}

vfloat xlogf_u1(vfloat d) {
  vfloat2 s = logkf(d);
  vfloat x = vadd_vf_vf_vf(s.x, s.y);

  x = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(INFINITY), x);
#ifdef ENABLE_NEON32
  x = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(d, vcast_vf_f(1e-37f)), vcast_vf_f(-INFINITY), x);
#else
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0)), vcast_vf_f(-INFINITY), x);
#endif
  x = (vfloat)vor_vm_vo32_vm(vgt_vo_vf_vf(vcast_vf_f(0), d), (vmask)x);

  return x;
}

static INLINE vfloat expkf(vfloat2 d) {
  vfloat u = vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(R_LN2f));
  vint2 q = vrint_vi2_vf(u);
  vfloat2 s, t;

  s = dfadd2_vf2_vf2_vf(d, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf)));

  s = dfnormalize_vf2_vf2(s);

  u = vcast_vf_f(0.00136324646882712841033936f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.00836596917361021041870117f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0416710823774337768554688f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.166665524244308471679688f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.499999850988388061523438f));

  t = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfsqu_vf2_vf2(s), u));

  t = dfadd_vf2_vf_vf2(vcast_vf_f(1), t);
  u = vadd_vf_vf_vf(t.x, t.y);
  u = vldexp_vf_vf_vi2(u, q);

  u = (vfloat)vandnot_vm_vo32_vm(vlt_vo_vf_vf(d.x, vcast_vf_f(-104)), (vmask)u);
  
  return u;
}

vfloat xpowf(vfloat x, vfloat y) {
#if 1
  vopmask yisint = vor_vo_vo_vo(veq_vo_vf_vf(vtruncate_vf_vf(y), y), vgt_vo_vf_vf(vabs_vf_vf(y), vcast_vf_f(1 << 23)));
  vopmask yisodd = vand_vo_vo_vo(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vtruncate_vi2_vf(y), vcast_vi2_i(1)), vcast_vi2_i(1)), yisint);

#ifdef ENABLE_NEON32
  yisodd = vandnot_vm_vo32_vm(visinf_vo_vf(y), yisodd);
#endif

  vfloat result = expkf(dfmul_vf2_vf2_vf(logkf(vabs_vf_vf(x)), y));

  result = vmul_vf_vf_vf(result,
			 vsel_vf_vo_vf_vf(vgt_vo_vf_vf(x, vcast_vf_f(0)),
					  vcast_vf_f(1),
					  vsel_vf_vo_vf_vf(yisint, vsel_vf_vo_vf_vf(yisodd, vcast_vf_f(-1.0f), vcast_vf_f(1)), vcast_vf_f(NANf))));

  vfloat efx = vmulsign_vf_vf_vf(vsub_vf_vf_vf(vabs_vf_vf(x), vcast_vf_f(1)), y);

  result = vsel_vf_vo_vf_vf(visinf_vo_vf(y),
			    (vfloat)vandnot_vm_vo32_vm(vlt_vo_vf_vf(efx, vcast_vf_f(0.0f)),
						       (vmask)vsel_vf_vo_vf_vf(veq_vo_vf_vf(efx, vcast_vf_f(0.0f)),
									       vcast_vf_f(1.0f),
									       vcast_vf_f(INFINITYf))),
			    result);

  result = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0))),
			    vmul_vf_vf_vf(vsel_vf_vo_vf_vf(yisodd, vsign_vf_vf(x), vcast_vf_f(1)),
					  (vfloat)vandnot_vm_vo32_vm(vlt_vo_vf_vf(vsel_vf_vo_vf_vf(veq_vo_vf_vf(x, vcast_vf_f(0)), vneg_vf_vf(y), y), vcast_vf_f(0)),
								     (vmask)vcast_vf_f(INFINITYf))),
			    result);

  result = (vfloat)vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), (vmask)result);

  result = vsel_vf_vo_vf_vf(vor_vo_vo_vo(veq_vo_vf_vf(y, vcast_vf_f(0)), veq_vo_vf_vf(x, vcast_vf_f(1))), vcast_vf_f(1), result);

  return result;
#else
  return expkf(dfmul_vf2_vf2_vf(logkf(x), y));
#endif
}

static INLINE vfloat2 expk2f(vfloat2 d) {
  vfloat u = vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(R_LN2f));
  vint2 q = vrint_vi2_vf(u);
  vfloat2 s, t;

  s = dfadd2_vf2_vf2_vf(d, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf)));

  u = vcast_vf_f(0.00136324646882712841033936f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.00836596917361021041870117f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0416710823774337768554688f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.166665524244308471679688f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.499999850988388061523438f));

  t = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfsqu_vf2_vf2(s), u));

  t = dfadd_vf2_vf_vf2(vcast_vf_f(1), t);
  return dfscale_vf2_vf2_vf(dfscale_vf2_vf2_vf(t, vcast_vf_f(2)),
			    vpow2i_vf_vi2(vsub_vi2_vi2_vi2(q, vcast_vi2_i(1))));
}

vfloat xsinhf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vfloat2 d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0)));
  d = dfsub_vf2_vf2_vf2(d, dfrec_vf2_vf2(d));
  y = vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(0.5));

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(89)),
				    visnan_vo_vf(y)), vcast_vf_f(INFINITYf), y);
  y = vmulsign_vf_vf_vf(y, x);
  y = (vfloat)vor_vm_vo32_vm(visnan_vo_vf(x), (vmask)y);

  return y;
}

vfloat xcoshf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vfloat2 d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0)));
  d = dfadd_vf2_vf2_vf2(d, dfrec_vf2_vf2(d));
  y = vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(0.5));

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(89)),
				    visnan_vo_vf(y)), vcast_vf_f(INFINITYf), y);
  y = (vfloat)vor_vm_vo32_vm(visnan_vo_vf(x), (vmask)y);

  return y;
}

vfloat xtanhf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vfloat2 d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0)));
  vfloat2 e = dfrec_vf2_vf2(d);
  d = dfdiv_vf2_vf2_vf2(dfadd_vf2_vf2_vf2(d, dfneg_vf2_vf2(e)), dfadd_vf2_vf2_vf2(d, e));
  y = vadd_vf_vf_vf(d.x, d.y);

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(8.664339742f)),
				    visnan_vo_vf(y)), vcast_vf_f(1.0f), y);
  y = vmulsign_vf_vf_vf(y, x);
  y = (vfloat)vor_vm_vo32_vm(visnan_vo_vf(x), (vmask)y);

  return y;
}

static INLINE vfloat2 logk2f(vfloat2 d) {
  vfloat2 x, x2, m;
  vfloat t;
  vint2 e;

#ifndef ENABLE_AVX512F
  e = vilogbk_vi2_vf(vmul_vf_vf_vf(d.x, vcast_vf_f(1.0f/0.75f)));
#else
  e = vrint_vi2_vf(_mm512_getexp_ps(vmul_vf_vf_vf(d.x, vcast_vf_f(1.0f/0.75f))));
#endif
  m = dfscale_vf2_vf2_vf(d, vpow2i_vf_vi2(vneg_vi2_vi2(e)));

  x = dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf2_vf(m, vcast_vf_f(-1)), dfadd2_vf2_vf2_vf(m, vcast_vf_f(1)));
  x2 = dfsqu_vf2_vf2(x);

  t = vcast_vf_f(0.2392828464508056640625f);
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.28518211841583251953125f));
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.400005877017974853515625f));
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.666666686534881591796875f));

  return dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf(vcast_vf2_vf_vf(vcast_vf_f(0.69314718246459960938f), vcast_vf_f(-1.904654323148236017e-09f)),
					     vcast_vf_vi2(e)),
			    dfadd2_vf2_vf2_vf2(dfscale_vf2_vf2_vf(x, vcast_vf_f(2)), dfmul_vf2_vf2_vf(dfmul_vf2_vf2_vf2(x2, x), t)));
}

vfloat xasinhf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vopmask o = vgt_vo_vf_vf(y, vcast_vf_f(1));
  vfloat2 d;
  
  d = vsel_vf2_vo_vf2_vf2(o, dfrec_vf2_vf(x), vcast_vf2_vf_vf(y, vcast_vf_f(0)));
  d = dfsqrt_vf2_vf2(dfadd2_vf2_vf2_vf(dfsqu_vf2_vf2(d), vcast_vf_f(1)));
  d = vsel_vf2_vo_vf2_vf2(o, dfmul_vf2_vf2_vf(d, y), d);

  d = logk2f(dfnormalize_vf2_vf2(dfadd2_vf2_vf2_vf(d, x)));
  y = vadd_vf_vf_vf(d.x, d.y);

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(SQRT_FLT_MAX)),
				    visnan_vo_vf(y)),
		       vmulsign_vf_vf_vf(vcast_vf_f(INFINITYf), x), y);
  y = (vfloat)vor_vm_vo32_vm(visnan_vo_vf(x), (vmask)y);
  y = vsel_vf_vo_vf_vf(visnegzero_vo_vf(x), vcast_vf_f(-0.0), y);

  return y;
}

vfloat xacoshf(vfloat x) {
  vfloat2 d = logk2f(dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf2(dfsqrt_vf2_vf2(dfadd2_vf2_vf_vf(x, vcast_vf_f(1))), dfsqrt_vf2_vf2(dfadd2_vf2_vf_vf(x, vcast_vf_f(-1)))), x));
  vfloat y = vadd_vf_vf_vf(d.x, d.y);

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(SQRT_FLT_MAX)),
				    visnan_vo_vf(y)),
		       vcast_vf_f(INFINITYf), y);

  y = (vfloat)vandnot_vm_vo32_vm(veq_vo_vf_vf(x, vcast_vf_f(1.0f)), (vmask)y);

  y = (vfloat)vor_vm_vo32_vm(vlt_vo_vf_vf(x, vcast_vf_f(1.0f)), (vmask)y);
  y = (vfloat)vor_vm_vo32_vm(visnan_vo_vf(x), (vmask)y);

  return y;
}

vfloat xatanhf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vfloat2 d = logk2f(dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf_vf(vcast_vf_f(1), y), dfadd2_vf2_vf_vf(vcast_vf_f(1), vneg_vf_vf(y))));
  y = (vfloat)vor_vm_vo32_vm(vgt_vo_vf_vf(y, vcast_vf_f(1.0)), (vmask)vsel_vf_vo_vf_vf(veq_vo_vf_vf(y, vcast_vf_f(1.0)), vcast_vf_f(INFINITYf), vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(0.5))));

  y = (vfloat)vor_vm_vo32_vm(vor_vo_vo_vo(visinf_vo_vf(x), visnan_vo_vf(y)), (vmask)y);
  y = vmulsign_vf_vf_vf(y, x);
  y = (vfloat)vor_vm_vo32_vm(visnan_vo_vf(x), (vmask)y);

  return y;
}

vfloat xexp2f(vfloat a) {
  vfloat u = expkf(dfmul_vf2_vf2_vf(vcast_vf2_vf_vf(vcast_vf_f(0.69314718246459960938f), vcast_vf_f(-1.904654323148236017e-09f)), a));
  u = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(a, vcast_vf_f(128.0f)), vcast_vf_f(INFINITYf), u);
  u = (vfloat)vandnot_vm_vo32_vm(visminf_vo_vf(a), (vmask)u);
  return u;
}

vfloat xexp10f(vfloat a) {
  vfloat u = expkf(dfmul_vf2_vf2_vf(vcast_vf2_vf_vf(vcast_vf_f(2.3025851249694824219f), vcast_vf_f(-3.1975436520781386207e-08f)), a));
  u = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(a, vcast_vf_f(38.531839419103626f)), vcast_vf_f(INFINITYf), u);
  u = (vfloat)vandnot_vm_vo32_vm(visminf_vo_vf(a), (vmask)u);
  return u;
}

vfloat xexpm1f(vfloat a) {
  vfloat2 d = dfadd2_vf2_vf2_vf(expk2f(vcast_vf2_vf_vf(a, vcast_vf_f(0))), vcast_vf_f(-1.0));
  vfloat x = vadd_vf_vf_vf(d.x, d.y);
  x = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(a, vcast_vf_f(88.72283905206835f)), vcast_vf_f(INFINITYf), x);
  x = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(a, vcast_vf_f(-16.635532333438687426013570f)), vcast_vf_f(-1), x);
  x = vsel_vf_vo_vf_vf(visnegzero_vo_vf(a), vcast_vf_f(-0.0f), x);
  return x;
}

vfloat xlog10f(vfloat a) {
  vfloat2 d = dfmul_vf2_vf2_vf2(logkf(a), vcast_vf2_vf_vf(vcast_vf_f(0.43429449200630187988f), vcast_vf_f(-1.0103050118726031315e-08f)));
  vfloat x = vadd_vf_vf_vf(d.x, d.y);

  x = vsel_vf_vo_vf_vf(vispinf_vo_vf(a), vcast_vf_f(INFINITYf), x);
  x = (vfloat)vor_vm_vo32_vm(vgt_vo_vf_vf(vcast_vf_f(0), a), (vmask)x);
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(a, vcast_vf_f(0)), vcast_vf_f(-INFINITYf), x);

  return x;
}

vfloat xlog1pf(vfloat a) {
  vfloat2 d = logk2f(dfadd2_vf2_vf_vf(a, vcast_vf_f(1)));
  vfloat x = vadd_vf_vf_vf(d.x, d.y);

  x = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(a, vcast_vf_f(1e+38)), vcast_vf_f(INFINITYf), x);
  x = (vfloat)vor_vm_vo32_vm(vgt_vo_vf_vf(vcast_vf_f(-1), a), (vmask)x);
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(a, vcast_vf_f(-1)), vcast_vf_f(-INFINITYf), x);
  x = vsel_vf_vo_vf_vf(visnegzero_vo_vf(a), vcast_vf_f(-0.0f), x);

  return x;
}
