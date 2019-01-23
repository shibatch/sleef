//          Copyright Naoki Shibata 2010 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Always use -ffp-contract=off option to compile SLEEF.

#include <stdint.h>
#include <assert.h>
#include <limits.h>
#include <float.h>

#include "misc.h"

#define __SLEEFSIMDQP_C__

#if (defined(_MSC_VER))
#pragma fp_contract (off)
#endif

#ifdef ENABLE_PUREC_SCALAR
#define CONFIG 1
#include "helperpurec_scalar.h"
#ifdef DORENAME
#include "qrenamepurec_scalar.h"
#endif
#endif

#ifdef ENABLE_PURECFMA_SCALAR
#define CONFIG 2
#include "helperpurec_scalar.h"
#ifdef DORENAME
#include "qrenamepurecfma_scalar.h"
#endif
#endif

#ifdef ENABLE_SSE2
#define CONFIG 2
#include "helpersse2.h"
#ifdef DORENAME
#include "qrenamesse2.h"
#endif
#endif

#ifdef ENABLE_AVX2128
#define CONFIG 1
#include "helperavx2_128.h"
#ifdef DORENAME
#include "qrenameavx2128.h"
#endif
#endif

#ifdef ENABLE_AVX
#define CONFIG 1
#include "helperavx.h"
#ifdef DORENAME
#include "qrenameavx.h"
#endif
#endif

#ifdef ENABLE_FMA4
#define CONFIG 4
#include "helperavx.h"
#ifdef DORENAME
#include "qrenamefma4.h"
#endif
#endif

#ifdef ENABLE_AVX2
#define CONFIG 1
#include "helperavx2.h"
#ifdef DORENAME
#include "qrenameavx2.h"
#endif
#endif

#ifdef ENABLE_AVX512F
#define CONFIG 1
#include "helperavx512f.h"
#ifdef DORENAME
#include "qrenameavx512f.h"
#endif
#endif

#ifdef ENABLE_ADVSIMD
#define CONFIG 1
#include "helperadvsimd.h"
#ifdef DORENAME
#include "qrenameadvsimd.h"
#endif
#endif

#ifdef ENABLE_SVE
#define CONFIG 1
#include "helpersve.h"
#ifdef DORENAME
#include "qrenamesve.h"
#endif
#endif

#ifdef ENABLE_VSX
#define CONFIG 1
#include "helperpower_128.h"
#ifdef DORENAME
#include "qrenamevsx.h"
#endif
#endif

//

#include "dd.h"

#ifdef ENABLE_SVE
typedef __sizeless_struct vdouble3 {
  svfloat64_t x;
  svfloat64_t y;
  svfloat64_t z;
} vdouble3;

typedef __sizeless_struct {
  vmask e;
  vdouble3 dd;
} tdx;
#else
typedef struct {
  vdouble x, y, z;
} vdouble3;

typedef struct {
  vmask e;
  vdouble3 dd;
} tdx;
#endif

//

static INLINE CONST vopmask visnonnumber(vdouble x) {
  return veq64_vo_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(x), vcast_vm_i_i(0x7ff00000, 0)), vcast_vm_i_i(0x7ff00000, 0));
}

static INLINE CONST vmask vsignbit_vm_vd(vdouble d) {
  return vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
}

static INLINE CONST vdouble vmulsign_vd_vd_vd(vdouble x, vdouble y) {
  return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(x), vsignbit_vm_vd(y)));
}

static INLINE CONST vopmask vnot_vo64_vo64(vopmask x) {
  return vxor_vo_vo_vo(x, veq64_vo_vm_vm(vcast_vm_i_i(0, 0), vcast_vm_i_i(0, 0)));
}

static INLINE CONST vopmask vugt64_vo_vm_vm(vmask x, vmask y) { // unsigned compare
  x = vxor_vm_vm_vm(vcast_vm_i_i(0x80000000, 0), x);
  y = vxor_vm_vm_vm(vcast_vm_i_i(0x80000000, 0), y);
  return vgt64_vo_vm_vm(x, y);
}

static INLINE CONST vmask vilogb2k_vm_vd(vdouble d) {
  vmask m = vreinterpret_vm_vd(d);
  m = vsrl64_vm_vm_i(m, 20 + 32);
  m = vand_vm_vm_vm(m, vcast_vm_i_i(0, 0x7ff));
  m = vsub64_vm_vm_vm(m, vcast_vm_i_i(0, 0x3ff));
  return m;
}

static INLINE CONST vmask vilogb3k_vm_vd(vdouble d) {
  vmask m = vreinterpret_vm_vd(d);
  m = vsrl64_vm_vm_i(m, 20 + 32);
  m = vand_vm_vm_vm(m, vcast_vm_i_i(0, 0x7ff));
  return m;
}

static INLINE CONST vdouble vldexp3_vd_vd_vm(vdouble d, vmask q) {
  return vreinterpret_vd_vm(vadd64_vm_vm_vm(vreinterpret_vm_vd(d), vsll64_vm_vm_i(q, 52)));
}

static INLINE CONST vopmask vsignbit_vo_vd(vdouble d) {
  return veq64_vo_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
}

static INLINE CONST vmask2 vsel_vm2_vo_vm2_vm2(vopmask o, vmask2 x, vmask2 y) {
  return (vmask2) { .y = vsel_vm_vo64_vm_vm(o, x.y, y.y), .x = vsel_vm_vo64_vm_vm(o, x.x, y.x) };
}

static INLINE CONST vmask2 vadd128_vm2_vm2_vm2(vmask2 x, vmask2 y) {
  vmask2 r = { .y = vadd64_vm_vm_vm(x.y, y.y), .x = vadd64_vm_vm_vm(x.x, y.x) };
  r.y = vadd64_vm_vm_vm(r.y, vand_vm_vo64_vm(vugt64_vo_vm_vm(x.x, r.x), vcast_vm_i_i(0, 1)));
  return r;
}

static INLINE CONST vmask2 imdvm2(vmask x, vmask y) { vmask2 r = { x, y }; return r; }

// imm must be smaller than 64
#define vsrl128_vm2_vm2_i(m, imm)					\
  imdvm2(vor_vm_vm_vm(vsrl64_vm_vm_i(m.x, imm), vsll64_vm_vm_i(m.y, 64-imm)), vsrl64_vm_vm_i(m.y, imm))

static INLINE CONST vopmask visnonnumberq_vo_vm2(vmask2 a) {
  return veq64_vo_vm_vm(vand_vm_vm_vm(a.y, vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
}

static INLINE CONST vopmask visnonnumberq_vo_vm2_vm2(vmask2 a, vmask2 b) {
  vmask ma = vxor_vm_vm_vm(vand_vm_vm_vm(a.y, vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  vmask mb = vxor_vm_vm_vm(vand_vm_vm_vm(b.y, vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  return veq64_vo_vm_vm(vand_vm_vm_vm(ma, mb), vcast_vm_i_i(0, 0));
}

static INLINE CONST vopmask visnonnumberq_vo_vm2_vm2_vm2(vmask2 a, vmask2 b, vmask2 c) {
  vmask ma = vxor_vm_vm_vm(vand_vm_vm_vm(a.y, vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  vmask mb = vxor_vm_vm_vm(vand_vm_vm_vm(b.y, vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  vmask mc = vxor_vm_vm_vm(vand_vm_vm_vm(c.y, vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  return veq64_vo_vm_vm(vand_vm_vm_vm(vand_vm_vm_vm(ma, mb), mc), vcast_vm_i_i(0, 0));
}

static INLINE CONST vopmask visinfq_vo_vm2(vmask2 a) {
  vopmask o = veq64_vo_vm_vm(vand_vm_vm_vm(a.y, vcast_vm_i_i(0x7fffffff, 0xffffffff)), vcast_vm_i_i(0x7fff0000, 0));
  return vand_vo_vo_vo(o, veq64_vo_vm_vm(a.x, vcast_vm_i_i(0, 0)));
}

static INLINE CONST vopmask vispinfq_vo_vm2(vmask2 a) {
  return vand_vo_vo_vo(veq64_vo_vm_vm(a.y, vcast_vm_i_i(0x7fff0000, 0)), veq64_vo_vm_vm(a.x, vcast_vm_i_i(0, 0)));
}

static INLINE CONST vopmask visnanq_vo_vm2(vmask2 a) {
  return vandnot_vo_vo_vo(visinfq_vo_vm2(a), visnonnumberq_vo_vm2(a));
}

static INLINE CONST vopmask viszeroq_vo_vm2(vmask2 a) {
  return veq64_vo_vm_vm(vor_vm_vm_vm(vand_vm_vm_vm(a.y, vcast_vm_i_i(~0x80000000, ~0)), a.x), vcast_vm_i_i(0, 0));
}

//

static INLINE CONST vdouble2 ddscale_vd2_vd2_d(vdouble2 d, double s) { return ddscale_vd2_vd2_vd(d, vcast_vd_d(s)); }

static INLINE CONST vdouble3 vsel_vd3_vo_vd3_vd3(vopmask m, vdouble3 x, vdouble3 y) {
  vdouble3 r;
  r.x = vsel_vd_vo_vd_vd(m, x.x, y.x);
  r.y = vsel_vd_vo_vd_vd(m, x.y, y.y);
  r.z = vsel_vd_vo_vd_vd(m, x.z, y.z);
  return r;
}

// TD algorithms are based on Y. Hida et al., Library for double-double and quad-double arithmetic (2007).
static INLINE CONST vdouble2 twosum_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble2 r;
  r.x  = vadd_vd_vd_vd(x, y);
  vdouble v = vsub_vd_vd_vd(r.x, x);
  r.y = vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(r.x, v)), vsub_vd_vd_vd(y, v));
  return r;
}

static INLINE CONST vdouble2 twosumx_vd2_vd_vd_vd(vdouble x, vdouble y, vdouble s) {
  vdouble2 r;
  r.x  = vmla_vd_vd_vd_vd(y, s, x);
  vdouble v = vsub_vd_vd_vd(r.x, x);
  r.y = vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(r.x, v)), vmlapn_vd_vd_vd_vd(y, s, v));
  return r;
}

static INLINE CONST vdouble2 quicktwosum_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble2 r;
  r.x = vadd_vd_vd_vd(x, y);
  r.y = vadd_vd_vd_vd(vsub_vd_vd_vd(x, r.x), y);
  return r;
}

static INLINE CONST vdouble2 twoprod_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble2 r;
#ifdef ENABLE_FMA_DP
  r.x = vmul_vd_vd_vd(x, y);
  r.y = vfmapn_vd_vd_vd_vd(x, y, r.x);
#else
  vdouble xh = vmul_vd_vd_vd(x, vcast_vd_d((1 << 27)+1));
  xh = vsub_vd_vd_vd(xh, vsub_vd_vd_vd(xh, x));
  vdouble xl = vsub_vd_vd_vd(x, xh);
  vdouble yh = vmul_vd_vd_vd(y, vcast_vd_d((1 << 27)+1));
  yh = vsub_vd_vd_vd(yh, vsub_vd_vd_vd(yh, y));
  vdouble yl = vsub_vd_vd_vd(y, yh);

  r.x = vmul_vd_vd_vd(x, y);
  r.y = vadd_vd_5vd(vmul_vd_vd_vd(xh, yh), vneg_vd_vd(r.x), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yl));
#endif
  return r;
}

static INLINE CONST vdouble3 tdscale_vd3_vd3_vd(vdouble3 d, vdouble s) {
  return (vdouble3) { vmul_vd_vd_vd(d.x, s), vmul_vd_vd_vd(d.y, s), vmul_vd_vd_vd(d.z, s) };
}

static INLINE CONST vdouble3 tdscale_vd3_vd3_d(vdouble3 d, double s) { return tdscale_vd3_vd3_vd(d, vcast_vd_d(s)); }

static INLINE CONST vdouble3 tdquickrenormalize_vd3_vd3(vdouble3 td) {
  vdouble2 u = quicktwosum_vd2_vd_vd(td.x, td.y);
  vdouble2 v = quicktwosum_vd2_vd_vd(u.y, td.z);
  return (vdouble3) { u.x, v.x, v.y };
}

static INLINE CONST vdouble3 tdnormalize_vd3_vd3(vdouble3 td) {
  vdouble2 u = quicktwosum_vd2_vd_vd(td.x, td.y);
  vdouble2 v = quicktwosum_vd2_vd_vd(u.y, td.z);
  td.x = u.x; td.y = v.x; td.z = v.y;
  u = quicktwosum_vd2_vd_vd(td.x, td.y);
  return (vdouble3) { u.x, u.y, td.z };
}

static INLINE CONST vdouble3 tdadd2_vd3_vd3_vd3(vdouble3 x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(x.x, y.x);
  vdouble2 d1 = twosum_vd2_vd_vd(x.y, y.y);
  vdouble2 d3 = twosum_vd2_vd_vd(d0.y, d1.x);
  return tdnormalize_vd3_vd3((vdouble3) { d0.x, d3.x, vadd_vd_4vd(x.z, y.z, d1.y, d3.y) });
}

static INLINE CONST vdouble3 tdadd2_vd3_vd2_vd3(vdouble2 x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(x.x, y.x);
  vdouble2 d1 = twosum_vd2_vd_vd(x.y, y.y);
  vdouble2 d3 = twosum_vd2_vd_vd(d0.y, d1.x);
  return tdnormalize_vd3_vd3((vdouble3) { d0.x, d3.x, vadd_vd_3vd(d1.y, d3.y, y.z) });
}

static INLINE CONST vdouble3 tdadd_vd3_vd2_vd3(vdouble2 x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(x.x, y.x);
  vdouble2 d1 = twosum_vd2_vd_vd(x.y, y.y);
  vdouble2 d3 = twosum_vd2_vd_vd(d0.y, d1.x);
  return (vdouble3) { d0.x, d3.x, vadd_vd_3vd(d1.y, d3.y, y.z) };
}

static INLINE CONST vdouble3 tdadd2_vd3_vd_vd3(vdouble x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(x, y.x);
  vdouble2 d3 = twosum_vd2_vd_vd(d0.y, y.y);
  return tdnormalize_vd3_vd3((vdouble3) { d0.x, d3.x, vadd_vd_vd_vd(d3.y, y.z) });
}

static INLINE CONST vdouble3 tdadd_vd3_vd_vd3(vdouble x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(x, y.x);
  vdouble2 d3 = twosum_vd2_vd_vd(d0.y, y.y);
  return (vdouble3) { d0.x, d3.x, vadd_vd_vd_vd(d3.y, y.z) };
}

static INLINE CONST vdouble3 tdscaleadd2_vd3_vd3_vd3_vd(vdouble3 x, vdouble3 y, vdouble s) {
  vdouble2 d0 = twosumx_vd2_vd_vd_vd(x.x, y.x, s);
  vdouble2 d1 = twosumx_vd2_vd_vd_vd(x.y, y.y, s);
  vdouble2 d3 = twosum_vd2_vd_vd(d0.y, d1.x);
  return tdnormalize_vd3_vd3((vdouble3) { d0.x, d3.x, vadd_vd_3vd(vmla_vd_vd_vd_vd(y.z, s, x.z), d1.y, d3.y) });
}

static INLINE CONST vdouble3 tdmul2_vd3_vd3_vd3(vdouble3 x, vdouble3 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(x.x, y.x);
  vdouble2 d1 = twoprod_vd2_vd_vd(x.x, y.y);
  vdouble2 d2 = twoprod_vd2_vd_vd(x.y, y.x);
  vdouble2 d4 = twosum_vd2_vd_vd(d0.y, d1.x);
  vdouble2 d5 = twosum_vd2_vd_vd(d4.x, d2.x);

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(x.z, y.x, vmla_vd_vd_vd_vd(x.y, y.y, vmla_vd_vd_vd_vd(x.x, y.z, vadd_vd_vd_vd(d1.y, d2.y)))), d4.y, d5.y);

  return tdnormalize_vd3_vd3((vdouble3){ d0.x, d5.x, t2 });
}

static INLINE CONST vdouble3 tdmul_vd3_vd3_vd3(vdouble3 x, vdouble3 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(x.x, y.x);
  vdouble2 d1 = twoprod_vd2_vd_vd(x.x, y.y);
  vdouble2 d2 = twoprod_vd2_vd_vd(x.y, y.x);
  vdouble2 d4 = twosum_vd2_vd_vd(d0.y, d1.x);
  vdouble2 d5 = twosum_vd2_vd_vd(d4.x, d2.x);

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(x.z, y.x, vmla_vd_vd_vd_vd(x.y, y.y, vmla_vd_vd_vd_vd(x.x, y.z, vadd_vd_vd_vd(d1.y, d2.y)))), d4.y, d5.y);

  return tdquickrenormalize_vd3_vd3((vdouble3){ d0.x, d5.x, t2 });
}

static INLINE CONST vdouble3 tdmul_vd3_vd2_vd3(vdouble2 x, vdouble3 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(x.x, y.x);
  vdouble2 d1 = twoprod_vd2_vd_vd(x.x, y.y);
  vdouble2 d2 = twoprod_vd2_vd_vd(x.y, y.x);
  vdouble2 d4 = twosum_vd2_vd_vd(d0.y, d1.x);
  vdouble2 d5 = twosum_vd2_vd_vd(d4.x, d2.x);

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(x.y, y.y, vmla_vd_vd_vd_vd(x.x, y.z, vadd_vd_vd_vd(d1.y, d2.y))), d4.y, d5.y);

  return (vdouble3){ d0.x, d5.x, t2 };
}

static INLINE CONST vdouble3 tdmul_vd3_vd2_vd2(vdouble2 x, vdouble2 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(x.x, y.x);
  vdouble2 d1 = twoprod_vd2_vd_vd(x.x, y.y);
  vdouble2 d2 = twoprod_vd2_vd_vd(x.y, y.x);
  vdouble2 d4 = twosum_vd2_vd_vd(d0.y, d1.x);
  vdouble2 d5 = twosum_vd2_vd_vd(d4.x, d2.x);

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(x.y, y.y, vadd_vd_vd_vd(d1.y, d2.y)), d4.y, d5.y);

  return (vdouble3){ d0.x, d5.x, t2 };
}

static INLINE CONST vdouble3 tddiv2_vd3_vd3_vd3(vdouble3 n, vdouble3 q) {
  vdouble2 d = ddrec_vd2_vd2((vdouble2) {q.x, q.y});
  return tdmul2_vd3_vd3_vd3(n, tdadd_vd3_vd2_vd3(d, tdmul_vd3_vd2_vd3(ddscale_vd2_vd2_d(d, -1),
								      tdadd_vd3_vd_vd3(vcast_vd_d(-1), tdmul_vd3_vd2_vd3(d, q)))));
}

static INLINE CONST vdouble3 tddiv_vd3_vd3_vd3(vdouble3 n, vdouble3 q) {
  vdouble2 d = ddrec_vd2_vd2((vdouble2) {q.x, q.y});
  return tdmul_vd3_vd3_vd3(n, tdadd_vd3_vd2_vd3(d, tdmul_vd3_vd2_vd3(ddscale_vd2_vd2_d(d, -1),
								     tdadd_vd3_vd_vd3(vcast_vd_d(-1), tdmul_vd3_vd2_vd3(d, q)))));
}

static INLINE CONST vdouble3 tdrec_vd3_vd3(vdouble3 q) {
  vdouble2 d = ddrec_vd2_vd2((vdouble2) {q.x, q.y});
  return tdadd2_vd3_vd2_vd3(d, tdmul_vd3_vd2_vd3(ddscale_vd2_vd2_d(d, -1),
						 tdadd_vd3_vd_vd3(vcast_vd_d(-1), tdmul_vd3_vd2_vd3(d, q))));
}

static INLINE CONST vdouble3 tdrec_vd3_vd2(vdouble2 q) {
  vdouble2 d = ddrec_vd2_vd2((vdouble2) {q.x, q.y});
  return tdadd2_vd3_vd2_vd3(d, tdmul_vd3_vd2_vd3(ddscale_vd2_vd2_d(d, -1),
						 tdadd_vd3_vd_vd3(vcast_vd_d(-1), tdmul_vd3_vd2_vd2(d, q))));
}

static INLINE CONST vdouble3 tdsqrt_vd3_vd3(vdouble3 d) {
  vdouble2 t = ddsqrt_vd2_vd2((vdouble2) {d.x, d.y});
  vdouble3 r = tdmul2_vd3_vd3_vd3(tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd2_vd2(t, t)), tdrec_vd3_vd2(t));
  return tdscale_vd3_vd3_d(r, 0.5);
}

//

static INLINE CONST tdx vsel_tdx_vo64_tdx_tdx(vopmask o, tdx x, tdx y) {
  tdx r = { .dd = vsel_vd3_vo_vd3_vd3(o, x.dd, y.dd) };
  r.e = vsel_vm_vo64_vm_vm(o, x.e, y.e);
  return r;
}

static INLINE CONST tdx add2_tdx_tdx_tdx(tdx dd0, tdx dd1) {
  vmask ed = vsub64_vm_vm_vm(dd1.e, dd0.e);
  vdouble t = vldexp3_vd_vd_vm(vcast_vd_d(1), ed);

  tdx r = { .dd = tdscaleadd2_vd3_vd3_vd3_vd(dd0.dd, dd1.dd, t) };
  r.e = vilogb2k_vm_vd(r.dd.x);
  t = vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(r.e));

  vopmask o = veq_vo_vd_vd(dd0.dd.x, vcast_vd_d(0));
  r.e = vsel_vm_vo64_vm_vm(o, dd1.e, vadd64_vm_vm_vm(r.e, dd0.e));
  r.dd = tdscale_vd3_vd3_vd(r.dd, t);

  r = vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(ed, vcast_vm_i_i(0, 200)), dd1, r);
  r = vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(vcast_vm_i_i(-1, -200), ed), dd0, r);

  return r;
}

static INLINE CONST tdx mul2_tdx_tdx_tdx(tdx dd0, tdx dd1) {
  tdx r = { .dd = tdmul2_vd3_vd3_vd3(dd0.dd, dd1.dd) };
  r.e = vilogb2k_vm_vd(r.dd.x);
  r.dd = tdscale_vd3_vd3_vd(r.dd, vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(r.e)));
  r.e = vadd64_vm_vm_vm(vadd64_vm_vm_vm(vadd64_vm_vm_vm(dd0.e, dd1.e), vcast_vm_i_i(-1, -16383)), r.e);
  return r;
}

static INLINE CONST tdx mul_tdx_tdx_tdx(tdx dd0, tdx dd1) {
  tdx r = { .dd = tdmul_vd3_vd3_vd3(dd0.dd, dd1.dd) };
  r.e = vilogb2k_vm_vd(r.dd.x);
  r.dd = tdscale_vd3_vd3_vd(r.dd, vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(r.e)));
  r.e = vadd64_vm_vm_vm(vadd64_vm_vm_vm(vadd64_vm_vm_vm(dd0.e, dd1.e), vcast_vm_i_i(-1, -16383)), r.e);
  return r;
}

static INLINE CONST tdx div2_tdx_tdx_tdx(tdx dd0, tdx dd1) {
  tdx r = { .dd = tddiv2_vd3_vd3_vd3(dd0.dd, dd1.dd) };
  r.e = vilogb2k_vm_vd(r.dd.x);
  r.dd = tdscale_vd3_vd3_vd(r.dd, vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(r.e)));
  r.e = vadd64_vm_vm_vm(vadd64_vm_vm_vm(vsub64_vm_vm_vm(dd0.e, dd1.e), vcast_vm_i_i(0, 16383)), r.e);
  return r;
}

static INLINE CONST tdx sqrt_tdx_tdx(tdx dd0) {
  vopmask o = veq64_vo_vm_vm(vand_vm_vm_vm(dd0.e, vcast_vm_i_i(0, 1)), vcast_vm_i_i(0, 1));
  dd0.dd = tdscale_vd3_vd3_vd(dd0.dd, vsel_vd_vo_vd_vd(o, vcast_vd_d(1), vcast_vd_d(2)));
  tdx r = { .dd = tdsqrt_vd3_vd3(dd0.dd) };
  r.e = vilogb2k_vm_vd(r.dd.x);
  r.dd = tdscale_vd3_vd3_vd(r.dd, vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(r.e)));
  r.e = vadd64_vm_vm_vm(vcast_vm_vi2(vsra_vi2_vi2_i(vcast_vi2_vm(vadd64_vm_vm_vm(dd0.e, vcast_vm_i_i(0, 16383))), 1)), r.e);
  o = vneq_vo_vd_vd(dd0.dd.x, vcast_vd_d(0));
  r.dd.x = vreinterpret_vd_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(r.dd.x)));
  r.dd.y = vreinterpret_vd_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(r.dd.y)));
  r.dd.z = vreinterpret_vd_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(r.dd.z)));
  return r;
}

//

static tdx vcast_tdx_vf128(vmask2 f) {
  tdx r = { .e = vand_vm_vm_vm(vsrl64_vm_vm_i(f.y, 48), vcast_vm_i_i(0, 0x7fff)) };

  vmask signbit = vand_vm_vm_vm(f.y, vcast_vm_i_i(0x80000000, 0)), mx, my, mz;
  vopmask iszero = viszeroq_vo_vm2(f);

  mx = vand_vm_vm_vm(vsrl128_vm2_vm2_i(f, 60).x, vcast_vm_i_i(0xfffff, 0xffffffff));
  my = vand_vm_vm_vm(vsrl64_vm_vm_i(f.x, 8), vcast_vm_i_i(0xfffff, 0xffffffff));
  mz = vand_vm_vm_vm(vsll64_vm_vm_i(f.x, 44), vcast_vm_i_i(0xfffff, 0xffffffff));

  mx = vor_vm_vm_vm(mx, vcast_vm_i_i(0x3ff00000, 0));
  my = vor_vm_vm_vm(my, vcast_vm_i_i(0x3cb00000, 0));
  mz = vor_vm_vm_vm(mz, vcast_vm_i_i(0x39700000, 0));

  mx = vandnot_vm_vo64_vm(iszero, mx);
  my = vreinterpret_vm_vd(vsub_vd_vd_vd(vreinterpret_vd_vm(my), vcast_vd_d(1.0 / (1ULL << 52))));
  mz = vreinterpret_vm_vd(vsub_vd_vd_vd(vreinterpret_vd_vm(mz), vcast_vd_d(1.0 / ((1ULL << 52) * (double)(1ULL << 52)))));

  r.dd.x = vreinterpret_vd_vm(vor_vm_vm_vm(mx, signbit));
  r.dd.y = vreinterpret_vd_vm(vor_vm_vm_vm(my, signbit));
  r.dd.z = vreinterpret_vd_vm(vor_vm_vm_vm(mz, signbit));

  vopmask fisdenorm = veq64_vo_vm_vm(r.e, vcast_vm_i_i(0, 0));

  if (UNLIKELY(!vtestallzeros_i_vo64(vor_vo_vo_vo(veq64_vo_vm_vm(r.e, vcast_vm_i_i(0, 0x7fff)),
						  vandnot_vo_vo_vo(iszero, fisdenorm))))) {
    vopmask fisinf = vand_vo_vo_vo(veq64_vo_vm_vm(vand_vm_vm_vm(f.y, vcast_vm_i_i(0x7fffffff, 0xffffffff)),
						  vcast_vm_i_i(0x7fff0000, 0)),
				   veq64_vo_vm_vm(f.x, vcast_vm_i_i(0, 0)));
    vopmask fisnan = vandnot_vo_vo_vo(fisinf, veq64_vo_vm_vm(r.e, vcast_vm_i_i(0, 0x7fff)));

    tdx g = r;
    g.dd.x = vsub_vd_vd_vd(g.dd.x, vmulsign_vd_vd_vd(vcast_vd_d(1), g.dd.x));
    g.dd = tdnormalize_vd3_vd3(g.dd);
    g.e = vilogb2k_vm_vd(g.dd.x);
    g.dd = tdscale_vd3_vd3_vd(g.dd, vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(g.e)));
    g.e = vadd64_vm_vm_vm(g.e, vcast_vm_i_i(0, 1));
    r = vsel_tdx_vo64_tdx_tdx(fisdenorm, g, r);

    vdouble t = vreinterpret_vd_vm(vor_vm_vm_vm(signbit, vreinterpret_vm_vd(vcast_vd_d(SLEEF_INFINITY))));
    r.dd.x = vsel_vd_vo_vd_vd(fisnan, vcast_vd_d(SLEEF_INFINITY - SLEEF_INFINITY), vsel_vd_vo_vd_vd(fisinf, t, r.dd.x));
    r.dd.x = vreinterpret_vd_vm(vandnot_vm_vo64_vm(iszero, vreinterpret_vm_vd(r.dd.x)));
  }

  return r;
}

static INLINE CONST tdx vcast_tdx_vf128_fast(vmask2 f) {
  tdx r = { .e = vand_vm_vm_vm(vsrl64_vm_vm_i(f.y, 48), vcast_vm_i_i(0, 0x7fff)) };

  vmask signbit = vand_vm_vm_vm(f.y, vcast_vm_i_i(0x80000000, 0)), mx, my, mz;
  vopmask iszero = viszeroq_vo_vm2(f);

  mx = vand_vm_vm_vm(vsrl128_vm2_vm2_i(f, 60).x, vcast_vm_i_i(0xfffff, 0xffffffff));
  my = vand_vm_vm_vm(vsrl64_vm_vm_i(f.x, 8), vcast_vm_i_i(0xfffff, 0xffffffff));
  mz = vand_vm_vm_vm(vsll64_vm_vm_i(f.x, 44), vcast_vm_i_i(0xfffff, 0xffffffff));

  mx = vor_vm_vm_vm(mx, vcast_vm_i_i(0x3ff00000, 0));
  my = vor_vm_vm_vm(my, vcast_vm_i_i(0x3cb00000, 0));
  mz = vor_vm_vm_vm(mz, vcast_vm_i_i(0x39700000, 0));

  mx = vandnot_vm_vo64_vm(iszero, mx);
  my = vreinterpret_vm_vd(vsub_vd_vd_vd(vreinterpret_vd_vm(my), vcast_vd_d(1.0 / (1ULL << 52))));
  mz = vreinterpret_vm_vd(vsub_vd_vd_vd(vreinterpret_vd_vm(mz), vcast_vd_d(1.0 / ((1ULL << 52) * (double)(1ULL << 52)))));

  r.dd.x = vreinterpret_vd_vm(vor_vm_vm_vm(mx, signbit));
  r.dd.y = vreinterpret_vd_vm(vor_vm_vm_vm(my, signbit));
  r.dd.z = vreinterpret_vd_vm(vor_vm_vm_vm(mz, signbit));

  return r;
}

#define HBX 1.0
#define LOGXSCALE 1
#define XSCALE (1 << LOGXSCALE)
#define SX 61
#define HBY (1.0 / (1ULL << 53))
#define LOGYSCALE 4
#define YSCALE (1 << LOGYSCALE)
#define SY 11
#define HBZ (1.0 / ((1ULL << 53) * (double)(1ULL << 53)))
#define LOGZSCALE 10
#define ZSCALE (1 << LOGZSCALE)
#define SZ 36
#define HBR (1.0 / (1ULL << 60))

static vmask2 vcast_vf128_tdx_slowpath(tdx f) {
  vmask signbit = vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), vcast_vm_i_i(0x80000000, 0));
  vopmask iszero = veq_vo_vd_vd(f.dd.x, vcast_vd_d(0.0));

  f.dd.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), signbit));
  f.dd.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.y), signbit));
  f.dd.z = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.z), signbit));

  vopmask denorm = vgt64_vo_vm_vm(vcast_vm_i_i(0, 1), f.e);
  vdouble t = vldexp3_vd_vd_vm(vcast_vd_d(0.5), f.e);
  t = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vgt64_vo_vm_vm(vcast_vm_i_i(-1, -120), f.e), vreinterpret_vm_vd(t)));
  t = vsel_vd_vo_vd_vd(denorm, t, vcast_vd_d(1));
  f.e = vsel_vm_vo64_vm_vm(denorm, vcast_vm_i_i(0, 1), f.e);

  vopmask o = vlt_vo_vd_vd(f.dd.y, vcast_vd_d(-1.0/(1ULL << 57)/(1ULL << 57)));
  o = vand_vo_vo_vo(o, veq_vo_vd_vd(f.dd.x, vcast_vd_d(1)));
  o = vandnot_vo_vo_vo(veq64_vo_vm_vm(f.e, vcast_vm_i_i(0, 1)), o);
  t = vsel_vd_vo_vd_vd(o, vcast_vd_d(2), t);
  f.e = vsub64_vm_vm_vm(f.e, vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0, 2), vcast_vm_i_i(0, 1)));

  f.dd.x = vmul_vd_vd_vd(f.dd.x, t);
  f.dd.y = vmul_vd_vd_vd(f.dd.y, t);
  f.dd.z = vmul_vd_vd_vd(f.dd.z, t);

  t = vadd_vd_vd_vd(f.dd.y, vcast_vd_d(HBY * (1 << LOGYSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGYSCALE)));
  f.dd.z = vadd_vd_vd_vd(f.dd.z, vsub_vd_vd_vd(f.dd.y, vsub_vd_vd_vd(t, vcast_vd_d(HBZ * (1 << LOGZSCALE) + HBY * (1 << LOGYSCALE)))));
  f.dd.y = t;

  vdouble c = vsel_vd_vo_vd_vd(denorm, vcast_vd_d(HBX * XSCALE + HBX), vcast_vd_d(HBX * XSCALE));
  vdouble d = vadd_vd_vd_vd(f.dd.x, c);
  d = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGXSCALE)));
  t = vadd_vd_vd_vd(f.dd.y, vsub_vd_vd_vd(f.dd.x, vsub_vd_vd_vd(d, c)));
  f.dd.z = vadd_vd_vd_vd(f.dd.z, vadd_vd_vd_vd(vsub_vd_vd_vd(f.dd.y, t), vsub_vd_vd_vd(f.dd.x, vsub_vd_vd_vd(d, c))));
  f.dd.x = d;

  d = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGYSCALE)));
  f.dd.z = vadd_vd_vd_vd(f.dd.z, vsub_vd_vd_vd(t, d));
  f.dd.y = d;

  t = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(f.dd.z, vcast_vd_d(HBZ * ZSCALE)),
		       vcast_vd_d(HBZ * (ZSCALE/2)), vcast_vd_d(0));
  f.dd.y = vsub_vd_vd_vd(f.dd.y, t);
  f.dd.z = vadd_vd_vd_vd(f.dd.z, t);

  t = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(f.dd.y, vcast_vd_d(HBY * YSCALE)),
		       vcast_vd_d(HBY * (YSCALE/2)), vcast_vd_d(0));
  f.dd.x = vsub_vd_vd_vd(f.dd.x, t);
  f.dd.y = vadd_vd_vd_vd(f.dd.y, t);

  f.dd.z = vadd_vd_vd_vd(f.dd.z, vcast_vd_d(HBR));
  f.dd.z = vsub_vd_vd_vd(f.dd.z, vcast_vd_d(HBR));

  //

  vmask m = vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), vcast_vm_i_i(0xfffff, 0xffffffff));
  vmask2 r = { .y = vsrl64_vm_vm_i(vreinterpret_vm_vd(f.dd.x), 64-SX), .x = vsll64_vm_vm_i(m, SX) };

  f.dd.z = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.z), vcast_vm_i_i(0xfffff, 0xffffffff)));
  r.x = vor_vm_vm_vm(r.x, vsrl64_vm_vm_i(vreinterpret_vm_vd(f.dd.z), SZ));

  f.dd.y = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.y), vcast_vm_i_i(0xfffff, 0xffffffff)));
  vmask2 s = { .y = vsrl64_vm_vm_i(vreinterpret_vm_vd(f.dd.y), 64-SY), .x = vsll64_vm_vm_i(vreinterpret_vm_vd(f.dd.y), SY) };
  r = vadd128_vm2_vm2_vm2(r, s);

  r.y = vand_vm_vm_vm(r.y, vsel_vm_vo64_vm_vm(denorm, vcast_vm_i_i(0xffff, 0xffffffff), vcast_vm_i_i(0x3ffff, 0xffffffff)));
  m = vsll64_vm_vm_i(vand_vm_vm_vm(f.e, vcast_vm_i_i(0xffffffff, ~(~0UL << 15))), 48);
  r.y = vcast_vm_vi2(vadd_vi2_vi2_vi2(vcast_vi2_vm(r.y), vcast_vi2_vm(m)));

  r.y = vandnot_vm_vo64_vm(iszero, r.y);
  r.x = vandnot_vm_vo64_vm(iszero, r.x);

  o = vor_vo_vo_vo(vgt64_vo_vm_vm(f.e, vcast_vm_i_i(0, 32765)), veq_vo_vd_vd(f.dd.x, vcast_vd_d(SLEEF_INFINITY)));
  r.y = vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0x7fff0000, 0), r.y);
  r.x = vandnot_vm_vo64_vm(o, r.x);

  o = vandnot_vo_vo_vo(veq_vo_vd_vd(f.dd.x, vcast_vd_d(SLEEF_INFINITY)), visnonnumber(f.dd.x));
  r.y = vor_vm_vo64_vm(o, r.y);
  r.x = vor_vm_vo64_vm(o, r.x);

  r.y = vor_vm_vm_vm(r.y, signbit);

  return r;
}

static vmask2 vcast_vf128_tdx(tdx f) {
  vopmask o = vor_vo_vo_vo(vgt64_vo_vm_vm(vcast_vm_i_i(0, 2), f.e), vgt64_vo_vm_vm(f.e, vcast_vm_i_i(0, 0x7ffd)));
  o = vor_vo_vo_vo(o, visnonnumber(f.dd.x));
  o = vandnot_vo_vo_vo(veq_vo_vd_vd(vcast_vd_d(0), f.dd.x), o);
  if (UNLIKELY(!vtestallzeros_i_vo64(o))) return vcast_vf128_tdx_slowpath(f);

  vmask signbit = vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), vcast_vm_i_i(0x80000000, 0));
  vopmask iszero = veq_vo_vd_vd(f.dd.x, vcast_vd_d(0.0));

  f.dd.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), signbit));
  f.dd.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.y), signbit));
  f.dd.z = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.z), signbit));

  o = vand_vo_vo_vo(veq_vo_vd_vd(f.dd.x, vcast_vd_d(1.0)), vlt_vo_vd_vd(f.dd.y, vcast_vd_d(0.0)));
  vint2 i2 = vcast_vi2_vm(vand_vm_vo64_vm(o, vcast_vm_vi2(vcast_vi2_vm(vcast_vm_i_i(1 << 20, 0)))));
  f.dd.x = vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(f.dd.x), i2));
  f.dd.y = vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(f.dd.y), i2));
  f.dd.z = vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(f.dd.z), i2));
  f.e = vsub64_vm_vm_vm(f.e, vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0, 2), vcast_vm_i_i(0, 1)));
  
  vdouble t = vadd_vd_vd_vd(f.dd.y, vcast_vd_d(HBY * (1 << LOGYSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGYSCALE)));
  f.dd.z = vadd_vd_vd_vd(f.dd.z, vsub_vd_vd_vd(f.dd.y, vsub_vd_vd_vd(t, vcast_vd_d(HBZ * (1 << LOGZSCALE) + HBY * (1 << LOGYSCALE)))));
  f.dd.y = t;

  t = vadd_vd_vd_vd(f.dd.x, vcast_vd_d(HBX * (1 << LOGXSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGXSCALE)));
  f.dd.y = vadd_vd_vd_vd(f.dd.y, vsub_vd_vd_vd(f.dd.x, vsub_vd_vd_vd(t, vcast_vd_d(HBX * (1 << LOGXSCALE)))));
  f.dd.x = t;

  f.dd.z = vadd_vd_vd_vd(f.dd.z, vcast_vd_d(HBZ * ((1 << LOGZSCALE)/2) + HBR));
  f.dd.z = vsub_vd_vd_vd(f.dd.z, vcast_vd_d(HBR));
  f.dd.y = vadd_vd_vd_vd(f.dd.y, vcast_vd_d(HBY * ((1 << LOGYSCALE)/2) - HBZ * ((1 << LOGZSCALE)/2)));
  f.dd.x = vsub_vd_vd_vd(f.dd.x, vcast_vd_d(HBY * ((1 << LOGYSCALE)/2)));

  //

  f.dd.x = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), vcast_vm_i_i(0xfffff, 0xffffffff)));
  vmask2 r = { .y = vsrl64_vm_vm_i(vreinterpret_vm_vd(f.dd.x), 64-SX), .x = vsll64_vm_vm_i(vreinterpret_vm_vd(f.dd.x), SX) };

  f.dd.z = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.z), vcast_vm_i_i(0xfffff, 0xffffffff)));
  r.x = vor_vm_vm_vm(r.x, vsrl64_vm_vm_i(vreinterpret_vm_vd(f.dd.z), SZ));

  f.dd.y = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.y), vcast_vm_i_i(0xfffff, 0xffffffff)));
  vmask2 s = { .y = vsrl64_vm_vm_i(vreinterpret_vm_vd(f.dd.y), 64-SY), .x = vsll64_vm_vm_i(vreinterpret_vm_vd(f.dd.y), SY) };
  r = vadd128_vm2_vm2_vm2(r, s);

  r.y = vand_vm_vm_vm(r.y, vcast_vm_i_i(0x3ffff, 0xffffffff));
  f.e = vsll64_vm_vm_i(vand_vm_vm_vm(f.e, vcast_vm_i_i(0xffffffff, ~(~0UL << 15))), 48);
  r.y = vcast_vm_vi2(vadd_vi2_vi2_vi2(vcast_vi2_vm(r.y), vcast_vi2_vm(f.e)));

  r.y = vandnot_vm_vo64_vm(iszero, r.y);
  r.x = vandnot_vm_vo64_vm(iszero, r.x);

  r.y = vor_vm_vm_vm(r.y, signbit);

  return r;
}

static INLINE CONST vmask2 vcast_vf128_tdx_fast(tdx f) {
  vmask signbit = vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), vcast_vm_i_i(0x80000000, 0));
  vopmask iszero = veq_vo_vd_vd(f.dd.x, vcast_vd_d(0.0));

  f.dd.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), signbit));
  f.dd.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.y), signbit));
  f.dd.z = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.z), signbit));

  vopmask o = vand_vo_vo_vo(veq_vo_vd_vd(f.dd.x, vcast_vd_d(1.0)), vlt_vo_vd_vd(f.dd.y, vcast_vd_d(0.0)));
  vint2 i2 = vcast_vi2_vm(vand_vm_vo64_vm(o, vcast_vm_vi2(vcast_vi2_vm(vcast_vm_i_i(1 << 20, 0)))));
  f.dd.x = vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(f.dd.x), i2));
  f.dd.y = vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(f.dd.y), i2));
  f.dd.z = vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(f.dd.z), i2));
  f.e = vsub64_vm_vm_vm(f.e, vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0, 2), vcast_vm_i_i(0, 1)));
  
  vdouble t = vadd_vd_vd_vd(f.dd.y, vcast_vd_d(HBY * (1 << LOGYSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGYSCALE)));
  f.dd.z = vadd_vd_vd_vd(f.dd.z, vsub_vd_vd_vd(f.dd.y, vsub_vd_vd_vd(t, vcast_vd_d(HBZ * (1 << LOGZSCALE) + HBY * (1 << LOGYSCALE)))));
  f.dd.y = t;

  t = vadd_vd_vd_vd(f.dd.x, vcast_vd_d(HBX * (1 << LOGXSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGXSCALE)));
  f.dd.y = vadd_vd_vd_vd(f.dd.y, vsub_vd_vd_vd(f.dd.x, vsub_vd_vd_vd(t, vcast_vd_d(HBX * (1 << LOGXSCALE)))));
  f.dd.x = t;

  f.dd.z = vadd_vd_vd_vd(f.dd.z, vcast_vd_d(HBZ * ((1 << LOGZSCALE)/2) + HBR));
  f.dd.z = vsub_vd_vd_vd(f.dd.z, vcast_vd_d(HBR));
  f.dd.y = vadd_vd_vd_vd(f.dd.y, vcast_vd_d(HBY * ((1 << LOGYSCALE)/2) - HBZ * ((1 << LOGZSCALE)/2)));
  f.dd.x = vsub_vd_vd_vd(f.dd.x, vcast_vd_d(HBY * ((1 << LOGYSCALE)/2)));

  //

  f.dd.x = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), vcast_vm_i_i(0xfffff, 0xffffffff)));
  vmask2 r = { .y = vsrl64_vm_vm_i(vreinterpret_vm_vd(f.dd.x), 64-SX), .x = vsll64_vm_vm_i(vreinterpret_vm_vd(f.dd.x), SX) };

  f.dd.z = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.z), vcast_vm_i_i(0xfffff, 0xffffffff)));
  r.x = vor_vm_vm_vm(r.x, vsrl64_vm_vm_i(vreinterpret_vm_vd(f.dd.z), SZ));

  f.dd.y = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.y), vcast_vm_i_i(0xfffff, 0xffffffff)));
  vmask2 s = { .y = vsrl64_vm_vm_i(vreinterpret_vm_vd(f.dd.y), 64-SY), .x = vsll64_vm_vm_i(vreinterpret_vm_vd(f.dd.y), SY) };
  r = vadd128_vm2_vm2_vm2(r, s);

  r.y = vand_vm_vm_vm(r.y, vcast_vm_i_i(0x3ffff, 0xffffffff));
  f.e = vsll64_vm_vm_i(vand_vm_vm_vm(f.e, vcast_vm_i_i(0xffffffff, ~(~0UL << 15))), 48);
  r.y = vcast_vm_vi2(vadd_vi2_vi2_vi2(vcast_vi2_vm(r.y), vcast_vi2_vm(f.e)));

  r.y = vandnot_vm_vo64_vm(iszero, r.y);
  r.x = vandnot_vm_vo64_vm(iszero, r.x);

  r.y = vor_vm_vm_vm(r.y, signbit);

  return r;
}

//

EXPORT CONST vargquad xaddq_u05(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 b = vcast_vm2_aq(ab);

  vmask ea = vand_vm_vm_vm(vsrl64_vm_vm_i(a.y, 48), vcast_vm_i_i(0, 0x7fff));
  vmask eb = vand_vm_vm_vm(vsrl64_vm_vm_i(b.y, 48), vcast_vm_i_i(0, 0x7fff));

  vopmask oa = vor_vo_vo_vo(viszeroq_vo_vm2(a), 
			    vand_vo_vo_vo(vgt64_vo_vm_vm(ea, vcast_vm_i_i(0, 120)),
					  vgt64_vo_vm_vm(vcast_vm_i_i(0, 0x7ffe), ea)));
  vopmask ob = vor_vo_vo_vo(viszeroq_vo_vm2(b), 
			    vand_vo_vo_vo(vgt64_vo_vm_vm(eb, vcast_vm_i_i(0, 120)),
					  vgt64_vo_vm_vm(vcast_vm_i_i(0, 0x7ffe), eb)));

  if (LIKELY(vtestallones_i_vo64(vand_vo_vo_vo(oa, ob)))) {
    vmask2 r = vcast_vf128_tdx_fast(add2_tdx_tdx_tdx(vcast_tdx_vf128_fast(a), vcast_tdx_vf128_fast(b)));
    r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(vand_vm_vm_vm(a.y, b.y), vcast_vm_i_i(0x80000000, 0)));
    return vcast_aq_vm2(r);
  }

  vmask2 r = vcast_vf128_tdx(add2_tdx_tdx_tdx(vcast_tdx_vf128(a), vcast_tdx_vf128(b)));

  r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(vand_vm_vm_vm(a.y, b.y), vcast_vm_i_i(0x80000000, 0)));

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonnumberq_vo_vm2_vm2(a, b)))) {
    vopmask aisinf = visinfq_vo_vm2(a), bisinf = visinfq_vo_vm2(b);
    vopmask aisnan = visnanq_vo_vm2(a), bisnan = visnanq_vo_vm2(b);
    vopmask o = veq64_vo_vm_vm(a.y, vxor_vm_vm_vm(b.y, vcast_vm_i_i(0x80000000, 0)));
    r = vsel_vm2_vo_vm2_vm2(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(bisnan, aisinf)), a, r);
    r = vsel_vm2_vo_vm2_vm2(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(aisnan, bisinf)), b, r);
  }

  return vcast_aq_vm2(r);
}

EXPORT CONST vargquad xmulq_u05(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 b = vcast_vm2_aq(ab);

  vmask ea = vand_vm_vm_vm(vsrl64_vm_vm_i(a.y, 48), vcast_vm_i_i(0, 0x7fff));
  vmask eb = vand_vm_vm_vm(vsrl64_vm_vm_i(b.y, 48), vcast_vm_i_i(0, 0x7fff));
  vopmask oa = vand_vo_vo_vo(vgt64_vo_vm_vm(ea, vcast_vm_i_i(0, 120)),
			     vgt64_vo_vm_vm(vcast_vm_i_i(0, 0x7ffe), ea));
  vopmask ob = vand_vo_vo_vo(vgt64_vo_vm_vm(eb, vcast_vm_i_i(0, 120)),
			     vgt64_vo_vm_vm(vcast_vm_i_i(0, 0x7ffe), eb));
  vopmask oc = vand_vo_vo_vo(vgt64_vo_vm_vm(vadd64_vm_vm_vm(ea, eb), vcast_vm_i_i(0, 120+16383)),
			     vgt64_vo_vm_vm(vcast_vm_i_i(0, 0x7ffe +16383), vadd64_vm_vm_vm(ea, eb)));
  if (LIKELY(vtestallones_i_vo64(vandnot_vo_vo_vo(visnonnumberq_vo_vm2_vm2(a, b), 
						  vor_vo_vo_vo(vor_vo_vo_vo(viszeroq_vo_vm2(a), viszeroq_vo_vm2(b)),
							       vand_vo_vo_vo(vand_vo_vo_vo(oa, ob), oc)))))) {
    vmask2 r = vcast_vf128_tdx_fast(mul_tdx_tdx_tdx(vcast_tdx_vf128_fast(a), vcast_tdx_vf128_fast(b)));
    r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(vxor_vm_vm_vm(a.y, b.y), vcast_vm_i_i(0x80000000, 0)));
    return vcast_aq_vm2(r);
  }

  vmask2 r = vcast_vf128_tdx(mul2_tdx_tdx_tdx(vcast_tdx_vf128(a), vcast_tdx_vf128(b)));

  r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(vxor_vm_vm_vm(a.y, b.y), vcast_vm_i_i(0x80000000, 0)));

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonnumberq_vo_vm2_vm2(a, b)))) {
    vopmask aisinf = visinfq_vo_vm2(a), bisinf = visinfq_vo_vm2(b);
    vopmask aisnan = visnanq_vo_vm2(a), bisnan = visnanq_vo_vm2(b);
    vopmask aiszero = viszeroq_vo_vm2(a), biszero = viszeroq_vo_vm2(b);
    vopmask o = vor_vo_vo_vo(aisinf, bisinf);
    r.y = vsel_vm_vo64_vm_vm(o, vor_vm_vm_vm(vcast_vm_i_i(0x7fff0000, 0),
					   vand_vm_vm_vm(vxor_vm_vm_vm(a.y, b.y), vcast_vm_i_i(0x80000000, 0))), r.y);
    r.x = vandnot_vm_vo64_vm(o, r.x);

    o = vor_vo_vo_vo(vand_vo_vo_vo(aiszero, bisinf), vand_vo_vo_vo(biszero, aisinf));
    o = vor_vo_vo_vo(vor_vo_vo_vo(o, aisnan), bisnan);
    r.y = vor_vm_vo64_vm(o, r.y);
    r.x = vor_vm_vo64_vm(o, r.x);
  }

  return vcast_aq_vm2(r);
}

EXPORT CONST vargquad xdivq_u05(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 b = vcast_vm2_aq(ab);
  vmask2 r = vcast_vf128_tdx(div2_tdx_tdx_tdx(vcast_tdx_vf128(a), vcast_tdx_vf128(b)));

  r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(vxor_vm_vm_vm(a.y, b.y), vcast_vm_i_i(0x80000000, 0)));

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonnumberq_vo_vm2_vm2_vm2(a, b, r)))) {
    vopmask aisinf = visinfq_vo_vm2(a), bisinf = visinfq_vo_vm2(b);
    vopmask aisnan = visnanq_vo_vm2(a), bisnan = visnanq_vo_vm2(b);
    vopmask aiszero = viszeroq_vo_vm2(a), biszero = viszeroq_vo_vm2(b);
    vmask signbit = vand_vm_vm_vm(vxor_vm_vm_vm(a.y, b.y), vcast_vm_i_i(0x80000000, 0));

    r.y = vsel_vm_vo64_vm_vm(bisinf, signbit, r.y);
    r.x = vandnot_vm_vo64_vm(bisinf, r.x);

    vopmask o = vor_vo_vo_vo(aisinf, biszero);
    vmask m = vor_vm_vm_vm(vcast_vm_i_i(0x7fff0000, 0), signbit);
    r.y = vsel_vm_vo64_vm_vm(o, m, r.y);
    r.x = vandnot_vm_vo64_vm(o, r.x);

    o = vand_vo_vo_vo(aiszero, biszero);
    o = vor_vo_vo_vo(o, vand_vo_vo_vo(aisinf, bisinf));
    o = vor_vo_vo_vo(o, vor_vo_vo_vo(aisnan, bisnan));
    r.y = vor_vm_vo64_vm(o, r.y);
    r.x = vor_vm_vo64_vm(o, r.x);
  }  

  return vcast_aq_vm2(r);
}

EXPORT CONST vargquad xsqrtq_u05(vargquad aa) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 r = vcast_vf128_tdx(sqrt_tdx_tdx(vcast_tdx_vf128(a)));

  r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(a.y, vcast_vm_i_i(0x80000000, 0)));
  vopmask aispinf = vispinfq_vo_vm2(a);

  r.y = vsel_vm_vo64_vm_vm(aispinf, vcast_vm_i_i(0x7fff0000, 0), r.y);
  r.x = vandnot_vm_vo64_vm(aispinf, r.x);

  return vcast_aq_vm2(r);
}

//

#ifdef ENABLE_MAIN
// gcc -DENABLE_MAIN -Wno-attributes -I../libm -I../quad-tester -I../common -I../arch -DUSEMPFR -DENABLE_AVX2 -mavx2 -mfma sleefsimdqp.c ../common/common.c ../quad-tester/qtesterutil.c -lm -lmpfr
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include <time.h>
#include <unistd.h>

#include "qtesterutil.h"

int main(int argc, char **argv) {
  xsrand(time(NULL) + (int)getpid());
  int lane = xrand() % VECTLENDP;
  printf("lane = %d\n", lane);

  char s[200];
  mpfr_set_default_prec(1024);
  mpfr_t fr0, fr1, fr2;
  mpfr_inits(fr0, fr1, fr2, NULL);

  mpfr_set_d(fr0, 0, GMP_RNDN);
  if (argc >= 2) mpfr_set_str(fr0, argv[1], 10, GMP_RNDN);
  Sleef_quad q0 = mpfr_get_f128(fr0, GMP_RNDN);
  mpfr_set_f128(fr0, q0, GMP_RNDN);
  if (argc >= 2) printf("arg0 : %s\n", sprintfr(fr0));
  vargquad a0;
  memrand(&a0, sizeof(vargquad));
  a0.s[lane] = q0;

  mpfr_set_d(fr1, 0, GMP_RNDN);
  if (argc >= 3) mpfr_set_str(fr1, argv[2], 10, GMP_RNDN);
  Sleef_quad q1 = mpfr_get_f128(fr1, GMP_RNDN);
  mpfr_set_f128(fr1, q1, GMP_RNDN);
  if (argc >= 3) printf("arg1 : %s\n", sprintfr(fr1));
  vargquad a1;
  memrand(&a1, sizeof(vargquad));
  a1.s[lane] = q1;

  //

#if 1
  vargquad a2 = xaddq_u05(a0, a1);
  mpfr_add(fr2, fr0, fr1, GMP_RNDN);
#endif

#if 0
  vargquad a2 = xmulq_u05(a0, a1);
  mpfr_mul(fr2, fr0, fr1, GMP_RNDN);
#endif

#if 0
  vargquad a2 = xdivq_u05(a0, a1);
  mpfr_div(fr2, fr0, fr1, GMP_RNDN);
#endif

#if 0
  vargquad a2 = xsqrtq_u05(a0);
  mpfr_sqrt(fr2, fr0, GMP_RNDN);
#endif

  //

  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  Sleef_quad q2 = a2.s[lane];
  mpfr_set_f128(fr2, q2, GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
}
#endif
