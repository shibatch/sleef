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

#if defined(ENABLE_MAIN)
#include <stdio.h>

static void printvmask(char *mes, vmask g) {
  uint64_t u[VECTLENDP];
  vstoreu_v_p_vd((double *)u, vreinterpret_vd_vm(g));
  printf("%s ", mes);
  for(int i=0;i<VECTLENDP;i++) printf("%016lx : ", (unsigned long)u[i]);
  printf("\n");
}

#if !defined(ENABLE_SVE)
static void printvopmask(char *mes, vopmask g) {
  union {
    vopmask g;
    uint8_t u[sizeof(vopmask)];
  } cnv = { .g = g };
  printf("%s ", mes);
  for(int i=0;i<sizeof(vopmask);i++) printf("%02x", cnv.u[i]);
  printf("\n");
}
#else
static void printvopmask(char *mes, vopmask g) {
  vmask m = vand_vm_vo64_vm(g, vcast_vm_i_i(-1, -1));
  printvmask(mes, m);
}
#endif

static void printvdouble(char *mes, vdouble vd) {
  double u[VECTLENDP];
  vstoreu_v_p_vd((double *)u, vd);
  printf("%s ", mes);
  for(int i=0;i<VECTLENDP;i++) printf("%.20g : ", u[i]);
  printf("\n");
}

static void printvint(char *mes, vint vi) {
  uint32_t u[VECTLENDP];
  vstoreu_v_p_vi((int32_t *)u, vi);
  printf("%s ", mes);
  for(int i=0;i<VECTLENDP;i++) printf("%08x : ", (unsigned)u[i]);
  printf("\n");
}

static void printvmask2(char *mes, vmask2 g) {
  uint64_t u[VECTLENDP*2];
  vstoreu_v_p_vd((double *)u, vreinterpret_vd_vm(g.x));
  vstoreu_v_p_vd((double *)&u[VECTLENDP], vreinterpret_vd_vm(g.y));
  printf("%s ", mes);
  for(int i=0;i<VECTLENDP*2;i++) printf("%016lx : ", (unsigned long)(u[i]));
  printf("\n");
}
#endif

//

static INLINE CONST vopmask visnonfinite(vdouble x) {
  return veq64_vo_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(x), vcast_vm_i_i(0x7ff00000, 0)), vcast_vm_i_i(0x7ff00000, 0));
}

static INLINE CONST vmask vsignbit_vm_vd(vdouble d) {
  return vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
}

static INLINE CONST vdouble vclearlsb_vd_vd_i(vdouble d, int n) {
  return vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vcast_vm_i_i(-1, ~0U << n)));
}

static INLINE CONST vdouble vcast_vd_vm(vmask m) { return vcast_vd_vi(vcast_vi_vm(m)); } // 32 bit only
static INLINE CONST vmask vtruncate_vm_vd(vdouble d) { return vcast_vm_vi(vtruncate_vi_vd(d)); }

static INLINE CONST vdouble vmulsign_vd_vd_vd(vdouble x, vdouble y) {
  return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(x), vsignbit_vm_vd(y)));
}

static INLINE CONST vopmask vlt64_vo_vm_vm(vmask x, vmask y) { return vgt64_vo_vm_vm(y, x); }

static INLINE CONST vopmask vnot_vo64_vo64(vopmask x) {
  return vxor_vo_vo_vo(x, veq64_vo_vm_vm(vcast_vm_i_i(0, 0), vcast_vm_i_i(0, 0)));
}

static INLINE CONST vopmask vugt64_vo_vm_vm(vmask x, vmask y) { // unsigned compare
  x = vxor_vm_vm_vm(vcast_vm_i_i(0x80000000, 0), x);
  y = vxor_vm_vm_vm(vcast_vm_i_i(0x80000000, 0), y);
  return vgt64_vo_vm_vm(x, y);
}

static INLINE CONST vmask vilogbk_vm_vd(vdouble d) {
  vopmask o = vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(4.9090934652977266E-91));
  d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(vcast_vd_d(2.037035976334486E90), d), d);
  vmask q = vreinterpret_vm_vd(d);
  q = vsrl64_vm_vm_i(q, 20 + 32);
  q = vand_vm_vm_vm(q, vcast_vm_i_i(0, 0x7ff));
  q = vsub64_vm_vm_vm(q, vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0, 300 + 0x3ff), vcast_vm_i_i(0, 0x3ff)));
  return q;
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

static INLINE CONST VECTOR_CC vdouble vpow2i_vd_vm(vmask q) {
  q = vadd64_vm_vm_vm(vcast_vm_i_i(0, 0x3ff), q);
  return vreinterpret_vd_vm(vsll64_vm_vm_i(q, 52));
}

static INLINE CONST vdouble vldexp1_vd_vd_vm(vdouble d, vmask e) {
  vmask m = vsrl64_vm_vm_i(e, 2);
  e = vsub64_vm_vm_vm(vsub64_vm_vm_vm(vsub64_vm_vm_vm(e, m), m), m);
  d = vmul_vd_vd_vd(d, vpow2i_vd_vm(m));
  d = vmul_vd_vd_vd(d, vpow2i_vd_vm(m));
  d = vmul_vd_vd_vd(d, vpow2i_vd_vm(m));
  d = vmul_vd_vd_vd(d, vpow2i_vd_vm(e));
  return d;
}

static INLINE CONST vdouble vldexp2_vd_vd_vm(vdouble d, vmask e) {
  return vmul_vd_vd_vd(vmul_vd_vd_vd(d, vpow2i_vd_vm(vsrl64_vm_vm_i(e, 1))), vpow2i_vd_vm(vsub64_vm_vm_vm(e, vsrl64_vm_vm_i(e, 1))));
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

static INLINE CONST vmask2 vneg128_vm2_vm2(vmask2 ax) {
  vmask2 x = { .y = vxor_vm_vm_vm(ax.y, vcast_vm_i_i(0xffffffff, 0xffffffff)),
	       .x = vxor_vm_vm_vm(ax.x, vcast_vm_i_i(0xffffffff, 0xffffffff)) }; 
  vmask2 r = { .y = x.y, .x = vadd64_vm_vm_vm(x.x, vcast_vm_i_i(0, 1)) };
  r.y = vadd64_vm_vm_vm(r.y, vand_vm_vo64_vm(veq64_vo_vm_vm(r.x, vcast_vm_i_i(0, 0)), vcast_vm_i_i(0, 1)));
  return r;
}

static INLINE CONST vmask2 imdvm2(vmask x, vmask y) { vmask2 r = { x, y }; return r; }

// imm must be smaller than 64
#define vsrl128_vm2_vm2_i(m, imm)					\
  imdvm2(vor_vm_vm_vm(vsrl64_vm_vm_i(m.x, imm), vsll64_vm_vm_i(m.y, 64-imm)), vsrl64_vm_vm_i(m.y, imm))

static INLINE CONST vopmask visnonfiniteq_vo_vm2(vmask2 a) {
  return veq64_vo_vm_vm(vand_vm_vm_vm(a.y, vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
}

static INLINE CONST vopmask visnonfiniteq_vo_vm2_vm2(vmask2 a, vmask2 b) {
  vmask ma = vxor_vm_vm_vm(vand_vm_vm_vm(a.y, vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  vmask mb = vxor_vm_vm_vm(vand_vm_vm_vm(b.y, vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  return veq64_vo_vm_vm(vand_vm_vm_vm(ma, mb), vcast_vm_i_i(0, 0));
}

static INLINE CONST vopmask visnonfiniteq_vo_vm2_vm2_vm2(vmask2 a, vmask2 b, vmask2 c) {
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
  return vandnot_vo_vo_vo(visinfq_vo_vm2(a), visnonfiniteq_vo_vm2(a));
}

static INLINE CONST vopmask viszeroq_vo_vm2(vmask2 a) {
  return veq64_vo_vm_vm(vor_vm_vm_vm(vand_vm_vm_vm(a.y, vcast_vm_i_i(~0x80000000, ~0)), a.x), vcast_vm_i_i(0, 0));
}

static INLINE CONST vmask2 vcmpcnv_vm2_vm2(vmask2 x) {
  vmask2 t = { .y = vxor_vm_vm_vm(x.y, vcast_vm_i_i(0x7fffffff, 0xffffffff)),
	       .x = vxor_vm_vm_vm(x.x, vcast_vm_i_i(0xffffffff, 0xffffffff)) }; 
  t.x = vadd64_vm_vm_vm(t.x, vcast_vm_i_i(0, 1));
  t.y = vadd64_vm_vm_vm(t.y, vand_vm_vo64_vm(veq64_vo_vm_vm(t.x, vcast_vm_i_i(0, 0)), vcast_vm_i_i(0, 1)));

  return vsel_vm2_vo_vm2_vm2(veq64_vo_vm_vm(vand_vm_vm_vm(x.y, vcast_vm_i_i(0x80000000, 0)), vcast_vm_i_i(0x80000000, 0)),
			     t, x);
}

// double3 functions ------------------------------------------------------------------------------------------------------------

static INLINE CONST vdouble2 ddscale_vd2_vd2_d(vdouble2 d, double s) { return ddscale_vd2_vd2_vd(d, vcast_vd_d(s)); }

static INLINE CONST vdouble3 vcast_vd3_vd_vd_vd(vdouble d0, vdouble d1, vdouble d2) {
  vdouble3 ret = { d0, d1, d2 };
  return ret;
}

static INLINE CONST vdouble3 vcast_vd3_d_d_d(double d0, double d1, double d2) {
  vdouble3 ret = { vcast_vd_d(d0), vcast_vd_d(d1), vcast_vd_d(d2) };
  return ret;
}

static INLINE CONST vdouble3 tdmulsign_vd3_vd3_vd(vdouble3 d, vdouble s) {
  return vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(d.x, s), vmulsign_vd_vd_vd(d.y, s), vmulsign_vd_vd_vd(d.z, s));
}

static INLINE CONST vdouble3 tdabs_vd3_vd3(vdouble3 d) { return tdmulsign_vd3_vd3_vd(d, d.x); }

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

static INLINE CONST vdouble2 twosub_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble2 r;
  r.x  = vsub_vd_vd_vd(x, y);
  vdouble v = vsub_vd_vd_vd(r.x, x);
  r.y = vsub_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(r.x, v)), vadd_vd_vd_vd(y, v));
  return r;
}

static INLINE CONST vdouble2 twosumx_vd2_vd_vd_vd(vdouble x, vdouble y, vdouble s) {
  vdouble2 r;
  r.x  = vmla_vd_vd_vd_vd(y, s, x);
  vdouble v = vsub_vd_vd_vd(r.x, x);
  r.y = vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(r.x, v)), vmlapn_vd_vd_vd_vd(y, s, v));
  return r;
}

static INLINE CONST vdouble2 twosubx_vd2_vd_vd_vd(vdouble x, vdouble y, vdouble s) {
  vdouble2 r;
  r.x  = vmlanp_vd_vd_vd_vd(y, s, x);
  vdouble v = vsub_vd_vd_vd(r.x, x);
  r.y = vsub_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(r.x, v)), vmla_vd_vd_vd_vd(y, s, v));
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

static INLINE CONST vdouble3 tdsub2_vd3_vd3_vd3(vdouble3 x, vdouble3 y) {
  vdouble2 d0 = twosub_vd2_vd_vd(x.x, y.x);
  vdouble2 d1 = twosub_vd2_vd_vd(x.y, y.y);
  vdouble2 d3 = twosum_vd2_vd_vd(d0.y, d1.x);
  return tdnormalize_vd3_vd3((vdouble3) { d0.x, d3.x, vadd_vd_4vd(x.z, vneg_vd_vd(y.z), d1.y, d3.y) });
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

static INLINE CONST vdouble3 tdscalesub2_vd3_vd3_vd3_vd(vdouble3 x, vdouble3 y, vdouble s) {
  vdouble2 d0 = twosubx_vd2_vd_vd_vd(x.x, y.x, s);
  vdouble2 d1 = twosubx_vd2_vd_vd_vd(x.y, y.y, s);
  vdouble2 d3 = twosum_vd2_vd_vd(d0.y, d1.x);
  return tdnormalize_vd3_vd3((vdouble3) { d0.x, d3.x, vadd_vd_3vd(vmlanp_vd_vd_vd_vd(y.z, s, x.z), d1.y, d3.y) });
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

static INLINE CONST vdouble3 tdsqu_vd3_vd3(vdouble3 x) { return tdmul_vd3_vd3_vd3(x, x); }

static INLINE CONST vdouble3 tdmul_vd3_vd2_vd3(vdouble2 x, vdouble3 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(x.x, y.x);
  vdouble2 d1 = twoprod_vd2_vd_vd(x.x, y.y);
  vdouble2 d2 = twoprod_vd2_vd_vd(x.y, y.x);
  vdouble2 d4 = twosum_vd2_vd_vd(d0.y, d1.x);
  vdouble2 d5 = twosum_vd2_vd_vd(d4.x, d2.x);

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(x.y, y.y, vmla_vd_vd_vd_vd(x.x, y.z, vadd_vd_vd_vd(d1.y, d2.y))), d4.y, d5.y);

  return (vdouble3){ d0.x, d5.x, t2 };
}

static INLINE CONST vdouble3 tdmul_vd3_vd3_vd2(vdouble3 x, vdouble2 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(y.x, x.x);
  vdouble2 d1 = twoprod_vd2_vd_vd(y.x, x.y);
  vdouble2 d2 = twoprod_vd2_vd_vd(y.y, x.x);
  vdouble2 d4 = twosum_vd2_vd_vd(d0.y, d1.x);
  vdouble2 d5 = twosum_vd2_vd_vd(d4.x, d2.x);

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(y.y, x.y, vmla_vd_vd_vd_vd(y.x, x.z, vadd_vd_vd_vd(d1.y, d2.y))), d4.y, d5.y);

  return (vdouble3){ d0.x, d5.x, t2 };
}

static INLINE CONST vdouble3 tdmul_vd3_vd3_vd(vdouble3 x, vdouble y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(y, x.x);
  vdouble2 d1 = twoprod_vd2_vd_vd(y, x.y);
  vdouble2 d4 = twosum_vd2_vd_vd(d0.y, d1.x);

  return (vdouble3){ d0.x, d4.x, vadd_vd_vd_vd(vmla_vd_vd_vd_vd(y, x.z, d1.y), d4.y) };
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

static INLINE CONST vdouble3 tdneg_vd3_vd3(vdouble3 d) {
  d.x = vneg_vd_vd(d.x);
  d.y = vneg_vd_vd(d.y);
  d.z = vneg_vd_vd(d.z);
  return d;
}

//

static INLINE CONST vdouble mla(vdouble x, vdouble y, vdouble z) { return vmla_vd_vd_vd_vd(x, y, z); }

static INLINE CONST vdouble poly2d(vdouble x, double c1, double c0) { return mla(x, vcast_vd_d(c1), vcast_vd_d(c0)); }
static INLINE CONST vdouble poly3d(vdouble x, double c2, double c1, double c0) { return mla(vmul_vd_vd_vd(x, x), vcast_vd_d(c2), poly2d(x, c1, c0)); }
static INLINE CONST vdouble poly4d(vdouble x, double c3, double c2, double c1, double c0) {
  return mla(vmul_vd_vd_vd(x, x), poly2d(x, c3, c2), poly2d(x, c1, c0));
}
static INLINE CONST vdouble poly5d(vdouble x, double c4, double c3, double c2, double c1, double c0) {
  return mla(vmul_vd_vd_vd(vmul_vd_vd_vd(x, x),vmul_vd_vd_vd(x, x)), vcast_vd_d(c4), poly4d(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble poly6d(vdouble x, double c5, double c4, double c3, double c2, double c1, double c0) {
  return mla(vmul_vd_vd_vd(vmul_vd_vd_vd(x, x),vmul_vd_vd_vd(x, x)), poly2d(x, c5, c4), poly4d(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble poly7d(vdouble x, double c6, double c5, double c4, double c3, double c2, double c1, double c0) {
  return mla(vmul_vd_vd_vd(vmul_vd_vd_vd(x, x),vmul_vd_vd_vd(x, x)), poly3d(x, c6, c5, c4), poly4d(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble poly8d(vdouble x, double c7, double c6, double c5, double c4, double c3, double c2, double c1, double c0) {
  return mla(vmul_vd_vd_vd(vmul_vd_vd_vd(x, x),vmul_vd_vd_vd(x, x)), poly4d(x, c7, c6, c5, c4), poly4d(x, c3, c2, c1, c0));
}

//

typedef struct {
  double x, y;
} double2;

static INLINE CONST double2 dd(double h, double l) {
  double2 ret = { h, l };
  return ret;
}

static INLINE CONST vdouble2 vcast_vd2_d2(double2 dd) {
  vdouble2 ret = { vcast_vd_d(dd.x), vcast_vd_d(dd.y) };
  return ret;
}

static INLINE CONST vdouble2 ddmla(vdouble2 x, vdouble2 y, vdouble2 z) {
  return ddadd_vd2_vd2_vd2(z, ddmul_vd2_vd2_vd2(x, y));
}

static INLINE CONST vdouble2 poly2dd_b(vdouble2 x, double2 c1, double2 c0) { return ddmla(x, vcast_vd2_d2(c1), vcast_vd2_d2(c0)); }
static INLINE CONST vdouble2 poly2dd(vdouble2 x, vdouble c1, double2 c0) { return ddmla(x, vcast_vd2_vd_vd(c1, vcast_vd_d(0)), vcast_vd2_d2(c0)); }
static INLINE CONST vdouble2 poly3dd_b(vdouble2 x, double2 c2, double2 c1, double2 c0) { return ddmla(ddsqu_vd2_vd2(x), vcast_vd2_d2(c2), poly2dd_b(x, c1, c0)); }
static INLINE CONST vdouble2 poly3dd(vdouble2 x, vdouble c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(x), vcast_vd2_vd_vd(c2, vcast_vd_d(0)), poly2dd_b(x, c1, c0));
}
static INLINE CONST vdouble2 poly4dd_b(vdouble2 x, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(x), poly2dd_b(x, c3, c2), poly2dd_b(x, c1, c0));
}
static INLINE CONST vdouble2 poly4dd(vdouble2 x, vdouble c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(x), poly2dd(x, c3, c2), poly2dd_b(x, c1, c0));
}
static INLINE CONST vdouble2 poly5dd_b(vdouble2 x, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), vcast_vd2_d2(c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly5dd(vdouble2 x, vdouble c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), vcast_vd2_vd_vd(c4, vcast_vd_d(0)), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly6dd_b(vdouble2 x, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly2dd_b(x, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly6dd(vdouble2 x, vdouble c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly2dd(x, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly7dd_b(vdouble2 x, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly3dd_b(x, c6, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly7dd(vdouble2 x, vdouble c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly3dd(x, c6, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly8dd_b(vdouble2 x, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly4dd_b(x, c7, c6, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly8dd(vdouble2 x, vdouble c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly4dd(x, c7, c6, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly9dd_b(vdouble2 x,
				     double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), vcast_vd2_d2(c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly9dd(vdouble2 x,
				       vdouble c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), vcast_vd2_vd_vd(c8, vcast_vd_d(0)), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly10dd_b(vdouble2 x, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly2dd_b(x, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly10dd(vdouble2 x, vdouble c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly2dd(x, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly11dd_b(vdouble2 x, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly3dd_b(x, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly11dd(vdouble2 x, vdouble c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly3dd(x, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly12dd_b(vdouble2 x, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly4dd_b(x, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly12dd(vdouble2 x, vdouble c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly4dd(x, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly13dd_b(vdouble2 x, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly5dd_b(x, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly13dd(vdouble2 x, vdouble c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly5dd(x, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly14dd_b(vdouble2 x, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly6dd_b(x, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly14dd(vdouble2 x, vdouble c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly6dd(x, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly15dd_b(vdouble2 x, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly7dd_b(x, c14, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly15dd(vdouble2 x, vdouble c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly7dd(x, c14, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly16dd_b(vdouble2 x, double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly8dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly16dd(vdouble2 x, vdouble c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly8dd(x, c15, c14, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly17dd_b(vdouble2 x, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), vcast_vd2_d2(c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly17dd(vdouble2 x, vdouble c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), vcast_vd2_vd_vd(c16, vcast_vd_d(0)),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly18dd_b(vdouble2 x, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly2dd_b(x, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly18dd(vdouble2 x, vdouble c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly2dd(x, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly19dd_b(vdouble2 x, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly3dd_b(x, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly19dd(vdouble2 x, vdouble c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly3dd(x, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly20dd_b(vdouble2 x, double2 c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly4dd_b(x, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST vdouble2 poly20dd(vdouble2 x, vdouble c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9,
				      double2 c8, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly4dd(x, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}

//

typedef struct {
  double x, y, z;
} double3;

static INLINE CONST vdouble3 vcast_vd3_d3(double3 td) {
  vdouble3 ret = { vcast_vd_d(td.x), vcast_vd_d(td.y), vcast_vd_d(td.z) };
  return ret;
}

static INLINE double3 td(double x, double y, double z) {
  double3 ret = { x, y, z };
  return ret;
}

static INLINE CONST vdouble3 tdmla(vdouble3 x, vdouble3 y, vdouble3 z) { return tdadd2_vd3_vd3_vd3(z, tdmul_vd3_vd3_vd3(x, y)); }
static INLINE CONST vdouble3 poly2td_b(vdouble3 x, double3 c1, double3 c0) {
  if (c0.y == 0 && c0.z == 0) {
    if (c1.x == 1 && c1.y == 0 && c1.z == 0) return tdadd2_vd3_vd_vd3(vcast_vd_d(c0.x), x);
    return tdadd2_vd3_vd_vd3(vcast_vd_d(c0.x), tdmul_vd3_vd3_vd3(x, vcast_vd3_d3(c1)));
  }
  return tdmla(x, vcast_vd3_d3(c1), vcast_vd3_d3(c0));
}
static INLINE CONST vdouble3 poly2td(vdouble3 x, vdouble2 c1, double3 c0) {
  return tdadd2_vd3_vd3_vd3(vcast_vd3_d3(c0), tdmul_vd3_vd3_vd2(x, c1));
}
static INLINE CONST vdouble3 poly3td_b(vdouble3 x, double3 c2, double3 c1, double3 c0) { return tdmla(tdsqu_vd3_vd3(x), vcast_vd3_d3(c2), poly2td_b(x, c1, c0)); }
static INLINE CONST vdouble3 poly3td(vdouble3 x, vdouble2 c2, double3 c1, double3 c0) {
  return tdadd2_vd3_vd3_vd3(vcast_vd3_d3(c0), tdmul_vd3_vd3_vd2(tdsqu_vd3_vd3(x), c2));
}
static INLINE CONST vdouble3 poly4td_b(vdouble3 x, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla(tdsqu_vd3_vd3(x), poly2td_b(x, c3, c2), poly2td_b(x, c1, c0));
}
static INLINE CONST vdouble3 poly4td(vdouble3 x, vdouble2 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla(tdsqu_vd3_vd3(x), poly2td(x, c3, c2), poly2td_b(x, c1, c0));
}
static INLINE CONST vdouble3 poly5td_b(vdouble3 x, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), vcast_vd3_d3(c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble3 poly5td(vdouble3 x, vdouble2 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdadd2_vd3_vd3_vd3(poly4td_b(x, c3, c2, c1, c0), tdmul_vd3_vd3_vd2(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), c4));
}
static INLINE CONST vdouble3 poly6td_b(vdouble3 x, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly2td_b(x, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble3 poly6td(vdouble3 x, vdouble2 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly2td(x, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble3 poly7td_b(vdouble3 x, double3 c6, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly3td_b(x, c6, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble3 poly7td(vdouble3 x, vdouble2 c6, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly3td(x, c6, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble3 poly8td_b(vdouble3 x, double3 c7, double3 c6, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly4td_b(x, c7, c6, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST vdouble3 poly8td(vdouble3 x, vdouble2 c7, double3 c6, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly4td(x, c7, c6, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}

// TDX functions ------------------------------------------------------------------------------------------------------------

// TDX Compare and select functions

static INLINE CONST tdx vsel_tdx_vo64_tdx_tdx(vopmask o, tdx x, tdx y) {
  tdx r = { .dd = vsel_vd3_vo_vd3_vd3(o, x.dd, y.dd) };
  r.e = vsel_vm_vo64_vm_vm(o, x.e, y.e);
  return r;
}

static INLINE CONST vmask vcmp_vm_tdx_tdx(tdx t0, tdx t1) {
  vmask r = vcast_vm_i_i(0, 0);
  r = vsel_vm_vo64_vm_vm(vlt_vo_vd_vd(t0.dd.z, t1.dd.z), vcast_vm_i_i(-1, -1), r);
  r = vsel_vm_vo64_vm_vm(vgt_vo_vd_vd(t0.dd.z, t1.dd.z), vcast_vm_i_i( 0,  1), r);
  r = vsel_vm_vo64_vm_vm(vlt_vo_vd_vd(t0.dd.y, t1.dd.y), vcast_vm_i_i(-1, -1), r);
  r = vsel_vm_vo64_vm_vm(vgt_vo_vd_vd(t0.dd.y, t1.dd.y), vcast_vm_i_i( 0,  1), r);
  r = vsel_vm_vo64_vm_vm(vlt_vo_vd_vd(t0.dd.x, t1.dd.x), vcast_vm_i_i(-1, -1), r);
  r = vsel_vm_vo64_vm_vm(vgt_vo_vd_vd(t0.dd.x, t1.dd.x), vcast_vm_i_i( 0,  1), r);
  r = vsel_vm_vo64_vm_vm(vgt64_vo_vm_vm(t1.e, t0.e)    , vcast_vm_i_i(-1, -1), r);
  r = vsel_vm_vo64_vm_vm(vgt64_vo_vm_vm(t0.e, t1.e)    , vcast_vm_i_i( 0,  1), r);
  r = vsel_vm_vo64_vm_vm(vand_vo_vo_vo(veq_vo_vd_vd(t0.dd.x, vcast_vd_d(0)), veq_vo_vd_vd(t1.dd.x, vcast_vd_d(0))),
			 vcast_vm_i_i( 0,  0), r);
  r = vsel_vm_vo64_vm_vm(vand_vo_vo_vo(vlt_vo_vd_vd(t0.dd.x, vcast_vd_d(0)), vge_vo_vd_vd(t1.dd.x, vcast_vd_d(0))),
			 vcast_vm_i_i(-1, -1), r);
  r = vsel_vm_vo64_vm_vm(vand_vo_vo_vo(vge_vo_vd_vd(t0.dd.x, vcast_vd_d(0)), vlt_vo_vd_vd(t1.dd.x, vcast_vd_d(0))),
			 vcast_vm_i_i( 0,  1), r);
  return r;
}

static INLINE CONST vopmask vgt_vo_tdx_tdx(tdx x, tdx y) {
  return vgt64_vo_vm_vm(vcmp_vm_tdx_tdx(x, y), vcast_vm_i_i(0, 0));
}

static INLINE CONST vopmask vge_vo_tdx_tdx(tdx x, tdx y) {
  return vgt64_vo_vm_vm(vcmp_vm_tdx_tdx(x, y), vcast_vm_i_i(-1, -1));
}

// TDX Cast operators

static INLINE CONST tdx vcast_tdx_vd(vdouble d) {
  tdx r;
  r.dd.x = d;
  r.dd.y = vcast_vd_d(0);
  r.dd.z = vcast_vd_d(0);
  r.e = vilogbk_vm_vd(r.dd.x);
  r.dd.x = vsel_vd_vo_vd_vd(visnonfinite(r.dd.x), r.dd.x, vldexp2_vd_vd_vm(r.dd.x, vneg64_vm_vm(r.e)));
  r.e = vadd64_vm_vm_vm(r.e, vcast_vm_i_i(0, 16383));
  return r;
}

static INLINE CONST tdx vcast_tdx_vd3(vdouble3 d) {
  tdx r;
  r.e = vilogbk_vm_vd(d.x);
  r.dd.x = vldexp2_vd_vd_vm(d.x, vneg64_vm_vm(r.e));
  r.dd.y = vldexp2_vd_vd_vm(d.y, vneg64_vm_vm(r.e));
  r.dd.z = vldexp2_vd_vd_vm(d.z, vneg64_vm_vm(r.e));
  r.e = vadd64_vm_vm_vm(r.e, vcast_vm_i_i(0, 16383));
  return r;
}

static INLINE CONST tdx vcast_tdx_vd3_fast(vdouble3 d) {
  tdx r;
  r.e = vadd64_vm_vm_vm(vilogb2k_vm_vd(d.x), vcast_vm_i_i(0, 16383));
  vdouble t = vldexp3_vd_vd_vm(vcast_vd_d(0.5), vneg64_vm_vm(r.e));
  r.dd.x = vmul_vd_vd_vd(d.x, t);
  r.dd.y = vmul_vd_vd_vd(d.y, t);
  r.dd.z = vmul_vd_vd_vd(d.z, t);
  return r;
}

static INLINE CONST vdouble vcast_vd_tdx(tdx t) {
  vdouble ret = vldexp2_vd_vd_vm(t.dd.x, vadd64_vm_vm_vm(t.e, vcast_vm_i_i(-1, -16383)));

  vopmask o = vor_vo_vo_vo(veq_vo_vd_vd(t.dd.x, vcast_vd_d(0)),
			   vlt64_vo_vm_vm(t.e, vcast_vm_i_i(0, 16383 - 0x500)));
  ret = vsel_vd_vo_vd_vd(o, vmulsign_vd_vd_vd(vcast_vd_d(0), t.dd.x), ret);

  o = vgt64_vo_vm_vm(t.e, vcast_vm_i_i(0, 16383 + 0x400));
  ret = vsel_vd_vo_vd_vd(o, vmulsign_vd_vd_vd(vcast_vd_d(SLEEF_INFINITY), t.dd.x), ret);

  return vsel_vd_vo_vd_vd(visnonfinite(t.dd.x), t.dd.x, ret);
}

static INLINE CONST vdouble3 vcast_vd3_tdx(tdx t) {
  vmask e = vadd64_vm_vm_vm(t.e, vcast_vm_i_i(-1, -16383));
  vdouble3 r = vcast_vd3_vd_vd_vd(vldexp2_vd_vd_vm(t.dd.x, e),
				  vldexp2_vd_vd_vm(t.dd.y, e),
				  vldexp2_vd_vd_vm(t.dd.z, e));

  vopmask o = vor_vo_vo_vo(veq_vo_vd_vd(t.dd.x, vcast_vd_d(0)),
			   vlt64_vo_vm_vm(t.e, vcast_vm_i_i(0, 16383 - 0x500)));
  r = vsel_vd3_vo_vd3_vd3(o, vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(0), t.dd.x), vcast_vd_d(0), vcast_vd_d(0)), r);

  o = vgt64_vo_vm_vm(t.e, vcast_vm_i_i(0, 16383 + 0x400));
  r = vsel_vd3_vo_vd3_vd3(o, vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(SLEEF_INFINITY), t.dd.x), vcast_vd_d(0), vcast_vd_d(0)), r);

  r = vsel_vd3_vo_vd3_vd3(visnonfinite(t.dd.x), t.dd, r);

  return r;
}

// TDX Arithmetic functions

static INLINE CONST tdx vneg_tdx_tdx(tdx x) {
  x.dd = tdneg_vd3_vd3(x.dd);
  return x;
}

static INLINE CONST vmask vilogb_vm_tdx(tdx t) {
  vmask e = vadd64_vm_vm_vm(t.e, vcast_vm_i_i(-1, -16383));
  e = vsel_vm_vo64_vm_vm(vor_vo_vo_vo(veq_vo_vd_vd(t.dd.x, vcast_vd_d(1.0)), vlt_vo_vd_vd(t.dd.y, vcast_vd_d(0))),
			 vadd64_vm_vm_vm(e, vcast_vm_i_i(-1, -1)), e);
  return e;
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

static INLINE CONST tdx sub2_tdx_tdx_tdx(tdx dd0, tdx dd1) {
  vmask ed = vsub64_vm_vm_vm(dd1.e, dd0.e);
  vdouble t = vldexp3_vd_vd_vm(vcast_vd_d(1), ed);

  tdx r = { .dd = tdscalesub2_vd3_vd3_vd3_vd(dd0.dd, dd1.dd, t) };
  r.e = vilogb2k_vm_vd(r.dd.x);
  t = vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(r.e));

  vopmask o = veq_vo_vd_vd(dd0.dd.x, vcast_vd_d(0));
  r.e = vsel_vm_vo64_vm_vm(o, dd1.e, vadd64_vm_vm_vm(r.e, dd0.e));
  r.dd = tdscale_vd3_vd3_vd(r.dd, t);

  r = vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(ed, vcast_vm_i_i(0, 200)), vneg_tdx_tdx(dd1), r);
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

// TDX math functions

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

#if defined(ENABLE_SVE) || defined(ENABLE_SVENOFMA)
typedef __sizeless_struct {
  vdouble d;
  vint i;
} di_t;

typedef __sizeless_struct {
  vdouble3 td;
  vint i;
} tdi_t;
#else
typedef struct {
  vdouble d;
  vint i;
} di_t;

typedef struct {
  vdouble3 td;
  vint i;
} tdi_t;
#endif

static INLINE CONST di_t rempisub(vdouble x) {
#ifdef FULL_FP_ROUNDING
  vdouble y = vrint_vd_vd(vmul_vd_vd_vd(x, vcast_vd_d(4)));
  vint vi = vtruncate_vi_vd(vsub_vd_vd_vd(y, vmul_vd_vd_vd(vrint_vd_vd(x), vcast_vd_d(4))));
  di_t ret = { vsub_vd_vd_vd(x, vmul_vd_vd_vd(y, vcast_vd_d(0.25))), vi };
#else
  vdouble fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d(1LL << 28), vtruncate_vd_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / (1LL << 28))))));
  vint vi = vadd_vi_vi_vi(vsel_vi_vo_vi_vi(vcast_vo32_vo64(vgt_vo_vd_vd(x, vcast_vd_d(0))), vcast_vi_i(4), vcast_vi_i(3)), vtruncate_vi_vd(vmul_vd_vd_vd(fr, vcast_vd_d(8))));
  vi = vsra_vi_vi_i(vsub_vi_vi_vi(vand_vi_vi_vi(vcast_vi_i(7), vi), vcast_vi_i(3)), 1);
  fr = vsub_vd_vd_vd(fr, vmul_vd_vd_vd(vcast_vd_d(0.25), vtruncate_vd_vd(vmla_vd_vd_vd_vd(fr, vcast_vd_d(4), vmulsign_vd_vd_vd(vcast_vd_d(0.5), x)))));
  fr = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(vabs_vd_vd(fr), vcast_vd_d(0.25)), vsub_vd_vd_vd(fr, vmulsign_vd_vd_vd(vcast_vd_d(0.5), x)), fr);
  fr = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(vabs_vd_vd(fr), vcast_vd_d(1e+10)), vcast_vd_d(0), fr);
  vopmask o = veq_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(0.12499999999999998612));
  fr = vsel_vd_vo_vd_vd(o, x, fr);
  vi = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vcast_vi_i(0), vi);
  di_t ret = { fr, vi };
#endif
  return ret;
}

static CONST tdi_t rempio2q(tdx a) {
  const int N = 8, B = 8;
  const int NCOL = 53-B, NROW = (16385+(53-B)*N-106)/NCOL+1;
  extern const double Sleef_rempitabqp[];

  vmask e = vilogb_vm_tdx(a);
  e = vsel_vm_vo64_vm_vm(vgt64_vo_vm_vm(e, vcast_vm_i_i(0, 106)), e, vcast_vm_i_i(0, 106));
  a.e = vadd64_vm_vm_vm(a.e, vsub64_vm_vm_vm(vcast_vm_i_i(0, 106), e));

  vdouble row = vtruncate_vd_vd(vmul_vd_vd_vd(vcast_vd_vm(vsub64_vm_vm_vm(e, vcast_vm_i_i(0, 106))), vcast_vd_d(1.0 / NCOL))); // (e - 106) / NCOL;
  vdouble col = vsub_vd_vd_vd(vcast_vd_vm(vsub64_vm_vm_vm(e, vcast_vm_i_i(0, 106))), vmul_vd_vd_vd(row, vcast_vd_d(NCOL)));    // (e - 106) % NCOL;
  vint p = vtruncate_vi_vd(vmla_vd_vd_vd_vd(col, vcast_vd_d(NROW), row));

  vint q = vcast_vi_i(0);
  vdouble3 d = tdnormalize_vd3_vd3(vcast_vd3_tdx(a)), x = vcast_vd3_d_d_d(0, 0, 0);

  for(int i=0;i<N;i++) {
    vdouble t = vldexp3_vd_vd_vm(vgather_vd_p_vi(Sleef_rempitabqp+i, p), vcast_vm_i_i(i == 0 ? 0 : -1, -(53-B)*i));
    x = tdadd2_vd3_vd3_vd3(x, tdnormalize_vd3_vd3(tdmul_vd3_vd3_vd(d, t)));
    di_t di = rempisub(x.x);
    q = vadd_vi_vi_vi(q, di.i);
    x.x = di.d;
    x = tdnormalize_vd3_vd3(x);
  }

  x = tdmul2_vd3_vd3_vd3(x, vcast_vd3_d_d_d(3.141592653589793116*2, 1.2246467991473532072e-16*2, -2.9947698097183396659e-33*2));
  x = vsel_vd3_vo_vd3_vd3(vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383), a.e), d, x);

  tdi_t ret = { .i = q, .td = x };
  return ret;
}

static INLINE CONST tdx sin_tdx_tdx(tdx a) {
  const double3 npiu = { -3.141592653589793116, -1.2246467991473532072e-16, 0 };
  const double3 npil = { +2.9947698097183396659e-33, -1.1124542208633652815e-49, -5.6722319796403157441e-66 };

  vdouble3 d = vcast_vd3_tdx(a);
  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(d.x, vcast_vd_d(1.0 / M_PI)));
  vint q = vrint_vi_vd(dq);
  d = tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd3_vd(vcast_vd3_d3(npiu), dq));
  d = tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd3_vd(vcast_vd3_d3(npil), dq));

  vopmask o = vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 + 28), a.e);
  if (!LIKELY(vtestallones_i_vo64(o))) {
    tdi_t tdi = rempio2q(a);
    vint qw = vand_vi_vi_vi(tdi.i, vcast_vi_i(3));
    qw = vadd_vi_vi_vi(vadd_vi_vi_vi(qw, qw), vsel_vi_vo_vi_vi(vcast_vo32_vo64(vgt_vo_vd_vd(tdi.td.x, vcast_vd_d(0))),
							       vcast_vi_i(2), vcast_vi_i(1)));
    qw = vsra_vi_vi_i(qw, 2);
    vdouble3 dw = vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(3.141592653589793116      *-0.5), tdi.td.x),
				     vmulsign_vd_vd_vd(vcast_vd_d(1.2246467991473532072e-16 *-0.5), tdi.td.x),
				     vmulsign_vd_vd_vd(vcast_vd_d(-2.9947698097183396659e-33*-0.5), tdi.td.x));
    dw = vsel_vd3_vo_vd3_vd3(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(tdi.i, vcast_vi_i(1)), vcast_vi_i(1))),
			     tdadd2_vd3_vd3_vd3(tdi.td, dw), tdi.td);
    d = vsel_vd3_vo_vd3_vd3(o, d, dw);
    q = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), q, qw);
  }

  vdouble3 s = tdsqu_vd3_vd3(d);

  vmask m = vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1))), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
  d.x = vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(d.x)));
  d.y = vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(d.y)));
  d.z = vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(d.z)));

  vdouble u = poly4d(s.x,
		     -1.1940250944959890417e-34,
		     1.1308027528153266305e-31,
		     -9.183679676378987613e-29,
		     6.4469502484797539906e-26);

  vdouble2 u2 = poly9dd(vcast_vd2_vd_vd(s.x, s.y),
			u,
			dd(-3.868170170541284842e-23, -5.0031797333103428885e-40),
			dd(1.957294106338964361e-20, 1.7861752657707958995e-37),
			dd(-8.2206352466243279548e-18, 3.9191951527123122798e-34),
			dd(2.8114572543455205981e-15, 1.6297259344381721363e-31),
			dd(-7.6471637318198164055e-13, -7.0372077527137340446e-30),
			dd(1.6059043836821613341e-10, 1.2585293802741673201e-26),
			dd(-2.5052108385441720224e-08, 1.4488140712190297804e-24),
			dd(2.7557319223985892511e-06, -1.8583932740471482254e-22));

  vdouble3 u3 = poly5td(s,
			u2,
			td(-0.00019841269841269841253, -1.7209558293419717872e-22, -2.7335161110921010284e-39),
			td(0.0083333333333333332177, 1.1564823173178713802e-19, 8.4649335998891595007e-37),
			td(-0.16666666666666665741, -9.2518585385429706566e-18, -5.1355955723371960468e-34),
			td(1, 0, 0));

  u3 = tdmul_vd3_vd3_vd3(u3, d);
  return vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 - 0x300), a.e), a, vcast_tdx_vd3_fast(u3));
}

static INLINE CONST tdx cos_tdx_tdx(tdx a) {
  const double3 npiu = { -3.141592653589793116*0.5, -1.2246467991473532072e-16*0.5, 0 };
  const double3 npil = { +2.9947698097183396659e-33*0.5, -1.1124542208633652815e-49*0.5, -5.6722319796403157441e-66*0.5 };

  vdouble3 d = vcast_vd3_tdx(a);
  vdouble dq = vmla_vd_vd_vd_vd(vcast_vd_d(2),
				vrint_vd_vd(vmla_vd_vd_vd_vd(d.x, vcast_vd_d(1.0 / M_PI), vcast_vd_d(-0.5))),
				vcast_vd_d(1));
  vint q = vrint_vi_vd(dq);
  d = tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd3_vd(vcast_vd3_d3(npiu), dq));
  d = tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd3_vd(vcast_vd3_d3(npil), dq));

  vopmask o = vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 + 28), a.e);
  if (!LIKELY(vtestallones_i_vo64(o))) {
    tdi_t tdi = rempio2q(a);
    vint qw = vand_vi_vi_vi(tdi.i, vcast_vi_i(3));
    qw = vadd_vi_vi_vi(vadd_vi_vi_vi(qw, qw), vsel_vi_vo_vi_vi(vcast_vo32_vo64(vgt_vo_vd_vd(tdi.td.x, vcast_vd_d(0))), vcast_vi_i(8), vcast_vi_i(7)));
    qw = vsra_vi_vi_i(qw, 1);
    vdouble3 dw = vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(3.141592653589793116      *-0.5), tdi.td.x),
				     vmulsign_vd_vd_vd(vcast_vd_d(1.2246467991473532072e-16 *-0.5), tdi.td.x),
				     vmulsign_vd_vd_vd(vcast_vd_d(-2.9947698097183396659e-33*-0.5), tdi.td.x));
    dw = vsel_vd3_vo_vd3_vd3(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(tdi.i, vcast_vi_i(1)), vcast_vi_i(0))),
			     tdadd2_vd3_vd3_vd3(tdi.td, dw), tdi.td);
    d = vsel_vd3_vo_vd3_vd3(o, d, dw);
    q = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), q, qw);
  }

  vdouble3 s = tdsqu_vd3_vd3(d);

  vdouble u = poly4d(s.x,
		     -1.1940250944959890417e-34,
		     1.1308027528153266305e-31,
		     -9.183679676378987613e-29,
		     6.4469502484797539906e-26);

  vdouble2 u2 = poly9dd(vcast_vd2_vd_vd(s.x, s.y),
			u,
			dd(-3.868170170541284842e-23, -5.0031797333103428885e-40),
			dd(1.957294106338964361e-20, 1.7861752657707958995e-37),
			dd(-8.2206352466243279548e-18, 3.9191951527123122798e-34),
			dd(2.8114572543455205981e-15, 1.6297259344381721363e-31),
			dd(-7.6471637318198164055e-13, -7.0372077527137340446e-30),
			dd(1.6059043836821613341e-10, 1.2585293802741673201e-26),
			dd(-2.5052108385441720224e-08, 1.4488140712190297804e-24),
			dd(2.7557319223985892511e-06, -1.8583932740471482254e-22));

  vdouble3 u3 = poly5td(s,
			u2,
			td(-0.00019841269841269841253, -1.7209558293419717872e-22, -2.7335161110921010284e-39),
			td(0.0083333333333333332177, 1.1564823173178713802e-19, 8.4649335998891595007e-37),
			td(-0.16666666666666665741, -9.2518585385429706566e-18, -5.1355955723371960468e-34),
			td(1, 0, 0));

  u3 = tdmul_vd3_vd3_vd3(u3, d);

  vmask m = vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(0))), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
  u3.x = vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(u3.x)));
  u3.y = vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(u3.y)));
  u3.z = vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(u3.z)));

  return vcast_tdx_vd3_fast(u3);
}

static INLINE CONST tdx tan_tdx_tdx(tdx a) {
  const double3 npiu = { -3.141592653589793116*0.5, -1.2246467991473532072e-16*0.5, 0 };
  const double3 npil = { +2.9947698097183396659e-33*0.5, -1.1124542208633652815e-49*0.5, -5.6722319796403157441e-66*0.5 };

  vdouble3 x = vcast_vd3_tdx(a);
  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(x.x, vcast_vd_d(2.0 / M_PI)));
  vint q = vrint_vi_vd(dq);
  x = tdadd2_vd3_vd3_vd3(x, tdmul_vd3_vd3_vd(vcast_vd3_d3(npiu), dq));
  x = tdadd2_vd3_vd3_vd3(x, tdmul_vd3_vd3_vd(vcast_vd3_d3(npil), dq));

  vopmask o = vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 + 28), a.e);
  if (!LIKELY(vtestallones_i_vo64(o))) {
    tdi_t tdi = rempio2q(a);
    x = vsel_vd3_vo_vd3_vd3(o, x, tdi.td);
    q = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), q, tdi.i);
  }

  x = tdscale_vd3_vd3_d(x, 0.5);
  vdouble3 s = tdsqu_vd3_vd3(x);

  vdouble u = poly5d(s.x,
		     2.2015831737910052452e-08,
		     2.0256594378812907225e-08,
		     7.4429817298004292868e-08,
		     1.728455913166476866e-07,
		     4.2976852952607503818e-07);

  vdouble2 u2 = poly14dd(vcast_vd2_vd_vd(s.x, s.y),
			 u,
			 dd(1.0596794286215624247e-06, -5.5786255180009979924e-23),
			 dd(2.6147773828883112431e-06, -1.1480390409818038282e-22),
			 dd(6.4516885768946985146e-06, 3.2627944831502214901e-22),
			 dd(1.5918905120725069629e-05, -1.1535340858735760272e-21),
			 dd(3.9278323880066157397e-05, -3.0151337837994129323e-21),
			 dd(9.691537956945558128e-05, -6.7065314026885621303e-21),
			 dd(0.00023912911424354627086, 8.2076644671207424279e-21),
			 dd(0.00059002744094558616343, 1.1011612305688670223e-21),
			 dd(0.0014558343870513183304, -6.6211292607098418407e-20),
			 dd(0.0035921280365724811423, -1.2531638332150681915e-19),
			 dd(0.0088632355299021973322, -7.6330133111459338275e-19),
			 dd(0.021869488536155202996, -1.7377828965915248127e-19),
			 dd(0.053968253968253970809, -2.5552752154325148981e-18));

  vdouble3 u3 = poly4td(s,
			u2,
			td(0.13333333333333333148, 1.8503717077086519863e-18, 1.8676215451093490329e-34),
			td(0.33333333333333331483, 1.8503717077085941313e-17, 9.8074108858570314539e-34),
			td(1, 0, 0));

  u3 = tdmul_vd3_vd3_vd3(u3, x);
  vdouble3 y = tdadd2_vd3_vd_vd3(vcast_vd_d(-1), tdsqu_vd3_vd3(u3));
  x = tdscale_vd3_vd3_d(u3, -2);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1)));
  u3 = tddiv2_vd3_vd3_vd3(vsel_vd3_vo_vd3_vd3(o, tdneg_vd3_vd3(y), x), vsel_vd3_vo_vd3_vd3(o, x, y));

  return vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 - 0x300), a.e), a, vcast_tdx_vd3_fast(u3));
}

static INLINE CONST tdx exp_tdx_tdx(tdx a) {
  const double3 nln2u = { -0.69314718055994528623, -2.3190468138462995584e-17, 0 };
  const double3 nln2l = { -5.7077084384162120658e-34, +3.5824322106018114234e-50, +1.3521696757988629569e-66 };

  vdouble3 s = vcast_vd3_tdx(a);
  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(s.x, vcast_vd_d(R_LN2)));
  vint q = vrint_vi_vd(dq);
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nln2u), dq));
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nln2l), dq));

  s = tdscale_vd3_vd3_d(s, 0.5);

  vdouble u = poly6d(s.x,
		     1.5620530411202639902e-16,
		     2.8125634200750730004e-15,
		     4.7794775039652234692e-14,
		     7.6471631094741035269e-13,
		     1.1470745597601740926e-11,
		     1.6059043837011404763e-10);

  vdouble2 u2 = poly10dd(vcast_vd2_vd_vd(s.x, s.y),
			 u,
			 dd(2.0876756987868133274e-09, 1.7071714288407538431e-25),
			 dd(2.5052108385441683828e-08, 9.711602107176752226e-25),
			 dd(2.7557319223985888276e-07, 2.3716807526092479675e-23),
			 dd(2.7557319223985892511e-06, -1.8547855362888887935e-22),
			 dd(2.4801587301587301566e-05, 2.1512320051964885027e-23),
			 dd(0.00019841269841269841253, 1.7209340449757701664e-22),
			 dd(0.0013888888888888889419, -5.3005439545025066336e-20),
			 dd(0.0083333333333333332177, 1.1564823173844765377e-19),
			 dd(0.041666666666666664354, 2.3129646346357442049e-18));

  vdouble3 u3 = poly5td(s,
			u2,
			td(0.16666666666666665741, 9.2518585385429629529e-18, 6.1848790332762276811e-34),
			td(0.5, 0, 0),
			td(1, 0, 0),
			td(1, 0, 0));

  u3 = tdsqu_vd3_vd3(u3);

  tdx r = vcast_tdx_vd3_fast(u3);
  r.e = vadd64_vm_vm_vm(r.e, vcast_vm_vi(q));

  vopmask p = vor_vo_vo_vo(visinf_vo_vd(a.dd.x), vgt64_vo_vm_vm(a.e, vcast_vm_i_i(0, 16397)));
  vopmask o = vgt_vo_vd_vd(a.dd.x, vcast_vd_d(0));

  r.dd = vsel_vd3_vo_vd3_vd3(vand_vo_vo_vo(o, p), vcast_vd3_d_d_d(SLEEF_INFINITY, SLEEF_INFINITY, SLEEF_INFINITY), r.dd);
  r.dd = vsel_vd3_vo_vd3_vd3(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(visnan_vo_vd(a.dd.x), p)), vcast_vd3_d_d_d(0, 0, 0), r.dd);

  return r;
}

static INLINE CONST tdx exp2_tdx_tdx(tdx a) {
  vdouble3 s = vcast_vd3_tdx(a);

  vdouble dq = vrint_vd_vd(s.x);
  vint q = vrint_vi_vd(dq);
  s = tdadd2_vd3_vd_vd3(vneg_vd_vd(dq), s);

  s = tdscale_vd3_vd3_d(s, 0.5);

  vdouble u = poly6d(s.x,
		     2.1312038164967297247e-19,
		     5.5352570141139560433e-18,
		     1.357024745958052877e-16,
		     3.132436443693084597e-15,
		     6.7787263548592201519e-14,
		     1.3691488854074843157e-12);

  vdouble2 u2 = poly10dd(vcast_vd2_vd_vd(s.x, s.y),
			 u,
			 dd(2.5678435993488179662e-11, 1.1022345333981114638e-27),
			 dd(4.4455382718708049041e-10, 5.8760174994123609884e-27),
			 dd(7.054911620801123359e-09, -2.7991025824670962009e-26),
			 dd(1.0178086009239699922e-07, -1.9345131218603885369e-24),
			 dd(1.3215486790144309509e-06, -2.0163036061290568906e-24),
			 dd(1.5252733804059841083e-05, -8.0274487610413408036e-22),
			 dd(0.00015403530393381608776, 1.1783618440356898175e-20),
			 dd(0.0013333558146428443284, 1.3928059564606790402e-20),
			 dd(0.0096181291076284768787, 2.8324606784380676049e-19));

  vdouble3 u3 = poly5td(s,
			u2,
			td(0.055504108664821583119, -3.1658222903912850146e-18, 1.6443777641435022298e-34),
			td(0.24022650695910072183, -9.4939312531828755586e-18, -2.3317045736512889737e-34),
			td(0.69314718055994528623, 2.3190468138462995584e-17, 5.7491470631463543202e-34),
			td(1, 0, 0));

  u3 = tdsqu_vd3_vd3(u3);

  tdx r = vcast_tdx_vd3_fast(u3);
  r.e = vadd64_vm_vm_vm(r.e, vcast_vm_vi(q));

  vopmask p = vor_vo_vo_vo(visinf_vo_vd(a.dd.x), vgt64_vo_vm_vm(a.e, vcast_vm_i_i(0, 16397)));
  vopmask o = vgt_vo_vd_vd(a.dd.x, vcast_vd_d(0));

  r.dd = vsel_vd3_vo_vd3_vd3(vand_vo_vo_vo(o, p), vcast_vd3_d_d_d(SLEEF_INFINITY, SLEEF_INFINITY, SLEEF_INFINITY), r.dd);
  r.dd = vsel_vd3_vo_vd3_vd3(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(visnan_vo_vd(a.dd.x), p)), vcast_vd3_d_d_d(0, 0, 0), r.dd);

  return r;
}

static INLINE CONST tdx exp10_tdx_tdx(tdx a) {
  const double3 nlog_10_2u = { -0.30102999566398119802, 2.8037281277851703937e-18, 0 };
  const double3 nlog_10_2l = { -5.4719484023146385333e-35, -5.1051389831070924689e-51, -1.2459153896093320861e-67 };

  vdouble3 s = vcast_vd3_tdx(a);
  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(s.x, vcast_vd_d(LOG10_2)));
  vint q = vrint_vi_vd(dq);
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nlog_10_2u), dq));
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nlog_10_2l), dq));

  s = tdscale_vd3_vd3_d(s, 0.5);

  vdouble u = poly6d(s.x,
		     5.1718894362277323603e-10,
		     4.0436341626932450119e-09,
		     2.9842239377609726639e-08,
		     2.073651082488697668e-07,
		     1.3508629476297046323e-06,
		     8.2134125355421453926e-06);

  vdouble2 u2 = poly10dd(vcast_vd2_vd_vd(s.x, s.y),
			 u,
			 dd(4.6371516642572155309e-05, -2.8553006623261362986e-21),
			 dd(0.00024166672554424659006, -5.090444533686828711e-21),
			 dd(0.0011544997789984347888, 8.7682157503137164069e-20),
			 dd(0.0050139288337754401442, -4.5412598169138237919e-20),
			 dd(0.019597694626478524144, -4.4678161257271451055e-19),
			 dd(0.068089365074437066538, -4.1730121698237928055e-18),
			 dd(0.20699584869686810107, -4.3690361006122386817e-18),
			 dd(0.53938292919558139538, 1.4823411613429006666e-17),
			 dd(1.1712551489122668968, 6.6334809422055596804e-17));

  vdouble3 u3 = poly5td(s,
			u2,
			td(2.0346785922934760293, 1.6749086434898761064e-16, 8.7840626642020613972e-33),
			td(2.6509490552391992146, -2.0935887830503358319e-16, 4.4428498991402792409e-33),
			td(2.3025850929940459011, -2.1707562233822493508e-16, -9.9703562308677605865e-33),
			td(1, 0, 0));

  u3 = tdsqu_vd3_vd3(u3);

  tdx r = vcast_tdx_vd3_fast(u3);
  r.e = vadd64_vm_vm_vm(r.e, vcast_vm_vi(q));

  vopmask p = vor_vo_vo_vo(visinf_vo_vd(a.dd.x), vgt64_vo_vm_vm(a.e, vcast_vm_i_i(0, 16395)));
  vopmask o = vgt_vo_vd_vd(a.dd.x, vcast_vd_d(0));

  r.dd = vsel_vd3_vo_vd3_vd3(vand_vo_vo_vo(o, p), vcast_vd3_d_d_d(SLEEF_INFINITY, SLEEF_INFINITY, SLEEF_INFINITY), r.dd);
  r.dd = vsel_vd3_vo_vd3_vd3(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(visnan_vo_vd(a.dd.x), p)), vcast_vd3_d_d_d(0, 0, 0), r.dd);

  return r;
}

static INLINE CONST tdx expm1_tdx_tdx(tdx a) {
  const double3 nln2u = { -0.69314718055994528623, -2.3190468138462995584e-17, 0 };
  const double3 nln2l = { -5.7077084384162120658e-34, +3.5824322106018114234e-50, +1.3521696757988629569e-66 };

  vdouble3 s = vcast_vd3_tdx(a);
  vopmask o = vlt_vo_vd_vd(s.x, vcast_vd_d(-100));

  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(s.x, vcast_vd_d(R_LN2)));
  vint q = vrint_vi_vd(dq);
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nln2u), dq));
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nln2l), dq));

  vdouble u = poly7d(s.x,
		     8.9624718038949319757e-22,
		     1.9596996271605003749e-20,
		     4.1102802184721881474e-19,
		     8.2206288447936758107e-18,
		     1.561920706551789357e-16,
		     2.8114572552972840413e-15,
		     4.7794773323730898317e-14);

  vdouble2 u2 = poly12dd(vcast_vd2_vd_vd(s.x, s.y),
			 u,
			 dd(7.6471637318189520664e-13, -1.5551332723390327262e-29),
			 dd(1.1470745597729737432e-11, 3.0337593188564885149e-28),
			 dd(1.6059043836821615926e-10, -8.3414062100037129366e-27),
			 dd(2.0876756987868100187e-09, -1.2148616582619574456e-25),
			 dd(2.5052108385441720224e-08, -1.4489867507419584246e-24),
			 dd(2.7557319223985888276e-07, 2.3767741816880403292e-23),
			 dd(2.7557319223985892511e-06, -1.8583932392536878791e-22),
			 dd(2.4801587301587301566e-05, 2.151194728002721518e-23),
			 dd(0.00019841269841269841253, 1.7209558290108237438e-22),
			 dd(0.0013888888888888889419, -5.3005439543729035854e-20),
			 dd(0.0083333333333333332177, 1.1564823173178718617e-19));

  vdouble3 u3 = poly5td(s,
			u2,
			td(0.041666666666666664354, 2.3129646346357426642e-18, 9.7682684964787852124e-35),
			td(0.16666666666666665741, 9.2518585385429706566e-18, 5.1444521483914181353e-34),
			td(0.5, 0, 0),
			td(1, 0, 0));

  u3 = tdmul_vd3_vd3_vd3(s, u3);

  tdx r = vcast_tdx_vd3_fast(u3);

  vopmask p = vneq_vo_vd_vd(dq, vcast_vd_d(0));

  r = vsel_tdx_vo64_tdx_tdx(p, add2_tdx_tdx_tdx(r, vcast_tdx_vd(vcast_vd_d(1))), r);
  r.e = vsel_vm_vo64_vm_vm(p, vadd64_vm_vm_vm(r.e, vcast_vm_vi(q)), r.e);
  r = vsel_tdx_vo64_tdx_tdx(p, sub2_tdx_tdx_tdx(r, vcast_tdx_vd(vcast_vd_d(1))), r);

  p = vand_vo_vo_vo(vgt_vo_vd_vd(a.dd.x, vcast_vd_d(0)),
		    vor_vo_vo_vo(visinf_vo_vd(a.dd.x), vgt64_vo_vm_vm(a.e, vcast_vm_i_i(0, 16397))));
  r.dd = vsel_vd3_vo_vd3_vd3(vand_vo_vo_vo(vgt_vo_vd_vd(a.dd.x, vcast_vd_d(0)), p),
			     vcast_vd3_d_d_d(SLEEF_INFINITY, SLEEF_INFINITY, SLEEF_INFINITY), r.dd);

  p = vandnot_vo_vo_vo(vgt_vo_vd_vd(a.dd.x, vcast_vd_d(0)), 
		       vor_vo_vo_vo(visinf_vo_vd(a.dd.x), vgt64_vo_vm_vm(a.e, vcast_vm_i_i(0, 16389))));
  r = vsel_tdx_vo64_tdx_tdx(vor_vo_vo_vo(o, p), vcast_tdx_vd(vcast_vd_d(-1)), r);

  r = vsel_tdx_vo64_tdx_tdx(vor_vo_vo_vo(visnan_vo_vd(a.dd.x), vlt64_vo_vm_vm(a.e, vcast_vm_i_i(0, 16000))),
			    a, r);

  return r;
}

// Float128 functions ------------------------------------------------------------------------------------------------------------

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

static vmask2 vcast_vf128_tdx_slow(tdx f) {
  vmask signbit = vand_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), vcast_vm_i_i(0x80000000, 0));
  vopmask iszero = veq_vo_vd_vd(f.dd.x, vcast_vd_d(0.0));

  f.dd.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.x), signbit));
  f.dd.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.y), signbit));
  f.dd.z = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(f.dd.z), signbit));

  vopmask denorm = vgt64_vo_vm_vm(vcast_vm_i_i(0, 1), f.e), fisinf = visinf_vo_vd(f.dd.x);
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

  o = vor_vo_vo_vo(vgt64_vo_vm_vm(f.e, vcast_vm_i_i(0, 32765)), fisinf);
  r.y = vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0x7fff0000, 0), r.y);
  r.x = vandnot_vm_vo64_vm(o, r.x);

  o = vandnot_vo_vo_vo(fisinf, visnonfinite(f.dd.x));
  r.y = vor_vm_vo64_vm(o, r.y);
  r.x = vor_vm_vo64_vm(o, r.x);

  r.y = vor_vm_vm_vm(r.y, signbit);

  return r;
}

static vmask2 vcast_vf128_tdx(tdx f) {
  vopmask o = vor_vo_vo_vo(vgt64_vo_vm_vm(vcast_vm_i_i(0, 2), f.e), vgt64_vo_vm_vm(f.e, vcast_vm_i_i(0, 0x7ffd)));
  o = vor_vo_vo_vo(o, visnonfinite(f.dd.x));
  o = vandnot_vo_vo_vo(veq_vo_vd_vd(vcast_vd_d(0), f.dd.x), o);
  if (UNLIKELY(!vtestallzeros_i_vo64(o))) return vcast_vf128_tdx_slow(f);

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

// Float128 conversion functions

EXPORT CONST vargquad xcast_from_doubleq(vdouble d) {
  return vcast_aq_vm2(vcast_vf128_tdx(vcast_tdx_vd(vinterleave_vd_vd(d))));
}

EXPORT CONST vdouble xcast_to_doubleq(vargquad q) {
  tdx t = vcast_tdx_vf128(vcast_vm2_aq(q));
  t.dd.x = vadd_vd_vd_vd(vadd_vd_vd_vd(t.dd.z, t.dd.y), t.dd.x);
  t.dd.y = t.dd.z = vcast_vd_d(0);
  return vuninterleave_vd_vd(vcast_vd_tdx(t));
}

// Float128 comparison functions

EXPORT CONST vint xcmpltq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vor_vo_vo_vo(vgt64_vo_vm_vm(cy.y, cx.y),
				       vand_vo_vo_vo(veq64_vo_vm_vm(cy.y, cx.y), vugt64_vo_vm_vm(cy.x, cx.x))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vuninterleave_vi_vi(vi);
}

EXPORT CONST vint xcmpgtq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vor_vo_vo_vo(vgt64_vo_vm_vm(cx.y, cy.y),
				       vand_vo_vo_vo(veq64_vo_vm_vm(cx.y, cy.y), vugt64_vo_vm_vm(cx.x, cy.x))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vuninterleave_vi_vi(vi);
}

EXPORT CONST vint xcmpleq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vor_vo_vo_vo(vgt64_vo_vm_vm(cy.y, cx.y),
				       vand_vo_vo_vo(veq64_vo_vm_vm(cy.y, cx.y),
						     vor_vo_vo_vo(vugt64_vo_vm_vm(cy.x, cx.x),
								  veq64_vo_vm_vm(cy.x, cx.x)))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vuninterleave_vi_vi(vi);
}

EXPORT CONST vint xcmpgeq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vor_vo_vo_vo(vgt64_vo_vm_vm(cx.y, cy.y),
				       vand_vo_vo_vo(veq64_vo_vm_vm(cx.y, cy.y),
						     vor_vo_vo_vo(vugt64_vo_vm_vm(cx.x, cy.x),
								  veq64_vo_vm_vm(cx.x, cy.x)))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vuninterleave_vi_vi(vi);
}

EXPORT CONST vint xcmpeqq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vand_vo_vo_vo(veq64_vo_vm_vm(cy.y, cx.y), veq64_vo_vm_vm(cx.x, cy.x)));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vuninterleave_vi_vi(vi);
}

EXPORT CONST vint xcmpneqq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vnot_vo64_vo64(vand_vo_vo_vo(veq64_vo_vm_vm(cy.y, cx.y), veq64_vo_vm_vm(cx.x, cy.x))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vuninterleave_vi_vi(vi);
}

EXPORT CONST vint xunordq(vargquad ax, vargquad ay) {
  vopmask o = vor_vo_vo_vo(visnanq_vo_vm2(vcast_vm2_aq(ax)), visnanq_vo_vm2(vcast_vm2_aq(ay)));
  vint vi = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vcast_vi_i(1), vcast_vi_i(0));
  return vuninterleave_vi_vi(vi);
}

// Float128 arithmetic functions

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

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonfiniteq_vo_vm2_vm2(a, b)))) {
    vopmask aisinf = visinfq_vo_vm2(a), bisinf = visinfq_vo_vm2(b);
    vopmask aisnan = visnanq_vo_vm2(a), bisnan = visnanq_vo_vm2(b);
    vopmask o = veq64_vo_vm_vm(a.y, vxor_vm_vm_vm(b.y, vcast_vm_i_i(0x80000000, 0)));
    o = vand_vo_vo_vo(o, veq64_vo_vm_vm(a.x, b.x));
    r = vsel_vm2_vo_vm2_vm2(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(bisnan, aisinf)), a, r);
    r = vsel_vm2_vo_vm2_vm2(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(aisnan, bisinf)), b, r);
  }

  return vcast_aq_vm2(r);
}

EXPORT CONST vargquad xsubq_u05(vargquad aa, vargquad ab) {
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
    vmask2 r = vcast_vf128_tdx_fast(sub2_tdx_tdx_tdx(vcast_tdx_vf128_fast(a), vcast_tdx_vf128_fast(b)));
    r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(vandnot_vm_vm_vm(b.y, a.y), vcast_vm_i_i(0x80000000, 0)));
    return vcast_aq_vm2(r);
  }

  vmask2 r = vcast_vf128_tdx(sub2_tdx_tdx_tdx(vcast_tdx_vf128(a), vcast_tdx_vf128(b)));

  r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(vandnot_vm_vm_vm(b.y, a.y), vcast_vm_i_i(0x80000000, 0)));

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonfiniteq_vo_vm2_vm2(a, b)))) {
    vopmask aisinf = visinfq_vo_vm2(a), bisinf = visinfq_vo_vm2(b);
    vopmask aisnan = visnanq_vo_vm2(a), bisnan = visnanq_vo_vm2(b);
    vopmask o = vand_vo_vo_vo(veq64_vo_vm_vm(a.x, b.x), veq64_vo_vm_vm(a.y, b.y));
    r = vsel_vm2_vo_vm2_vm2(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(bisnan, aisinf)), a, r);
    b.y = vxor_vm_vm_vm(b.y, vcast_vm_i_i(0x80000000, 0));
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
  if (LIKELY(vtestallones_i_vo64(vandnot_vo_vo_vo(visnonfiniteq_vo_vm2_vm2(a, b), 
						  vor_vo_vo_vo(vor_vo_vo_vo(viszeroq_vo_vm2(a), viszeroq_vo_vm2(b)),
							       vand_vo_vo_vo(vand_vo_vo_vo(oa, ob), oc)))))) {
    vmask2 r = vcast_vf128_tdx_fast(mul_tdx_tdx_tdx(vcast_tdx_vf128_fast(a), vcast_tdx_vf128_fast(b)));
    r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(vxor_vm_vm_vm(a.y, b.y), vcast_vm_i_i(0x80000000, 0)));
    return vcast_aq_vm2(r);
  }

  vmask2 r = vcast_vf128_tdx(mul2_tdx_tdx_tdx(vcast_tdx_vf128(a), vcast_tdx_vf128(b)));

  r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(vxor_vm_vm_vm(a.y, b.y), vcast_vm_i_i(0x80000000, 0)));

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonfiniteq_vo_vm2_vm2(a, b)))) {
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

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonfiniteq_vo_vm2_vm2_vm2(a, b, r)))) {
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

EXPORT CONST vargquad xnegq(vargquad aa) {
  vmask2 a = vcast_vm2_aq(aa);
  a.y = vxor_vm_vm_vm(a.y, vcast_vm_i_i(0x80000000, 0));
  return vcast_aq_vm2(a);
}

// Float128 math functions

EXPORT CONST vargquad xsqrtq_u05(vargquad aa) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 r = vcast_vf128_tdx(sqrt_tdx_tdx(vcast_tdx_vf128(a)));

  r.y = vor_vm_vm_vm(r.y, vand_vm_vm_vm(a.y, vcast_vm_i_i(0x80000000, 0)));
  vopmask aispinf = vispinfq_vo_vm2(a);

  r.y = vsel_vm_vo64_vm_vm(aispinf, vcast_vm_i_i(0x7fff0000, 0), r.y);
  r.x = vandnot_vm_vo64_vm(aispinf, r.x);

  return vcast_aq_vm2(r);
}

EXPORT CONST vargquad xsinq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(sin_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST vargquad xcosq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(cos_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST vargquad xtanq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(tan_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST vargquad xexpq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(exp_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST vargquad xexp2q_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(exp2_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST vargquad xexp10q_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(exp10_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST vargquad xexpm1q_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(expm1_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

// Float128 string functions

#ifdef ENABLE_PUREC_SCALAR
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static const tdx pow10tab[14] = {
  { 16386, { 1.25, 0, 0 } }, // 10
  { 16389, { 1.5625, 0, 0 } }, // 100
  { 16396, { 1.220703125, 0, 0 } }, // 10000
  { 16409, { 1.490116119384765625, 0, 0 } }, // 1e+08
  { 16436, { 1.1102230246251565404, 0, 0 } }, // 1e+16
  { 16489, { 1.2325951644078310121, -6.6143055845634601483e-17, 0 } }, // 1e+32
  { 16595, { 1.519290839321567832, -3.2391917291561477915e-17, -1.8687814275678753633e-33 } }, // 1e+64
  { 16808, { 1.1541223272232170594, -8.6760553787903265289e-17, -5.7759618887794337486e-33 } }, // 1e+128
  { 17233, { 1.3319983461951343529, -4.0129993161716667573e-17, -4.1720927621797370111e-34 } }, // 1e+256
  { 18083, { 1.7742195942665728303, 4.9309343678620668082e-17, 1.3386888736008621608e-34 } }, // 1e+512
  { 19784, { 1.5739275843397213528, -1.0848584040002990893e-16, 4.3586291506346591213e-33 } }, // 1e+1024
  { 23186, { 1.2386240203727352238, -5.8476062413608067671e-17, -2.0006771920677486581e-33 } }, // 1e+2048
  { 29989, { 1.5341894638443178689, -1.0973609479387666102e-17, -6.5816871252891901643e-34 } }, // 1e+4096
  { 43596, { 1.1768686554854577153, 3.0579788864750933707e-17, -2.6771867381968692559e-34 } }, // 1e+8192
};

static tdx pow10i(int n) {
  tdx r = vcast_tdx_vd(1);
  for(int i=0;i<14;i++)
    if ((n & (1 << i)) != 0) r = mul2_tdx_tdx_tdx(r, pow10tab[i]);
  return r;
}

static int ilog10(tdx t) {
  int r = 0, p = 1;
  if ((int)vcmp_vm_tdx_tdx(t, vcast_tdx_vd(1)) < 0) {
    t = div2_tdx_tdx_tdx(vcast_tdx_vd(1), t);
    p = -1;
  }
  for(int i=12;i>=0;i--) {
    int c = vcmp_vm_tdx_tdx(t, pow10tab[i]);
    if ((p > 0 && c >= 0) || (p < 0 && c > 0)) {
      t = div2_tdx_tdx_tdx(t, pow10tab[i]);
      r |= (1 << i);
    }
  }
  if (p < 0) r++;
  return r * p;
}

EXPORT vargquad Sleef_strtoq(char *str, char **endptr, int base) {
  while(isspace(*str)) str++;
  char *p = str;

  int positive = 1, bp = 0, e = 0, error = 0, mf = 0;
  tdx n = vcast_tdx_vd(0), d = vcast_tdx_vd(1);

  if (*p == '-') {
    positive = 0;
    p++;
  } else if (*p == '+') p++;

  if (tolower(p[0]) == 'n' && tolower(p[1]) == 'a' && tolower(p[2]) == 'n') {
    if (endptr != NULL) *endptr = p+3;
    vmask2 r = { vcast_vm_i_i(-1, -1), vcast_vm_i_i(-1, -1) };
    return vcast_aq_vm2(r);
  }

  if (tolower(p[0]) == 'i' && tolower(p[1]) == 'n' && tolower(p[2]) == 'f') {
    if (endptr != NULL) *endptr = p+3;
    if (positive) {
      vmask2 r = { vcast_vm_i_i(0, 0), vcast_vm_i_i(0x7fff0000, 0) };
      return vcast_aq_vm2(r);
    } else {
      vmask2 r = { vcast_vm_i_i(0, 0), vcast_vm_i_i(0xffff0000, 0) };
      return vcast_aq_vm2(r);
    }
  }

  while(*p != '\0' && !error) {
    if ('0' <= *p && *p <= '9') {
      n = add2_tdx_tdx_tdx(mul2_tdx_tdx_tdx(n, vcast_tdx_vd(10)), vcast_tdx_vd(*p - '0'));
      if (bp) d = mul2_tdx_tdx_tdx(d, vcast_tdx_vd(10));
      p++;
      mf = 1;
      continue;
    }

    if (*p == '.') {
      if (bp) break;
      bp = 1;
      p++;
      continue;
    }

    if (*p == 'e' || *p == 'E') {
      char *q;
      e = strtol(p+1, &q, 10);
      if (p+1 == q || isspace(*(p+1))) {
	e = 0;
      } else {
	p = q;
      }
      break;
    }

    error = 1;
    break;
  }

  if (error || !mf) {
    if (endptr != NULL) *endptr = str;
    vmask2 r = { vcast_vm_i_i(0, 0), vcast_vm_i_i(0, 0) };
    return vcast_aq_vm2(r);
  }

  n = div2_tdx_tdx_tdx(n, d);
  if (e > 0) n = mul2_tdx_tdx_tdx(n, pow10i(+e));
  if (e < 0) n = div2_tdx_tdx_tdx(n, pow10i(-e));
  if (!positive) n = vneg_tdx_tdx(n);

  if (endptr != NULL) *endptr = str;

  return vcast_aq_vm2(vcast_vf128_tdx(n));
}

EXPORT void Sleef_qtostr(char *s, int n, vargquad a, int base) {
  if (n <= 0) return;
  if (n > 48) n = 48;
  if (n < 9) { *s = '\0'; return; }

  union {
    vmask2 q;
    struct {
      uint64_t l, h;
    };
  } c128 = { .q = vcast_vm2_aq(a) };

  char *p = s;

  if (visnanq_vo_vm2(c128.q)) { sprintf(p, "nan"); return; }

  if ((c128.h & 0x8000000000000000ULL) != 0) {
    *p++ = '-';
    c128.h ^= 0x8000000000000000ULL;
  } else {
    *p++ = '+';
  }

  if (visinfq_vo_vm2(c128.q)) { sprintf(p, "inf"); return; }

  tdx t = vcast_tdx_vf128(c128.q);
  int e = ilog10(t);

  if (e < 0) t = mul2_tdx_tdx_tdx(t, pow10i(-e-1));
  if (e >= 0) t = div2_tdx_tdx_tdx(t, pow10i(+e+1));

  t = add2_tdx_tdx_tdx(t, div2_tdx_tdx_tdx(vcast_tdx_vd(0.5), pow10i(n-8)));

  *p++ = '.';

  if ((int)vcmp_vm_tdx_tdx(t, vcast_tdx_vd(1)) >= 0) {
    t = div2_tdx_tdx_tdx(t, vcast_tdx_vd(10));
    e++;
  }

  for(;n>=9;n--) {
    t = mul2_tdx_tdx_tdx(t, vcast_tdx_vd(10));
    int ia = (int)vcast_vd_tdx(t);
    if ((int)vcmp_vm_tdx_tdx(t, vcast_tdx_vd(ia)) < 0) ia--;
    *p++ = ia + '0';
    t = add2_tdx_tdx_tdx(t, vcast_tdx_vd(-ia));
  }

  if (viszeroq_vo_vm2(c128.q)) {
    *p++ = '\0';
    return;
  }

  *p++ = 'e';
  e++;
  if (e >= 0) *p++ = '+';
  if (e < 0) { *p++ = '-'; e = -e; }

  sprintf(p, "%d", e);
}
#endif

// Functions for debugging ------------------------------------------------------------------------------------------------------------

#ifdef ENABLE_MAIN
// gcc -DENABLE_MAIN -DENABLEFLOAT128 -Wno-attributes -I../libm -I../quad-tester -I../common -I../arch -DUSEMPFR -DENABLE_AVX2 -mavx2 -mfma sleefsimdqp.c ../common/common.c ../quad-tester/qtesterutil.c -lm -lmpfr
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
  double ad[32];
  mpfr_set_default_prec(18000);
  mpfr_t fr0, fr1, fr2;
  mpfr_inits(fr0, fr1, fr2, NULL);

  mpfr_set_d(fr0, 0, GMP_RNDN);
  if (argc >= 2) mpfr_set_str(fr0, argv[1], 10, GMP_RNDN);
  Sleef_quad q0 = mpfr_get_f128(fr0, GMP_RNDN);
  mpfr_set_f128(fr0, q0, GMP_RNDN);
  if (argc >= 2) printf("arg0 : %s\n", sprintfr(fr0));
  vargquad a0;
#if 1
  memrand(&a0, sizeof(vargquad));
#else
  memset(&a0, 0, sizeof(vargquad));
#endif
  a0.s[lane] = q0;
#if 0
  memrand(ad, sizeof(ad));
  ad[lane] = mpfr_get_d(fr0, GMP_RNDN);
  a0 = xcast_from_doubleq(vloadu_vd_p(ad));
#endif

  mpfr_set_d(fr1, 0, GMP_RNDN);
  if (argc >= 3) mpfr_set_str(fr1, argv[2], 10, GMP_RNDN);
  Sleef_quad q1 = mpfr_get_f128(fr1, GMP_RNDN);
  mpfr_set_f128(fr1, q1, GMP_RNDN);
  if (argc >= 3) printf("arg1 : %s\n", sprintfr(fr1));
  vargquad a1;
#if 1
  memrand(&a1, sizeof(vargquad));
#else
  memset(&a1, 0, sizeof(vargquad));
#endif
  a1.s[lane] = q1;
#if 0
  memrand(ad, sizeof(ad));
  ad[lane] = mpfr_get_d(fr1, GMP_RNDN);
  a1 = xcast_from_doubleq(vloadu_vd_p(ad));
#endif

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
