//   Copyright Naoki Shibata and contributors 2010 - 2020.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Always use -ffp-contract=off option to compile SLEEF.

#if !defined(SLEEF_GENHEADER)
#include <stdint.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#endif

#include "misc.h"

#ifndef ENABLE_CUDA
extern const double Sleef_rempitabqp[];
#endif

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

#ifdef ENABLE_CUDA
#define CONFIG 3
#include "helperpurec_scalar.h"
#ifdef DORENAME
#include "qrenamecuda.h"
#endif
typedef vmask2 Sleef_quadx1;
#endif

#ifdef ENABLE_SSE2
#define CONFIG 2
#include "helpersse2.h"
#ifdef DORENAME
#include "qrenamesse2.h"
#endif
typedef vmask2 Sleef_quadx2;
#endif

#ifdef ENABLE_AVX2
#define CONFIG 1
#include "helperavx2.h"
#ifdef DORENAME
#include "qrenameavx2.h"
#endif
typedef vmask2 Sleef_quadx4;
#endif

#ifdef ENABLE_AVX512F
#define CONFIG 1
#include "helperavx512f.h"
#ifdef DORENAME
#include "qrenameavx512f.h"
#endif
typedef vmask2 Sleef_quadx8;
#endif

#ifdef ENABLE_ADVSIMD
#define CONFIG 1
#include "helperadvsimd.h"
#ifdef DORENAME
#include "qrenameadvsimd.h"
#endif
typedef vmask2 Sleef_quadx2;
#endif

#ifdef ENABLE_SVE
#define CONFIG 1
#include "helpersve.h"
#ifdef DORENAME
#include "qrenamesve.h"
#endif
typedef vmask2 Sleef_svquad;
#endif

#ifdef ENABLE_VSX
#define CONFIG 1
#include "helperpower_128.h"
#ifdef DORENAME
#include "qrenamevsx.h"
#endif
typedef vmask2 Sleef_quadx2;
#endif

#ifdef ENABLE_VSX3
#define CONFIG 3
#include "helperpower_128.h"
#ifdef DORENAME
#include "qrenamevsx3.h"
#endif
#endif

#ifdef ENABLE_VXE
#define CONFIG 140
#include "helpers390x_128.h"
#ifdef DORENAME
#include "qrenamevxe.h"
#endif
typedef vmask2 Sleef_quadx2;
#endif

#ifdef ENABLE_VXE2
#define CONFIG 150
#include "helpers390x_128.h"
#ifdef DORENAME
#include "qrenamevxe2.h"
#endif
typedef vmask2 Sleef_quadx2;
#endif

#include "dd.h"
#include "commonfuncs.h"

//

static INLINE CONST VECTOR_CC vopmask visnonfiniteq_vo_vm2(vmask2 a) {
  return veq64_vo_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
}

static INLINE CONST VECTOR_CC vopmask visnonfiniteq_vo_vm2_vm2(vmask2 a, vmask2 b) {
  vmask ma = vxor_vm_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  vmask mb = vxor_vm_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(b), vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  return veq64_vo_vm_vm(vand_vm_vm_vm(ma, mb), vcast_vm_i_i(0, 0));
}

static INLINE CONST VECTOR_CC vopmask visnonfiniteq_vo_vm2_vm2_vm2(vmask2 a, vmask2 b, vmask2 c) {
  vmask ma = vxor_vm_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  vmask mb = vxor_vm_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(b), vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  vmask mc = vxor_vm_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(c), vcast_vm_i_i(0x7fff0000, 0)), vcast_vm_i_i(0x7fff0000, 0));
  return veq64_vo_vm_vm(vand_vm_vm_vm(vand_vm_vm_vm(ma, mb), mc), vcast_vm_i_i(0, 0));
}

static INLINE CONST VECTOR_CC vopmask visinfq_vo_vm2(vmask2 a) {
  vopmask o = veq64_vo_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(0x7fffffff, 0xffffffff)), vcast_vm_i_i(0x7fff0000, 0));
  return vand_vo_vo_vo(o, veq64_vo_vm_vm(vm2getx_vm_vm2(a), vcast_vm_i_i(0, 0)));
}

static INLINE CONST VECTOR_CC vopmask vispinfq_vo_vm2(vmask2 a) {
  return vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(0x7fff0000, 0)), veq64_vo_vm_vm(vm2getx_vm_vm2(a), vcast_vm_i_i(0, 0)));
}

static INLINE CONST VECTOR_CC vopmask visnanq_vo_vm2(vmask2 a) {
  return vandnot_vo_vo_vo(visinfq_vo_vm2(a), visnonfiniteq_vo_vm2(a));
}

static INLINE CONST VECTOR_CC vopmask viszeroq_vo_vm2(vmask2 a) {
  return veq64_vo_vm_vm(vor_vm_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(~0x80000000, ~0)), vm2getx_vm_vm2(a)), vcast_vm_i_i(0, 0));
}

static INLINE CONST VECTOR_CC vmask2 vcmpcnv_vm2_vm2(vmask2 x) {
  vmask2 t = vm2setxy_vm2_vm_vm(vxor_vm_vm_vm(vm2getx_vm_vm2(x), vcast_vm_i_i(0xffffffff, 0xffffffff)), 
				vxor_vm_vm_vm(vm2gety_vm_vm2(x), vcast_vm_i_i(0x7fffffff, 0xffffffff)));
  t = vm2setx_vm2_vm2_vm(t, vadd64_vm_vm_vm(vm2getx_vm_vm2(t), vcast_vm_i_i(0, 1)));
  t = vm2sety_vm2_vm2_vm(t, vadd64_vm_vm_vm(vm2gety_vm_vm2(t), vand_vm_vo64_vm(veq64_vo_vm_vm(vm2getx_vm_vm2(t), vcast_vm_i_i(0, 0)), vcast_vm_i_i(0, 1))));

  return vsel_vm2_vo_vm2_vm2(veq64_vo_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(x), vcast_vm_i_i(0x80000000, 0)), vcast_vm_i_i(0x80000000, 0)),
			     t, x);
}

// double3 functions ------------------------------------------------------------------------------------------------------------

static INLINE CONST VECTOR_CC vdouble2 ddscale_vd2_vd2_d(vdouble2 d, double s) { return ddscale_vd2_vd2_vd(d, vcast_vd_d(s)); }

static INLINE CONST VECTOR_CC vdouble3 vcast_vd3_vd_vd_vd(vdouble d0, vdouble d1, vdouble d2) {
  return vd3setxyz_vd3_vd_vd_vd(d0, d1, d2);
}

static INLINE CONST VECTOR_CC vdouble3 vcast_vd3_d_d_d(double d0, double d1, double d2) {
  return vd3setxyz_vd3_vd_vd_vd(vcast_vd_d(d0), vcast_vd_d(d1), vcast_vd_d(d2));
}

static INLINE CONST VECTOR_CC vdouble3 tdmulsign_vd3_vd3_vd(vdouble3 d, vdouble s) {
  return vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(vd3getx_vd_vd3(d), s), vmulsign_vd_vd_vd(vd3gety_vd_vd3(d), s), vmulsign_vd_vd_vd(vd3getz_vd_vd3(d), s));
}

static INLINE CONST VECTOR_CC vdouble3 tdabs_vd3_vd3(vdouble3 d) { return tdmulsign_vd3_vd3_vd(d, vd3getx_vd_vd3(d)); }

static INLINE CONST VECTOR_CC vdouble3 vsel_vd3_vo_vd3_vd3(vopmask m, vdouble3 x, vdouble3 y) {
  return vd3setxyz_vd3_vd_vd_vd(vsel_vd_vo_vd_vd(m, vd3getx_vd_vd3(x), vd3getx_vd_vd3(y)), 
				vsel_vd_vo_vd_vd(m, vd3gety_vd_vd3(x), vd3gety_vd_vd3(y)),
				vsel_vd_vo_vd_vd(m, vd3getz_vd_vd3(x), vd3getz_vd_vd3(y)));
}

// TD algorithms are based on Y. Hida et al., Library for double-double and quad-double arithmetic (2007).
static INLINE CONST VECTOR_CC vdouble2 twosum_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble rx = vadd_vd_vd_vd(x, y);
  vdouble v = vsub_vd_vd_vd(rx, x);
  return vd2setxy_vd2_vd_vd(rx, vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(rx, v)), vsub_vd_vd_vd(y, v)));
}

static INLINE CONST VECTOR_CC vdouble2 twosub_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble rx = vsub_vd_vd_vd(x, y);
  vdouble v = vsub_vd_vd_vd(rx, x);
  return vd2setxy_vd2_vd_vd(rx, vsub_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(rx, v)), vadd_vd_vd_vd(y, v)));
}

static INLINE CONST VECTOR_CC vdouble2 twosumx_vd2_vd_vd_vd(vdouble x, vdouble y, vdouble s) {
  vdouble rx = vmla_vd_vd_vd_vd(y, s, x);
  vdouble v = vsub_vd_vd_vd(rx, x);
  return vd2setxy_vd2_vd_vd(rx, vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(rx, v)), vmlapn_vd_vd_vd_vd(y, s, v)));
}

static INLINE CONST VECTOR_CC vdouble2 twosubx_vd2_vd_vd_vd(vdouble x, vdouble y, vdouble s) {
  vdouble rx = vmlanp_vd_vd_vd_vd(y, s, x);
  vdouble v = vsub_vd_vd_vd(rx, x);
  return vd2setxy_vd2_vd_vd(rx, vsub_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(rx, v)), vmla_vd_vd_vd_vd(y, s, v)));
}

static INLINE CONST VECTOR_CC vdouble2 quicktwosum_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble rx = vadd_vd_vd_vd(x, y);
  return vd2setxy_vd2_vd_vd(rx, vadd_vd_vd_vd(vsub_vd_vd_vd(x, rx), y));
}

static INLINE CONST VECTOR_CC vdouble2 twoprod_vd2_vd_vd(vdouble x, vdouble y) {
#ifdef ENABLE_FMA_DP
  vdouble rx = vmul_vd_vd_vd(x, y);
  return vd2setxy_vd2_vd_vd(rx, vfmapn_vd_vd_vd_vd(x, y, rx));
#else
  vdouble xh = vmul_vd_vd_vd(x, vcast_vd_d((1 << 27)+1));
  xh = vsub_vd_vd_vd(xh, vsub_vd_vd_vd(xh, x));
  vdouble xl = vsub_vd_vd_vd(x, xh);
  vdouble yh = vmul_vd_vd_vd(y, vcast_vd_d((1 << 27)+1));
  yh = vsub_vd_vd_vd(yh, vsub_vd_vd_vd(yh, y));
  vdouble yl = vsub_vd_vd_vd(y, yh);

  vdouble rx = vmul_vd_vd_vd(x, y);
  return vd2setxy_vd2_vd_vd(rx, vadd_vd_5vd(vmul_vd_vd_vd(xh, yh), vneg_vd_vd(rx), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yl)));
#endif
}

static INLINE CONST VECTOR_CC vdouble3 tdscale_vd3_vd3_vd(vdouble3 d, vdouble s) {
  return (vdouble3) { vmul_vd_vd_vd(vd3getx_vd_vd3(d), s), vmul_vd_vd_vd(vd3gety_vd_vd3(d), s), vmul_vd_vd_vd(vd3getz_vd_vd3(d), s) };
}

static INLINE CONST VECTOR_CC vdouble3 tdscale_vd3_vd3_d(vdouble3 d, double s) { return tdscale_vd3_vd3_vd(d, vcast_vd_d(s)); }

static INLINE CONST VECTOR_CC vdouble3 tdquickrenormalize_vd3_vd3(vdouble3 td) {
  vdouble2 u = quicktwosum_vd2_vd_vd(vd3getx_vd_vd3(td), vd3gety_vd_vd3(td));
  vdouble2 v = quicktwosum_vd2_vd_vd(vd2gety_vd_vd2(u), vd3getz_vd_vd3(td));
  return (vdouble3) { vd2getx_vd_vd2(u), vd2getx_vd_vd2(v), vd2gety_vd_vd2(v) };
}

static INLINE CONST VECTOR_CC vdouble3 tdnormalize_vd3_vd3(vdouble3 td) {
  vdouble2 u = quicktwosum_vd2_vd_vd(vd3getx_vd_vd3(td), vd3gety_vd_vd3(td));
  vdouble2 v = quicktwosum_vd2_vd_vd(vd2gety_vd_vd2(u), vd3getz_vd_vd3(td));
  td = vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(u), vd2getx_vd_vd2(v), vd2gety_vd_vd2(v));
  u = quicktwosum_vd2_vd_vd(vd3getx_vd_vd3(td), vd3gety_vd_vd3(td));
  return vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(u), vd2gety_vd_vd2(u), vd3getz_vd_vd3(td));
}

static INLINE CONST VECTOR_CC vdouble3 tdadd2_vd3_vd3_vd3(vdouble3 x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(vd3getx_vd_vd3(x), vd3getx_vd_vd3(y));
  vdouble2 d1 = twosum_vd2_vd_vd(vd3gety_vd_vd3(x), vd3gety_vd_vd3(y));
  vdouble2 d3 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  return tdnormalize_vd3_vd3(vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d3), vadd_vd_4vd(vd3getz_vd_vd3(x), vd3getz_vd_vd3(y), vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d3))));
}

static INLINE CONST VECTOR_CC vdouble3 tdsub2_vd3_vd3_vd3(vdouble3 x, vdouble3 y) {
  vdouble2 d0 = twosub_vd2_vd_vd(vd3getx_vd_vd3(x), vd3getx_vd_vd3(y));
  vdouble2 d1 = twosub_vd2_vd_vd(vd3gety_vd_vd3(x), vd3gety_vd_vd3(y));
  vdouble2 d3 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  return tdnormalize_vd3_vd3(vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d3), vadd_vd_4vd(vd3getz_vd_vd3(x), vneg_vd_vd(vd3getz_vd_vd3(y)), vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d3))));
}

static INLINE CONST VECTOR_CC vdouble3 tdadd2_vd3_vd2_vd3(vdouble2 x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(vd2getx_vd_vd2(x), vd3getx_vd_vd3(y));
  vdouble2 d1 = twosum_vd2_vd_vd(vd2gety_vd_vd2(x), vd3gety_vd_vd3(y));
  vdouble2 d3 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  return tdnormalize_vd3_vd3(vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d3), vadd_vd_3vd(vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d3), vd3getz_vd_vd3(y))));
}

static INLINE CONST VECTOR_CC vdouble3 tdadd_vd3_vd2_vd3(vdouble2 x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(vd2getx_vd_vd2(x), vd3getx_vd_vd3(y));
  vdouble2 d1 = twosum_vd2_vd_vd(vd2gety_vd_vd2(x), vd3gety_vd_vd3(y));
  vdouble2 d3 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  return vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d3), vadd_vd_3vd(vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d3), vd3getz_vd_vd3(y)));
}

static INLINE CONST VECTOR_CC vdouble3 tdadd2_vd3_vd_vd3(vdouble x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(x, vd3getx_vd_vd3(y));
  vdouble2 d3 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd3gety_vd_vd3(y));
  return tdnormalize_vd3_vd3(vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d3), vadd_vd_vd_vd(vd2gety_vd_vd2(d3), vd3getz_vd_vd3(y))));
}

static INLINE CONST VECTOR_CC vdouble3 tdadd_vd3_vd_vd3(vdouble x, vdouble3 y) {
  vdouble2 d0 = twosum_vd2_vd_vd(x, vd3getx_vd_vd3(y));
  vdouble2 d3 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd3gety_vd_vd3(y));
  return vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d3), vadd_vd_vd_vd(vd2gety_vd_vd2(d3), vd3getz_vd_vd3(y)));
}

static INLINE CONST VECTOR_CC vdouble3 tdscaleadd2_vd3_vd3_vd3_vd(vdouble3 x, vdouble3 y, vdouble s) {
  vdouble2 d0 = twosumx_vd2_vd_vd_vd(vd3getx_vd_vd3(x), vd3getx_vd_vd3(y), s);
  vdouble2 d1 = twosumx_vd2_vd_vd_vd(vd3gety_vd_vd3(x), vd3gety_vd_vd3(y), s);
  vdouble2 d3 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  return tdnormalize_vd3_vd3(vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d3), vadd_vd_3vd(vmla_vd_vd_vd_vd(vd3getz_vd_vd3(y), s, vd3getz_vd_vd3(x)), vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d3))));
}

static INLINE CONST VECTOR_CC vdouble3 tdscalesub2_vd3_vd3_vd3_vd(vdouble3 x, vdouble3 y, vdouble s) {
  vdouble2 d0 = twosubx_vd2_vd_vd_vd(vd3getx_vd_vd3(x), vd3getx_vd_vd3(y), s);
  vdouble2 d1 = twosubx_vd2_vd_vd_vd(vd3gety_vd_vd3(x), vd3gety_vd_vd3(y), s);
  vdouble2 d3 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  return tdnormalize_vd3_vd3(vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d3), vadd_vd_3vd(vmlanp_vd_vd_vd_vd(vd3getz_vd_vd3(y), s, vd3getz_vd_vd3(x)), vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d3))));
}

static INLINE CONST VECTOR_CC vdouble3 tdmul2_vd3_vd3_vd3(vdouble3 x, vdouble3 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(vd3getx_vd_vd3(x), vd3getx_vd_vd3(y));
  vdouble2 d1 = twoprod_vd2_vd_vd(vd3getx_vd_vd3(x), vd3gety_vd_vd3(y));
  vdouble2 d2 = twoprod_vd2_vd_vd(vd3gety_vd_vd3(x), vd3getx_vd_vd3(y));
  vdouble2 d4 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  vdouble2 d5 = twosum_vd2_vd_vd(vd2getx_vd_vd2(d4), vd2getx_vd_vd2(d2));

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(vd3getz_vd_vd3(x), vd3getx_vd_vd3(y), vmla_vd_vd_vd_vd(vd3gety_vd_vd3(x), vd3gety_vd_vd3(y), vmla_vd_vd_vd_vd(vd3getx_vd_vd3(x), vd3getz_vd_vd3(y), vadd_vd_vd_vd(vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d2))))), vd2gety_vd_vd2(d4), vd2gety_vd_vd2(d5));

  return tdnormalize_vd3_vd3(vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d5), t2));
}

static INLINE CONST VECTOR_CC vdouble3 tdmul_vd3_vd3_vd3(vdouble3 x, vdouble3 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(vd3getx_vd_vd3(x), vd3getx_vd_vd3(y));
  vdouble2 d1 = twoprod_vd2_vd_vd(vd3getx_vd_vd3(x), vd3gety_vd_vd3(y));
  vdouble2 d2 = twoprod_vd2_vd_vd(vd3gety_vd_vd3(x), vd3getx_vd_vd3(y));
  vdouble2 d4 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  vdouble2 d5 = twosum_vd2_vd_vd(vd2getx_vd_vd2(d4), vd2getx_vd_vd2(d2));

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(vd3getz_vd_vd3(x), vd3getx_vd_vd3(y), vmla_vd_vd_vd_vd(vd3gety_vd_vd3(x), vd3gety_vd_vd3(y), vmla_vd_vd_vd_vd(vd3getx_vd_vd3(x), vd3getz_vd_vd3(y), vadd_vd_vd_vd(vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d2))))), vd2gety_vd_vd2(d4), vd2gety_vd_vd2(d5));

  return tdquickrenormalize_vd3_vd3(vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d5), t2));
}

static INLINE CONST VECTOR_CC vdouble3 tdsqu_vd3_vd3(vdouble3 x) { return tdmul_vd3_vd3_vd3(x, x); }

static INLINE CONST VECTOR_CC vdouble3 tdmul_vd3_vd2_vd3(vdouble2 x, vdouble3 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(vd2getx_vd_vd2(x), vd3getx_vd_vd3(y));
  vdouble2 d1 = twoprod_vd2_vd_vd(vd2getx_vd_vd2(x), vd3gety_vd_vd3(y));
  vdouble2 d2 = twoprod_vd2_vd_vd(vd2gety_vd_vd2(x), vd3getx_vd_vd3(y));
  vdouble2 d4 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  vdouble2 d5 = twosum_vd2_vd_vd(vd2getx_vd_vd2(d4), vd2getx_vd_vd2(d2));

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(vd2gety_vd_vd2(x), vd3gety_vd_vd3(y), vmla_vd_vd_vd_vd(vd2getx_vd_vd2(x), vd3getz_vd_vd3(y), vadd_vd_vd_vd(vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d2)))), vd2gety_vd_vd2(d4), vd2gety_vd_vd2(d5));

  return vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d5), t2);
}

static INLINE CONST VECTOR_CC vdouble3 tdmul_vd3_vd3_vd2(vdouble3 x, vdouble2 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(vd2getx_vd_vd2(y), vd3getx_vd_vd3(x));
  vdouble2 d1 = twoprod_vd2_vd_vd(vd2getx_vd_vd2(y), vd3gety_vd_vd3(x));
  vdouble2 d2 = twoprod_vd2_vd_vd(vd2gety_vd_vd2(y), vd3getx_vd_vd3(x));
  vdouble2 d4 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  vdouble2 d5 = twosum_vd2_vd_vd(vd2getx_vd_vd2(d4), vd2getx_vd_vd2(d2));

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(vd2gety_vd_vd2(y), vd3gety_vd_vd3(x), vmla_vd_vd_vd_vd(vd2getx_vd_vd2(y), vd3getz_vd_vd3(x), vadd_vd_vd_vd(vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d2)))), vd2gety_vd_vd2(d4), vd2gety_vd_vd2(d5));

  return vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d5), t2);
}

static INLINE CONST VECTOR_CC vdouble3 tdmul_vd3_vd3_vd(vdouble3 x, vdouble y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(y, vd3getx_vd_vd3(x));
  vdouble2 d1 = twoprod_vd2_vd_vd(y, vd3gety_vd_vd3(x));
  vdouble2 d4 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));

  return vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d4), vadd_vd_vd_vd(vmla_vd_vd_vd_vd(y, vd3getz_vd_vd3(x), vd2gety_vd_vd2(d1)), vd2gety_vd_vd2(d4)));
}

static INLINE CONST VECTOR_CC vdouble3 tdmul_vd3_vd2_vd2(vdouble2 x, vdouble2 y) {
  vdouble2 d0 = twoprod_vd2_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
  vdouble2 d1 = twoprod_vd2_vd_vd(vd2getx_vd_vd2(x), vd2gety_vd_vd2(y));
  vdouble2 d2 = twoprod_vd2_vd_vd(vd2gety_vd_vd2(x), vd2getx_vd_vd2(y));
  vdouble2 d4 = twosum_vd2_vd_vd(vd2gety_vd_vd2(d0), vd2getx_vd_vd2(d1));
  vdouble2 d5 = twosum_vd2_vd_vd(vd2getx_vd_vd2(d4), vd2getx_vd_vd2(d2));

  vdouble t2 = vadd_vd_3vd(vmla_vd_vd_vd_vd(vd2gety_vd_vd2(x), vd2gety_vd_vd2(y), vadd_vd_vd_vd(vd2gety_vd_vd2(d1), vd2gety_vd_vd2(d2))), vd2gety_vd_vd2(d4), vd2gety_vd_vd2(d5));

  return vd3setxyz_vd3_vd_vd_vd(vd2getx_vd_vd2(d0), vd2getx_vd_vd2(d5), t2);
}

static INLINE CONST VECTOR_CC vdouble3 tddiv2_vd3_vd3_vd3(vdouble3 n, vdouble3 q) {
  vdouble2 d = ddrec_vd2_vd2((vdouble2) {vd3getx_vd_vd3(q), vd3gety_vd_vd3(q)});
  return tdmul2_vd3_vd3_vd3(n, tdadd_vd3_vd2_vd3(d, tdmul_vd3_vd2_vd3(ddscale_vd2_vd2_d(d, -1),
								      tdadd_vd3_vd_vd3(vcast_vd_d(-1), tdmul_vd3_vd2_vd3(d, q)))));
}

static INLINE CONST VECTOR_CC vdouble3 tddiv_vd3_vd3_vd3(vdouble3 n, vdouble3 q) {
  vdouble2 d = ddrec_vd2_vd2((vdouble2) {vd3getx_vd_vd3(q), vd3gety_vd_vd3(q)});
  return tdmul_vd3_vd3_vd3(n, tdadd_vd3_vd2_vd3(d, tdmul_vd3_vd2_vd3(ddscale_vd2_vd2_d(d, -1),
								     tdadd_vd3_vd_vd3(vcast_vd_d(-1), tdmul_vd3_vd2_vd3(d, q)))));
}

static INLINE CONST VECTOR_CC vdouble3 tdrec_vd3_vd3(vdouble3 q) {
  vdouble2 d = ddrec_vd2_vd2((vdouble2) {vd3getx_vd_vd3(q), vd3gety_vd_vd3(q)});
  return tdadd2_vd3_vd2_vd3(d, tdmul_vd3_vd2_vd3(ddscale_vd2_vd2_d(d, -1),
						 tdadd_vd3_vd_vd3(vcast_vd_d(-1), tdmul_vd3_vd2_vd3(d, q))));
}

static INLINE CONST VECTOR_CC vdouble3 tdrec_vd3_vd2(vdouble2 q) {
  vdouble2 d = ddrec_vd2_vd2((vdouble2) {vd2getx_vd_vd2(q), vd2gety_vd_vd2(q)});
  return tdadd2_vd3_vd2_vd3(d, tdmul_vd3_vd2_vd3(ddscale_vd2_vd2_d(d, -1),
						 tdadd_vd3_vd_vd3(vcast_vd_d(-1), tdmul_vd3_vd2_vd2(d, q))));
}

static INLINE CONST VECTOR_CC vdouble3 tdsqrt_vd3_vd3(vdouble3 d) {
  vdouble2 t = ddsqrt_vd2_vd2((vdouble2) {vd3getx_vd_vd3(d), vd3gety_vd_vd3(d)});
  vdouble3 r = tdmul2_vd3_vd3_vd3(tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd2_vd2(t, t)), tdrec_vd3_vd2(t));
  r = vsel_vd3_vo_vd3_vd3(veq_vo_vd_vd(vd3getx_vd_vd3(d), vcast_vd_d(0)), vcast_vd3_d_d_d(0, 0, 0), tdscale_vd3_vd3_d(r, 0.5));
  return r;
}

static INLINE CONST VECTOR_CC vdouble3 tdneg_vd3_vd3(vdouble3 d) {
  return vd3setxyz_vd3_vd_vd_vd(vneg_vd_vd(vd3getx_vd_vd3(d)),
				vneg_vd_vd(vd3gety_vd_vd3(d)),
				vneg_vd_vd(vd3getz_vd_vd3(d)));
}

//

static INLINE CONST VECTOR_CC vdouble poly2d(vdouble x, double c1, double c0) { return vmla_vd_vd_vd_vd(x, vcast_vd_d(c1), vcast_vd_d(c0)); }
static INLINE CONST VECTOR_CC vdouble poly3d(vdouble x, double c2, double c1, double c0) { return vmla_vd_vd_vd_vd(vmul_vd_vd_vd(x, x), vcast_vd_d(c2), poly2d(x, c1, c0)); }
static INLINE CONST VECTOR_CC vdouble poly4d(vdouble x, double c3, double c2, double c1, double c0) {
  return vmla_vd_vd_vd_vd(vmul_vd_vd_vd(x, x), poly2d(x, c3, c2), poly2d(x, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble poly5d(vdouble x, double c4, double c3, double c2, double c1, double c0) {
  return vmla_vd_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(x, x),vmul_vd_vd_vd(x, x)), vcast_vd_d(c4), poly4d(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble poly6d(vdouble x, double c5, double c4, double c3, double c2, double c1, double c0) {
  return vmla_vd_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(x, x),vmul_vd_vd_vd(x, x)), poly2d(x, c5, c4), poly4d(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble poly7d(vdouble x, double c6, double c5, double c4, double c3, double c2, double c1, double c0) {
  return vmla_vd_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(x, x),vmul_vd_vd_vd(x, x)), poly3d(x, c6, c5, c4), poly4d(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble poly8d(vdouble x, double c7, double c6, double c5, double c4, double c3, double c2, double c1, double c0) {
  return vmla_vd_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(x, x),vmul_vd_vd_vd(x, x)), poly4d(x, c7, c6, c5, c4), poly4d(x, c3, c2, c1, c0));
}

//

#if !defined(ENABLE_CUDA)
typedef struct {
  double x, y;
} double2;
#endif

static INLINE CONST VECTOR_CC double2 dd(double h, double l) {
  double2 ret = { h, l };
  return ret;
}

static INLINE CONST VECTOR_CC vdouble2 vcast_vd2_d2(double2 dd) {
  return vd2setxy_vd2_vd_vd(vcast_vd_d(dd.x), vcast_vd_d(dd.y));
}

static INLINE CONST VECTOR_CC vdouble2 ddmla_vd2_vd2_vd2_vd2(vdouble2 x, vdouble2 y, vdouble2 z) {
  return ddadd_vd2_vd2_vd2(z, ddmul_vd2_vd2_vd2(x, y));
}

static INLINE CONST VECTOR_CC vdouble2 poly2dd_b(vdouble2 x, double2 c1, double2 c0) { return ddmla_vd2_vd2_vd2_vd2(x, vcast_vd2_d2(c1), vcast_vd2_d2(c0)); }
static INLINE CONST VECTOR_CC vdouble2 poly2dd(vdouble2 x, vdouble c1, double2 c0) { return ddmla_vd2_vd2_vd2_vd2(x, vcast_vd2_vd_vd(c1, vcast_vd_d(0)), vcast_vd2_d2(c0)); }
static INLINE CONST VECTOR_CC vdouble2 poly3dd_b(vdouble2 x, double2 c2, double2 c1, double2 c0) { return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(x), vcast_vd2_d2(c2), poly2dd_b(x, c1, c0)); }
static INLINE CONST VECTOR_CC vdouble2 poly3dd(vdouble2 x, vdouble c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(x), vcast_vd2_vd_vd(c2, vcast_vd_d(0)), poly2dd_b(x, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly4dd_b(vdouble2 x, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(x), poly2dd_b(x, c3, c2), poly2dd_b(x, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly4dd(vdouble2 x, vdouble c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(x), poly2dd(x, c3, c2), poly2dd_b(x, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly5dd_b(vdouble2 x, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), vcast_vd2_d2(c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly5dd(vdouble2 x, vdouble c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), vcast_vd2_vd_vd(c4, vcast_vd_d(0)), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly6dd_b(vdouble2 x, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly2dd_b(x, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly6dd(vdouble2 x, vdouble c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly2dd(x, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly7dd_b(vdouble2 x, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly3dd_b(x, c6, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly7dd(vdouble2 x, vdouble c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly3dd(x, c6, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly8dd_b(vdouble2 x, double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly4dd_b(x, c7, c6, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly8dd(vdouble2 x, vdouble c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)), poly4dd(x, c7, c6, c5, c4), poly4dd_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly9dd_b(vdouble2 x, double2 c8,
				       double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), vcast_vd2_d2(c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly9dd(vdouble2 x, vdouble c8,
				     double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), vcast_vd2_vd_vd(c8, vcast_vd_d(0)), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly10dd_b(vdouble2 x, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly2dd_b(x, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly10dd(vdouble2 x, vdouble c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly2dd(x, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly11dd_b(vdouble2 x, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly3dd_b(x, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly11dd(vdouble2 x, vdouble c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly3dd(x, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly12dd_b(vdouble2 x, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly4dd_b(x, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly12dd(vdouble2 x, vdouble c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly4dd(x, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly13dd_b(vdouble2 x, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly5dd_b(x, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly13dd(vdouble2 x, vdouble c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly5dd(x, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly14dd_b(vdouble2 x, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly6dd_b(x, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly14dd(vdouble2 x, vdouble c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly6dd(x, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly15dd_b(vdouble2 x, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly7dd_b(x, c14, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly15dd(vdouble2 x, vdouble c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly7dd(x, c14, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly16dd_b(vdouble2 x, double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly8dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly16dd(vdouble2 x, vdouble c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x))), poly8dd(x, c15, c14, c13, c12, c11, c10, c9, c8), poly8dd_b(x, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly17dd_b(vdouble2 x, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), vcast_vd2_d2(c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly17dd(vdouble2 x, vdouble c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), vcast_vd2_vd_vd(c16, vcast_vd_d(0)),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly18dd_b(vdouble2 x, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly2dd_b(x, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly18dd(vdouble2 x, vdouble c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly2dd(x, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly19dd_b(vdouble2 x, double2 c18, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly3dd_b(x, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly19dd(vdouble2 x, vdouble c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly3dd(x, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly20dd_b(vdouble2 x, double2 c19, double2 c18, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly4dd_b(x, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly20dd(vdouble2 x, vdouble c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly4dd(x, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly21dd_b(vdouble2 x, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly5dd_b(x, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly21dd(vdouble2 x, vdouble c20, double2 c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly5dd(x, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly22dd_b(vdouble2 x, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly6dd_b(x, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly22dd(vdouble2 x, vdouble c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly6dd(x, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly23dd_b(vdouble2 x, double2 c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly7dd_b(x, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly23dd(vdouble2 x, vdouble c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly7dd(x, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly24dd_b(vdouble2 x, double2 c23, double2 c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly8dd_b(x, c23, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly24dd(vdouble2 x, vdouble c23, double2 c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly8dd(x, c23, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly25dd_b(vdouble2 x, double2 c24,
					double2 c23, double2 c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly9dd_b(x, c24, c23, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly25dd(vdouble2 x, vdouble c24,
				      double2 c23, double2 c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly9dd(x, c24, c23, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly26dd_b(vdouble2 x, double2 c25, double2 c24,
					double2 c23, double2 c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly10dd_b(x, c25, c24, c23, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly26dd(vdouble2 x, vdouble c25, double2 c24,
				      double2 c23, double2 c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly10dd(x, c25, c24, c23, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly27dd_b(vdouble2 x, double2 c26, double2 c25, double2 c24,
					double2 c23, double2 c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
					double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
					double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly11dd_b(x, c26, c25, c24, c23, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble2 poly27dd(vdouble2 x, vdouble c26, double2 c25, double2 c24, 
				      double2 c23, double2 c22, double2 c21, double2 c20, double2 c19, double2 c18, double2 c17, double2 c16,
				      double2 c15, double2 c14, double2 c13, double2 c12, double2 c11, double2 c10, double2 c9, double2 c8,
				      double2 c7, double2 c6, double2 c5, double2 c4, double2 c3, double2 c2, double2 c1, double2 c0) {
  return ddmla_vd2_vd2_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(ddsqu_vd2_vd2(x)))), poly11dd(x, c26, c25, c24, c23, c22, c21, c20, c19, c18, c17, c16),
	       poly16dd_b(x, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0));
}

//

#ifndef ENABLE_CUDA
typedef struct {
  double x, y, z;
} double3;
#endif

static INLINE CONST VECTOR_CC vdouble3 vcast_vd3_d3(double3 td) {
  return vd3setxyz_vd3_vd_vd_vd(vcast_vd_d(td.x), vcast_vd_d(td.y), vcast_vd_d(td.z));
}

static INLINE CONST double3 td(double x, double y, double z) {
  double3 ret = { x, y, z };
  return ret;
}

static INLINE CONST VECTOR_CC vdouble3 tdmla_vd3_vd3_vd3_vd3(vdouble3 x, vdouble3 y, vdouble3 z) { return tdadd2_vd3_vd3_vd3(z, tdmul_vd3_vd3_vd3(x, y)); }
static INLINE CONST VECTOR_CC vdouble3 poly2td_b(vdouble3 x, double3 c1, double3 c0) {
  if (c0.y == 0 && c0.z == 0) {
    if (c1.x == 1 && c1.y == 0 && c1.z == 0) return tdadd2_vd3_vd_vd3(vcast_vd_d(c0.x), x);
    return tdadd2_vd3_vd_vd3(vcast_vd_d(c0.x), tdmul_vd3_vd3_vd3(x, vcast_vd3_d3(c1)));
  }
  return tdmla_vd3_vd3_vd3_vd3(x, vcast_vd3_d3(c1), vcast_vd3_d3(c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly2td(vdouble3 x, vdouble2 c1, double3 c0) {
  return tdadd2_vd3_vd3_vd3(vcast_vd3_d3(c0), tdmul_vd3_vd3_vd2(x, c1));
}
static INLINE CONST VECTOR_CC vdouble3 poly3td_b(vdouble3 x, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(x), vcast_vd3_d3(c2), poly2td_b(x, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly3td(vdouble3 x, vdouble2 c2, double3 c1, double3 c0) {
  return tdadd2_vd3_vd3_vd3(poly2td_b(x, c1, c0), tdmul_vd3_vd3_vd2(tdsqu_vd3_vd3(x), c2));
}
static INLINE CONST VECTOR_CC vdouble3 poly4td_b(vdouble3 x, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(x), poly2td_b(x, c3, c2), poly2td_b(x, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly4td(vdouble3 x, vdouble2 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(x), poly2td(x, c3, c2), poly2td_b(x, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly5td_b(vdouble3 x, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), vcast_vd3_d3(c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly5td(vdouble3 x, vdouble2 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdadd2_vd3_vd3_vd3(poly4td_b(x, c3, c2, c1, c0), tdmul_vd3_vd3_vd2(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), c4));
}
static INLINE CONST VECTOR_CC vdouble3 poly6td_b(vdouble3 x, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly2td_b(x, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly6td(vdouble3 x, vdouble2 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly2td(x, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly7td_b(vdouble3 x, double3 c6, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly3td_b(x, c6, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly7td(vdouble3 x, vdouble2 c6, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly3td(x, c6, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly8td_b(vdouble3 x, double3 c7, double3 c6, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly4td_b(x, c7, c6, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}
static INLINE CONST VECTOR_CC vdouble3 poly8td(vdouble3 x, vdouble2 c7, double3 c6, double3 c5, double3 c4, double3 c3, double3 c2, double3 c1, double3 c0) {
  return tdmla_vd3_vd3_vd3_vd3(tdsqu_vd3_vd3(tdsqu_vd3_vd3(x)), poly4td(x, c7, c6, c5, c4), poly4td_b(x, c3, c2, c1, c0));
}

// TDX functions ------------------------------------------------------------------------------------------------------------

// TDX Compare and select functions

static INLINE CONST VECTOR_CC tdx vsel_tdx_vo64_tdx_tdx(vopmask o, tdx x, tdx y) {
  return tdxseted3_tdx_vm_vd3(vsel_vm_vo64_vm_vm(o, tdxgete_vm_tdx(x), tdxgete_vm_tdx(y)), 
			      vsel_vd3_vo_vd3_vd3(o, tdxgetd3_vd3_tdx(x), tdxgetd3_vd3_tdx(y)));
}

static INLINE CONST VECTOR_CC vmask vcmp_vm_tdx_tdx(tdx t0, tdx t1) {
  vmask r = vcast_vm_i_i(0, 0);
  r = vsel_vm_vo64_vm_vm(vlt_vo_vd_vd(tdxgetd3z_vd_tdx(t0), tdxgetd3z_vd_tdx(t1)), vcast_vm_i_i(-1, -1), r);
  r = vsel_vm_vo64_vm_vm(vgt_vo_vd_vd(tdxgetd3z_vd_tdx(t0), tdxgetd3z_vd_tdx(t1)), vcast_vm_i_i( 0,  1), r);
  r = vsel_vm_vo64_vm_vm(vlt_vo_vd_vd(tdxgetd3y_vd_tdx(t0), tdxgetd3y_vd_tdx(t1)), vcast_vm_i_i(-1, -1), r);
  r = vsel_vm_vo64_vm_vm(vgt_vo_vd_vd(tdxgetd3y_vd_tdx(t0), tdxgetd3y_vd_tdx(t1)), vcast_vm_i_i( 0,  1), r);
  r = vsel_vm_vo64_vm_vm(vlt_vo_vd_vd(tdxgetd3x_vd_tdx(t0), tdxgetd3x_vd_tdx(t1)), vcast_vm_i_i(-1, -1), r);
  r = vsel_vm_vo64_vm_vm(vgt_vo_vd_vd(tdxgetd3x_vd_tdx(t0), tdxgetd3x_vd_tdx(t1)), vcast_vm_i_i( 0,  1), r);
  r = vsel_vm_vo64_vm_vm(vand_vo_vo_vo(vgt64_vo_vm_vm(tdxgete_vm_tdx(t1), tdxgete_vm_tdx(t0)),
				       veq64_vo_vm_vm(vsignbit_vm_vd(tdxgetd3x_vd_tdx(t0)), vsignbit_vm_vd(tdxgetd3x_vd_tdx(t1)))),
			 vsel_vm_vo64_vm_vm(vsignbit_vo_vd(tdxgetd3x_vd_tdx(t0)), vcast_vm_i_i(+0, +1), vcast_vm_i_i(-1, -1)), r);
  r = vsel_vm_vo64_vm_vm(vand_vo_vo_vo(vgt64_vo_vm_vm(tdxgete_vm_tdx(t0), tdxgete_vm_tdx(t1)),
				       veq64_vo_vm_vm(vsignbit_vm_vd(tdxgetd3x_vd_tdx(t0)), vsignbit_vm_vd(tdxgetd3x_vd_tdx(t1)))),
			 vsel_vm_vo64_vm_vm(vsignbit_vo_vd(tdxgetd3x_vd_tdx(t0)), vcast_vm_i_i(-1, -1), vcast_vm_i_i(+0, +1)), r);
  r = vsel_vm_vo64_vm_vm(vand_vo_vo_vo(veq_vo_vd_vd(tdxgetd3x_vd_tdx(t0), vcast_vd_d(0)), veq_vo_vd_vd(tdxgetd3x_vd_tdx(t1), vcast_vd_d(0))),
			 vcast_vm_i_i( 0,  0), r);
  r = vsel_vm_vo64_vm_vm(vand_vo_vo_vo(vlt_vo_vd_vd(tdxgetd3x_vd_tdx(t0), vcast_vd_d(0)), vge_vo_vd_vd(tdxgetd3x_vd_tdx(t1), vcast_vd_d(0))),
			 vcast_vm_i_i(-1, -1), r);
  r = vsel_vm_vo64_vm_vm(vand_vo_vo_vo(vge_vo_vd_vd(tdxgetd3x_vd_tdx(t0), vcast_vd_d(0)), vlt_vo_vd_vd(tdxgetd3x_vd_tdx(t1), vcast_vd_d(0))),
			 vcast_vm_i_i( 0,  1), r);
  return r;
}

static INLINE CONST VECTOR_CC vopmask vsignbit_vo_tdx(tdx x) {
  return vsignbit_vo_vd(tdxgetd3x_vd_tdx(x));
}

static INLINE CONST VECTOR_CC vopmask visnan_vo_tdx(tdx x) {
  return visnan_vo_vd(tdxgetd3x_vd_tdx(x));
}

static INLINE CONST VECTOR_CC vopmask veq_vo_tdx_tdx(tdx x, tdx y) {
  return veq64_vo_vm_vm(vcmp_vm_tdx_tdx(x, y), vcast_vm_i_i(0, 0));
}

static INLINE CONST VECTOR_CC vopmask vneq_vo_tdx_tdx(tdx x, tdx y) {
  return vnot_vo64_vo64(veq64_vo_vm_vm(vcmp_vm_tdx_tdx(x, y), vcast_vm_i_i(0, 0)));
}

static INLINE CONST VECTOR_CC vopmask vgt_vo_tdx_tdx(tdx x, tdx y) {
  return vgt64_vo_vm_vm(vcmp_vm_tdx_tdx(x, y), vcast_vm_i_i(0, 0));
}

static INLINE CONST VECTOR_CC vopmask vlt_vo_tdx_tdx(tdx x, tdx y) {
  return vgt64_vo_vm_vm(vcmp_vm_tdx_tdx(y, x), vcast_vm_i_i(0, 0));
}

static INLINE CONST VECTOR_CC vopmask vge_vo_tdx_tdx(tdx x, tdx y) {
  return vgt64_vo_vm_vm(vcmp_vm_tdx_tdx(x, y), vcast_vm_i_i(-1, -1));
}

static INLINE CONST VECTOR_CC vopmask vle_vo_tdx_tdx(tdx x, tdx y) {
  return vgt64_vo_vm_vm(vcmp_vm_tdx_tdx(y, x), vcast_vm_i_i(-1, -1));
}

// TDX Cast operators

static INLINE CONST VECTOR_CC tdx vcast_tdx_vd(vdouble d) {
  tdx r = tdxsetexyz_tdx_vm_vd_vd_vd(vilogbk_vm_vd(d), d, vcast_vd_d(0), vcast_vd_d(0));
  r = tdxsetx_tdx_tdx_vd(r, vsel_vd_vo_vd_vd(visnonfinite_vo_vd(tdxgetd3x_vd_tdx(r)), tdxgetd3x_vd_tdx(r), vldexp2_vd_vd_vm(tdxgetd3x_vd_tdx(r), vneg64_vm_vm(tdxgete_vm_tdx(r)))));
  r = tdxsete_tdx_tdx_vm(r, vadd64_vm_vm_vm(tdxgete_vm_tdx(r), vcast_vm_i_i(0, 16383)));
  return r;
}

static INLINE CONST VECTOR_CC tdx vcast_tdx_d(double d) {
  return vcast_tdx_vd(vcast_vd_d(d));
}

static INLINE CONST VECTOR_CC tdx vcast_tdx_vd3(vdouble3 d) {
  vmask re = vilogbk_vm_vd(vd3getx_vd_vd3(d));
  vdouble3 rd3 = vd3setxyz_vd3_vd_vd_vd(vldexp2_vd_vd_vm(vd3getx_vd_vd3(d), vneg64_vm_vm(re)), 
					vldexp2_vd_vd_vm(vd3gety_vd_vd3(d), vneg64_vm_vm(re)),
					vldexp2_vd_vd_vm(vd3getz_vd_vd3(d), vneg64_vm_vm(re)));
  re = vadd64_vm_vm_vm(re, vcast_vm_i_i(0, 16383));
  return tdxseted3_tdx_vm_vd3(re, rd3);
}

static INLINE CONST VECTOR_CC tdx vcast_tdx_vd3_fast(vdouble3 d) {
  vmask re = vadd64_vm_vm_vm(vilogb2k_vm_vd(vd3getx_vd_vd3(d)), vcast_vm_i_i(0, 16383));
  vdouble t = vldexp3_vd_vd_vm(vcast_vd_d(0.5), vneg64_vm_vm(re));
  return tdxsetexyz_tdx_vm_vd_vd_vd(re,
				    vmul_vd_vd_vd(vd3getx_vd_vd3(d), t),
				    vmul_vd_vd_vd(vd3gety_vd_vd3(d), t),
				    vmul_vd_vd_vd(vd3getz_vd_vd3(d), t));
}

static INLINE CONST VECTOR_CC vdouble vcast_vd_tdx(tdx t) {
  vdouble ret = vldexp2_vd_vd_vm(tdxgetd3x_vd_tdx(t), vadd64_vm_vm_vm(tdxgete_vm_tdx(t), vcast_vm_i_i(-1, -16383)));

  vopmask o = vor_vo_vo_vo(veq_vo_vd_vd(tdxgetd3x_vd_tdx(t), vcast_vd_d(0)),
			   vlt64_vo_vm_vm(tdxgete_vm_tdx(t), vcast_vm_i_i(0, 16383 - 0x500)));
  ret = vsel_vd_vo_vd_vd(o, vmulsign_vd_vd_vd(vcast_vd_d(0), tdxgetd3x_vd_tdx(t)), ret);

  o = vgt64_vo_vm_vm(tdxgete_vm_tdx(t), vcast_vm_i_i(0, 16383 + 0x400));
  ret = vsel_vd_vo_vd_vd(o, vmulsign_vd_vd_vd(vcast_vd_d(SLEEF_INFINITY), tdxgetd3x_vd_tdx(t)), ret);

  return vsel_vd_vo_vd_vd(visnonfinite_vo_vd(tdxgetd3x_vd_tdx(t)), tdxgetd3x_vd_tdx(t), ret);
}

static INLINE CONST VECTOR_CC vdouble3 vcast_vd3_tdx(tdx t) {
  vmask e = vadd64_vm_vm_vm(tdxgete_vm_tdx(t), vcast_vm_i_i(-1, -16383));
  vdouble3 r = vcast_vd3_vd_vd_vd(vldexp2_vd_vd_vm(tdxgetd3x_vd_tdx(t), e),
				  vldexp2_vd_vd_vm(tdxgetd3y_vd_tdx(t), e),
				  vldexp2_vd_vd_vm(tdxgetd3z_vd_tdx(t), e));

  vopmask o = vor_vo_vo_vo(veq_vo_vd_vd(tdxgetd3x_vd_tdx(t), vcast_vd_d(0)),
			   vlt64_vo_vm_vm(tdxgete_vm_tdx(t), vcast_vm_i_i(0, 16383 - 0x500)));
  r = vsel_vd3_vo_vd3_vd3(o, vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(0), tdxgetd3x_vd_tdx(t)), vcast_vd_d(0), vcast_vd_d(0)), r);

  o = vgt64_vo_vm_vm(tdxgete_vm_tdx(t), vcast_vm_i_i(0, 16383 + 0x400));
  r = vsel_vd3_vo_vd3_vd3(o, vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(SLEEF_INFINITY), tdxgetd3x_vd_tdx(t)), vcast_vd_d(0), vcast_vd_d(0)), r);

  r = vsel_vd3_vo_vd3_vd3(visnonfinite_vo_vd(tdxgetd3x_vd_tdx(t)), tdxgetd3_vd3_tdx(t), r);

  return r;
}

// TDX Arithmetic functions

static INLINE CONST VECTOR_CC tdx vneg_tdx_tdx(tdx x) {
  return tdxsetd3_tdx_tdx_vd3(x, tdneg_vd3_vd3(tdxgetd3_vd3_tdx(x)));
}

static INLINE CONST VECTOR_CC vmask vilogb_vm_tdx(tdx t) {
  vmask e = vadd64_vm_vm_vm(tdxgete_vm_tdx(t), vcast_vm_i_i(-1, -16383));
  e = vsel_vm_vo64_vm_vm(vor_vo_vo_vo(veq_vo_vd_vd(tdxgetd3x_vd_tdx(t), vcast_vd_d(1.0)), vlt_vo_vd_vd(tdxgetd3y_vd_tdx(t), vcast_vd_d(0))),
			 vadd64_vm_vm_vm(e, vcast_vm_i_i(-1, -1)), e);
  return e;
}

static INLINE CONST VECTOR_CC tdx add2_tdx_tdx_tdx(tdx dd0, tdx dd1) { // finite numbers only
  vmask ed = vsub64_vm_vm_vm(tdxgete_vm_tdx(dd1), tdxgete_vm_tdx(dd0));
  vdouble t = vldexp3_vd_vd_vm(vcast_vd_d(1), ed);

  vdouble3 rd3 = tdscaleadd2_vd3_vd3_vd3_vd(tdxgetd3_vd3_tdx(dd0), tdxgetd3_vd3_tdx(dd1), t);
  tdx r = tdxseted3_tdx_vm_vd3(vilogb2k_vm_vd(vd3getx_vd_vd3(rd3)), rd3);
  t = vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(tdxgete_vm_tdx(r)));

  vopmask o = veq_vo_vd_vd(tdxgetd3x_vd_tdx(dd0), vcast_vd_d(0));
  r = tdxsete_tdx_tdx_vm(r, vsel_vm_vo64_vm_vm(o, tdxgete_vm_tdx(dd1), vadd64_vm_vm_vm(tdxgete_vm_tdx(r), tdxgete_vm_tdx(dd0))));
  r = tdxsetd3_tdx_tdx_vd3(r, tdscale_vd3_vd3_vd(tdxgetd3_vd3_tdx(r), t));

  r = vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(ed, vcast_vm_i_i(0, 200)), dd1, r);
  r = vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(vcast_vm_i_i(-1, -200), ed), dd0, r);

  return r;
}

static INLINE CONST VECTOR_CC tdx sub2_tdx_tdx_tdx(tdx dd0, tdx dd1) {
  vmask ed = vsub64_vm_vm_vm(tdxgete_vm_tdx(dd1), tdxgete_vm_tdx(dd0));
  vdouble t = vldexp3_vd_vd_vm(vcast_vd_d(1), ed);

  vdouble3 rd3 = tdscalesub2_vd3_vd3_vd3_vd(tdxgetd3_vd3_tdx(dd0), tdxgetd3_vd3_tdx(dd1), t);
  tdx r = tdxseted3_tdx_vm_vd3(vilogb2k_vm_vd(vd3getx_vd_vd3(rd3)), rd3);
  t = vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(tdxgete_vm_tdx(r)));

  vopmask o = veq_vo_vd_vd(tdxgetd3x_vd_tdx(dd0), vcast_vd_d(0));
  r = tdxsete_tdx_tdx_vm(r, vsel_vm_vo64_vm_vm(o, tdxgete_vm_tdx(dd1), vadd64_vm_vm_vm(tdxgete_vm_tdx(r), tdxgete_vm_tdx(dd0))));
  r = tdxsetd3_tdx_tdx_vd3(r, tdscale_vd3_vd3_vd(tdxgetd3_vd3_tdx(r), t));

  r = vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(ed, vcast_vm_i_i(0, 200)), vneg_tdx_tdx(dd1), r);
  r = vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(vcast_vm_i_i(-1, -200), ed), dd0, r);

  return r;
}

static INLINE CONST VECTOR_CC tdx mul2_tdx_tdx_tdx(tdx dd0, tdx dd1) {
  vdouble3 rd3 = tdmul2_vd3_vd3_vd3(tdxgetd3_vd3_tdx(dd0), tdxgetd3_vd3_tdx(dd1));
  tdx r = tdxseted3_tdx_vm_vd3(vilogb2k_vm_vd(vd3getx_vd_vd3(rd3)), rd3);
  r = tdxsetd3_tdx_tdx_vd3(r, tdscale_vd3_vd3_vd(tdxgetd3_vd3_tdx(r), vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(tdxgete_vm_tdx(r)))));
  r = tdxsete_tdx_tdx_vm(r, vadd64_vm_vm_vm(vadd64_vm_vm_vm(vadd64_vm_vm_vm(tdxgete_vm_tdx(dd0), tdxgete_vm_tdx(dd1)), vcast_vm_i_i(-1, -16383)), tdxgete_vm_tdx(r)));
  return r;
}

static INLINE CONST VECTOR_CC tdx mul_tdx_tdx_tdx(tdx dd0, tdx dd1) {
  vdouble3 rd3 = tdmul_vd3_vd3_vd3(tdxgetd3_vd3_tdx(dd0), tdxgetd3_vd3_tdx(dd1));
  tdx r = tdxseted3_tdx_vm_vd3(vilogb2k_vm_vd(vd3getx_vd_vd3(rd3)), rd3);
  r = tdxsetd3_tdx_tdx_vd3(r, tdscale_vd3_vd3_vd(tdxgetd3_vd3_tdx(r), vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(tdxgete_vm_tdx(r)))));
  r = tdxsete_tdx_tdx_vm(r, vadd64_vm_vm_vm(vadd64_vm_vm_vm(vadd64_vm_vm_vm(tdxgete_vm_tdx(dd0), tdxgete_vm_tdx(dd1)), vcast_vm_i_i(-1, -16383)), tdxgete_vm_tdx(r)));
  return r;
}

static INLINE CONST VECTOR_CC tdx div2_tdx_tdx_tdx(tdx dd0, tdx dd1) {
  vdouble3 rd3 = tddiv2_vd3_vd3_vd3(tdxgetd3_vd3_tdx(dd0), tdxgetd3_vd3_tdx(dd1));
  tdx r = tdxseted3_tdx_vm_vd3(vilogb2k_vm_vd(vd3getx_vd_vd3(rd3)), rd3);
  r = tdxsetd3_tdx_tdx_vd3(r, tdscale_vd3_vd3_vd(tdxgetd3_vd3_tdx(r), vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(tdxgete_vm_tdx(r)))));
  r = tdxsete_tdx_tdx_vm(r, vadd64_vm_vm_vm(vadd64_vm_vm_vm(vsub64_vm_vm_vm(tdxgete_vm_tdx(dd0), tdxgete_vm_tdx(dd1)), vcast_vm_i_i(0, 16383)), tdxgete_vm_tdx(r)));
  return r;
}

// TDX math functions

static INLINE CONST VECTOR_CC tdx sqrt_tdx_tdx(tdx dd0) {
  vopmask o = veq64_vo_vm_vm(vand_vm_vm_vm(tdxgete_vm_tdx(dd0), vcast_vm_i_i(0, 1)), vcast_vm_i_i(0, 1));
  dd0 = tdxsetd3_tdx_tdx_vd3(dd0, tdscale_vd3_vd3_vd(tdxgetd3_vd3_tdx(dd0), vsel_vd_vo_vd_vd(o, vcast_vd_d(1), vcast_vd_d(2))));
  vdouble3 rd3 = tdsqrt_vd3_vd3(tdxgetd3_vd3_tdx(dd0));
  tdx r = tdxseted3_tdx_vm_vd3(vilogb2k_vm_vd(vd3getx_vd_vd3(rd3)), rd3);
  r = tdxsetd3_tdx_tdx_vd3(r, tdscale_vd3_vd3_vd(tdxgetd3_vd3_tdx(r), vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(tdxgete_vm_tdx(r)))));
  r = tdxsete_tdx_tdx_vm(r, vadd64_vm_vm_vm(vcast_vm_vi2(vsra_vi2_vi2_i(vcast_vi2_vm(vadd64_vm_vm_vm(tdxgete_vm_tdx(dd0), vcast_vm_i_i(0, 16383))), 1)), tdxgete_vm_tdx(r)));
  o = vneq_vo_vd_vd(tdxgetd3x_vd_tdx(dd0), vcast_vd_d(0));
  return tdxsetxyz_tdx_tdx_vd_vd_vd(r,
				    vreinterpret_vd_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(tdxgetd3x_vd_tdx(r)))),
				    vreinterpret_vd_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(tdxgetd3y_vd_tdx(r)))),
				    vreinterpret_vd_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(tdxgetd3z_vd_tdx(r)))));
}

static CONST VECTOR_CC tdi_t rempio2q(tdx a) {
  const int N = 8, B = 8;
  const int NCOL = 53-B, NROW = (16385+(53-B)*N-106)/NCOL+1;

  vmask e = vilogb_vm_tdx(a);
  e = vsel_vm_vo64_vm_vm(vgt64_vo_vm_vm(e, vcast_vm_i_i(0, 106)), e, vcast_vm_i_i(0, 106));
  a = tdxsete_tdx_tdx_vm(a, vadd64_vm_vm_vm(tdxgete_vm_tdx(a), vsub64_vm_vm_vm(vcast_vm_i_i(0, 106), e)));

  vdouble row = vtruncate_vd_vd(vmul_vd_vd_vd(vcast_vd_vm(vsub64_vm_vm_vm(e, vcast_vm_i_i(0, 106))), vcast_vd_d(1.0 / NCOL))); // (e - 106) / NCOL;
  vdouble col = vsub_vd_vd_vd(vcast_vd_vm(vsub64_vm_vm_vm(e, vcast_vm_i_i(0, 106))), vmul_vd_vd_vd(row, vcast_vd_d(NCOL)));    // (e - 106) % NCOL;
  vint p = vtruncate_vi_vd(vmla_vd_vd_vd_vd(col, vcast_vd_d(NROW), row));

  vint q = vcast_vi_i(0);
  vdouble3 d = tdnormalize_vd3_vd3(vcast_vd3_tdx(a)), x = vcast_vd3_d_d_d(0, 0, 0);

  for(int i=0;i<N;i++) {
    vdouble t = vldexp3_vd_vd_vm(vgather_vd_p_vi(Sleef_rempitabqp+i, p), vcast_vm_i_i(i == 0 ? 0 : -1, -(53-B)*i));
    x = tdadd2_vd3_vd3_vd3(x, tdnormalize_vd3_vd3(tdmul_vd3_vd3_vd(d, t)));
    di_t di = rempisub(vd3getx_vd_vd3(x));
    q = vadd_vi_vi_vi(q, digeti_vi_di(di));
    x = vd3setx_vd3_vd3_vd(x, digetd_vd_di(di));
    x = tdnormalize_vd3_vd3(x);
  }

  x = tdmul2_vd3_vd3_vd3(x, vcast_vd3_d_d_d(3.141592653589793116*2, 1.2246467991473532072e-16*2, -2.9947698097183396659e-33*2));
  x = vsel_vd3_vo_vd3_vd3(vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383), tdxgete_vm_tdx(a)), d, x);

  return tdisettdi_tdi_vd3_vi(x, q);
}

static INLINE CONST VECTOR_CC tdx sin_tdx_tdx(tdx a) {
  const double3 npiu = { -3.141592653589793116, -1.2246467991473532072e-16, 0 };
  const double3 npil = { +2.9947698097183396659e-33, -1.1124542208633652815e-49, -5.6722319796403157441e-66 };

  vdouble3 d = vcast_vd3_tdx(a);

  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(vd3getx_vd_vd3(d), vcast_vd_d(1.0 / M_PI)));
  vint q = vrint_vi_vd(dq);
  d = tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd3_vd(vcast_vd3_d3(npiu), dq));
  d = tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd3_vd(vcast_vd3_d3(npil), dq));

  vopmask o = vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 + 28), tdxgete_vm_tdx(a));
  if (!LIKELY(vtestallones_i_vo64(o))) {
    tdi_t tdi = rempio2q(a);
    vint qw = vand_vi_vi_vi(tdigeti_vi_tdi(tdi), vcast_vi_i(3));
    qw = vadd_vi_vi_vi(vadd_vi_vi_vi(qw, qw), vsel_vi_vo_vi_vi(vcast_vo32_vo64(vgt_vo_vd_vd(tdigetx_vd_tdi(tdi), vcast_vd_d(0))),
							       vcast_vi_i(2), vcast_vi_i(1)));
    qw = vsra_vi_vi_i(qw, 2);
    vdouble3 dw = vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(3.141592653589793116      *-0.5), tdigetx_vd_tdi(tdi)),
				     vmulsign_vd_vd_vd(vcast_vd_d(1.2246467991473532072e-16 *-0.5), tdigetx_vd_tdi(tdi)),
				     vmulsign_vd_vd_vd(vcast_vd_d(-2.9947698097183396659e-33*-0.5), tdigetx_vd_tdi(tdi)));
    dw = vsel_vd3_vo_vd3_vd3(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(tdigeti_vi_tdi(tdi), vcast_vi_i(1)), vcast_vi_i(1))),
			     tdadd2_vd3_vd3_vd3(tdigettd_vd3_tdi(tdi), dw), tdigettd_vd3_tdi(tdi));
    d = vsel_vd3_vo_vd3_vd3(o, d, dw);
    q = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), q, qw);
  }

  vdouble3 s = tdsqu_vd3_vd3(d);

  vmask m = vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1))), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
  d = vd3setxyz_vd3_vd_vd_vd(vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(vd3getx_vd_vd3(d)))),
			     vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(vd3gety_vd_vd3(d)))),
			     vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(vd3getz_vd_vd3(d)))));

  vdouble u = poly4d(vd3getx_vd_vd3(s),
		     -1.1940250944959890417e-34,
		     1.1308027528153266305e-31,
		     -9.183679676378987613e-29,
		     6.4469502484797539906e-26);

  vdouble2 u2 = poly9dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(s), vd3gety_vd_vd3(s)),
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
  return vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 - 0x300), tdxgete_vm_tdx(a)), a, vcast_tdx_vd3_fast(u3));
}

static INLINE CONST VECTOR_CC tdx cos_tdx_tdx(tdx a) {
  const double3 npiu = { -3.141592653589793116*0.5, -1.2246467991473532072e-16*0.5, 0 };
  const double3 npil = { +2.9947698097183396659e-33*0.5, -1.1124542208633652815e-49*0.5, -5.6722319796403157441e-66*0.5 };

  vdouble3 d = vcast_vd3_tdx(a);
  vdouble dq = vmla_vd_vd_vd_vd(vcast_vd_d(2),
				vrint_vd_vd(vmla_vd_vd_vd_vd(vd3getx_vd_vd3(d), vcast_vd_d(1.0 / M_PI), vcast_vd_d(-0.5))),
				vcast_vd_d(1));
  vint q = vrint_vi_vd(dq);
  d = tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd3_vd(vcast_vd3_d3(npiu), dq));
  d = tdadd2_vd3_vd3_vd3(d, tdmul_vd3_vd3_vd(vcast_vd3_d3(npil), dq));

  vopmask o = vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 + 28), tdxgete_vm_tdx(a));
  if (!LIKELY(vtestallones_i_vo64(o))) {
    tdi_t tdi = rempio2q(a);
    vint qw = vand_vi_vi_vi(tdigeti_vi_tdi(tdi), vcast_vi_i(3));
    qw = vadd_vi_vi_vi(vadd_vi_vi_vi(qw, qw), vsel_vi_vo_vi_vi(vcast_vo32_vo64(vgt_vo_vd_vd(tdigetx_vd_tdi(tdi), vcast_vd_d(0))), vcast_vi_i(8), vcast_vi_i(7)));
    qw = vsra_vi_vi_i(qw, 1);
    vdouble3 dw = vcast_vd3_vd_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(3.141592653589793116      *-0.5), tdigetx_vd_tdi(tdi)),
				     vmulsign_vd_vd_vd(vcast_vd_d(1.2246467991473532072e-16 *-0.5), tdigetx_vd_tdi(tdi)),
				     vmulsign_vd_vd_vd(vcast_vd_d(-2.9947698097183396659e-33*-0.5), tdigetx_vd_tdi(tdi)));
    dw = vsel_vd3_vo_vd3_vd3(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(tdigeti_vi_tdi(tdi), vcast_vi_i(1)), vcast_vi_i(0))),
			     tdadd2_vd3_vd3_vd3(tdigettd_vd3_tdi(tdi), dw), tdigettd_vd3_tdi(tdi));
    d = vsel_vd3_vo_vd3_vd3(o, d, dw);
    q = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), q, qw);
  }

  vdouble3 s = tdsqu_vd3_vd3(d);

  vdouble u = poly4d(vd3getx_vd_vd3(s),
		     -1.1940250944959890417e-34,
		     1.1308027528153266305e-31,
		     -9.183679676378987613e-29,
		     6.4469502484797539906e-26);

  vdouble2 u2 = poly9dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(s), vd3gety_vd_vd3(s)),
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
  u3 = vd3setxyz_vd3_vd_vd_vd(vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(vd3getx_vd_vd3(u3)))),
			      vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(vd3gety_vd_vd3(u3)))),
			      vreinterpret_vd_vm(vxor_vm_vm_vm(m, vreinterpret_vm_vd(vd3getz_vd_vd3(u3)))));

  return vcast_tdx_vd3_fast(u3);
}

static INLINE CONST VECTOR_CC tdx tan_tdx_tdx(tdx a) {
  const double3 npiu = { -3.141592653589793116*0.5, -1.2246467991473532072e-16*0.5, 0 };
  const double3 npil = { +2.9947698097183396659e-33*0.5, -1.1124542208633652815e-49*0.5, -5.6722319796403157441e-66*0.5 };

  vdouble3 x = vcast_vd3_tdx(a);
  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(vd3getx_vd_vd3(x), vcast_vd_d(2.0 / M_PI)));
  vint q = vrint_vi_vd(dq);
  x = tdadd2_vd3_vd3_vd3(x, tdmul_vd3_vd3_vd(vcast_vd3_d3(npiu), dq));
  x = tdadd2_vd3_vd3_vd3(x, tdmul_vd3_vd3_vd(vcast_vd3_d3(npil), dq));

  vopmask o = vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 + 28), tdxgete_vm_tdx(a));
  if (!LIKELY(vtestallones_i_vo64(o))) {
    tdi_t tdi = rempio2q(a);
    x = vsel_vd3_vo_vd3_vd3(o, x, tdigettd_vd3_tdi(tdi));
    q = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), q, tdigeti_vi_tdi(tdi));
  }

  x = tdscale_vd3_vd3_d(x, 0.5);
  vdouble3 s = tdsqu_vd3_vd3(x);

  vdouble u = poly5d(vd3getx_vd_vd3(s),
		     2.2015831737910052452e-08,
		     2.0256594378812907225e-08,
		     7.4429817298004292868e-08,
		     1.728455913166476866e-07,
		     4.2976852952607503818e-07);

  vdouble2 u2 = poly14dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(s), vd3gety_vd_vd3(s)),
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

  return vsel_tdx_vo64_tdx_tdx(vgt64_vo_vm_vm(vcast_vm_i_i(0, 16383 - 0x300), tdxgete_vm_tdx(a)), a, vcast_tdx_vd3_fast(u3));
}

static INLINE CONST VECTOR_CC tdx exp_tdx_tdx(tdx a) {
  const double3 nln2u = { -0.69314718055994528623, -2.3190468138462995584e-17, 0 };
  const double3 nln2l = { -5.7077084384162120658e-34, +3.5824322106018114234e-50, +1.3521696757988629569e-66 };

  vdouble3 s = vcast_vd3_tdx(a);
  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(vd3getx_vd_vd3(s), vcast_vd_d(R_LN2)));
  vint q = vrint_vi_vd(dq);
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nln2u), dq));
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nln2l), dq));

  s = tdscale_vd3_vd3_d(s, 0.5);

  vdouble u = poly6d(vd3getx_vd_vd3(s),
		     1.5620530411202639902e-16,
		     2.8125634200750730004e-15,
		     4.7794775039652234692e-14,
		     7.6471631094741035269e-13,
		     1.1470745597601740926e-11,
		     1.6059043837011404763e-10);

  vdouble2 u2 = poly10dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(s), vd3gety_vd_vd3(s)),
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
  r = tdxsete_tdx_tdx_vm(r, vadd64_vm_vm_vm(tdxgete_vm_tdx(r), vcast_vm_vi(q)));

  vopmask p = vor_vo_vo_vo(visinf_vo_vd(tdxgetd3x_vd_tdx(a)), vgt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16397)));
  vopmask o = vgt_vo_vd_vd(tdxgetd3x_vd_tdx(a), vcast_vd_d(0));

  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vand_vo_vo_vo(o, p), vcast_vd3_d_d_d(SLEEF_INFINITY, SLEEF_INFINITY, SLEEF_INFINITY), tdxgetd3_vd3_tdx(r)));
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(a)), p)), vcast_vd3_d_d_d(0, 0, 0), tdxgetd3_vd3_tdx(r)));

  return r;
}

static INLINE CONST VECTOR_CC tdx exp2_tdx_tdx(tdx a) {
  vdouble3 s = vcast_vd3_tdx(a);

  vdouble dq = vrint_vd_vd(vd3getx_vd_vd3(s));
  vint q = vrint_vi_vd(dq);
  s = tdadd2_vd3_vd_vd3(vneg_vd_vd(dq), s);

  s = tdscale_vd3_vd3_d(s, 0.5);

  vdouble u = poly6d(vd3getx_vd_vd3(s),
		     2.1312038164967297247e-19,
		     5.5352570141139560433e-18,
		     1.357024745958052877e-16,
		     3.132436443693084597e-15,
		     6.7787263548592201519e-14,
		     1.3691488854074843157e-12);

  vdouble2 u2 = poly10dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(s), vd3gety_vd_vd3(s)),
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
  r = tdxsete_tdx_tdx_vm(r, vadd64_vm_vm_vm(tdxgete_vm_tdx(r), vcast_vm_vi(q)));

  vopmask p = vor_vo_vo_vo(visinf_vo_vd(tdxgetd3x_vd_tdx(a)), vgt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16397)));
  vopmask o = vgt_vo_vd_vd(tdxgetd3x_vd_tdx(a), vcast_vd_d(0));

  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vand_vo_vo_vo(o, p), vcast_vd3_d_d_d(SLEEF_INFINITY, SLEEF_INFINITY, SLEEF_INFINITY), tdxgetd3_vd3_tdx(r)));
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(a)), p)), vcast_vd3_d_d_d(0, 0, 0), tdxgetd3_vd3_tdx(r)));

  return r;
}

static INLINE CONST VECTOR_CC tdx exp10_tdx_tdx(tdx a) {
  const double3 nlog_10_2u = { -0.30102999566398119802, 2.8037281277851703937e-18, 0 };
  const double3 nlog_10_2l = { -5.4719484023146385333e-35, -5.1051389831070924689e-51, -1.2459153896093320861e-67 };

  vdouble3 s = vcast_vd3_tdx(a);
  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(vd3getx_vd_vd3(s), vcast_vd_d(LOG10_2)));
  vint q = vrint_vi_vd(dq);
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nlog_10_2u), dq));
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nlog_10_2l), dq));

  s = tdscale_vd3_vd3_d(s, 0.5);

  vdouble u = poly6d(vd3getx_vd_vd3(s),
		     5.1718894362277323603e-10,
		     4.0436341626932450119e-09,
		     2.9842239377609726639e-08,
		     2.073651082488697668e-07,
		     1.3508629476297046323e-06,
		     8.2134125355421453926e-06);

  vdouble2 u2 = poly10dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(s), vd3gety_vd_vd3(s)),
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
  r = tdxsete_tdx_tdx_vm(r, vadd64_vm_vm_vm(tdxgete_vm_tdx(r), vcast_vm_vi(q)));

  vopmask p = vor_vo_vo_vo(visinf_vo_vd(tdxgetd3x_vd_tdx(a)), vgt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16395)));
  vopmask o = vgt_vo_vd_vd(tdxgetd3x_vd_tdx(a), vcast_vd_d(0));

  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vand_vo_vo_vo(o, p), vcast_vd3_d_d_d(SLEEF_INFINITY, SLEEF_INFINITY, SLEEF_INFINITY), tdxgetd3_vd3_tdx(r)));
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(a)), p)), vcast_vd3_d_d_d(0, 0, 0), tdxgetd3_vd3_tdx(r)));

  return r;
}

static INLINE CONST VECTOR_CC tdx expm1_tdx_tdx(tdx a) {
  const double3 nln2u = { -0.69314718055994528623, -2.3190468138462995584e-17, 0 };
  const double3 nln2l = { -5.7077084384162120658e-34, +3.5824322106018114234e-50, +1.3521696757988629569e-66 };

  vdouble3 s = vcast_vd3_tdx(a);
  vopmask o = vlt_vo_vd_vd(vd3getx_vd_vd3(s), vcast_vd_d(-100));

  vdouble dq = vrint_vd_vd(vmul_vd_vd_vd(vd3getx_vd_vd3(s), vcast_vd_d(R_LN2)));
  vint q = vrint_vi_vd(dq);
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nln2u), dq));
  s = tdadd2_vd3_vd3_vd3(s, tdmul_vd3_vd3_vd(vcast_vd3_d3(nln2l), dq));

  vdouble u = poly7d(vd3getx_vd_vd3(s),
		     8.9624718038949319757e-22,
		     1.9596996271605003749e-20,
		     4.1102802184721881474e-19,
		     8.2206288447936758107e-18,
		     1.561920706551789357e-16,
		     2.8114572552972840413e-15,
		     4.7794773323730898317e-14);

  vdouble2 u2 = poly12dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(s), vd3gety_vd_vd3(s)),
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

  r = vsel_tdx_vo64_tdx_tdx(p, add2_tdx_tdx_tdx(r, vcast_tdx_d(1)), r);
  r = tdxsete_tdx_tdx_vm(r, vsel_vm_vo64_vm_vm(p, vadd64_vm_vm_vm(tdxgete_vm_tdx(r), vcast_vm_vi(q)), tdxgete_vm_tdx(r)));
  r = vsel_tdx_vo64_tdx_tdx(p, sub2_tdx_tdx_tdx(r, vcast_tdx_d(1)), r);

  p = vand_vo_vo_vo(vgt_vo_vd_vd(tdxgetd3x_vd_tdx(a), vcast_vd_d(0)),
		    vor_vo_vo_vo(visinf_vo_vd(tdxgetd3x_vd_tdx(a)), vgt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16397))));
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vand_vo_vo_vo(vgt_vo_vd_vd(tdxgetd3x_vd_tdx(a), vcast_vd_d(0)), p),
						  vcast_vd3_d_d_d(SLEEF_INFINITY, SLEEF_INFINITY, SLEEF_INFINITY), tdxgetd3_vd3_tdx(r)));

  p = vandnot_vo_vo_vo(vgt_vo_vd_vd(tdxgetd3x_vd_tdx(a), vcast_vd_d(0)), 
		       vor_vo_vo_vo(visinf_vo_vd(tdxgetd3x_vd_tdx(a)), vgt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16389))));
  r = vsel_tdx_vo64_tdx_tdx(vor_vo_vo_vo(o, p), vcast_tdx_d(-1), r);

  r = vsel_tdx_vo64_tdx_tdx(vor_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(a)), vlt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16000))),
			    a, r);

  return r;
}

static INLINE CONST VECTOR_CC tdx log_tdx_tdx(tdx d) {
  vmask e = vilogb2k_vm_vd(vmul_vd_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(1/0.75)));
  e = vadd64_vm_vm_vm(e, vadd64_vm_vm_vm(tdxgete_vm_tdx(d), vcast_vm_i_i(-1, -16383)));
  d = tdxsete_tdx_tdx_vm(d, vsub64_vm_vm_vm(tdxgete_vm_tdx(d), e));

  vdouble3 x = vcast_vd3_tdx(d);

  x = tddiv_vd3_vd3_vd3(tdadd2_vd3_vd_vd3(vcast_vd_d(-1), x), tdadd2_vd3_vd_vd3(vcast_vd_d(1), x));

  vdouble3 x2 = tdsqu_vd3_vd3(x);

  vdouble t = poly6d(vd3getx_vd_vd3(x2),
		     0.077146495191485184306,
		     0.052965311163344339085,
		     0.061053005369706196681,
		     0.064483912078500099652,
		     0.06896718437842412619,
		     0.074074009929408698993);

  vdouble2 t2 = poly12dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(x2), vd3gety_vd_vd3(x2)),
			 t,
			 dd(0.080000001871751574845, -1.9217428179318222729e-18),
			 dd(0.086956521697293065465, 5.2385684246247988452e-18),
			 dd(0.095238095238813463839, -5.9627506176667020273e-19),
			 dd(0.10526315789472741324, -6.3998555142971381419e-18),
			 dd(0.11764705882352950728, -1.5923063236090721578e-18),
			 dd(0.13333333333333333148, 1.1538478835417353313e-18),
			 dd(0.15384615384615385469, -8.5364262380653974925e-18),
			 dd(0.18181818181818182323, -5.0464824142103336146e-18),
			 dd(0.22222222222222220989, 1.2335811419831364972e-17),
			 dd(0.28571428571428569843, 1.5860328923163793786e-17),
			 dd(0.4000000000000000222, -2.22044604925030889e-17));

  vdouble3 t3 = poly3td(x2,
			t2,
			td(0.66666666666666662966, 3.7007434154171882626e-17, 2.0425398205897696253e-33),
			td(2, 0, 0));

  x = tdmla_vd3_vd3_vd3_vd3(x, t3, tdmul_vd3_vd3_vd(vcast_vd3_d_d_d(0.69314718055994528623, 2.3190468138462995584e-17, 5.7077084384162120658e-34), vcast_vd_vm(e)));

  tdx r = vcast_tdx_vd3_fast(x);

  r = vsel_tdx_vo64_tdx_tdx(visinf_vo_vd(tdxgetd3x_vd_tdx(d)), d, r);
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vor_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(d)), vlt_vo_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(0))),
						  vcast_vd3_d_d_d(SLEEF_NAN, SLEEF_NAN, SLEEF_NAN), tdxgetd3_vd3_tdx(r)));
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(veq_vo_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(0)),
						  vcast_vd3_d_d_d(-SLEEF_INFINITY, -SLEEF_INFINITY, -SLEEF_INFINITY), tdxgetd3_vd3_tdx(r)));

  return r;
}

static INLINE CONST VECTOR_CC tdx log2_tdx_tdx(tdx d) {
  vmask e = vilogb2k_vm_vd(vmul_vd_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(1/0.75)));
  e = vadd64_vm_vm_vm(e, vadd64_vm_vm_vm(tdxgete_vm_tdx(d), vcast_vm_i_i(-1, -16383)));
  d = tdxsete_tdx_tdx_vm(d, vsub64_vm_vm_vm(tdxgete_vm_tdx(d), e));

  vdouble3 x = vcast_vd3_tdx(d);
  x = tddiv_vd3_vd3_vd3(tdadd2_vd3_vd_vd3(vcast_vd_d(-1), x), tdadd2_vd3_vd_vd3(vcast_vd_d(1), x));

  vdouble3 x2 = tdsqu_vd3_vd3(x);

  vdouble t = poly5d(vd3getx_vd_vd3(x2),
		     0.11283869194114022616,
		     0.075868674982939712792,
		     0.088168993990551183804,
		     0.093021951597299201708,
		     0.099499193384647965921);

  vdouble2 t2 = poly12dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(x2), vd3gety_vd_vd3(x2)),
			 t,
			 dd(0.10686617907247662751, -5.5612389810478289383e-18),
			 dd(0.11541560695469484099, -2.4324034365866797361e-18),
			 dd(0.12545174259935440442, 5.5468926503422675757e-19),
			 dd(0.13739952770528035542, -1.3233875890966274475e-18),
			 dd(0.15186263588302695293, -1.0242825966355216372e-18),
			 dd(0.16972882833987829043, -1.1621860275817517217e-17),
			 dd(0.19235933878519512197, -2.8146828788758641178e-18),
			 dd(0.22195308321368667492, 3.142152296591772438e-18),
			 dd(0.26230818925253879259, 8.7473840613060620439e-18),
			 dd(0.32059889797532520328, -1.6445114102151181092e-18),
			 dd(0.41219858311113238836, 1.3745956958819257564e-17));

  vdouble3 t3 = poly4td(x2,
			t2,
			td(0.57707801635558531039, 5.2551030481378853588e-17, 1.6992678371393930323e-33),
			td(0.96179669392597555433, 5.0577616648125906755e-17, -7.7776153307451268974e-34),
			td(2.885390081777926774, 4.0710547481862066222e-17, -2.1229267572708742309e-33));

  x = tdadd_vd3_vd_vd3(vcast_vd_vm(e), tdmul_vd3_vd3_vd3(x, t3));

  tdx r = vcast_tdx_vd3_fast(x);

  r = vsel_tdx_vo64_tdx_tdx(visinf_vo_vd(tdxgetd3x_vd_tdx(d)), d, r);
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vor_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(d)), vlt_vo_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(0))),
						  vcast_vd3_d_d_d(SLEEF_NAN, SLEEF_NAN, SLEEF_NAN), tdxgetd3_vd3_tdx(r)));
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(veq_vo_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(0)),
						  vcast_vd3_d_d_d(-SLEEF_INFINITY, -SLEEF_INFINITY, -SLEEF_INFINITY), tdxgetd3_vd3_tdx(r)));

  return r;
}

static INLINE CONST VECTOR_CC tdx log10_tdx_tdx(tdx d) {
  vmask e = vilogb2k_vm_vd(vmul_vd_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(1/0.75)));
  e = vadd64_vm_vm_vm(e, vadd64_vm_vm_vm(tdxgete_vm_tdx(d), vcast_vm_i_i(-1, -16383)));
  d = tdxsete_tdx_tdx_vm(d, vsub64_vm_vm_vm(tdxgete_vm_tdx(d), e));

  vdouble3 x = vcast_vd3_tdx(d);
  x = tddiv_vd3_vd3_vd3(tdadd2_vd3_vd_vd3(vcast_vd_d(-1), x), tdadd2_vd3_vd_vd3(vcast_vd_d(1), x));

  vdouble3 x2 = tdsqu_vd3_vd3(x);

  vdouble t = poly5d(vd3getx_vd_vd3(x2),
		     0.034780228527814822936,
		     0.024618640686533504319,
		     0.028190304121442043284,
		     0.029939794936408799936,
		     0.032170516619407903136);

  vdouble2 t2 = poly12dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(x2), vd3gety_vd_vd3(x2)),
			 t,
			 dd(0.034743538887401982651, -2.3450624418352980494e-18),
			 dd(0.037764738079930665338, 1.0563536287862622615e-18),
			 dd(0.041361379218350230458, -1.9723370672592275046e-18),
			 dd(0.045715208621555349089, -5.1139468818561952716e-19),
			 dd(0.051093468459204260945, 4.8336280836858013011e-19),
			 dd(0.057905930920433591746, 8.2942670572720828473e-19),
			 dd(0.066814535677423361748, -3.7431853326891197301e-18),
			 dd(0.078962633073318508337, 5.7822033728003665468e-18),
			 dd(0.096509884867389289509, 5.5246620283517822911e-18),
			 dd(0.12408413768664337817, 1.1555150300573091071e-18),
			 dd(0.17371779276130072667, 4.3932786008652476057e-18));

  vdouble3 t3 = poly3td(x2,
			t2,
			td(0.28952965460216789628, -1.1181586075640841342e-17, 4.1402925752337893924e-34),
			td(0.86858896380650363334, 2.1966393004335301455e-17, 7.4337789622442436909e-34));

  x = tdmla_vd3_vd3_vd3_vd3(x, t3, tdmul_vd3_vd3_vd(vcast_vd3_d_d_d(0.30102999566398119802, -2.8037281277851703937e-18, 5.4719484023146385333e-35), vcast_vd_vm(e)));

  tdx r = vcast_tdx_vd3_fast(x);

  r = vsel_tdx_vo64_tdx_tdx(visinf_vo_vd(tdxgetd3x_vd_tdx(d)), d, r);
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vor_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(d)), vlt_vo_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(0))),
						  vcast_vd3_d_d_d(SLEEF_NAN, SLEEF_NAN, SLEEF_NAN), tdxgetd3_vd3_tdx(r)));
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(veq_vo_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(0)),
						  vcast_vd3_d_d_d(-SLEEF_INFINITY, -SLEEF_INFINITY, -SLEEF_INFINITY), tdxgetd3_vd3_tdx(r)));

  return r;
}

static INLINE CONST VECTOR_CC tdx log1p_tdx_tdx(tdx d) {
  vmask cm1 = vcmp_vm_tdx_tdx(d, vcast_tdx_d(-1));
  vopmask fnan = vlt64_vo_vm_vm(cm1, vcast_vm_i_i(0, 0));
  vopmask fminf = vand_vo_vo_vo(veq64_vo_vm_vm(cm1, vcast_vm_i_i(0, 0)), vneq_vo_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(-SLEEF_INFINITY)));

  vopmask o = vlt64_vo_vm_vm(tdxgete_vm_tdx(d), vcast_vm_i_i(0, 16383 + 0x3f0));

  tdx dp1 = add2_tdx_tdx_tdx(d, vcast_tdx_d(1));

  vdouble s = vsel_vd_vo_vd_vd(o, vcast_vd_tdx(dp1), tdxgetd3x_vd_tdx(d));
  vmask e = vilogb2k_vm_vd(vmul_vd_vd_vd(s, vcast_vd_d(1/0.75)));
  e = vsel_vm_vo64_vm_vm(o, e, vadd64_vm_vm_vm(e, vadd64_vm_vm_vm(tdxgete_vm_tdx(d), vcast_vm_i_i(-1, -16383))));

  tdx f = d;
  f = tdxsete_tdx_tdx_vm(f, vsub64_vm_vm_vm(tdxgete_vm_tdx(f), e));

  vdouble3 x = vcast_vd3_tdx(f);

  x = vsel_vd3_vo_vd3_vd3(o,
			  tdadd2_vd3_vd3_vd3(tdadd_vd3_vd_vd3(vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(e)),
							      vcast_vd3_d_d_d(-1, 0, 0)), x), x);

  x = tddiv_vd3_vd3_vd3(tdadd2_vd3_vd_vd3(vsel_vd_vo_vd_vd(o, vcast_vd_d(0), vcast_vd_d(-1)), x),
			tdadd2_vd3_vd_vd3(vsel_vd_vo_vd_vd(o, vcast_vd_d(2), vcast_vd_d(+1)), x));

  vdouble3 x2 = tdsqu_vd3_vd3(x);

  vdouble t = poly6d(vd3getx_vd_vd3(x2),
		     0.077146495191485184306,
		     0.052965311163344339085,
		     0.061053005369706196681,
		     0.064483912078500099652,
		     0.06896718437842412619,
		     0.074074009929408698993);

  vdouble2 t2 = poly12dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(x2), vd3gety_vd_vd3(x2)),
			 t,
			 dd(0.080000001871751574845, -1.9217428179318222729e-18),
			 dd(0.086956521697293065465, 5.2385684246247988452e-18),
			 dd(0.095238095238813463839, -5.9627506176667020273e-19),
			 dd(0.10526315789472741324, -6.3998555142971381419e-18),
			 dd(0.11764705882352950728, -1.5923063236090721578e-18),
			 dd(0.13333333333333333148, 1.1538478835417353313e-18),
			 dd(0.15384615384615385469, -8.5364262380653974925e-18),
			 dd(0.18181818181818182323, -5.0464824142103336146e-18),
			 dd(0.22222222222222220989, 1.2335811419831364972e-17),
			 dd(0.28571428571428569843, 1.5860328923163793786e-17),
			 dd(0.4000000000000000222, -2.22044604925030889e-17));

  vdouble3 t3 = poly3td(x2,
			t2,
			td(0.66666666666666662966, 3.7007434154171882626e-17, 2.0425398205897696253e-33),
			td(2, 0, 0));

  x = tdmla_vd3_vd3_vd3_vd3(x, t3, tdmul_vd3_vd3_vd(vcast_vd3_d_d_d(0.69314718055994528623, 2.3190468138462995584e-17, 5.7077084384162120658e-34), vcast_vd_vm(e)));

  tdx r = vcast_tdx_vd3_fast(x);

  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(fnan, vcast_vd3_d_d_d(SLEEF_NAN, SLEEF_NAN, SLEEF_NAN), tdxgetd3_vd3_tdx(r)));
  r = vsel_tdx_vo64_tdx_tdx(vor_vo_vo_vo(veq_vo_vd_vd(tdxgetd3x_vd_tdx(d), vcast_vd_d(SLEEF_INFINITY)),
					 vlt64_vo_vm_vm(tdxgete_vm_tdx(d), vcast_vm_i_i(0, 16250))), d, r);
  r = tdxsetd3_tdx_tdx_vd3(r, vsel_vd3_vo_vd3_vd3(vandnot_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(d)), fminf), vcast_vd3_d_d_d(-SLEEF_INFINITY, -SLEEF_INFINITY, -SLEEF_INFINITY), tdxgetd3_vd3_tdx(r)));

  return r;
}

static INLINE CONST VECTOR_CC tdx asin_tdx_tdx(tdx a) {
  vdouble3 d = vcast_vd3_tdx(a);
  vopmask o = vle_vo_vd_vd(vabs_vd_vd(vd3getx_vd_vd3(d)), vcast_vd_d(0.5));
  vdouble3 x2 = vsel_vd3_vo_vd3_vd3(o, tdsqu_vd3_vd3(d), tdscale_vd3_vd3_d(tdadd2_vd3_vd_vd3(vcast_vd_d(-1), tdabs_vd3_vd3(d)), -0.5));
  vdouble3 x  = vsel_vd3_vo_vd3_vd3(o, tdabs_vd3_vd3(d), tdsqrt_vd3_vd3(x2));

  vdouble2 u2 = poly27dd_b(vcast_vd2_vd_vd(vd3getx_vd_vd3(x2), vd3gety_vd_vd3(x2)), 
			   dd(0.12093344446090691091, 5.0363120565637591991e-18),
			   dd(-0.36153378269275532331, -1.7556583708421419762e-17),
			   dd(0.55893099015865999046, -2.6907079627246343089e-17),
			   dd(-0.55448141966051567309, -3.7978309370893552801e-17),
			   dd(0.40424204356181753228, 1.5887847216842667733e-17),
			   dd(-0.22032590676598312607, 7.3324786328972556294e-18),
			   dd(0.09993532937851500042, -5.0227770446227564411e-18),
			   dd(-0.032226727408526410767, 2.2387871993717722738e-18),
			   dd(0.012832610577524721993, 6.7972341988853136857e-19),
			   dd(0.00036188912455060616088, -1.6944901896181012601e-20),
			   dd(0.003567016367626023917, -1.1672109598283198892e-19),
			   dd(0.0032090026267793956075, -2.1539544372108677509e-19),
			   dd(0.0035820918102690149989, 1.2353879271988841965e-19),
			   dd(0.0038793667965660583175, -4.155657554516947127e-20),
			   dd(0.004241074869665890576, 2.3022448888497653433e-19),
			   dd(0.0046601285950660783705, -1.7736471990811509808e-20),
			   dd(0.0051533107957558634687, 4.1203961363283285874e-20),
			   dd(0.005740037601071565701, -1.188207215521789741e-19),
			   dd(0.0064472103155285972673, -4.3758757814241262962e-20),
			   dd(0.0073125258734422823176, 1.6205809366671831942e-19),
			   dd(0.0083903358096223089324, -1.8031629602842976318e-19),
			   dd(0.0097616095291939240092, 2.8656907908010536683e-20),
			   dd(0.011551800896139708535, 7.931728093260218342e-19),
			   dd(0.013964843750000000694, -7.5293817685951843173e-19),
			   dd(0.017352764423076923878, -7.9988670374828518542e-19),
			   dd(0.022372159090909091855, -9.4621970154314115315e-19),
			   dd(0.030381944444444444059, 3.8549414809450573771e-19));

  vdouble3 u3 = poly4td(x2,
			u2,
			td(0.044642857142857143848, -9.9127055785953217514e-19, 2.1664954675370400492e-35),
			td(0.074999999999999997224, 2.7755575615631961873e-18, 1.1150555589813349703e-34),
			td(0.16666666666666665741, 9.2518585385429706566e-18, 3.1477646033755622006e-34));

  u3 = tdmla_vd3_vd3_vd3_vd3(u3, tdmul_vd3_vd3_vd3(x, x2), x);

  u3 = vsel_vd3_vo_vd3_vd3(o, u3,
			   tdsub2_vd3_vd3_vd3(vcast_vd3_d_d_d(3.141592653589793116*0.5, 1.2246467991473532072e-16*0.5, -2.9947698097183396659e-33*0.5),
					      tdscale_vd3_vd3_d(u3, 2)));
  u3 = tdmulsign_vd3_vd3_vd(u3, vd3getx_vd_vd3(d));

  tdx t = vsel_tdx_vo64_tdx_tdx(vlt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16383 - 0x300)), a, vcast_tdx_vd3_fast(u3));
  t = tdxsetx_tdx_tdx_vd(t, vsel_vd_vo_vd_vd(visinf_vo_vd(tdxgetd3x_vd_tdx(a)), vcast_vd_d(SLEEF_NAN), tdxgetd3x_vd_tdx(t)));

  return t;
}

static INLINE CONST VECTOR_CC tdx acos_tdx_tdx(tdx a) {
  vdouble3 d = vcast_vd3_tdx(a);
  vopmask o = vle_vo_vd_vd(vabs_vd_vd(vd3getx_vd_vd3(d)), vcast_vd_d(0.5));
  vdouble3 x2 = vsel_vd3_vo_vd3_vd3(o, tdsqu_vd3_vd3(d), tdscale_vd3_vd3_d(tdadd2_vd3_vd_vd3(vcast_vd_d(-1), tdabs_vd3_vd3(d)), -0.5));
  vdouble3 x  = vsel_vd3_vo_vd3_vd3(o, tdabs_vd3_vd3(d), tdsqrt_vd3_vd3(x2));

  vdouble2 u2 = poly27dd_b(vcast_vd2_vd_vd(vd3getx_vd_vd3(x2), vd3gety_vd_vd3(x2)), 
			   dd(0.12093344446090691091, 5.0363120565637591991e-18),
			   dd(-0.36153378269275532331, -1.7556583708421419762e-17),
			   dd(0.55893099015865999046, -2.6907079627246343089e-17),
			   dd(-0.55448141966051567309, -3.7978309370893552801e-17),
			   dd(0.40424204356181753228, 1.5887847216842667733e-17),
			   dd(-0.22032590676598312607, 7.3324786328972556294e-18),
			   dd(0.09993532937851500042, -5.0227770446227564411e-18),
			   dd(-0.032226727408526410767, 2.2387871993717722738e-18),
			   dd(0.012832610577524721993, 6.7972341988853136857e-19),
			   dd(0.00036188912455060616088, -1.6944901896181012601e-20),
			   dd(0.003567016367626023917, -1.1672109598283198892e-19),
			   dd(0.0032090026267793956075, -2.1539544372108677509e-19),
			   dd(0.0035820918102690149989, 1.2353879271988841965e-19),
			   dd(0.0038793667965660583175, -4.155657554516947127e-20),
			   dd(0.004241074869665890576, 2.3022448888497653433e-19),
			   dd(0.0046601285950660783705, -1.7736471990811509808e-20),
			   dd(0.0051533107957558634687, 4.1203961363283285874e-20),
			   dd(0.005740037601071565701, -1.188207215521789741e-19),
			   dd(0.0064472103155285972673, -4.3758757814241262962e-20),
			   dd(0.0073125258734422823176, 1.6205809366671831942e-19),
			   dd(0.0083903358096223089324, -1.8031629602842976318e-19),
			   dd(0.0097616095291939240092, 2.8656907908010536683e-20),
			   dd(0.011551800896139708535, 7.931728093260218342e-19),
			   dd(0.013964843750000000694, -7.5293817685951843173e-19),
			   dd(0.017352764423076923878, -7.9988670374828518542e-19),
			   dd(0.022372159090909091855, -9.4621970154314115315e-19),
			   dd(0.030381944444444444059, 3.8549414809450573771e-19));

  vdouble3 u3 = poly4td(x2,
			u2,
			td(0.044642857142857143848, -9.9127055785953217514e-19, 2.1664954675370400492e-35),
			td(0.074999999999999997224, 2.7755575615631961873e-18, 1.1150555589813349703e-34),
			td(0.16666666666666665741, 9.2518585385429706566e-18, 3.1477646033755622006e-34));

  u3 = tdmul_vd3_vd3_vd3(u3, tdmul_vd3_vd3_vd3(x, x2));

  vdouble3 y = tdsub2_vd3_vd3_vd3(vcast_vd3_d_d_d(3.141592653589793116*0.5, 1.2246467991473532072e-16*0.5, -2.9947698097183396659e-33*0.5),
				  tdmulsign_vd3_vd3_vd(tdadd2_vd3_vd3_vd3(x, u3), vd3getx_vd_vd3(d)));

  x = tdadd2_vd3_vd3_vd3(x, u3);

  vdouble3 r = vsel_vd3_vo_vd3_vd3(o, y, tdscale_vd3_vd3_d(x, 2));

  r = vsel_vd3_vo_vd3_vd3(vandnot_vo_vo_vo(o, vsignbit_vo_vd(vd3getx_vd_vd3(d))),
			  tdsub2_vd3_vd3_vd3(vcast_vd3_d_d_d(3.141592653589793116, 1.2246467991473532072e-16, -2.9947698097183396659e-33), r), r);

  return vcast_tdx_vd3_fast(r);
}

static INLINE CONST VECTOR_CC tdx atan_tdx_tdx(tdx a) {
  vdouble3 s = tdmulsign_vd3_vd3_vd(vcast_vd3_tdx(a), tdxgetd3x_vd_tdx(a));
  vopmask q1 = vgt_vo_vd_vd(vd3getx_vd_vd3(s), vcast_vd_d(1)), q2 = vsignbit_vo_vd(tdxgetd3x_vd_tdx(a));

  vopmask o = vand_vo_vo_vo(vlt64_vo_vm_vm(vcast_vm_i_i(0, 16380), tdxgete_vm_tdx(a)), vlt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16400)));
  vdouble3 r = vsel_vd3_vo_vd3_vd3(vgt_vo_vd_vd(vd3getx_vd_vd3(s), vcast_vd_d(1)), s, vcast_vd3_d_d_d(1, 0, 0));
  vdouble3 t = tdsqrt_vd3_vd3(tdadd2_vd3_vd_vd3(vcast_vd_d(1), tdsqu_vd3_vd3(s)));
  t = tdadd2_vd3_vd3_vd3(vsel_vd3_vo_vd3_vd3(vgt_vo_vd_vd(vd3getx_vd_vd3(s), vcast_vd_d(1)), tdneg_vd3_vd3(s), vcast_vd3_d_d_d(-1, 0, 0)), t);
  s = vsel_vd3_vo_vd3_vd3(vgt_vo_vd_vd(vd3getx_vd_vd3(s), vcast_vd_d(1)), vcast_vd3_d_d_d(1, 0, 0), s);
  t = vsel_vd3_vo_vd3_vd3(o, t, s);
  s = vsel_vd3_vo_vd3_vd3(o, s, r);

  s = tddiv_vd3_vd3_vd3(t, s);
  t = tdsqu_vd3_vd3(s);

  vdouble u = poly4d(vd3getx_vd_vd3(t),
		     0.0023517758707377683057,
		     -0.0078460926062957729588,
		     0.014024369351559842073,
		     -0.018609060689550000617);

  vdouble2 u2 = poly20dd(vcast_vd2_vd_vd(vd3getx_vd_vd3(t), vd3gety_vd_vd3(t)),
			 u,
			 dd(0.021347897644887127433, 1.2358082911909778912e-18),
			 dd(-0.023027057264421869898, 5.5318804440500140026e-19),
			 dd(0.024341787465173968935, 7.9381619255068479527e-19),
			 dd(-0.025632626385681419462, -7.766288910534071802e-19),
			 dd(0.027025826555703309079, 1.680006496143075514e-18),
			 dd(-0.028571286386108368793, 2.6025002105966532517e-19),
			 dd(0.030303016309661177236, -3.06479533039765102e-19),
			 dd(-0.032258063371087788984, 1.1290551848834762045e-19),
			 dd(0.034482758542893850173, 2.5481151890869902948e-18),
			 dd(-0.037037037032663137903, 7.5104308241678790957e-19),
			 dd(0.039999999999797607175, 2.9312705736517077064e-18),
			 dd(-0.043478260869557569523, -1.3752578122860787278e-19),
			 dd(0.047619047619047387421, -1.8989611696449893353e-18),
			 dd(-0.052631578947368418131, 2.7615076871062793522e-18),
			 dd(0.058823529411764705066, 7.0808541076268165848e-19),
			 dd(-0.066666666666666665741, -9.2360964897470583754e-19),
			 dd(0.076923076923076927347, -4.2701055521153751364e-18),
			 dd(-0.090909090909090911614, 2.523234276830114669e-18),
			 dd(0.11111111111111110494, 6.1679056916998595896e-18));

  vdouble3 u3 = poly4td(t,
			u2,
			td(-0.14285714285714284921, -7.9301644616062175363e-18, -5.9177134210390659384e-34),
			td(0.2000000000000000111, -1.1102230246251569102e-17, 4.6352657794908457067e-34),
			td(-0.33333333333333331483, -1.8503717077085941313e-17, -1.025344895736163915e-33));

  t = tdmul_vd3_vd3_vd3(s, tdadd2_vd3_vd_vd3(vcast_vd_d(1), tdmul_vd3_vd3_vd3(t, u3)));

  q1 = vor_vo_vo_vo(q1, vgt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16880)));
  t = vsel_vd3_vo_vd3_vd3(vgt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16880)), vcast_vd3_d_d_d(0, 0, 0), t);

  t = vsel_vd3_vo_vd3_vd3(vand_vo_vo_vo(vlt64_vo_vm_vm(vcast_vm_i_i(0, 16380), tdxgete_vm_tdx(a)), vlt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16400))),
			  tdscale_vd3_vd3_d(t, 2), t);

  t = vsel_vd3_vo_vd3_vd3(visinf_vo_vd(tdxgetd3x_vd_tdx(a)), vcast_vd3_d_d_d(0, 0, 0), t);

  t = vsel_vd3_vo_vd3_vd3(q1, tdsub2_vd3_vd3_vd3(vcast_vd3_d_d_d(3.141592653589793116*0.5, 1.2246467991473532072e-16*0.5, -2.9947698097183396659e-33*0.5), t), t);
  t = vsel_vd3_vo_vd3_vd3(q2, tdneg_vd3_vd3(t), t);

  o = vor_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(a)), veq_vo_vd_vd(tdxgetd3x_vd_tdx(a), vcast_vd_d(0)));
  o = vor_vo_vo_vo(o, vandnot_vo_vo_vo(visnan_vo_vd(tdxgetd3x_vd_tdx(a)), vlt64_vo_vm_vm(tdxgete_vm_tdx(a), vcast_vm_i_i(0, 16383 - 0x300))));
  return vsel_tdx_vo64_tdx_tdx(o, a, vcast_tdx_vd3_fast(t));
}

// Float128 functions ------------------------------------------------------------------------------------------------------------

static CONST VECTOR_CC tdx vcast_tdx_vf128(vmask2 f) {
  vmask re = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2gety_vm_vm2(f), 48), vcast_vm_i_i(0, 0x7fff));

  vmask signbit = vand_vm_vm_vm(vm2gety_vm_vm2(f), vcast_vm_i_i(0x80000000, 0)), mx, my, mz;
  vopmask iszero = viszeroq_vo_vm2(f);

  mx = vand_vm_vm_vm(vm2getx_vm_vm2(vsrl128_vm2_vm2_i(f, 60)), vcast_vm_i_i(0xfffff, 0xffffffff));
  my = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2getx_vm_vm2(f), 8), vcast_vm_i_i(0xfffff, 0xffffffff));
  mz = vand_vm_vm_vm(vsll64_vm_vm_i(vm2getx_vm_vm2(f), 44), vcast_vm_i_i(0xfffff, 0xffffffff));

  mx = vor_vm_vm_vm(mx, vcast_vm_i_i(0x3ff00000, 0));
  my = vor_vm_vm_vm(my, vcast_vm_i_i(0x3cb00000, 0));
  mz = vor_vm_vm_vm(mz, vcast_vm_i_i(0x39700000, 0));

  mx = vandnot_vm_vo64_vm(iszero, mx);
  my = vreinterpret_vm_vd(vsub_vd_vd_vd(vreinterpret_vd_vm(my), vcast_vd_d(1.0 / (UINT64_C(1) << 52))));
  mz = vreinterpret_vm_vd(vsub_vd_vd_vd(vreinterpret_vd_vm(mz), vcast_vd_d(1.0 / ((UINT64_C(1) << 52) * (double)(UINT64_C(1) << 52)))));

  tdx r = tdxsetexyz_tdx_vm_vd_vd_vd(re, 
				     vreinterpret_vd_vm(vor_vm_vm_vm(mx, signbit)), 
				     vreinterpret_vd_vm(vor_vm_vm_vm(my, signbit)), 
				     vreinterpret_vd_vm(vor_vm_vm_vm(mz, signbit)));

  vopmask fisdenorm = vandnot_vo_vo_vo(iszero, veq64_vo_vm_vm(tdxgete_vm_tdx(r), vcast_vm_i_i(0, 0))); // ??

  if (UNLIKELY(!vtestallzeros_i_vo64(vor_vo_vo_vo(veq64_vo_vm_vm(tdxgete_vm_tdx(r), vcast_vm_i_i(0, 0x7fff)),
						  vandnot_vo_vo_vo(iszero, fisdenorm))))) {
    vopmask fisinf = vand_vo_vo_vo(veq64_vo_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(f), vcast_vm_i_i(0x7fffffff, 0xffffffff)),
						  vcast_vm_i_i(0x7fff0000, 0)),
				   veq64_vo_vm_vm(vm2getx_vm_vm2(f), vcast_vm_i_i(0, 0)));
    vopmask fisnan = vandnot_vo_vo_vo(fisinf, veq64_vo_vm_vm(tdxgete_vm_tdx(r), vcast_vm_i_i(0, 0x7fff)));

    tdx g = r;
    g = tdxsetx_tdx_tdx_vd(g, vsub_vd_vd_vd(tdxgetd3x_vd_tdx(g), vmulsign_vd_vd_vd(vcast_vd_d(1), tdxgetd3x_vd_tdx(g))));
    g = tdxsetd3_tdx_tdx_vd3(g, tdnormalize_vd3_vd3(tdxgetd3_vd3_tdx(g)));
    g = tdxsete_tdx_tdx_vm(g, vilogb2k_vm_vd(tdxgetd3x_vd_tdx(g)));
    g = tdxsetd3_tdx_tdx_vd3(g, tdscale_vd3_vd3_vd(tdxgetd3_vd3_tdx(g), vldexp3_vd_vd_vm(vcast_vd_d(1), vneg64_vm_vm(tdxgete_vm_tdx(g)))));
    g = tdxsete_tdx_tdx_vm(g, vadd64_vm_vm_vm(tdxgete_vm_tdx(g), vcast_vm_i_i(0, 1)));
    r = vsel_tdx_vo64_tdx_tdx(fisdenorm, g, r);

    vdouble t = vreinterpret_vd_vm(vor_vm_vm_vm(signbit, vreinterpret_vm_vd(vcast_vd_d(SLEEF_INFINITY))));
    r = tdxsetx_tdx_tdx_vd(r, vsel_vd_vo_vd_vd(fisnan, vcast_vd_d(SLEEF_INFINITY - SLEEF_INFINITY), vsel_vd_vo_vd_vd(fisinf, t, tdxgetd3x_vd_tdx(r))));
    r = tdxsetx_tdx_tdx_vd(r, vreinterpret_vd_vm(vor_vm_vm_vm(signbit, vandnot_vm_vo64_vm(iszero, vreinterpret_vm_vd(tdxgetd3x_vd_tdx(r))))));
  }

  return r;
}

static INLINE CONST VECTOR_CC tdx vcast_tdx_vf128_fast(vmask2 f) {
  vmask re = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2gety_vm_vm2(f), 48), vcast_vm_i_i(0, 0x7fff));

  vmask signbit = vand_vm_vm_vm(vm2gety_vm_vm2(f), vcast_vm_i_i(0x80000000, 0)), mx, my, mz;
  vopmask iszero = viszeroq_vo_vm2(f);

  mx = vand_vm_vm_vm(vm2getx_vm_vm2(vsrl128_vm2_vm2_i(f, 60)), vcast_vm_i_i(0xfffff, 0xffffffff));
  my = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2getx_vm_vm2(f), 8), vcast_vm_i_i(0xfffff, 0xffffffff));
  mz = vand_vm_vm_vm(vsll64_vm_vm_i(vm2getx_vm_vm2(f), 44), vcast_vm_i_i(0xfffff, 0xffffffff));

  mx = vor_vm_vm_vm(mx, vcast_vm_i_i(0x3ff00000, 0));
  my = vor_vm_vm_vm(my, vcast_vm_i_i(0x3cb00000, 0));
  mz = vor_vm_vm_vm(mz, vcast_vm_i_i(0x39700000, 0));

  mx = vandnot_vm_vo64_vm(iszero, mx);
  my = vreinterpret_vm_vd(vsub_vd_vd_vd(vreinterpret_vd_vm(my), vcast_vd_d(1.0 / (UINT64_C(1) << 52))));
  mz = vreinterpret_vm_vd(vsub_vd_vd_vd(vreinterpret_vd_vm(mz), vcast_vd_d(1.0 / ((UINT64_C(1) << 52) * (double)(UINT64_C(1) << 52)))));

  return tdxsetexyz_tdx_vm_vd_vd_vd(re, 
				    vreinterpret_vd_vm(vor_vm_vm_vm(mx, signbit)), 
				    vreinterpret_vd_vm(vor_vm_vm_vm(my, signbit)), 
				    vreinterpret_vd_vm(vor_vm_vm_vm(mz, signbit)));
}

#define HBX 1.0
#define LOGXSCALE 1
#define XSCALE (1 << LOGXSCALE)
#define SX 61
#define HBY (1.0 / (UINT64_C(1) << 53))
#define LOGYSCALE 4
#define YSCALE (1 << LOGYSCALE)
#define SY 11
#define HBZ (1.0 / ((UINT64_C(1) << 53) * (double)(UINT64_C(1) << 53)))
#define LOGZSCALE 10
#define ZSCALE (1 << LOGZSCALE)
#define SZ 36
#define HBR (1.0 / (UINT64_C(1) << 60))

static CONST VECTOR_CC vmask2 vcast_vf128_tdx_slow(tdx f) {
  vmask signbit = vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), vcast_vm_i_i(0x80000000, 0));
  vopmask iszero = veq_vo_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(0.0));

  f = tdxsetxyz_tdx_tdx_vd_vd_vd(f, 
				 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), signbit)),
				 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), signbit)),
				 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3z_vd_tdx(f)), signbit)));

  vopmask denorm = vgt64_vo_vm_vm(vcast_vm_i_i(0, 1), tdxgete_vm_tdx(f)), fisinf = visinf_vo_vd(tdxgetd3x_vd_tdx(f));
  vdouble t = vldexp3_vd_vd_vm(vcast_vd_d(0.5), tdxgete_vm_tdx(f));
  t = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vgt64_vo_vm_vm(vcast_vm_i_i(-1, -120), tdxgete_vm_tdx(f)), vreinterpret_vm_vd(t)));
  t = vsel_vd_vo_vd_vd(denorm, t, vcast_vd_d(1));
  f = tdxsete_tdx_tdx_vm(f, vsel_vm_vo64_vm_vm(denorm, vcast_vm_i_i(0, 1), tdxgete_vm_tdx(f)));

  vopmask o = vlt_vo_vd_vd(tdxgetd3y_vd_tdx(f), vcast_vd_d(-1.0/(UINT64_C(1) << 57)/(UINT64_C(1) << 57)));
  o = vand_vo_vo_vo(o, veq_vo_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(1)));
  o = vandnot_vo_vo_vo(veq64_vo_vm_vm(tdxgete_vm_tdx(f), vcast_vm_i_i(0, 1)), o);
  t = vsel_vd_vo_vd_vd(o, vcast_vd_d(2), t);
  f = tdxsete_tdx_tdx_vm(f, vsub64_vm_vm_vm(tdxgete_vm_tdx(f), vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0, 2), vcast_vm_i_i(0, 1))));

  f = tdxsetxyz_tdx_tdx_vd_vd_vd(f, 
				 vmul_vd_vd_vd(tdxgetd3x_vd_tdx(f), t),
				 vmul_vd_vd_vd(tdxgetd3y_vd_tdx(f), t),
				 vmul_vd_vd_vd(tdxgetd3z_vd_tdx(f), t));

  t = vadd_vd_vd_vd(tdxgetd3y_vd_tdx(f), vcast_vd_d(HBY * (1 << LOGYSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGYSCALE)));
  f = tdxsetz_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3z_vd_tdx(f), vsub_vd_vd_vd(tdxgetd3y_vd_tdx(f), vsub_vd_vd_vd(t, vcast_vd_d(HBZ * (1 << LOGZSCALE) + HBY * (1 << LOGYSCALE))))));
  f = tdxsety_tdx_tdx_vd(f, t);

  vdouble c = vsel_vd_vo_vd_vd(denorm, vcast_vd_d(HBX * XSCALE + HBX), vcast_vd_d(HBX * XSCALE));
  vdouble d = vadd_vd_vd_vd(tdxgetd3x_vd_tdx(f), c);
  d = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGXSCALE)));
  t = vadd_vd_vd_vd(tdxgetd3y_vd_tdx(f), vsub_vd_vd_vd(tdxgetd3x_vd_tdx(f), vsub_vd_vd_vd(d, c)));
  f = tdxsetz_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3z_vd_tdx(f), vadd_vd_vd_vd(vsub_vd_vd_vd(tdxgetd3y_vd_tdx(f), t), vsub_vd_vd_vd(tdxgetd3x_vd_tdx(f), vsub_vd_vd_vd(d, c)))));
  f = tdxsetx_tdx_tdx_vd(f, d);

  d = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGYSCALE)));
  f = tdxsetz_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3z_vd_tdx(f), vsub_vd_vd_vd(t, d)));
  f = tdxsety_tdx_tdx_vd(f, d);

  t = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(tdxgetd3z_vd_tdx(f), vcast_vd_d(HBZ * ZSCALE)),
		       vcast_vd_d(HBZ * (ZSCALE/2)), vcast_vd_d(0));
  f = tdxsety_tdx_tdx_vd(f, vsub_vd_vd_vd(tdxgetd3y_vd_tdx(f), t));
  f = tdxsetz_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3z_vd_tdx(f), t));

  t = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(tdxgetd3y_vd_tdx(f), vcast_vd_d(HBY * YSCALE)),
		       vcast_vd_d(HBY * (YSCALE/2)), vcast_vd_d(0));
  f = tdxsetx_tdx_tdx_vd(f, vsub_vd_vd_vd(tdxgetd3x_vd_tdx(f), t));
  f = tdxsety_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3y_vd_tdx(f), t));

  f = tdxsetz_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3z_vd_tdx(f), vcast_vd_d(HBR)));
  f = tdxsetz_tdx_tdx_vd(f, vsub_vd_vd_vd(tdxgetd3z_vd_tdx(f), vcast_vd_d(HBR)));

  //

  vmask m = vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), vcast_vm_i_i(0xfffff, 0xffffffff));
  vmask2 r = vm2setxy_vm2_vm_vm(vsll64_vm_vm_i(m, SX), vsrl64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), 64-SX));

  f = tdxsetz_tdx_tdx_vd(f, vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3z_vd_tdx(f)), vcast_vm_i_i(0xfffff, 0xffffffff))));
  r = vm2setx_vm2_vm2_vm(r, vor_vm_vm_vm(vm2getx_vm_vm2(r), vsrl64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3z_vd_tdx(f)), SZ)));

  f = tdxsety_tdx_tdx_vd(f, vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), vcast_vm_i_i(0xfffff, 0xffffffff))));
  vmask2 s = vm2setxy_vm2_vm_vm(vsll64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), SY),
				vsrl64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), 64-SY));
  r = vadd128_vm2_vm2_vm2(r, s);

  r = vm2sety_vm2_vm2_vm(r, vand_vm_vm_vm(vm2gety_vm_vm2(r), vsel_vm_vo64_vm_vm(denorm, vcast_vm_i_i(0xffff, 0xffffffff), vcast_vm_i_i(0x3ffff, 0xffffffff))));
  m = vsll64_vm_vm_i(vand_vm_vm_vm(tdxgete_vm_tdx(f), vcast_vm_i_i(0xffffffff, ~(~0UL << 15))), 48);
  r = vm2sety_vm2_vm2_vm(r, vcast_vm_vi2(vadd_vi2_vi2_vi2(vcast_vi2_vm(vm2gety_vm_vm2(r)), vcast_vi2_vm(m))));

  r = vm2sety_vm2_vm2_vm(r, vandnot_vm_vo64_vm(iszero, vm2gety_vm_vm2(r)));
  r = vm2setx_vm2_vm2_vm(r, vandnot_vm_vo64_vm(iszero, vm2getx_vm_vm2(r)));

  o = vor_vo_vo_vo(vgt64_vo_vm_vm(tdxgete_vm_tdx(f), vcast_vm_i_i(0, 32765)), fisinf);
  r = vm2sety_vm2_vm2_vm(r, vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0x7fff0000, 0), vm2gety_vm_vm2(r)));
  r = vm2setx_vm2_vm2_vm(r, vandnot_vm_vo64_vm(o, vm2getx_vm_vm2(r)));

  o = vandnot_vo_vo_vo(fisinf, visnonfinite_vo_vd(tdxgetd3x_vd_tdx(f)));
  r = vm2sety_vm2_vm2_vm(r, vor_vm_vo64_vm(o, vm2gety_vm_vm2(r)));
  r = vm2setx_vm2_vm2_vm(r, vor_vm_vo64_vm(o, vm2getx_vm_vm2(r)));

  r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vandnot_vm_vm_vm(vcast_vm_i_i(0x80000000, 0), vm2gety_vm_vm2(r)), signbit));

  return r;
}

static CONST VECTOR_CC vmask2 vcast_vf128_tdx(tdx f) {
  vopmask o = vor_vo_vo_vo(vgt64_vo_vm_vm(vcast_vm_i_i(0, 2), tdxgete_vm_tdx(f)), vgt64_vo_vm_vm(tdxgete_vm_tdx(f), vcast_vm_i_i(0, 0x7ffd)));
  o = vor_vo_vo_vo(o, visnonfinite_vo_vd(tdxgetd3x_vd_tdx(f)));
  o = vandnot_vo_vo_vo(veq_vo_vd_vd(vcast_vd_d(0), tdxgetd3x_vd_tdx(f)), o);
  if (UNLIKELY(!vtestallzeros_i_vo64(o))) return vcast_vf128_tdx_slow(f);

  vmask signbit = vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), vcast_vm_i_i(0x80000000, 0));
  vopmask iszero = veq_vo_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(0.0));

  f = tdxsetxyz_tdx_tdx_vd_vd_vd(f, 
				 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), signbit)),
				 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), signbit)),
				 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3z_vd_tdx(f)), signbit)));

  o = vand_vo_vo_vo(veq_vo_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(1.0)), vlt_vo_vd_vd(tdxgetd3y_vd_tdx(f), vcast_vd_d(0.0)));
  vint2 i2 = vcast_vi2_vm(vand_vm_vo64_vm(o, vcast_vm_vi2(vcast_vi2_vm(vcast_vm_i_i(1 << 20, 0)))));
  f = tdxsetexyz_tdx_vm_vd_vd_vd(vsub64_vm_vm_vm(tdxgete_vm_tdx(f), vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0, 2), vcast_vm_i_i(0, 1))),
				 vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(tdxgetd3x_vd_tdx(f)), i2)),
				 vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(tdxgetd3y_vd_tdx(f)), i2)),
				 vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(tdxgetd3z_vd_tdx(f)), i2)));
  
  vdouble t = vadd_vd_vd_vd(tdxgetd3y_vd_tdx(f), vcast_vd_d(HBY * (1 << LOGYSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGYSCALE)));
  f = tdxsetz_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3z_vd_tdx(f), vsub_vd_vd_vd(tdxgetd3y_vd_tdx(f), vsub_vd_vd_vd(t, vcast_vd_d(HBZ * (1 << LOGZSCALE) + HBY * (1 << LOGYSCALE))))));
  f = tdxsety_tdx_tdx_vd(f, t);

  t = vadd_vd_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(HBX * (1 << LOGXSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGXSCALE)));
  f = tdxsety_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3y_vd_tdx(f), vsub_vd_vd_vd(tdxgetd3x_vd_tdx(f), vsub_vd_vd_vd(t, vcast_vd_d(HBX * (1 << LOGXSCALE))))));
  f = tdxsetx_tdx_tdx_vd(f, t);

  f = tdxsetz_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3z_vd_tdx(f), vcast_vd_d(HBZ * ((1 << LOGZSCALE)/2) + HBR)));
  f = tdxsetz_tdx_tdx_vd(f, vsub_vd_vd_vd(tdxgetd3z_vd_tdx(f), vcast_vd_d(HBR)));
  f = tdxsety_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3y_vd_tdx(f), vcast_vd_d(HBY * ((1 << LOGYSCALE)/2) - HBZ * ((1 << LOGZSCALE)/2))));
  f = tdxsetx_tdx_tdx_vd(f, vsub_vd_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(HBY * ((1 << LOGYSCALE)/2))));

  //

  f = tdxsetx_tdx_tdx_vd(f, vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), vcast_vm_i_i(0xfffff, 0xffffffff))));
  vmask2 r = vm2setxy_vm2_vm_vm(vsll64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), SX), 
				vsrl64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), 64-SX));

  f = tdxsetz_tdx_tdx_vd(f, vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3z_vd_tdx(f)), vcast_vm_i_i(0xfffff, 0xffffffff))));
  r = vm2setx_vm2_vm2_vm(r, vor_vm_vm_vm(vm2getx_vm_vm2(r), vsrl64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3z_vd_tdx(f)), SZ)));

  f = tdxsety_tdx_tdx_vd(f, vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), vcast_vm_i_i(0xfffff, 0xffffffff))));
  vmask2 s = vm2setxy_vm2_vm_vm(vsll64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), SY), 
				vsrl64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), 64-SY));
  r = vadd128_vm2_vm2_vm2(r, s);

  r = vm2sety_vm2_vm2_vm(r, vand_vm_vm_vm(vm2gety_vm_vm2(r), vcast_vm_i_i(0x3ffff, 0xffffffff)));
  f = tdxsete_tdx_tdx_vm(f, vsll64_vm_vm_i(vand_vm_vm_vm(tdxgete_vm_tdx(f), vcast_vm_i_i(0xffffffff, ~(~0UL << 15))), 48));
  r = vm2sety_vm2_vm2_vm(r, vcast_vm_vi2(vadd_vi2_vi2_vi2(vcast_vi2_vm(vm2gety_vm_vm2(r)), vcast_vi2_vm(tdxgete_vm_tdx(f)))));

  r = vm2sety_vm2_vm2_vm(r, vandnot_vm_vo64_vm(iszero, vm2gety_vm_vm2(r)));
  r = vm2setx_vm2_vm2_vm(r, vandnot_vm_vo64_vm(iszero, vm2getx_vm_vm2(r)));

  r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), signbit));

  return r;
}

static INLINE CONST VECTOR_CC vmask2 vcast_vf128_tdx_fast(tdx f) {
  vmask signbit = vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), vcast_vm_i_i(0x80000000, 0));
  vopmask iszero = veq_vo_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(0.0));

  f = tdxsetxyz_tdx_tdx_vd_vd_vd(f,
				 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), signbit)),
				 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), signbit)),
				 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3z_vd_tdx(f)), signbit)));

  vopmask o = vand_vo_vo_vo(veq_vo_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(1.0)), vlt_vo_vd_vd(tdxgetd3y_vd_tdx(f), vcast_vd_d(0.0)));
  vint2 i2 = vcast_vi2_vm(vand_vm_vo64_vm(o, vcast_vm_vi2(vcast_vi2_vm(vcast_vm_i_i(1 << 20, 0)))));
  f = tdxsetxyz_tdx_tdx_vd_vd_vd(f,
				 vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(tdxgetd3x_vd_tdx(f)), i2)),
				 vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(tdxgetd3y_vd_tdx(f)), i2)),
				 vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(tdxgetd3z_vd_tdx(f)), i2)));
  f = tdxsete_tdx_tdx_vm(f, vsub64_vm_vm_vm(tdxgete_vm_tdx(f), vsel_vm_vo64_vm_vm(o, vcast_vm_i_i(0, 2), vcast_vm_i_i(0, 1))));
  
  vdouble t = vadd_vd_vd_vd(tdxgetd3y_vd_tdx(f), vcast_vd_d(HBY * (1 << LOGYSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGYSCALE)));
  f = tdxsetz_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3z_vd_tdx(f), vsub_vd_vd_vd(tdxgetd3y_vd_tdx(f), vsub_vd_vd_vd(t, vcast_vd_d(HBZ * (1 << LOGZSCALE) + HBY * (1 << LOGYSCALE))))));
  f = tdxsety_tdx_tdx_vd(f, t);

  t = vadd_vd_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(HBX * (1 << LOGXSCALE)));
  t = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(t), vcast_vm_i_i(0xffffffff, 0xffffffff << LOGXSCALE)));
  f = tdxsety_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3y_vd_tdx(f), vsub_vd_vd_vd(tdxgetd3x_vd_tdx(f), vsub_vd_vd_vd(t, vcast_vd_d(HBX * (1 << LOGXSCALE))))));
  f = tdxsetx_tdx_tdx_vd(f, t);

  f = tdxsetz_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3z_vd_tdx(f), vcast_vd_d(HBZ * ((1 << LOGZSCALE)/2) + HBR)));
  f = tdxsetz_tdx_tdx_vd(f, vsub_vd_vd_vd(tdxgetd3z_vd_tdx(f), vcast_vd_d(HBR)));
  f = tdxsety_tdx_tdx_vd(f, vadd_vd_vd_vd(tdxgetd3y_vd_tdx(f), vcast_vd_d(HBY * ((1 << LOGYSCALE)/2) - HBZ * ((1 << LOGZSCALE)/2))));
  f = tdxsetx_tdx_tdx_vd(f, vsub_vd_vd_vd(tdxgetd3x_vd_tdx(f), vcast_vd_d(HBY * ((1 << LOGYSCALE)/2))));

  //

  f = tdxsetx_tdx_tdx_vd(f, vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), vcast_vm_i_i(0xfffff, 0xffffffff))));
  vmask2 r = vm2setxy_vm2_vm_vm(vsll64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), SX), 
				vsrl64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3x_vd_tdx(f)), 64-SX));

  f = tdxsetz_tdx_tdx_vd(f, vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3z_vd_tdx(f)), vcast_vm_i_i(0xfffff, 0xffffffff))));
  r = vm2setx_vm2_vm2_vm(r, vor_vm_vm_vm(vm2getx_vm_vm2(r), vsrl64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3z_vd_tdx(f)), SZ)));

  f = tdxsety_tdx_tdx_vd(f, vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), vcast_vm_i_i(0xfffff, 0xffffffff))));
  vmask2 s = vm2setxy_vm2_vm_vm(vsll64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), SY), 
				vsrl64_vm_vm_i(vreinterpret_vm_vd(tdxgetd3y_vd_tdx(f)), 64-SY));
  r = vadd128_vm2_vm2_vm2(r, s);

  r = vm2sety_vm2_vm2_vm(r, vand_vm_vm_vm(vm2gety_vm_vm2(r), vcast_vm_i_i(0x3ffff, 0xffffffff)));
  f = tdxsete_tdx_tdx_vm(f, vsll64_vm_vm_i(vand_vm_vm_vm(tdxgete_vm_tdx(f), vcast_vm_i_i(0xffffffff, ~(~0UL << 15))), 48));
  r = vm2sety_vm2_vm2_vm(r, vcast_vm_vi2(vadd_vi2_vi2_vi2(vcast_vi2_vm(vm2gety_vm_vm2(r)), vcast_vi2_vm(tdxgete_vm_tdx(f)))));

  r = vm2sety_vm2_vm2_vm(r, vandnot_vm_vo64_vm(iszero, vm2gety_vm_vm2(r)));
  r = vm2setx_vm2_vm2_vm(r, vandnot_vm_vo64_vm(iszero, vm2getx_vm_vm2(r)));

  r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), signbit));

  return r;
}

// Float128 conversion functions

EXPORT CONST VECTOR_CC vargquad xcast_from_doubleq(vdouble d) {
  return vcast_aq_vm2(vcast_vf128_tdx(vcast_tdx_vd(d)));
}

EXPORT CONST VECTOR_CC vdouble xcast_to_doubleq(vargquad q) {
  tdx t = vcast_tdx_vf128(vcast_vm2_aq(q));
  t = tdxsetx_tdx_tdx_vd(t, vadd_vd_vd_vd(vadd_vd_vd_vd(tdxgetd3z_vd_tdx(t), tdxgetd3y_vd_tdx(t)), tdxgetd3x_vd_tdx(t)));
  t = tdxsety_tdx_tdx_vd(t, vcast_vd_d(0));
  t = tdxsetz_tdx_tdx_vd(t, vcast_vd_d(0));
  return vcast_vd_tdx(t);
}

EXPORT CONST VECTOR_CC vint64 xcast_to_int64q(vargquad q) {
  tdx t = vcast_tdx_vf128(vcast_vm2_aq(q));
  vmask e = tdxgete_vm_tdx(t);
  vdouble3 d3 = vcast_vd3_tdx(t);

  vopmask positive = vge_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(0));
  vopmask mp = vand_vo_vo_vo(positive, vgt64_vo_vm_vm(e, vcast_vm_i_i(0, 16383 + 100)));
  mp = vor_vo_vo_vo(mp, vneq_vo_vd_vd(vd3getx_vd_vd3(d3), vd3getx_vd_vd3(d3)));

  vopmask mn = vand_vo_vo_vo(vlt_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(0)),
			     vgt64_vo_vm_vm(e, vcast_vm_i_i(0, 16383 + 100)));

  d3 = tdnormalize_vd3_vd3(d3);

  mp = vor_vo_vo_vo(mp, vand_vo_vo_vo(veq_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(+9223372036854775808.0)),
				      vgt_vo_vd_vd(vd3gety_vd_vd3(d3), vcast_vd_d(-1))));
  mp = vor_vo_vo_vo(mp, vgt_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(+9223372036854775808.0)));

  mn = vor_vo_vo_vo(mn, vand_vo_vo_vo(veq_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(-9223372036854775808.0)),
					      vlt_vo_vd_vd(vd3gety_vd_vd3(d3), vcast_vd_d(0))));
  mn = vor_vo_vo_vo(mn, vlt_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(-9223372036854775808.0)));

  vdouble2 d = vd2setxy_vd2_vd_vd(vd3getx_vd_vd3(d3), vd3gety_vd_vd3(d3));
  vint i0 = vtruncate_vi_vd(vmul_vd_vd_vd(vcast_vd_d(1.0 / (INT64_C(1) << (64 - 28*1))), vd2getx_vd_vd2(d)));
  d = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(vcast_vd_vi(i0), vcast_vd_d(-(double)(INT64_C(1) << (64 - 28*1)))));
  d = ddnormalize_vd2_vd2(d);
  vint i1 = vtruncate_vi_vd(vmul_vd_vd_vd(vcast_vd_d(1.0 / (INT64_C(1) << (64 - 28*2))), vd2getx_vd_vd2(d)));
  d = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(vcast_vd_vi(i1), vcast_vd_d(-(double)(INT64_C(1) << (64 - 28*2)))));
  d = ddnormalize_vd2_vd2(d);
  vint i2 = vtruncate_vi_vd(vd2getx_vd_vd2(d));
  d = ddnormalize_vd2_vd2(ddadd2_vd2_vd2_vd(d, vcast_vd_vi(vneg_vi_vi(i2))));
  vint i3 = vsel_vi_vo_vi_vi(vcast_vo32_vo64(vand_vo_vo_vo   (positive, vlt_vo_vd_vd(vd2getx_vd_vd2(d), vcast_vd_d(0)))),
			     vcast_vi_i(-1), vcast_vi_i(0));
  vint i4 = vsel_vi_vo_vi_vi(vcast_vo32_vo64(vandnot_vo_vo_vo(positive, vgt_vo_vd_vd(vd2getx_vd_vd2(d), vcast_vd_d(0)))),
			     vcast_vi_i(+1), vcast_vi_i(0));

  vmask ti = vsll64_vm_vm_i(vcast_vm_vi(i0), 64 - 28*1);
  ti = vadd64_vm_vm_vm(ti, vsll64_vm_vm_i(vcast_vm_vi(i1), 64 - 28*2));
  ti = vadd64_vm_vm_vm(ti, vcast_vm_vi(vadd_vi_vi_vi(vadd_vi_vi_vi(i2, i3), i4)));

  vmask r = vsel_vm_vo64_vm_vm(mp, vcast_vm_i_i(0x7fffffff, 0xffffffff),
			       vsel_vm_vo64_vm_vm(mn, vcast_vm_i_i(0x80000000, 0), ti));
  return vreinterpret_vi64_vm(r);
}

EXPORT CONST VECTOR_CC vuint64 xcast_to_uint64q(vargquad q) {
  tdx t = vcast_tdx_vf128(vcast_vm2_aq(q));
  vmask e = tdxgete_vm_tdx(t);
  vdouble3 d3 = vcast_vd3_tdx(t);

  vopmask mp = vand_vo_vo_vo(vge_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(0)),
			     vgt64_vo_vm_vm(e, vcast_vm_i_i(0, 16383 + 100)));
  mp = vor_vo_vo_vo(mp, vneq_vo_vd_vd(vd3getx_vd_vd3(d3), vd3getx_vd_vd3(d3)));

  vopmask mn = vand_vo_vo_vo(vlt_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(0)),
			     vgt64_vo_vm_vm(e, vcast_vm_i_i(0, 16383 + 100)));

  d3 = tdnormalize_vd3_vd3(d3);

  mp = vor_vo_vo_vo(mp, vor_vo_vo_vo(vgt_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(+18446744073709551616.0)),
				     vand_vo_vo_vo(veq_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(+18446744073709551616.0)),
						   vgt_vo_vd_vd(vd3gety_vd_vd3(d3), vcast_vd_d(-1)))));

  mn = vor_vo_vo_vo(mn, vlt_vo_vd_vd(vd3getx_vd_vd3(d3), vcast_vd_d(0)));

  vdouble2 d = vd2setxy_vd2_vd_vd(vd3getx_vd_vd3(d3), vd3gety_vd_vd3(d3));
  vint i0 = vtruncate_vi_vd(vmul_vd_vd_vd(vcast_vd_d(1.0 / (INT64_C(1) << (64 - 28*1))), vd2getx_vd_vd2(d)));
  d = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(vcast_vd_vi(i0), vcast_vd_d(-(double)(INT64_C(1) << (64 - 28*1)))));
  d = ddnormalize_vd2_vd2(d);
  vint i1 = vtruncate_vi_vd(vmul_vd_vd_vd(vcast_vd_d(1.0 / (INT64_C(1) << (64 - 28*2))), vd2getx_vd_vd2(d)));
  d = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(vcast_vd_vi(i1), vcast_vd_d(-(double)(INT64_C(1) << (64 - 28*2)))));
  d = ddnormalize_vd2_vd2(d);
  vint i2 = vtruncate_vi_vd(vd2getx_vd_vd2(d));
  d = ddnormalize_vd2_vd2(ddadd2_vd2_vd2_vd(d, vcast_vd_vi(vneg_vi_vi(i2))));
  vint i3 = vsel_vi_vo_vi_vi(vcast_vo32_vo64(vlt_vo_vd_vd(vd2getx_vd_vd2(d), vcast_vd_d(0))),
			     vcast_vi_i(-1), vcast_vi_i(0));

  vmask ti = vsll64_vm_vm_i(vcast_vm_vi(i0), 64 - 28*1);
  ti = vadd64_vm_vm_vm(ti, vsll64_vm_vm_i(vcast_vm_vi(i1), 64 - 28*2));
  ti = vadd64_vm_vm_vm(ti, vcast_vm_vi(vadd_vi_vi_vi(i2, i3)));

  vmask r = vsel_vm_vo64_vm_vm(mp, vcast_vm_i_i(0xffffffff, 0xffffffff),
			       vsel_vm_vo64_vm_vm(mn, vcast_vm_i_i(0, 0), ti));
  return vreinterpret_vu64_vm(r);
}

EXPORT CONST VECTOR_CC vargquad xcast_from_int64q(vint64 ia) {
  vmask a = vreinterpret_vm_vi64(ia);
  a = vadd64_vm_vm_vm(a, vcast_vm_i_i(0x80000000, 0));
  vint h = vcastu_vi_vi2(vcast_vi2_vm(vsrl64_vm_vm_i(a, 8)));
  vint m = vcastu_vi_vi2(vcast_vi2_vm(vand_vm_vm_vm(vsll64_vm_vm_i(a, 12), vcast_vm_i_i(0xfffff, 0))));
  vint l = vcastu_vi_vi2(vcast_vi2_vm(vand_vm_vm_vm(vsll64_vm_vm_i(a, 32), vcast_vm_i_i(0xfffff, 0))));

  vdouble3 d3 = vd3setxyz_vd3_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_vi(h), vcast_vd_d(INT64_C(1) << 40)),
				       vmul_vd_vd_vd(vcast_vd_vi(m), vcast_vd_d(INT64_C(1) << 20)),
				       vcast_vd_vi(l));

  d3 = tdsub2_vd3_vd3_vd3(d3, vd3setxyz_vd3_vd_vd_vd(vcast_vd_d(UINT64_C(1) << 63), vcast_vd_d(0), vcast_vd_d(0)));
  return vcast_aq_vm2(vcast_vf128_tdx(vcast_tdx_vd3(d3)));
}

EXPORT CONST VECTOR_CC vargquad xcast_from_uint64q(vuint64 ia) {
  vmask a = vreinterpret_vm_vu64(ia);
  vint h = vcastu_vi_vi2(vcast_vi2_vm(vsrl64_vm_vm_i(a, 8)));
  vint m = vcastu_vi_vi2(vcast_vi2_vm(vand_vm_vm_vm(vsll64_vm_vm_i(a, 12), vcast_vm_i_i(0xfffff, 0))));
  vint l = vcastu_vi_vi2(vcast_vi2_vm(vand_vm_vm_vm(vsll64_vm_vm_i(a, 32), vcast_vm_i_i(0xfffff, 0))));

  vdouble3 d3 = vd3setxyz_vd3_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_vi(h), vcast_vd_d(INT64_C(1) << 40)),
				       vmul_vd_vd_vd(vcast_vd_vi(m), vcast_vd_d(INT64_C(1) << 20)),
				       vcast_vd_vi(l));

  d3 = tdnormalize_vd3_vd3(d3);
  return vcast_aq_vm2(vcast_vf128_tdx(vcast_tdx_vd3(d3)));
}

// Float128 comparison functions

EXPORT CONST VECTOR_CC vint xicmpltq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vor_vo_vo_vo(vgt64_vo_vm_vm(vm2gety_vm_vm2(cy), vm2gety_vm_vm2(cx)),
				       vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(cy), vm2gety_vm_vm2(cx)), vugt64_vo_vm_vm(vm2getx_vm_vm2(cy), vm2getx_vm_vm2(cx)))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vi;
}

EXPORT CONST VECTOR_CC vint xicmpgtq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vor_vo_vo_vo(vgt64_vo_vm_vm(vm2gety_vm_vm2(cx), vm2gety_vm_vm2(cy)),
				       vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(cx), vm2gety_vm_vm2(cy)), vugt64_vo_vm_vm(vm2getx_vm_vm2(cx), vm2getx_vm_vm2(cy)))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vi;
}

EXPORT CONST VECTOR_CC vint xicmpleq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vor_vo_vo_vo(vgt64_vo_vm_vm(vm2gety_vm_vm2(cy), vm2gety_vm_vm2(cx)),
				       vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(cy), vm2gety_vm_vm2(cx)),
						     vor_vo_vo_vo(vugt64_vo_vm_vm(vm2getx_vm_vm2(cy), vm2getx_vm_vm2(cx)),
								  veq64_vo_vm_vm(vm2getx_vm_vm2(cy), vm2getx_vm_vm2(cx))))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vi;
}

EXPORT CONST VECTOR_CC vint xicmpgeq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vor_vo_vo_vo(vgt64_vo_vm_vm(vm2gety_vm_vm2(cx), vm2gety_vm_vm2(cy)),
				       vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(cx), vm2gety_vm_vm2(cy)),
						     vor_vo_vo_vo(vugt64_vo_vm_vm(vm2getx_vm_vm2(cx), vm2getx_vm_vm2(cy)),
								  veq64_vo_vm_vm(vm2getx_vm_vm2(cx), vm2getx_vm_vm2(cy))))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vi;
}

EXPORT CONST VECTOR_CC vint xicmpeqq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(cy), vm2gety_vm_vm2(cx)), veq64_vo_vm_vm(vm2getx_vm_vm2(cx), vm2getx_vm_vm2(cy))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vi;
}

EXPORT CONST VECTOR_CC vint xicmpneq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);
  vopmask o = visnanq_vo_vm2(x);
  o = vandnot_vo_vo_vo(o, vnot_vo64_vo64(vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(cy), vm2gety_vm_vm2(cx)), veq64_vo_vm_vm(vm2getx_vm_vm2(cx), vm2getx_vm_vm2(cy)))));
  o = vcast_vo32_vo64(vandnot_vo_vo_vo(visnanq_vo_vm2(y), o));
  vint vi = vsel_vi_vo_vi_vi(o, vcast_vi_i(1), vcast_vi_i(0));
  return vi;
}

EXPORT CONST VECTOR_CC vint xicmpq(vargquad ax, vargquad ay) {
  vmask2 x = vcast_vm2_aq(ax), cx = vcmpcnv_vm2_vm2(x);
  vmask2 y = vcast_vm2_aq(ay), cy = vcmpcnv_vm2_vm2(y);

  vopmask oeq = vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(cy), vm2gety_vm_vm2(cx)), veq64_vo_vm_vm(vm2getx_vm_vm2(cx), vm2getx_vm_vm2(cy)));
  oeq = vor_vo_vo_vo(oeq, vor_vo_vo_vo(visnanq_vo_vm2(x), visnanq_vo_vm2(y)));
  vopmask ogt = vor_vo_vo_vo(vgt64_vo_vm_vm(vm2gety_vm_vm2(cx), vm2gety_vm_vm2(cy)),
			     vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(cx), vm2gety_vm_vm2(cy)), vugt64_vo_vm_vm(vm2getx_vm_vm2(cx), vm2getx_vm_vm2(cy))));
  vmask m = vsel_vm_vo64_vm_vm(oeq, vcast_vm_i_i(0, 0), vsel_vm_vo64_vm_vm(ogt, vcast_vm_i_i(0, 1), vcast_vm_i_i(-1, -1)));
  return vcast_vi_vm(m);
}

EXPORT CONST VECTOR_CC vint xiunordq(vargquad ax, vargquad ay) {
  vopmask o = vor_vo_vo_vo(visnanq_vo_vm2(vcast_vm2_aq(ax)), visnanq_vo_vm2(vcast_vm2_aq(ay)));
  vint vi = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vcast_vi_i(1), vcast_vi_i(0));
  return vi;
}

EXPORT CONST VECTOR_CC vargquad xiselectq(vint vi, vargquad a0, vargquad a1) {
  vmask2 m0 = vcast_vm2_aq(a0), m1 = vcast_vm2_aq(a1);
  vopmask o = vcast_vo64_vo32(veq_vo_vi_vi(vi, vcast_vi_i(0)));
  vmask2 m2 = vm2setxy_vm2_vm_vm(
    vsel_vm_vo64_vm_vm(o, vm2getx_vm_vm2(m0), vm2getx_vm_vm2(m1)),
    vsel_vm_vo64_vm_vm(o, vm2gety_vm_vm2(m0), vm2gety_vm_vm2(m1))
  );
  return vcast_aq_vm2(m2);
}

// Float128 arithmetic functions

EXPORT CONST VECTOR_CC vargquad xaddq_u05(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 b = vcast_vm2_aq(ab);

  vmask ea = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2gety_vm_vm2(a), 48), vcast_vm_i_i(0, 0x7fff));
  vmask eb = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2gety_vm_vm2(b), 48), vcast_vm_i_i(0, 0x7fff));

  vopmask oa = vor_vo_vo_vo(viszeroq_vo_vm2(a), 
			    vand_vo_vo_vo(vgt64_vo_vm_vm(ea, vcast_vm_i_i(0, 120)),
					  vgt64_vo_vm_vm(vcast_vm_i_i(0, 0x7ffe), ea)));
  vopmask ob = vor_vo_vo_vo(viszeroq_vo_vm2(b), 
			    vand_vo_vo_vo(vgt64_vo_vm_vm(eb, vcast_vm_i_i(0, 120)),
					  vgt64_vo_vm_vm(vcast_vm_i_i(0, 0x7ffe), eb)));

  if (LIKELY(vtestallones_i_vo64(vand_vo_vo_vo(oa, ob)))) {
    vmask2 r = vcast_vf128_tdx_fast(add2_tdx_tdx_tdx(vcast_tdx_vf128_fast(a), vcast_tdx_vf128_fast(b)));
    r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), vand_vm_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)), vcast_vm_i_i(0x80000000, 0))));
    return vcast_aq_vm2(r);
  }

  vmask2 r = vcast_vf128_tdx(add2_tdx_tdx_tdx(vcast_tdx_vf128(a), vcast_tdx_vf128(b)));

  r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), vand_vm_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)), vcast_vm_i_i(0x80000000, 0))));

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonfiniteq_vo_vm2_vm2(a, b)))) {
    vopmask aisinf = visinfq_vo_vm2(a), bisinf = visinfq_vo_vm2(b);
    vopmask aisnan = visnanq_vo_vm2(a), bisnan = visnanq_vo_vm2(b);
    vopmask o = veq64_vo_vm_vm(vm2gety_vm_vm2(a), vxor_vm_vm_vm(vm2gety_vm_vm2(b), vcast_vm_i_i(0x80000000, 0)));
    o = vand_vo_vo_vo(o, veq64_vo_vm_vm(vm2getx_vm_vm2(a), vm2getx_vm_vm2(b)));
    r = vsel_vm2_vo_vm2_vm2(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(bisnan, aisinf)), a, r);
    r = vsel_vm2_vo_vm2_vm2(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(aisnan, bisinf)), b, r);
  }

  return vcast_aq_vm2(r);
}

EXPORT CONST VECTOR_CC vargquad xsubq_u05(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 b = vcast_vm2_aq(ab);

  vmask ea = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2gety_vm_vm2(a), 48), vcast_vm_i_i(0, 0x7fff));
  vmask eb = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2gety_vm_vm2(b), 48), vcast_vm_i_i(0, 0x7fff));

  vopmask oa = vor_vo_vo_vo(viszeroq_vo_vm2(a), 
			    vand_vo_vo_vo(vgt64_vo_vm_vm(ea, vcast_vm_i_i(0, 120)),
					  vgt64_vo_vm_vm(vcast_vm_i_i(0, 0x7ffe), ea)));
  vopmask ob = vor_vo_vo_vo(viszeroq_vo_vm2(b), 
			    vand_vo_vo_vo(vgt64_vo_vm_vm(eb, vcast_vm_i_i(0, 120)),
					  vgt64_vo_vm_vm(vcast_vm_i_i(0, 0x7ffe), eb)));

  if (LIKELY(vtestallones_i_vo64(vand_vo_vo_vo(oa, ob)))) {
    vmask2 r = vcast_vf128_tdx_fast(sub2_tdx_tdx_tdx(vcast_tdx_vf128_fast(a), vcast_tdx_vf128_fast(b)));
    r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), vand_vm_vm_vm(vandnot_vm_vm_vm(vm2gety_vm_vm2(b), vm2gety_vm_vm2(a)), vcast_vm_i_i(0x80000000, 0))));
    return vcast_aq_vm2(r);
  }

  vmask2 r = vcast_vf128_tdx(sub2_tdx_tdx_tdx(vcast_tdx_vf128(a), vcast_tdx_vf128(b)));

  r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), vand_vm_vm_vm(vandnot_vm_vm_vm(vm2gety_vm_vm2(b), vm2gety_vm_vm2(a)), vcast_vm_i_i(0x80000000, 0))));

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonfiniteq_vo_vm2_vm2(a, b)))) {
    vopmask aisinf = visinfq_vo_vm2(a), bisinf = visinfq_vo_vm2(b);
    vopmask aisnan = visnanq_vo_vm2(a), bisnan = visnanq_vo_vm2(b);
    vopmask o = vand_vo_vo_vo(veq64_vo_vm_vm(vm2getx_vm_vm2(a), vm2getx_vm_vm2(b)), veq64_vo_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)));
    r = vsel_vm2_vo_vm2_vm2(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(bisnan, aisinf)), a, r);
    b = vm2sety_vm2_vm2_vm(b, vxor_vm_vm_vm(vm2gety_vm_vm2(b), vcast_vm_i_i(0x80000000, 0)));
    r = vsel_vm2_vo_vm2_vm2(vandnot_vo_vo_vo(o, vandnot_vo_vo_vo(aisnan, bisinf)), b, r);
  }

  return vcast_aq_vm2(r);
}

EXPORT CONST VECTOR_CC vargquad xmulq_u05(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 b = vcast_vm2_aq(ab);

  vmask ea = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2gety_vm_vm2(a), 48), vcast_vm_i_i(0, 0x7fff));
  vmask eb = vand_vm_vm_vm(vsrl64_vm_vm_i(vm2gety_vm_vm2(b), 48), vcast_vm_i_i(0, 0x7fff));
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
    r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), vand_vm_vm_vm(vxor_vm_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)), vcast_vm_i_i(0x80000000, 0))));
    return vcast_aq_vm2(r);
  }

  vmask2 r = vcast_vf128_tdx(mul2_tdx_tdx_tdx(vcast_tdx_vf128(a), vcast_tdx_vf128(b)));

  r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), vand_vm_vm_vm(vxor_vm_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)), vcast_vm_i_i(0x80000000, 0))));

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonfiniteq_vo_vm2_vm2(a, b)))) {
    vopmask aisinf = visinfq_vo_vm2(a), bisinf = visinfq_vo_vm2(b);
    vopmask aisnan = visnanq_vo_vm2(a), bisnan = visnanq_vo_vm2(b);
    vopmask aiszero = viszeroq_vo_vm2(a), biszero = viszeroq_vo_vm2(b);
    vopmask o = vor_vo_vo_vo(aisinf, bisinf);
    r = vm2sety_vm2_vm2_vm(r, vsel_vm_vo64_vm_vm(o, vor_vm_vm_vm(vcast_vm_i_i(0x7fff0000, 0),
								 vand_vm_vm_vm(vxor_vm_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)), vcast_vm_i_i(0x80000000, 0))), vm2gety_vm_vm2(r)));
    r = vm2setx_vm2_vm2_vm(r, vandnot_vm_vo64_vm(o, vm2getx_vm_vm2(r)));

    o = vor_vo_vo_vo(vand_vo_vo_vo(aiszero, bisinf), vand_vo_vo_vo(biszero, aisinf));
    o = vor_vo_vo_vo(vor_vo_vo_vo(o, aisnan), bisnan);
    r = vm2sety_vm2_vm2_vm(r, vor_vm_vo64_vm(o, vm2gety_vm_vm2(r)));
    r = vm2setx_vm2_vm2_vm(r, vor_vm_vo64_vm(o, vm2getx_vm_vm2(r)));
  }

  return vcast_aq_vm2(r);
}

EXPORT CONST VECTOR_CC vargquad xdivq_u05(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 b = vcast_vm2_aq(ab);
  vmask2 r = vcast_vf128_tdx(div2_tdx_tdx_tdx(vcast_tdx_vf128(a), vcast_tdx_vf128(b)));

  r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), vand_vm_vm_vm(vxor_vm_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)), vcast_vm_i_i(0x80000000, 0))));

  if (UNLIKELY(!vtestallzeros_i_vo64(visnonfiniteq_vo_vm2_vm2_vm2(a, b, r)))) {
    vopmask aisinf = visinfq_vo_vm2(a), bisinf = visinfq_vo_vm2(b);
    vopmask aisnan = visnanq_vo_vm2(a), bisnan = visnanq_vo_vm2(b);
    vopmask aiszero = viszeroq_vo_vm2(a), biszero = viszeroq_vo_vm2(b);
    vmask signbit = vand_vm_vm_vm(vxor_vm_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)), vcast_vm_i_i(0x80000000, 0));

    r = vm2sety_vm2_vm2_vm(r, vsel_vm_vo64_vm_vm(bisinf, signbit, vm2gety_vm_vm2(r)));
    r = vm2setx_vm2_vm2_vm(r, vandnot_vm_vo64_vm(bisinf, vm2getx_vm_vm2(r)));

    vopmask o = vor_vo_vo_vo(aisinf, biszero);
    vmask m = vor_vm_vm_vm(vcast_vm_i_i(0x7fff0000, 0), signbit);
    r = vm2sety_vm2_vm2_vm(r, vsel_vm_vo64_vm_vm(o, m, vm2gety_vm_vm2(r)));
    r = vm2setx_vm2_vm2_vm(r, vandnot_vm_vo64_vm(o, vm2getx_vm_vm2(r)));

    o = vand_vo_vo_vo(aiszero, biszero);
    o = vor_vo_vo_vo(o, vand_vo_vo_vo(aisinf, bisinf));
    o = vor_vo_vo_vo(o, vor_vo_vo_vo(aisnan, bisnan));
    r = vm2sety_vm2_vm2_vm(r, vor_vm_vo64_vm(o, vm2gety_vm_vm2(r)));
    r = vm2setx_vm2_vm2_vm(r, vor_vm_vo64_vm(o, vm2getx_vm_vm2(r)));
  }  

  return vcast_aq_vm2(r);
}

EXPORT CONST VECTOR_CC vargquad xnegq(vargquad aa) {
  vmask2 a = vcast_vm2_aq(aa);
  a = vm2sety_vm2_vm2_vm(a, vxor_vm_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(0x80000000, 0)));
  return vcast_aq_vm2(a);
}

EXPORT CONST VECTOR_CC vargquad xfabsq(vargquad aa) {
  vmask2 a = vcast_vm2_aq(aa);
  a = vm2sety_vm2_vm2_vm(a, vand_vm_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(0x7fffffff, 0xffffffff)));
  return vcast_aq_vm2(a);
}

EXPORT CONST VECTOR_CC vargquad xcopysignq(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa), b = vcast_vm2_aq(ab);
  a = vm2sety_vm2_vm2_vm(a, vor_vm_vm_vm(vand_vm_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(0x7fffffff, 0xffffffff)),
					 vand_vm_vm_vm(vm2gety_vm_vm2(b), vcast_vm_i_i(0x80000000, 0))));
  return vcast_aq_vm2(a);
}

EXPORT CONST VECTOR_CC vargquad xfmaxq(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa), b = vcast_vm2_aq(ab);
  vopmask onana = visnanq_vo_vm2(a), onanb = visnanq_vo_vm2(b);
  a = vcmpcnv_vm2_vm2(a);
  b = vcmpcnv_vm2_vm2(b);

  vopmask ogt = vor_vo_vo_vo(vgt64_vo_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)),
			     vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(a), vm2gety_vm_vm2(b)), vugt64_vo_vm_vm(vm2getx_vm_vm2(a), vm2getx_vm_vm2(b))));

  vmask2 r = vsel_vm2_vo_vm2_vm2(ogt, vcast_vm2_aq(aa), vcast_vm2_aq(ab));
  r = vsel_vm2_vo_vm2_vm2(onana, vcast_vm2_aq(ab), r);
  r = vsel_vm2_vo_vm2_vm2(onanb, vcast_vm2_aq(aa), r);
  return vcast_aq_vm2(r);
}

EXPORT CONST VECTOR_CC vargquad xfminq(vargquad aa, vargquad ab) {
  vmask2 a = vcast_vm2_aq(aa), b = vcast_vm2_aq(ab);
  vopmask onana = visnanq_vo_vm2(vcast_vm2_aq(aa)), onanb = visnanq_vo_vm2(b);
  a = vcmpcnv_vm2_vm2(a);
  b = vcmpcnv_vm2_vm2(b);

  vopmask olt = vor_vo_vo_vo(vgt64_vo_vm_vm(vm2gety_vm_vm2(b), vm2gety_vm_vm2(a)),
			     vand_vo_vo_vo(veq64_vo_vm_vm(vm2gety_vm_vm2(b), vm2gety_vm_vm2(a)), vugt64_vo_vm_vm(vm2getx_vm_vm2(b), vm2getx_vm_vm2(a))));

  vmask2 r = vsel_vm2_vo_vm2_vm2(olt, vcast_vm2_aq(aa), vcast_vm2_aq(ab));
  r = vsel_vm2_vo_vm2_vm2(onana, vcast_vm2_aq(ab), r);
  r = vsel_vm2_vo_vm2_vm2(onanb, vcast_vm2_aq(aa), r);
  return vcast_aq_vm2(r);
}

EXPORT CONST VECTOR_CC vargquad xfdimq_u05(vargquad aa, vargquad ab) {
  tdx a = vcast_tdx_vf128(vcast_vm2_aq(aa)), b = vcast_tdx_vf128(vcast_vm2_aq(ab));
  tdx r = sub2_tdx_tdx_tdx(a, b);
  r = vsel_tdx_vo64_tdx_tdx(vsignbit_vo_tdx(r), vcast_tdx_d(0), r);
  r = vsel_tdx_vo64_tdx_tdx(visnan_vo_tdx(a), a, r);
  r = vsel_tdx_vo64_tdx_tdx(visnan_vo_tdx(b), b, r);
  return vcast_aq_vm2(vcast_vf128_tdx(r));
}

// Float128 math functions

EXPORT CONST VECTOR_CC vargquad xsqrtq_u05(vargquad aa) {
  vmask2 a = vcast_vm2_aq(aa);
  vmask2 r = vcast_vf128_tdx(sqrt_tdx_tdx(vcast_tdx_vf128(a)));

  r = vm2sety_vm2_vm2_vm(r, vor_vm_vm_vm(vm2gety_vm_vm2(r), vand_vm_vm_vm(vm2gety_vm_vm2(a), vcast_vm_i_i(0x80000000, 0))));
  vopmask aispinf = vispinfq_vo_vm2(a);

  r = vm2sety_vm2_vm2_vm(r, vsel_vm_vo64_vm_vm(aispinf, vcast_vm_i_i(0x7fff0000, 0), vm2gety_vm_vm2(r)));
  r = vm2setx_vm2_vm2_vm(r, vandnot_vm_vo64_vm(aispinf, vm2getx_vm_vm2(r)));

  return vcast_aq_vm2(r);
}

EXPORT CONST VECTOR_CC vargquad xsinq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(sin_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xcosq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(cos_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xtanq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(tan_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xexpq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(exp_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xexp2q_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(exp2_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xexp10q_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(exp10_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xexpm1q_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(expm1_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xlogq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(log_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xlog2q_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(log2_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xlog10q_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(log10_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xlog1pq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(log1p_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xasinq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(asin_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xacosq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(acos_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

EXPORT CONST VECTOR_CC vargquad xatanq_u10(vargquad aa) {
  return vcast_aq_vm2(vcast_vf128_tdx(atan_tdx_tdx(vcast_tdx_vf128(vcast_vm2_aq(aa)))));
}

//

#ifndef ENABLE_SVE

#ifndef ENABLE_CUDA
#define EXPORT2 EXPORT
#define CONST2 CONST
#else
#define EXPORT2
#define CONST2
#endif

EXPORT2 CONST2 vargquad xloadq(Sleef_quad *p) {
  vargquad a;
  for(int i=0;i<VECTLENDP;i++) {
    memcpy((char *)&a + (i            ) * sizeof(double), (char *)(p + i)                 , sizeof(double));
    memcpy((char *)&a + (i + VECTLENDP) * sizeof(double), (char *)(p + i) + sizeof(double), sizeof(double));
  }
  return a;
}

EXPORT2 void xstoreq(Sleef_quad *p, vargquad a) {
  for(int i=0;i<VECTLENDP;i++) {
    memcpy((char *)(p + i)                 , (char *)&a + (i            ) * sizeof(double), sizeof(double));
    memcpy((char *)(p + i) + sizeof(double), (char *)&a + (i + VECTLENDP) * sizeof(double), sizeof(double));
  }
}

EXPORT2 CONST2 Sleef_quad xgetq(vargquad a, int index) {
  Sleef_quad q;
  memcpy((char *)&q                 , (char *)&a + (index            ) * sizeof(double), sizeof(double));
  memcpy((char *)&q + sizeof(double), (char *)&a + (index + VECTLENDP) * sizeof(double), sizeof(double));
  return q;
}

EXPORT2 CONST2 vargquad xsetq(vargquad a, int index, Sleef_quad q) {
  memcpy((char *)&a + (index            ) * sizeof(double), (char *)&q                 , sizeof(double));
  memcpy((char *)&a + (index + VECTLENDP) * sizeof(double), (char *)&q + sizeof(double), sizeof(double));
  return a;
}

EXPORT2 CONST2 vargquad xsplatq(Sleef_quad p) {
  vargquad a;
  for(int i=0;i<VECTLENDP;i++) {
    memcpy((char *)&a + (i            ) * sizeof(double), (char *)(&p)                 , sizeof(double));
    memcpy((char *)&a + (i + VECTLENDP) * sizeof(double), (char *)(&p) + sizeof(double), sizeof(double));
  }
  return a;
}
#else
EXPORT CONST VECTOR_CC vargquad xloadq(Sleef_quad *p) {
  double a[VECTLENDP*2];
  for(int i=0;i<VECTLENDP;i++) {
    memcpy((char *)a + (i            ) * sizeof(double), (char *)(p + i)                 , sizeof(double));
    memcpy((char *)a + (i + VECTLENDP) * sizeof(double), (char *)(p + i) + sizeof(double), sizeof(double));
  }

  return vm2setxy_vm2_vm_vm(svld1_s32(ptrue, (int32_t *)&a[0]), svld1_s32(ptrue, (int32_t *)&a[VECTLENDP]));
}

EXPORT void xstoreq(Sleef_quad *p, vargquad m) {
  double a[VECTLENDP*2];
  svst1_s32(ptrue, (int32_t *)&a[0        ], vm2getx_vm_vm2(m));
  svst1_s32(ptrue, (int32_t *)&a[VECTLENDP], vm2gety_vm_vm2(m));

  for(int i=0;i<VECTLENDP;i++) {
    memcpy((char *)(p + i)                 , (char *)a + (i            ) * sizeof(double), sizeof(double));
    memcpy((char *)(p + i) + sizeof(double), (char *)a + (i + VECTLENDP) * sizeof(double), sizeof(double));
  }
}

EXPORT CONST VECTOR_CC Sleef_quad xgetq(vargquad m, int index) {
  double a[VECTLENDP*2];
  svst1_s32(ptrue, (int32_t *)&a[0        ], vm2getx_vm_vm2(m));
  svst1_s32(ptrue, (int32_t *)&a[VECTLENDP], vm2gety_vm_vm2(m));

  Sleef_quad q;
  memcpy((char *)&q                 , (char *)a + (index            ) * sizeof(double), sizeof(double));
  memcpy((char *)&q + sizeof(double), (char *)a + (index + VECTLENDP) * sizeof(double), sizeof(double));
  return q;
}

EXPORT CONST VECTOR_CC vargquad xsetq(vargquad m, int index, Sleef_quad q) {
  double a[VECTLENDP*2];
  svst1_s32(ptrue, (int32_t *)&a[0        ], vm2getx_vm_vm2(m));
  svst1_s32(ptrue, (int32_t *)&a[VECTLENDP], vm2gety_vm_vm2(m));

  memcpy((char *)a + (index            ) * sizeof(double), (char *)&q                 , sizeof(double));
  memcpy((char *)a + (index + VECTLENDP) * sizeof(double), (char *)&q + sizeof(double), sizeof(double));

  return vm2setxy_vm2_vm_vm(svld1_s32(ptrue, (int32_t *)&a[0]), svld1_s32(ptrue, (int32_t *)&a[VECTLENDP]));
}

EXPORT CONST VECTOR_CC vargquad xsplatq(Sleef_quad p) {
  double a[VECTLENDP*2];

  for(int i=0;i<VECTLENDP;i++) {
    memcpy((char *)a + (i            ) * sizeof(double), (char *)(&p)                 , sizeof(double));
    memcpy((char *)a + (i + VECTLENDP) * sizeof(double), (char *)(&p) + sizeof(double), sizeof(double));
  }

  return vm2setxy_vm2_vm_vm(svld1_s32(ptrue, (int32_t *)&a[0]), svld1_s32(ptrue, (int32_t *)&a[VECTLENDP]));
}
#endif

#ifdef ENABLE_PUREC_SCALAR
#if !defined(SLEEF_GENHEADER)
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#endif

static const tdx exp10tab[14] = {
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

static CONST tdx exp10i(int n) {
  int neg = 0;
  if (n < 0) { neg = 1; n = -n; }
  tdx r = vcast_tdx_d(1);
  for(int i=0;i<14;i++)
    if ((n & (1 << i)) != 0) r = mul2_tdx_tdx_tdx(r, exp10tab[i]);
  if (neg) r = div2_tdx_tdx_tdx(vcast_tdx_d(1), r);
  return r;
}

static CONST int ilog10(tdx t) {
  int r = 0, p = 1;
  if ((int)vcmp_vm_tdx_tdx(t, vcast_tdx_d(1)) < 0) {
    t = div2_tdx_tdx_tdx(vcast_tdx_d(1), t);
    p = -1;
  }
  for(int i=12;i>=0;i--) {
    int c = vcmp_vm_tdx_tdx(t, exp10tab[i]);
    if ((p > 0 && c >= 0) || (p < 0 && c > 0)) {
      t = div2_tdx_tdx_tdx(t, exp10tab[i]);
      r |= (1 << i);
    }
  }
  if (p < 0) r++;
  return r * p;
}

static CONST vmask2 xsll128(vmask2 m, int c) {
  if (c == 0) return m;
  if (c >= 128) return imdvm2_vm2_vm_vm(0, 0);

  if (c < 64) {
    return imdvm2_vm2_vm_vm(vsll64_vm_vm_i(vm2getx_vm_vm2(m), c),
		  vor_vm_vm_vm(vsll64_vm_vm_i(vm2gety_vm_vm2(m), c), vsrl64_vm_vm_i(vm2getx_vm_vm2(m), 64 - c)));
  }

  return imdvm2_vm2_vm_vm(0, vsll64_vm_vm_i(vm2getx_vm_vm2(m), c - 64));
}

static CONST vmask2 xsrl128(vmask2 m, int c) {
  if (c == 0) return m;
  if (c >= 128) return imdvm2_vm2_vm_vm(0, 0);

  if (c < 64) {
    return imdvm2_vm2_vm_vm(vor_vm_vm_vm(vsrl64_vm_vm_i(vm2getx_vm_vm2(m), c), vsll64_vm_vm_i(vm2gety_vm_vm2(m), 64 - c)),
			    vsrl64_vm_vm_i(vm2gety_vm_vm2(m), c));
  }

  return imdvm2_vm2_vm_vm(vsrl64_vm_vm_i(vm2gety_vm_vm2(m), c - 64), 0);
}

static int xclz128(vmask2 m) {
  int n = m.y == 0 ? 128 : 64;
  uint64_t u = m.y == 0 ? m.x : m.y, v;
  v = u >> 32; if (v != 0) { n -= 32; u = v; }
  v = u >> 16; if (v != 0) { n -= 16; u = v; }
  v = u >>  8; if (v != 0) { n -=  8; u = v; }
  v = u >>  4; if (v != 0) { n -=  4; u = v; }
  v = u >>  2; if (v != 0) { n -=  2; u = v; }
  v = u >>  1; if (v != 0) return n - 2;
  return n - u;
}

// Float128 string functions

EXPORT vargquad Sleef_strtoq(const char *str, const char **endptr) {
  while(isspace(*str)) str++;
  const char *p = str;

  int positive = 1, bp = 0, e = 0, mf = 0;
  tdx n = vcast_tdx_d(0), d = vcast_tdx_d(1);

  if (*p == '-') {
    positive = 0;
    p++;
  } else if (*p == '+') p++;

  if (tolower(p[0]) == 'n' && tolower(p[1]) == 'a' && tolower(p[2]) == 'n') {
    if (endptr != NULL) *endptr = p+3;
    vmask2 r = { vcast_vm_i_i(-1, -1), vcast_vm_i_i(0x7fffffff, -1) };
    r.y |= ((uint64_t)!positive) << 63;
    return vcast_aq_vm2(r);
  }

  if (tolower(p[0]) == 'i' && tolower(p[1]) == 'n' && tolower(p[2]) == 'f') {
    if (endptr != NULL) *endptr = p+3;
    if (tolower(p[3]) == 'i' && tolower(p[4]) == 'n' && tolower(p[5]) == 'i' &&
	tolower(p[6]) == 't' && tolower(p[7]) == 'y' && endptr != NULL) *endptr = p+8;
    vmask2 r = { vcast_vm_i_i(0, 0), vcast_vm_i_i(0x7fff0000, 0) };
    r.y |= ((uint64_t)!positive) << 63;
    return vcast_aq_vm2(r);
  }

  if (*p == '0' && *(p+1) == 'x') {
    p += 2;
    vmask2 m = { 0, 0 };
    int pp = 0;

    while(*p != '\0') {
      int d = 0;
      if (('0' <= *p && *p <= '9')) {
	d = *p++ - '0';
      } else if ('a' <= *p && *p <= 'f') {
	d = *p++ - 'a' + 10;
      } else if ('A' <= *p && *p <= 'F') {
	d = *p++ - 'A' + 10;
      } else if (*p == '.') {
	if (bp) break;
	bp = 1;
	p++;
	continue;
      } else break;

      mf = 1;
      m = xsll128(m, 4);
      m.x += d;
      pp += bp * 4;
      if (m.y >> 60) break;
    }

    while(('0' <= *p && *p <= '9') || ('a' <= *p && *p <= 'f') || ('A' <= *p && *p <= 'F')) p++;
    if (*p == '.' && !bp) p++;
    while(('0' <= *p && *p <= '9') || ('a' <= *p && *p <= 'f') || ('A' <= *p && *p <= 'F')) p++;

    if (*p == 'p' || *p == 'P') {
      char *q;
      e = strtol(p+1, &q, 10);
      if (p+1 == q || isspace(*(p+1))) {
	e = 0;
      } else {
	p = q;
      }
    }

    int nsl = xclz128(m) - 15;
    e = e - pp - nsl + 0x3fff + 112;

    if (e <= 0 || nsl == 128 - 15) {
      nsl += e - 1;
      e = 0;
    }

    if (nsl >= 0) {
      m = xsll128(m, nsl);
    } else {
      if (-nsl-1 < 64) {
	uint64_t u = m.x + ((1ULL) << (-nsl - 1));
	if (u < m.x) m.y++;
	m.x = u;
      } else {
	m.y += ((1ULL) << (-nsl - 1 - 64));
      }
      m = xsrl128(m, -nsl);
    }

    m.y &= 0x0000ffffffffffff;
    m.y |= (((uint64_t)!positive) << 63) | ((((uint64_t)e) & 0x7fff) << 48);

    if (endptr != NULL) *endptr = p;
    return vcast_aq_vm2(m);
  } 

  while(*p != '\0') {
    if ('0' <= *p && *p <= '9') {
      n = add2_tdx_tdx_tdx(mul2_tdx_tdx_tdx(n, vcast_tdx_d(10)), vcast_tdx_d(*p - '0'));
      if (bp) d = mul2_tdx_tdx_tdx(d, vcast_tdx_d(10));
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
    }

    break;
  }

  if (!mf && !bp) {
    if (endptr != NULL) *endptr = str;
    vmask2 r = { vcast_vm_i_i(0, 0), vcast_vm_i_i(0, 0) };
    return vcast_aq_vm2(r);
  }

  n = div2_tdx_tdx_tdx(n, d);
  if (e > 0) n = mul2_tdx_tdx_tdx(n, exp10i(+e));
  if (e < 0) n = div2_tdx_tdx_tdx(n, exp10i(-e));
  if (!positive) n = vneg_tdx_tdx(n);

  if (endptr != NULL) *endptr = p;

  return vcast_aq_vm2(vcast_vf128_tdx(n));
}

//

#define FLAG_SIGN     (1 << 0)
#define FLAG_BLANK    (1 << 1)
#define FLAG_ALT      (1 << 2)
#define FLAG_LEFT     (1 << 3)
#define FLAG_ZERO     (1 << 4)
#define FLAG_UPPER    (1 << 5)

static int snprintquad(char *buf, size_t bufsize, vargquad argvalue, int typespec, int width, int precision, int flags) {
  if (width > bufsize) width = bufsize;

  union {
    vmask2 q;
    struct { uint64_t l, h; };
  } c128 = { .q = vcast_vm2_aq(argvalue) };

  char *ptr = buf;
  char prefix = 0;
  int length = 0, flag_rtz = 0;

  assert(typespec == 'e' || typespec == 'f' || typespec == 'g');

  if ((c128.h & UINT64_C(0x8000000000000000)) != 0) {
    c128.h ^= UINT64_C(0x8000000000000000);
    prefix = '-';
  } else if (flags & FLAG_SIGN) {
    prefix = '+';
  } else if (flags & FLAG_BLANK) {
    prefix = ' ';
  }

  tdx value = vcast_tdx_vf128(c128.q);

  if (visnanq_vo_vm2(c128.q)) {
    if (prefix) *ptr++ = prefix;
    ptr += snprintf(ptr, buf + bufsize - ptr, (flags & FLAG_UPPER) ? "NAN" : "nan");
    flags &= ~FLAG_ZERO;
  } else if (visinfq_vo_vm2(c128.q)) {
    if (prefix) *ptr++ = prefix;
    ptr += snprintf(ptr, buf + bufsize - ptr, (flags & FLAG_UPPER) ? "INF" : "inf");
    flags &= ~FLAG_ZERO;
  } else {
    if (precision < 0) precision = 6;
    if (precision > bufsize/2 - 10) precision = bufsize/2 - 10;
    if (typespec == 'g' && precision > 0) precision--;

    tdx rounder = mul2_tdx_tdx_tdx(vcast_tdx_d(0.5), exp10i(-precision));

    if (typespec == 'f') value = add2_tdx_tdx_tdx(value, rounder);

    int exp = 0, e2 = 0;
    if (!viszeroq_vo_vm2(c128.q)) {
      exp = ilog10(value);
      value = mul2_tdx_tdx_tdx(value, exp10i(-exp));
    }

    int flag_exp = typespec == 'e';

    if (typespec != 'f') {
      value = add2_tdx_tdx_tdx(value, rounder);
    }

    if ((int)vcmp_vm_tdx_tdx(value, vcast_tdx_d(10.0)) >= 0) {
      value = div2_tdx_tdx_tdx(value, vcast_tdx_d(10));
      exp++;
    }

    if (typespec == 'g') {
      flag_rtz = !(flags & FLAG_ALT);
      if (exp < -4 || exp > precision) {
	typespec = 'e';
      } else {
	precision = precision - exp;
	typespec = 'f';
      }
    }

    if (typespec != 'e') e2 = exp;

    int flag_dp = (precision > 0) | (flags & FLAG_ALT);

    if (prefix) *ptr++ = prefix;

    if (e2 < 0) {
      *ptr++ = '0';
    } else {
      for(;e2>=0;e2--) {
	int digit = (int)vcast_vd_tdx(value);
	if ((int)vcmp_vm_tdx_tdx(value, vcast_tdx_d(digit)) < 0) digit--;
	if (ptr - buf >= bufsize-1) { *ptr = '\0'; return -1; }
	*ptr++ = digit + '0';
	value = mul2_tdx_tdx_tdx(add2_tdx_tdx_tdx(value, vcast_tdx_d(-digit)), vcast_tdx_d(10));
      }
    }

    if (flag_dp) {
      if (ptr - buf >= bufsize-1) { *ptr = '\0'; return -1; }
      *ptr++ = '.';
    }

    for(e2++;e2 < 0 && precision > 0;precision--, e2++) {
      if (ptr - buf >= bufsize-1) { *ptr = '\0'; return -1; }
      *ptr++ = '0';
    }

    while (precision-- > 0) {
      int digit = (int)vcast_vd_tdx(value);
      if ((int)vcmp_vm_tdx_tdx(value, vcast_tdx_d(digit)) < 0) digit--;
      if (ptr - buf >= bufsize-1) { *ptr = '\0'; return -1; }
      *ptr++ = digit + '0';
      value = mul2_tdx_tdx_tdx(add2_tdx_tdx_tdx(value, vcast_tdx_d(-digit)), vcast_tdx_d(10));
    }

    if (flag_rtz && flag_dp) {
      while(ptr[-1] == '0') *(--ptr) = 0;
      assert(ptr > buf);
      if (ptr[-1] == '.') *(--ptr) = 0;
    }

    if (flag_exp || (typespec == 'e' && exp)) {
      if (ptr - buf >= bufsize-8) { *ptr = '\0'; return -1; }
      *ptr++ = (flags & FLAG_UPPER) ? 'E' : 'e';
      if (exp < 0){
	*ptr++ = '-'; exp = -exp;
      } else {
	*ptr++ = '+';
      }
      if (exp >= 1000) {
	*ptr++ = exp / 1000 + '0';
	exp %= 1000;
	*ptr++ = exp / 100 + '0';
	exp %= 100;
      } else if (exp >= 100) {
	*ptr++ = exp / 100 + '0';
	exp %= 100;
      }
      *ptr++ = exp / 10 + '0';
      *ptr++ = exp % 10 + '0';
    }
  }

  *ptr = 0;

  length = ptr - buf;
  ptr = buf;

  if (!(flags & FLAG_LEFT) && length < width) {
    int i;
    int nPad = width - length;
    for(i=width; i>=nPad; i--) {
      ptr[i] = ptr[i-nPad];
    }
    i = prefix != 0 && (flags & FLAG_ZERO);
    while (nPad--) {
      ptr[i++] = (flags & FLAG_ZERO) ? '0' : ' ';
    }
    length = width;
  }

  if (flags & FLAG_LEFT) {
    while (length < width) {
      ptr[length++] = ' ';
    }
    ptr[length] = '\0';
  }

  return length;
}

static int snprintquadhex(char *buf, size_t bufsize, vargquad argvalue, int width, int precision, int flags) {
  if (width > bufsize) width = bufsize;
  char *bufend = buf + bufsize, *ptr = buf;

  union {
    vmask2 q;
    struct { uint64_t l, h; };
  } c128 = { .q = vcast_vm2_aq(argvalue) };

  int mainpos = 0;

  if (c128.h >> 63) {
    ptr += snprintf(ptr, bufend - ptr, "-");
  } else if (flags & FLAG_SIGN) {
    ptr += snprintf(ptr, bufend - ptr, "+");
  } else if (flags & FLAG_BLANK) {
    ptr += snprintf(ptr, bufend - ptr, " ");
  }

  if (visnanq_vo_vm2(c128.q)) {
    ptr += snprintf(ptr, bufend - ptr, (flags & FLAG_UPPER) ? "NAN" : "nan");
    flags &= ~FLAG_ZERO;
  } else if (visinfq_vo_vm2(c128.q)) {
    ptr += snprintf(ptr, bufend - ptr, (flags & FLAG_UPPER) ? "INF" : "inf");
    flags &= ~FLAG_ZERO;
  } else {
    int iszero = viszeroq_vo_vm2(c128.q);
    if (precision >= 0 && precision < 28) {
      int s = (28 - precision) * 4 - 1;
      if (s < 64) {
	uint64_t u = c128.l + (((uint64_t)1) << s);
	if (u < c128.l) c128.h++;
	c128.l = u;
      } else {
	c128.h += ((uint64_t)1) << (s - 64);
      }
    }

    int exp = (c128.h >> 48) & 0x7fff;
    uint64_t h = c128.h << 16, l = c128.l;

    ptr += snprintf(ptr, bufend - ptr, (flags & FLAG_UPPER) ? "0X" :"0x");
    mainpos = ptr - buf;
    ptr += snprintf(ptr, bufend - ptr, exp != 0 ? "1" : "0");

    if ((!(h == 0 && l == 0 && precision < 0) && precision != 0) || (flags & FLAG_ALT)) ptr += snprintf(ptr, bufend - ptr, ".");

    const char *digits = (flags & FLAG_UPPER) ? "0123456789ABCDEF" : "0123456789abcdef";

    int niter = (precision < 0 || precision > 28) ? 28 : precision;

    for(int i=0;i<12 && i < niter;i++) {
      if (h == 0 && l == 0 && precision < 0) break;
      ptr += snprintf(ptr, bufend - ptr, "%c", digits[(h >> 60) & 0xf]);
      h <<= 4;
    }

    for(int i=0;i<16 && i < niter-12;i++) {
      if (l == 0 && precision < 0) break;
      ptr += snprintf(ptr, bufend - ptr, "%c", digits[(l >> 60) & 0xf]);
      l <<= 4;
    }

    if (exp == 0) exp++;
    if (iszero) exp = 0x3fff;

    ptr += snprintf(ptr, bufend - ptr, "%c%+d", (flags & FLAG_UPPER) ? 'P' : 'p', exp - 0x3fff);
  }

  int length = ptr - buf;
  ptr = buf;

  if (!(flags & FLAG_ZERO)) mainpos = 0;

  if (!(flags & FLAG_LEFT) && length < width) {
    int i;
    int nPad = width - length;
    for(i=width;i-nPad>=mainpos; i--) {
      ptr[i] = ptr[i-nPad];
    }
    i = mainpos;
    while (nPad--) {
      ptr[i++] = (flags & FLAG_ZERO) ? '0' : ' ';
    }
    length = width;
  }

  if (flags & FLAG_LEFT) {
    while (length < width) {
      ptr[length++] = ' ';
    }
    ptr[length] = '\0';
  }

  return length;
}

#define XBUFSIZE 5000

static int xvprintf(size_t (*consumer)(const char *ptr, size_t size, void *arg), void *arg, const char *fmt, va_list ap) {
  char *xbuf = malloc(XBUFSIZE+10);

  int outlen = 0, errorflag = 0;

  while(*fmt != '\0' && !errorflag) {
    // Copy the format string until a '%' is read

    if (*fmt != '%') {
      do {
	outlen += (*consumer)(fmt++, 1, arg);
      } while(*fmt != '%' && *fmt != '\0');

      if (*fmt == '\0') break;
    }

    const char *subfmtstart = fmt;

    if ((*++fmt) == '\0') {
      errorflag = 1;
      outlen += (*consumer)("%", 1, arg);
      break;
    }

    // Read flags

    int flags = 0, done = 0;
    do {
      switch(*fmt) {
        case '-': flags |= FLAG_LEFT;  break;
        case '+': flags |= FLAG_SIGN;  break;
        case ' ': flags |= FLAG_BLANK; break;
        case '#': flags |= FLAG_ALT;   break;
        case '0': flags |= FLAG_ZERO;  break;
        default:  done = 1; break;
      }
    } while(!done && (*++fmt) != 0);

    // Read width

    int width = 0;

    if (*fmt == '*') {
      width = va_arg(ap, int);
      if (width < 0) {
	flags |= FLAG_LEFT;
        width = -width;
      }
      fmt++;
    } else {
      while(*fmt >= '0' && *fmt <= '9') {
        width = width*10 + *fmt - '0';
        fmt++;
      }
    }

    // Read precision

    int precision = -1;

    if (*fmt == '.') {
      precision = 0;
      fmt++;
      if (*fmt == '*') {
        precision = va_arg(ap, int);
        if (precision < 0) precision = -precision;
        fmt++;
      } else {
        while(*fmt >= '0' && *fmt <= '9') {
          precision = precision*10 + *fmt - '0';
          fmt++;
        }
      }
    }

    // Read size prefix

    int flag_quad = 0, flag_ptrquad = 0;
    int size_prefix = 0, subfmt_processed = 0;

    if (*fmt == 'Q') {
      flag_quad = 1;
      fmt++;
    } else if (*fmt == 'P') {
      flag_ptrquad = 1;
      fmt++;
    } else {
      int pl = 0;
      if (*fmt == 'h' || *fmt == 'l' || *fmt == 'j' || *fmt == 'z' || *fmt == 't' || *fmt == 'L') {
	size_prefix = *fmt;
	pl = 1;
      }
      if ((*fmt == 'h' && *(fmt+1) == 'h') || (*fmt == 'l' && *(fmt+1) == 'l')) {
	size_prefix = *fmt + 256 * *(fmt+1);
	pl = 2;
      }
      fmt += pl;
    }

    // Call type-specific function

    va_list ap2;
    va_copy(ap2, ap);

    switch(*fmt) {
    case 'E': case 'F': case 'G': case 'A':
      flags |= FLAG_UPPER;
      // fall through
    case 'e': case 'f': case 'g': case 'a':
      {
	vargquad value;
	if (flag_quad || flag_ptrquad) {
	  if (flag_quad) {
	    value = va_arg(ap, vargquad);
	  } else {
	    value = *(vargquad *)va_arg(ap, vargquad *);
	  }
	  if (tolower(*fmt) == 'a') {
	    snprintquadhex(xbuf, XBUFSIZE, value, width, precision, flags);
	  } else {
	    snprintquad(xbuf, XBUFSIZE, value, tolower(*fmt), width, precision, flags);
	  }
	  outlen += (*consumer)(xbuf, strlen(xbuf), arg);
	  subfmt_processed = 1;
	  break;
	}

	if (size_prefix == 0) { // double and long double
	  va_arg(ap, double);
	} else if (size_prefix == 'L') {
	  va_arg(ap, long double);
	} else errorflag = 1;
      }
      break;

    case 'd': case 'i': case 'u': case 'o': case 'x': case 'X':
      {
	switch(size_prefix) {
	case 0: case 'h': case 'h' + 256*'h': va_arg(ap, int); break;
	case 'l': va_arg(ap, long int); break;
	case 'j': va_arg(ap, intmax_t); break;
	case 'z': va_arg(ap, size_t); break;
	case 't': va_arg(ap, ptrdiff_t); break;
	case 'l' + 256*'l': va_arg(ap, long long int); break;
	default: errorflag = 1; break;
	}
      }
      break;

    case 'c':
      if (size_prefix == 0) {
	va_arg(ap, int);
      } else
#if 0
	if (size_prefix == 'l') {
	  va_arg(ap, wint_t); // wint_t is not defined
	} else
#endif
	  errorflag = 1;
      break;

    case 's':
      if (size_prefix == 0 || size_prefix == 'l') {
	va_arg(ap, void *);
      } else errorflag = 1;
      break;

    case 'p': case 'n':
      {
	if ((*fmt == 'p' && size_prefix != 0) || size_prefix == 'L') { errorflag = 1; break; }
	va_arg(ap, void *);
      }
      break;

    default:
      errorflag = 1;
    }

    if (!subfmt_processed) {
      char *subfmt = malloc(fmt - subfmtstart + 2);
      memcpy(subfmt, subfmtstart, fmt - subfmtstart + 1);
      subfmt[fmt - subfmtstart + 1] = 0;
      int ret = vsnprintf(xbuf, XBUFSIZE, subfmt, ap2);
      free(subfmt);
      if (ret < 0) { errorflag = 1; break; }
      outlen += (*consumer)(xbuf, strlen(xbuf), arg);
    }

    fmt++;
  }

  free(xbuf);

  return errorflag ? -1 : outlen;
}

//

typedef struct { FILE *fp; } stream_consumer_t;

static size_t stream_consumer(const char *ptr, size_t size, void *varg) {
  stream_consumer_t *arg = (stream_consumer_t *)varg;
  return fwrite(ptr, size, 1, arg->fp);
}

EXPORT int Sleef_vfprintf(FILE *fp, const char *fmt, va_list ap) {
  stream_consumer_t arg = { fp };
  return xvprintf(stream_consumer, &arg, fmt, ap);
}

EXPORT int Sleef_vprintf(const char *fmt, va_list ap) { return Sleef_vfprintf(stdout, fmt, ap); }

EXPORT int Sleef_fprintf(FILE *fp, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = Sleef_vfprintf(fp, fmt, ap);
  va_end(ap);
  return ret;
}

EXPORT int Sleef_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = Sleef_vfprintf(stdout, fmt, ap);
  va_end(ap);
  return ret;
}

typedef struct {
  char *buf;
  size_t pos, size;
} buf_consumer_t;

static size_t buf_consumer(const char *ptr, size_t size, void *varg) {
  buf_consumer_t *arg = (buf_consumer_t *)varg;

  size_t p = 0;
  while(p < size) {
    if (arg->pos >= arg->size - 1) break;
    arg->buf[arg->pos++] = ptr[p++];
  }

  arg->buf[arg->pos] = '\0';

  return p;
}

EXPORT int Sleef_vsnprintf(char *str, size_t size, const char *fmt, va_list ap) {
  buf_consumer_t arg = { str, 0, size };
  return xvprintf(buf_consumer, &arg, fmt, ap);
}

EXPORT int Sleef_snprintf(char *str, size_t size, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = Sleef_vsnprintf(str, size, fmt, ap);
  va_end(ap);
  return ret;
}

#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 13)
#include <printf.h>

static int pa_quad = -1, printf_Qmodifier = -1, printf_Pmodifier = -1;

static void Sleef_quad_va(void *ptr, va_list *ap) { 
  *(Sleef_quad *)ptr = va_arg(*ap, Sleef_quad);
}

static int printf_arginfo(const struct printf_info *info, size_t n, int *argtypes, int *s) {
  if (info->user & printf_Qmodifier) {
    argtypes[0] = pa_quad;
    return 1;
  } else if (info->user & printf_Pmodifier) {
    argtypes[0] = PA_FLAG_PTR | pa_quad;
    return 1;
  }
  return -1;
}

static int printf_output(FILE *fp, const struct printf_info *info, const void *const *args) {
  if (!(info->user & printf_Qmodifier || info->user & printf_Pmodifier)) return -2;

  int flags = 0;
  if (info->showsign)      flags |= FLAG_SIGN;
  if (info->space)         flags |= FLAG_BLANK;
  if (info->alt)           flags |= FLAG_ALT;
  if (info->left)          flags |= FLAG_LEFT;
  if (isupper(info->spec)) flags |= FLAG_UPPER;

  vargquad q = **(const vargquad **)args[0];

  char *xbuf = malloc(XBUFSIZE+10);

  int len = 0;
  if (tolower(info->spec) == 'a') {
    len = snprintquadhex(xbuf, XBUFSIZE, q, info->width, info->prec, flags);
  } else {
    len = snprintquad(xbuf, XBUFSIZE, q, tolower(info->spec), info->width, info->prec, flags);
  }

  size_t wlen = fwrite(xbuf, len, 1, fp);

  free(xbuf);

  return (int)wlen;
}

EXPORT int Sleef_registerPrintfHook() {
  printf_Qmodifier = register_printf_modifier(L"Q");
  printf_Pmodifier = register_printf_modifier(L"P");
  if (printf_Qmodifier == -1 || printf_Pmodifier == -1) return -1;

  pa_quad = register_printf_type(Sleef_quad_va);
  if (pa_quad == -1) return -2;

  if (register_printf_specifier('a', printf_output, printf_arginfo)) return -3;
  if (register_printf_specifier('e', printf_output, printf_arginfo)) return -4;
  if (register_printf_specifier('f', printf_output, printf_arginfo)) return -5;
  if (register_printf_specifier('g', printf_output, printf_arginfo)) return -6;
  if (register_printf_specifier('A', printf_output, printf_arginfo)) return -7;
  if (register_printf_specifier('E', printf_output, printf_arginfo)) return -8;
  if (register_printf_specifier('F', printf_output, printf_arginfo)) return -9;
  if (register_printf_specifier('G', printf_output, printf_arginfo)) return -10;

  return 0;
}

EXPORT void Sleef_unregisterPrintfHook() {
  register_printf_specifier('a', NULL, NULL);
  register_printf_specifier('e', NULL, NULL);
  register_printf_specifier('f', NULL, NULL);
  register_printf_specifier('g', NULL, NULL);
  register_printf_specifier('A', NULL, NULL);
  register_printf_specifier('E', NULL, NULL);
  register_printf_specifier('F', NULL, NULL);
  register_printf_specifier('G', NULL, NULL);
}
#endif // #ifdef __GLIBC__

#endif // #ifdef ENABLE_PUREC_SCALAR

// Functions for debugging ------------------------------------------------------------------------------------------------------------

#ifdef ENABLE_MAIN
// gcc -DENABLE_MAIN -DENABLEFLOAT128 -Wno-attributes -I../libm -I../quad-tester -I../common -I../arch -DUSEMPFR -DENABLE_AVX2 -mavx2 -mfma sleefsimdqp.c rempitabqp.c ../common/common.c ../quad-tester/qtesterutil.c -lm -lmpfr
// gcc-10.2.0 -DENABLE_MAIN -Wno-attributes -I../libm -I../quad-tester -I../common -I../arch -DUSEMPFR -DENABLE_SVE -march=armv8-a+sve sleefsimdqp.c rempitabqp.c ../common/common.c ../quad-tester/qtesterutil.c -lm -lmpfr
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include <time.h>
#include <unistd.h>
#include <locale.h>
#include <wchar.h>

#include "qtesterutil.h"

#if 0
int main(int argc, char **argv) {
  Sleef_quad q0 = atof(argv[1]);

  int lane = 0;
  vargquad a0;
  memset(&a0, 0, sizeof(vargquad));
  a0 = xsetq(a0, lane, q0);

  tdx t = vcast_tdx_vf128(vcast_vm2_aq(a0));

  printvdouble("t.d3.x", t.d3.x);
}
#endif

#if 1
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
#if 0
  memrand(&a0, sizeof(vargquad));
#elif 0
  memset(&a0, 0, sizeof(vargquad));
#endif
  a0 = xsetq(a0, lane, q0);

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
#if 0
  memrand(&a1, sizeof(vargquad));
#elif 0
  memset(&a1, 0, sizeof(vargquad));
#endif
  a1 = xsetq(a1, lane, q1);

#if 0
  memrand(ad, sizeof(ad));
  ad[lane] = mpfr_get_d(fr1, GMP_RNDN);
  a1 = xcast_from_doubleq(vloadu_vd_p(ad));
#endif

  //

  vargquad a2;

#if 0
  a2 = xaddq_u05(a0, a1);
  mpfr_add(fr2, fr0, fr1, GMP_RNDN);

  printf("\nadd\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 1
  a2 = xsubq_u05(a0, a1);
  mpfr_sub(fr2, fr0, fr1, GMP_RNDN);

  printf("\nsub\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 0
  a2 = xmulq_u05(a0, a1);
  mpfr_mul(fr2, fr0, fr1, GMP_RNDN);

  printf("\nmul\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 0
  a2 = xdivq_u05(a0, a1);
  mpfr_div(fr2, fr0, fr1, GMP_RNDN);

  printf("\ndiv\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 0
  a2 = xsqrtq_u05(a0);
  mpfr_sqrt(fr2, fr0, GMP_RNDN);

  printf("\nsqrt\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 0
  a2 = xsinq_u10(a0);
  mpfr_sin(fr2, fr0, GMP_RNDN);

  printf("\nsin\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 1
  a2 = xfabsq(a0);
  mpfr_abs(fr2, fr0, GMP_RNDN);

  printf("\nfabs\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 1
  a2 = xcopysignq(a0, a1);
  mpfr_copysign(fr2, fr0, fr1, GMP_RNDN);

  printf("\ncopysign\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 1
  a2 = xfmaxq(a0, a1);
  mpfr_max(fr2, fr0, fr1, GMP_RNDN);

  printf("\nmax\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 1
  a2 = xfminq(a0, a1);
  mpfr_min(fr2, fr0, fr1, GMP_RNDN);

  printf("\nmin\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif

#if 1
  a2 = xfdimq_u05(a0, a1);
  mpfr_dim(fr2, fr0, fr1, GMP_RNDN);

  printf("\nfdim\n");
  mpfr_set_f128(fr2, mpfr_get_f128(fr2, GMP_RNDN), GMP_RNDN);
  printf("corr : %s\n", sprintfr(fr2));
  mpfr_set_f128(fr2, xgetq(a2, lane), GMP_RNDN);
  printf("test : %s\n", sprintfr(fr2));
#endif
}
#endif
#endif
