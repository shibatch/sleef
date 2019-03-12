//          Copyright Naoki Shibata 2010 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if CONFIG == 1 || CONFIG == 2

#ifndef __VSX__
#error Please specify -mvsx.
#endif

#else
#error CONFIG macro invalid or not defined
#endif

#define ENABLE_DP
#define LOG2VECTLENDP 1
#define VECTLENDP (1 << LOG2VECTLENDP)

#define ENABLE_SP
#define LOG2VECTLENSP (LOG2VECTLENDP+1)
#define VECTLENSP (1 << LOG2VECTLENSP)

#if CONFIG == 1
#define ENABLE_FMA_DP
#define ENABLE_FMA_SP
//#define SPLIT_KERNEL // Benchmark comparison is needed to determine whether this option should be enabled.
#endif

#define ACCURATE_SQRT
#define FULL_FP_ROUNDING

#include <altivec.h>

#include <stdint.h>
#include "misc.h"

typedef vector unsigned int vmask;
typedef vector unsigned int vopmask;

typedef vector double vdouble;
typedef vector int vint;

typedef vector float  vfloat;
typedef vector int vint2;

//

static INLINE int vavailability_i(int name) { return 3; }

#define ISANAME "VSX"
#define DFTPRIORITY 25

static INLINE void vprefetch_v_p(const void *ptr) { }

static vint2 vloadu_vi2_p(int32_t *p) { return vec_ld(0, p); }
static void vstoreu_v_p_vi2(int32_t *p, vint2 v) { vec_st(v, 0, p); }
static vint vloadu_vi_p(int32_t *p) { return vec_ld(0, p); }
static void vstoreu_v_p_vi(int32_t *p, vint v) { vec_st(v, 0, p); }

static INLINE vdouble vload_vd_p(const double *ptr) { return (vector double)vec_ld(0, (const int *)ptr); }
static INLINE void vstore_v_p_vd(double *ptr, vdouble v) { vec_st((vector int)v, 0, (int *)ptr); }
static INLINE vdouble vloadu_vd_p(const double *ptr) { return (vector double) ( ptr[0], ptr[1] ); }
static INLINE void vstoreu_v_p_vd(double *ptr, vdouble v) { ptr[0] = v[0]; ptr[1] = v[1]; }

static INLINE vfloat vload_vf_p(const float *ptr) { return (vector float)vec_ld(0, (const int *)ptr); }
static INLINE void vstore_v_p_vf(float *ptr, vfloat v) { vec_st((vector int)v, 0, (int *)ptr); }
static INLINE void vscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) {
  *(ptr+(offset + step * 0)*2 + 0) = v[0];
  *(ptr+(offset + step * 0)*2 + 1) = v[1];
  *(ptr+(offset + step * 1)*2 + 0) = v[2];
  *(ptr+(offset + step * 1)*2 + 1) = v[3];
}

static INLINE vfloat vloadu_vf_p(const float *ptr) { return (vfloat) ( ptr[0], ptr[1], ptr[2], ptr[3] ); }
static INLINE void vstoreu_v_p_vf(float *ptr, vfloat v) { ptr[0] = v[0]; ptr[1] = v[1]; ptr[2] = v[2]; ptr[3] = v[3]; }

static INLINE void vscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) { vstore_v_p_vd((double *)(&ptr[2*offset]), v); }

static INLINE vdouble vgather_vd_p_vi(const double *ptr, vint vi) {
  int a[VECTLENDP];
  vstoreu_v_p_vi(a, vi);
  return ((vdouble) { ptr[a[0]], ptr[a[1]] });
}

static INLINE vfloat vgather_vf_p_vi2(const float *ptr, vint2 vi2) {
  int a[VECTLENSP];
  vstoreu_v_p_vi2(a, vi2);
  return ((vfloat) { ptr[a[0]], ptr[a[1]], ptr[a[2]], ptr[a[3]] });
}

static INLINE vint vcast_vi_i(int i) { return (vint) { i, i }; }
static INLINE vint2 vcast_vi2_i(int i) { return (vint2) { i, i, i, i }; }
static INLINE vfloat vcast_vf_f(float f) { return (vfloat) { f, f, f, f }; }
static INLINE vdouble vcast_vd_d(double d) { return (vdouble) { d, d }; }

static INLINE vdouble vcast_vd_vi(vint vi) { return vec_doubleh(vi); }
static INLINE vfloat vcast_vf_vi2(vint2 vi) { return vec_float(vi); }
static INLINE vint vrint_vi_vd(vdouble vd) {
  vd = vec_signed(vec_round(vd));
  return vec_perm(vd, vd, (vector unsigned char)(0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15));
}
static INLINE vint2 vrint_vi2_vf(vfloat vf) { return vec_signed(vec_round(vf)); }
static INLINE vint vtruncate_vi_vd(vdouble vd) {
  return vec_perm(vec_signed(vd), vec_signed(vd), (vector unsigned char)(0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15));
}
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return vec_signed(vf); }
static INLINE vdouble vtruncate_vd_vd(vdouble vd) { return vec_trunc(vd); }
static INLINE vfloat vtruncate_vf_vf(vfloat vf) { return vec_trunc(vf); }
static INLINE vdouble vrint_vd_vd(vdouble vd) { return vec_round(vd); }
static INLINE vfloat vrint_vf_vf(vfloat vf) { return vec_round(vf); }

static INLINE vmask vreinterpret_vm_vd(vdouble vd) { return (vmask)vd; }
static INLINE vdouble vreinterpret_vd_vm(vmask vm) { return (vdouble)vm; }
static INLINE vint2 vreinterpret_vi2_vd(vdouble vd) { return (vint2)vd; }
static INLINE vdouble vreinterpret_vd_vi2(vint2 vi) { return (vdouble)vi; }

static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return (vmask)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return (vfloat)vm; }
static INLINE vfloat vreinterpret_vf_vi2(vint2 vi) { return (vfloat)vi; }
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) { return (vint2)vf; }

static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) { return vec_add(x, y); }
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) { return vec_sub(x, y); }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) { return vec_mul(x, y); }
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) { return vec_div(x, y); }
static INLINE vdouble vrec_vd_vd(vdouble x) { return vec_div(vcast_vd_d(1.0), x); }
static INLINE vdouble vneg_vd_vd(vdouble d) { return vec_neg(d); }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return vec_add(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return vec_sub(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return vec_mul(x, y); }
static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return vec_div(x, y); }
static INLINE vfloat vrec_vf_vf(vfloat x) { return vec_div(vcast_vf_f(1.0f), x); }
static INLINE vfloat vneg_vf_vf(vfloat d) { return vec_neg(d); }

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return vec_and(x, y); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return vec_andc(y, x); }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return vec_or(x, y); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return vec_xor(x, y); }

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) { return vec_and(x, y); }
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) { return vec_andc(y, x); }
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) { return vec_or(x, y); }
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) { return vec_xor(x, y); }

static INLINE vmask vand_vm_vo64_vm(vopmask x, vmask y) { return vec_and((vmask)x, y); }
static INLINE vmask vandnot_vm_vo64_vm(vopmask x, vmask y) { return vec_andc(y, x); }
static INLINE vmask vor_vm_vo64_vm(vopmask x, vmask y) { return vec_or((vmask)x, y); }
static INLINE vmask vxor_vm_vo64_vm(vopmask x, vmask y) { return vec_xor((vmask)x, y); }

static INLINE vmask vand_vm_vo32_vm(vopmask x, vmask y) { return vec_and((vmask)x, y); }
static INLINE vmask vandnot_vm_vo32_vm(vopmask x, vmask y) { return vec_andc(y, x); }
static INLINE vmask vor_vm_vo32_vm(vopmask x, vmask y) { return vec_or((vmask)x, y); }
static INLINE vmask vxor_vm_vo32_vm(vopmask x, vmask y) { return vec_xor((vmask)x, y); }

static INLINE vdouble vsel_vd_vo_vd_vd(vopmask o, vdouble x, vdouble y) { return vec_sel(y, x, (vector unsigned long long)o); }
static INLINE vfloat vsel_vf_vo_vf_vf(vopmask o, vfloat x, vfloat y) { return vec_sel(y, x, o); }
static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask o, vint2 x, vint2 y) { return vec_sel(y, x, o); }

static INLINE int vtestallones_i_vo64(vopmask g) {
  return vec_all_ne(vec_and(g, (vector unsigned int)(0, 0, 0xffffffff, 0xffffffff)), (vector unsigned int)(0, 0, 0, 0));
}
static INLINE int vtestallones_i_vo32(vopmask g) { return vec_all_ne(g, (vector unsigned int)(0, 0, 0, 0)); }

static INLINE vopmask vcast_vo32_vo64(vopmask m) { return vec_perm(m, m, (vector unsigned char)(4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15 )); }
static INLINE vopmask vcast_vo64_vo32(vopmask m) { return vec_perm(m, m, (vector unsigned char)(0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7)); }

static INLINE vmask vcast_vm_i_i(int h, int l) { return (vmask){ l, h, l, h }; }
static INLINE vint2 vcastu_vi2_vi(vint vi) { return (vint2){ 0, vi[0], 0, vi[1] }; }
static INLINE vint vcastu_vi_vi2(vint2 vi2) { return (vint){ vi2[1], vi2[3] }; }

static INLINE vint vreinterpretFirstHalf_vi_vi2(vint2 vi2) { return (vint){ vi2[0], vi2[1] }; }
static INLINE vint2 vreinterpretFirstHalf_vi2_vi(vint vi) { return (vint2){ vi[0], vi[1], 0, 0 }; }

static INLINE vdouble vrev21_vd_vd(vdouble vd) { return vec_perm(vd, vd, (vector unsigned char)(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7)); }
static INLINE vdouble vreva2_vd_vd(vdouble vd) { return vd; }
static INLINE vfloat vrev21_vf_vf(vfloat vf) { return vec_perm(vf, vf, (vector unsigned char)(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11)); }
static INLINE vfloat vreva2_vf_vf(vfloat vf) { return vec_perm(vf, vf, (vector unsigned char)(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7)); }
static INLINE vint2 vrev21_vi2_vi2(vint2 i) { return vreinterpret_vi2_vf(vrev21_vf_vf(vreinterpret_vf_vi2(i))); }

static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) {
  vopmask o = vec_cmpeq(x, y);
  return o & vec_perm(o, o, (vector unsigned char)(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11));
}

static INLINE vmask vadd64_vm_vm_vm(vmask x, vmask y) {
  return (vmask)vec_add((vector long long)x, (vector long long)y);
}

//

#define PNMASK ((vdouble) { +0.0, -0.0 })
#define NPMASK ((vdouble) { -0.0, +0.0 })
#define PNMASKf ((vfloat) { +0.0f, -0.0f, +0.0f, -0.0f })
#define NPMASKf ((vfloat) { -0.0f, +0.0f, -0.0f, +0.0f })

static INLINE vdouble vposneg_vd_vd(vdouble d) { return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(PNMASK))); }
static INLINE vdouble vnegpos_vd_vd(vdouble d) { return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(NPMASK))); }
static INLINE vfloat vposneg_vf_vf(vfloat d) { return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(PNMASKf))); }
static INLINE vfloat vnegpos_vf_vf(vfloat d) { return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(NPMASKf))); }

//

static INLINE vdouble vabs_vd_vd(vdouble d) { return vec_abs(d); }
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) { return vec_max(x, y); }
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) { return vec_min(x, y); }
static INLINE vdouble vsubadd_vd_vd_vd(vdouble x, vdouble y) { return vadd_vd_vd_vd(x, vnegpos_vd_vd(y)); }
static INLINE vdouble vsqrt_vd_vd(vdouble d) { return vec_sqrt(d); }

#if CONFIG == 1
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vec_madd(x, y, z); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vec_msub(x, y, z); }
static INLINE vdouble vmlanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vec_nmsub(x, y, z); }
#else
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vsub_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }
#endif

static INLINE vdouble vmlsubadd_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vmla_vd_vd_vd_vd(x, y, vnegpos_vd_vd(z)); }
static INLINE vdouble vfma_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vec_madd(x, y, z); }
static INLINE vdouble vfmapp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vec_madd(x, y, z); }
static INLINE vdouble vfmapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vec_msub(x, y, z); }
static INLINE vdouble vfmanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vec_nmsub(x, y, z); }
static INLINE vdouble vfmann_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vec_nmadd(x, y, z); }

static INLINE vfloat vabs_vf_vf(vfloat f) { return vec_abs(f); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return vec_max(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return vec_min(x, y); }
static INLINE vfloat vsubadd_vf_vf_vf(vfloat x, vfloat y) { return vadd_vf_vf_vf(x, vnegpos_vf_vf(y)); }
static INLINE vfloat vsqrt_vf_vf(vfloat d) { return vec_sqrt(d); }

#if CONFIG == 1
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vec_madd(x, y, z); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vec_nmsub(x, y, z); }
static INLINE vfloat vmlapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vec_msub(x, y, z); }
#else
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vsub_vf_vf_vf(z, vmul_vf_vf_vf(x, y)); }
static INLINE vfloat vmlapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vsub_vf_vf_vf(vmul_vf_vf_vf(x, y), z); }
#endif

static INLINE vfloat vmlsubadd_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vmla_vf_vf_vf_vf(x, y, vnegpos_vf_vf(z)); }
static INLINE vfloat vfma_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vec_madd(x, y, z); }
static INLINE vfloat vfmapp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vec_madd(x, y, z); }
static INLINE vfloat vfmapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vec_msub(x, y, z); }
static INLINE vfloat vfmanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vec_nmsub(x, y, z); }
static INLINE vfloat vfmann_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vec_nmadd(x, y, z); }

//

static INLINE CONST vdouble vsel_vd_vo_d_d(vopmask o, double v1, double v0) {
  return vsel_vd_vo_vd_vd(o, vcast_vd_d(v1), vcast_vd_d(v0));
}

static INLINE vdouble vsel_vd_vo_vo_d_d_d(vopmask o0, vopmask o1, double d0, double d1, double d2) {
  return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_d_d(o1, d1, d2));
}

static INLINE vdouble vsel_vd_vo_vo_vo_d_d_d_d(vopmask o0, vopmask o1, vopmask o2, double d0, double d1, double d2, double d3) {
  return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_vd_vd(o1, vcast_vd_d(d1), vsel_vd_vo_d_d(o2, d2, d3)));
}

//

static INLINE vopmask vnot_vo_vo(vopmask o) { return vec_nand(o, o); }

static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)vec_cmpeq(x, y); }
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)vnot_vo_vo(vec_cmpeq(x, y)); }
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)vec_cmplt(x, y); }
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)vec_cmple(x, y); }
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)vec_cmpgt(x, y); }
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)vec_cmpge(x, y); }

static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return vec_add(x, y); }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return vec_sub(x, y); }
static INLINE vint vneg_vi_vi(vint e) { return vec_neg(e); }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return vec_and(x, y); }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return vec_andc(y, x); }
static INLINE vint vor_vi_vi_vi(vint x, vint y) { return vec_or(x, y); }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return vec_xor(x, y); }

static INLINE vint vand_vi_vo_vi(vopmask x, vint y) { return vreinterpretFirstHalf_vi_vi2((vint2)x) & y; }
static INLINE vint vandnot_vi_vo_vi(vopmask x, vint y) { return vec_andc(y, vreinterpretFirstHalf_vi_vi2((vint2)x)); }

static INLINE vint vsll_vi_vi_i(vint x, int c) { return vec_sl (x, (vector unsigned int)(c, c, c, c)); }
static INLINE vint vsrl_vi_vi_i(vint x, int c) { return vec_sr (x, (vector unsigned int)(c, c, c, c)); }
static INLINE vint vsra_vi_vi_i(vint x, int c) { return vec_sra(x, (vector unsigned int)(c, c, c, c)); }

static INLINE vint veq_vi_vi_vi(vint x, vint y) { return vec_cmpeq(x, y); }
static INLINE vint vgt_vi_vi_vi(vint x, vint y) { return vec_cmpgt(x, y); }

static INLINE vopmask veq_vo_vi_vi(vint x, vint y) { return (vopmask)vreinterpretFirstHalf_vi2_vi(vec_cmpeq(x, y)); }
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) { return (vopmask)vreinterpretFirstHalf_vi2_vi(vec_cmpgt(x, y));}

static INLINE vint vsel_vi_vo_vi_vi(vopmask m, vint x, vint y) {
  return vor_vi_vi_vi(vand_vi_vi_vi(vreinterpretFirstHalf_vi_vi2((vint2)m), x),
		      vandnot_vi_vi_vi(vreinterpretFirstHalf_vi_vi2((vint2)m), y));
}

static INLINE vopmask visinf_vo_vd(vdouble d) { return (vopmask)(vec_cmpeq(vabs_vd_vd(d), vcast_vd_d(SLEEF_INFINITY))); }
static INLINE vopmask vispinf_vo_vd(vdouble d) { return (vopmask)(vec_cmpeq(d, vcast_vd_d(SLEEF_INFINITY))); }
static INLINE vopmask visminf_vo_vd(vdouble d) { return (vopmask)(vec_cmpeq(d, vcast_vd_d(-SLEEF_INFINITY))); }
static INLINE vopmask visnan_vo_vd(vdouble d) { return (vopmask)(vnot_vo_vo(vec_cmpeq(d, d))); }

static INLINE double vcast_d_vd(vdouble v) { return v[0]; }
static INLINE float vcast_f_vf(vfloat v) { return v[0]; }

static INLINE void vstream_v_p_vd(double *ptr, vdouble v) { vstore_v_p_vd(ptr, v); }
static INLINE void vsscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) { vscatter2_v_p_i_i_vd(ptr, offset, step, v); }

//

static INLINE CONST vfloat vsel_vf_vo_f_f(vopmask o, float v1, float v0) {
  return vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0));
}

static INLINE vfloat vsel_vf_vo_vo_f_f_f(vopmask o0, vopmask o1, float d0, float d1, float d2) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_f_f(o1, d1, d2));
}

static INLINE vfloat vsel_vf_vo_vo_vo_f_f_f_f(vopmask o0, vopmask o1, vopmask o2, float d0, float d1, float d2, float d3) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_vf_vf(o1, vcast_vf_f(d1), vsel_vf_vo_f_f(o2, d2, d3)));
}

static INLINE vint2 vcast_vi2_vm(vmask vm) { return (vint2)vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return (vmask)vi; }

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)vec_cmpeq(x, y); }
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)vnot_vo_vo(vec_cmpeq(x, y)); }
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)vec_cmplt(x, y); }
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)vec_cmple(x, y); }
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)vec_cmpgt(x, y); }
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)vec_cmpge(x, y); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return vec_add(x, y); }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return vec_sub(x, y); }
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return vec_neg(e); }

static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return vec_and(x, y); }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return vec_andc(y, x); }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return vec_or(x, y); }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return vec_xor(x, y); }

static INLINE vint2 vand_vi2_vo_vi2(vopmask x, vint2 y) { return (vint2)vec_and((vint2)x, y); }
static INLINE vint2 vandnot_vi2_vo_vi2(vopmask x, vint2 y) { return vec_andc(y, (vint2)x); }

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return vec_sl (x, (vector unsigned int)(c, c, c, c)); }
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return vec_sr (x, (vector unsigned int)(c, c, c, c)); }
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return vec_sra(x, (vector unsigned int)(c, c, c, c)); }

static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) { return (vopmask)vec_cmpeq(x, y); }
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) { return (vopmask)vec_cmpgt(x, y); }
static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) { return vec_cmpeq(x, y); }
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) { return vec_cmpgt(x, y); }

static INLINE vopmask visinf_vo_vf(vfloat d) { return (vopmask)vec_cmpeq(vabs_vf_vf(d), vcast_vf_f(SLEEF_INFINITYf)); }
static INLINE vopmask vispinf_vo_vf(vfloat d) { return (vopmask)vec_cmpeq(d, vcast_vf_f(SLEEF_INFINITYf)); }
static INLINE vopmask visminf_vo_vf(vfloat d) { return (vopmask)vec_cmpeq(d, vcast_vf_f(-SLEEF_INFINITYf)); }
static INLINE vopmask visnan_vo_vf(vfloat d) { return (vopmask)vnot_vo_vo(vec_cmpeq(d, d)); }

static INLINE void vsscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) { vscatter2_v_p_i_i_vf(ptr, offset, step, v); }
static INLINE void vstream_v_p_vf(float *ptr, vfloat v) { vstore_v_p_vf(ptr, v); }
