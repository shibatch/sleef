/*********************************************************************/
/*          Copyright ARM Ltd. 2010 - 2017.                          */
/* Distributed under the Boost Software License, Version 1.0.        */
/*    (See accompanying file LICENSE.txt or copy at                  */
/*          http://www.boost.org/LICENSE_1_0.txt)                    */
/*********************************************************************/

#ifndef __ARM_FEATURE_SVE
#error Please specify SVE flags.
#endif

#include <arm_sve.h>
#include <stdint.h>

#include "misc.h"

#if defined(VECTLENDP) || defined(VECTLENSP)
#error VECTLENDP or VECTLENSP already defined
#endif

#if CONFIG == 1
// Vector length agnostic
#define VECTLENSP (svcntw())
#define VECTLENDP (svcntd())
#define ISANAME "AArch64 SVE"
#define ptrue svptrue_b8()
#elif CONFIG == 8
// 256-bit vector length
#define ISANAME "AArch64 SVE 256-bit"
#define LOG2VECTLENDP 2
#define ptrue svptrue_pat_b8(SV_VL32)
#define DFTPRIORITY 20
#elif CONFIG == 9
// 512-bit vector length
#define ISANAME "AArch64 SVE 512-bit"
#define LOG2VECTLENDP 3
#define ptrue svptrue_pat_b8(SV_VL64)
#define DFTPRIORITY 21
#elif CONFIG == 10
// 1024-bit vector length
#define ISANAME "AArch64 SVE 1024-bit"
#define LOG2VECTLENDP 4
#define ptrue svptrue_pat_b8(SV_VL128)
#define DFTPRIORITY 22
#elif CONFIG == 11
// 2048-bit vector length
#define ISANAME "AArch64 SVE 2048-bit"
#define LOG2VECTLENDP 5
#define ptrue svptrue_pat_b8(SV_VL256)
#define DFTPRIORITY 23
#else
#error CONFIG macro invalid or not defined
#endif

#ifdef LOG2VECTLENDP
// For DFT, VECTLENDP and VECTLENSP are not the size of the available
// vector length, but the size of the partial vectors utilized in the
// computation. The appropriate VECTLENDP and VECTLENSP are chosen by
// the dispatcher according to the value of svcntd().

#define LOG2VECTLENSP (LOG2VECTLENDP+1)
#define VECTLENDP (1 << LOG2VECTLENDP)
#define VECTLENSP (1 << LOG2VECTLENSP)
static INLINE int vavailability_i(int name) { return svcntd() >= VECTLENDP ? 3 : 0; }
#else
static INLINE int vavailability_i(int name) { return 3; }
#endif

#define ENABLE_SP
#define ENABLE_FMA_SP

#define ENABLE_DP
#define ENABLE_FMA_DP

#define FULL_FP_ROUNDING
#define ACCURATE_SQRT

// Mask definition
typedef svint32_t vmask;
typedef svbool_t vopmask;

// Single precision definitions
typedef svfloat32_t vfloat;
typedef svint32_t vint2;

// Double precision definitions
typedef svfloat64_t vdouble;
typedef svint32_t vint;

// masking predicates
#define ALL_TRUE_MASK svdup_n_s32(0xffffffff)
#define ALL_FALSE_MASK svdup_n_s32(0x0)

static INLINE void vprefetch_v_p(const void *ptr) {}

//
//
//
// Test if all lanes are active
//
//
//
static INLINE int vtestallones_i_vo32(vopmask g) {
  svbool_t pg = svptrue_b32();
  return (svcntp_b32(pg, g) == svcntw());
}

static INLINE int vtestallones_i_vo64(vopmask g) {
  svbool_t pg = svptrue_b64();
  return (svcntp_b64(pg, g) == svcntd());
}
//
//
//
//
//
//

// Vector load / store
static INLINE void vstoreu_v_p_vi2(int32_t *p, vint2 v) { svst1_s32(ptrue, p, v); }

static INLINE vfloat vload_vf_p(const float *ptr) {
  return svld1_f32(ptrue, ptr);
}
static INLINE vfloat vloadu_vf_p(const float *ptr) {
  return svld1_f32(ptrue, ptr);
}
static INLINE void vstoreu_v_p_vf(float *ptr, vfloat v) {
  svst1_f32(ptrue, ptr, v);
}

// Basic logical operations for mask
static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) {
  return svand_s32_x(ptrue, x, y);
}
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) {
  return svbic_s32_x(ptrue, y, x);
}
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) {
  return svorr_s32_x(ptrue, x, y);
}
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) {
  return sveor_s32_x(ptrue, x, y);
}

static INLINE vmask vadd64_vm_vm_vm(vmask x, vmask y) {
  return svreinterpret_s32_s64(
           svadd_s64_x(ptrue, svreinterpret_s64_s32(x),
                              svreinterpret_s64_s32(y)));
}

// Mask <--> single precision reinterpret
static INLINE vmask vreinterpret_vm_vf(vfloat vf) {
  return svreinterpret_s32_f32(vf);
}
static INLINE vfloat vreinterpret_vf_vm(vmask vm) {
  return svreinterpret_f32_s32(vm);
}
static INLINE vfloat vreinterpret_vf_vi2(vint2 vm) {
  return svreinterpret_f32_s32(vm);
}
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) {
  return svreinterpret_s32_f32(vf);
}
static INLINE vint2 vcast_vi2_vm(vmask vm) { return vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return vi; }

// Conditional select
static INLINE vint2 vsel_vi2_vm_vi2_vi2(vmask m, vint2 x, vint2 y) {
  return svsel_s32(svcmpeq_s32(ptrue, m, ALL_TRUE_MASK), x, y);
}

/****************************************/
/* Single precision FP operations */
/****************************************/
// Broadcast
static INLINE vfloat vcast_vf_f(float f) { return svdup_n_f32(f); }

// Add, Sub, Mul, Reciprocal 1/x, Division, Square root
static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) {
  return svadd_f32_x(ptrue, x, y);
}
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) {
  return svsub_f32_x(ptrue, x, y);
}
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) {
  return svmul_f32_x(ptrue, x, y);
}
static INLINE vfloat vrec_vf_vf(vfloat d) {
  return svdivr_n_f32_x(ptrue, d, 1.0f);
}
static INLINE vfloat vdiv_vf_vf_vf(vfloat n, vfloat d) {
  return svdiv_f32_x(ptrue, n, d);
}
static INLINE vfloat vsqrt_vf_vf(vfloat d) { return svsqrt_f32_x(ptrue, d); }

// |x|, -x
static INLINE vfloat vabs_vf_vf(vfloat f) { return svabs_f32_x(ptrue, f); }
static INLINE vfloat vneg_vf_vf(vfloat f) { return svneg_f32_x(ptrue, f); }

// max, min
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) {
  return svmax_f32_x(ptrue, x, y);
}
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) {
  return svmin_f32_x(ptrue, x, y);
}

// int <--> float conversions
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) {
  return svcvt_s32_f32_x(ptrue, vf);
}
static INLINE vfloat vcast_vf_vi2(vint2 vi) {
  return svcvt_f32_s32_x(ptrue, vi);
}
static INLINE vint2 vcast_vi2_i(int i) { return svdup_n_s32(i); }
static INLINE vint2 vrint_vi2_vf(vfloat d) {
  return svcvt_s32_f32_x(ptrue, svrinta_f32_x(ptrue, d));
}

// Multiply accumulate: z = z + x * y
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) {
  return svmad_f32_x(ptrue, x, y, z);
}
// Multiply subtract: z = z - x * y
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) {
  return svmsb_f32_x(ptrue, x, y, z);
}

// fused multiply add / sub
static INLINE vfloat vfma_vf_vf_vf_vf(vfloat x, vfloat y,
                                      vfloat z) { // z + x * y
  return svmad_f32_x(ptrue, x, y, z);
}
static INLINE vfloat vfmanp_vf_vf_vf_vf(vfloat x, vfloat y,
                                        vfloat z) { // z - x * y
  return svmsb_f32_x(ptrue, x, y, z);
}
static INLINE vfloat vfmapn_vf_vf_vf_vf(vfloat x, vfloat y,
                                        vfloat z) { // x * y - z
  return svnmsb_f32_x(ptrue, x, y, z);
}

// conditional select
static INLINE vfloat vsel_vf_vo_vf_vf(vopmask mask, vfloat x, vfloat y) {
  return svsel_f32(mask, x, y);
}

//
//
//
//
//
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
//
//
//
//
//
//

// truncate
static INLINE vfloat vtruncate_vf_vf(vfloat vd) {
  return svrintz_f32_x(ptrue, vd);
}

//
//
//
// Round float
//
//
//
static INLINE vfloat vrint_vf_vf(vfloat vf) {
  return svrinta_f32_x(svptrue_b32(), vf);
}
//
//
//
//
//
//

/***************************************/
/* Single precision integer operations */
/***************************************/

// Add, Sub, Neg (-x)
static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) {
  return svadd_s32_x(ptrue, x, y);
}
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) {
  return svsub_s32_x(ptrue, x, y);
}
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return svneg_s32_x(ptrue, e); }

// Logical operations
static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) {
  return svand_s32_x(ptrue, x, y);
}
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) {
  return svbic_s32_x(ptrue, y, x);
}
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) {
  return svorr_s32_x(ptrue, x, y);
}
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) {
  return sveor_s32_x(ptrue, x, y);
}

// Shifts
#define vsll_vi2_vi2_i(x, c) svlsl_n_s32_x(ptrue, x, c)
#define vsrl_vi2_vi2_i(x, c)                                                   \
  svreinterpret_s32_u32(svlsr_n_u32_x(ptrue, svreinterpret_u32_s32(x), c))

#define vsra_vi2_vi2_i(x, c) svasr_n_s32_x(ptrue, x, c)

// Comparison returning integers
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) {
  return svsel_s32(svcmpge_s32(ptrue, x, y), ALL_TRUE_MASK, ALL_FALSE_MASK);
}

// conditional select
static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask m, vint2 x, vint2 y) {
  return svsel_s32(m, x, y);
}

/****************************************/
/* opmask operations                    */
/****************************************/
// single precision FP
static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) {
  return svcmpeq_f32(ptrue, x, y);
}
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) {
  return svcmpne_f32(ptrue, x, y);
}
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) {
  return svcmplt_f32(ptrue, x, y);
}
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) {
  return svcmple_f32(ptrue, x, y);
}
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) {
  return svcmpgt_f32(ptrue, x, y);
}
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) {
  return svcmpge_f32(ptrue, x, y);
}
static INLINE vopmask visinf_vo_vf(vfloat d) {
  return svcmpeq_n_f32(ptrue, vabs_vf_vf(d), SLEEF_INFINITYf);
}
static INLINE vopmask vispinf_vo_vf(vfloat d) {
  return svcmpeq_n_f32(ptrue, d, SLEEF_INFINITYf);
}
static INLINE vopmask visminf_vo_vf(vfloat d) {
  return svcmpeq_n_f32(ptrue, d, -SLEEF_INFINITYf);
}
static INLINE vopmask visnan_vo_vf(vfloat d) { return vneq_vo_vf_vf(d, d); }

// integers
static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) {
  return svcmpeq_s32(ptrue, x, y);
}
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) {
  return svcmpgt_s32(ptrue, x, y);
}

// logical opmask
static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) {
  return svand_b_z(ptrue, x, y);
}
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) {
  return svbic_b_z(ptrue, y, x);
}
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) {
  return svorr_b_z(ptrue, x, y);
}
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) {
  return sveor_b_z(ptrue, x, y);
}

static INLINE vint2 vand_vi2_vo_vi2(vopmask x, vint2 y) {
  // This needs to be zeroing to prevent asinf and atanf denormal test
  // failing.
  return svand_s32_z(x, y, y);
}

// bitmask logical operations
static INLINE vmask vand_vm_vo32_vm(vopmask x, vmask y) {
  return svsel_s32(x, y, ALL_FALSE_MASK);
}
static INLINE vmask vandnot_vm_vo32_vm(vopmask x, vmask y) {
  return svsel_s32(x, ALL_FALSE_MASK, y);
}
static INLINE vmask vor_vm_vo32_vm(vopmask x, vmask y) {
  return svsel_s32(x, ALL_TRUE_MASK, y);
}

// broadcast bitmask
static INLINE vmask vcast_vm_i_i(int i0, int i1) {
  return svreinterpret_s32_u64(
      svdup_n_u64((0xffffffff & (uint64_t)i1) | (((uint64_t)i0) << 32)));
}

/*********************************/
/* SVE for double precision math */
/*********************************/

// Vector load/store
static INLINE vdouble vload_vd_p(const double *ptr) {
  return svld1_f64(ptrue, ptr);
}
static INLINE vdouble vloadu_vd_p(const double *ptr) {
  return svld1_f64(ptrue, ptr);
}
static INLINE void vstoreu_v_p_vd(double *ptr, vdouble v) {
  svst1_f64(ptrue, ptr, v);
}

static INLINE void vstoreu_v_p_vi(int *ptr, vint v) {
  svst1w_s64(ptrue, ptr, svreinterpret_s64_s32(v));
}
static vint vloadu_vi_p(int32_t *p) {
  return svreinterpret_s32_s64(svld1uw_s64(ptrue, (uint32_t *)p));
}

// Reinterpret
static INLINE vdouble vreinterpret_vd_vm(vmask vm) {
  return svreinterpret_f64_s32(vm);
}
static INLINE vmask vreinterpret_vm_vd(vdouble vd) {
  return svreinterpret_s32_f64(vd);
}
static INLINE vdouble vreinterpret_vd_vi2(vint2 x) {
  return svreinterpret_f64_s32(x);
}
static INLINE vint2 vreinterpret_vi2_vd(vdouble x) {
  return svreinterpret_s32_f64(x);
}
static INLINE vint2 vcastu_vi2_vi(vint x) {
  return svreinterpret_s32_s64(
      svlsl_n_s64_x(ptrue, svreinterpret_s64_s32(x), 32));
}
static INLINE vint vcastu_vi_vi2(vint2 x) {
  return svreinterpret_s32_s64(
      svlsr_n_s64_x(ptrue, svreinterpret_s64_s32(x), 32));
}
static INLINE vdouble vcast_vd_vi(vint vi) {
  return svcvt_f64_s32_x(ptrue, vi);
}

// Splat
static INLINE vdouble vcast_vd_d(double d) { return svdup_n_f64(d); }

// Conditional select
static INLINE vdouble vsel_vd_vo_vd_vd(vopmask o, vdouble x, vdouble y) {
  return svsel_f64(o, x, y);
}

static INLINE CONST vdouble vsel_vd_vo_d_d(vopmask o, double v1, double v0) {
  return vsel_vd_vo_vd_vd(o, vcast_vd_d(v1), vcast_vd_d(v0));
}

static INLINE vdouble vsel_vd_vo_vo_d_d_d(vopmask o0, vopmask o1, double d0, double d1, double d2) {
  return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_d_d(o1, d1, d2));
}

static INLINE vdouble vsel_vd_vo_vo_vo_d_d_d_d(vopmask o0, vopmask o1, vopmask o2, double d0, double d1, double d2, double d3) {
  return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_vd_vd(o1, vcast_vd_d(d1), vsel_vd_vo_d_d(o2, d2, d3)));
}

static INLINE vint vsel_vi_vo_vi_vi(vopmask o, vint x, vint y) {
  return svsel_s32(o, x, y);
}
// truncate
static INLINE vdouble vtruncate_vd_vd(vdouble vd) {
  return svrintz_f64_x(ptrue, vd);
}
static INLINE vint vtruncate_vi_vd(vdouble vd) {
  return svcvt_s32_f64_x(ptrue, vd);
}
static INLINE vint vrint_vi_vd(vdouble vd) {
  return svcvt_s32_f64_x(ptrue, svrinta_f64_x(ptrue, vd));
}
static INLINE vdouble vrint_vd_vd(vdouble vd) {
  return svrinta_f64_x(ptrue, vd);
}

// FP math operations
static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) {
  return svadd_f64_x(ptrue, x, y);
}
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) {
  return svsub_f64_x(ptrue, x, y);
}
static INLINE vdouble vneg_vd_vd(vdouble x) { return svneg_f64_x(ptrue, x); }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) {
  return svmul_f64_x(ptrue, x, y);
}
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) {
  return svdiv_f64_x(ptrue, x, y);
}
static INLINE vdouble vrec_vd_vd(vdouble x) {
  return svdivr_n_f64_x(ptrue, x, 1.0);
}
static INLINE vdouble vsqrt_vd_vd(vdouble x) { return svsqrt_f64_x(ptrue, x); }
static INLINE vdouble vabs_vd_vd(vdouble x) { return svabs_f64_x(ptrue, x); }
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) {
  return svmax_f64_x(ptrue, x, y);
}
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) {
  return svmin_f64_x(ptrue, x, y);
}

// Multiply accumulate / subtract
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y,
                                       vdouble z) { // z = x*y + z
  return svmad_f64_x(ptrue, x, y, z);
}
static INLINE vdouble vfma_vd_vd_vd_vd(vdouble x, vdouble y,
                                       vdouble z) { // z + x * y
  return svmad_f64_x(ptrue, x, y, z);
}
static INLINE vdouble vfmanp_vd_vd_vd_vd(vdouble x, vdouble y,
                                         vdouble z) { // z - x * y
  return svmsb_f64_x(ptrue, x, y, z);
}
static INLINE vdouble vfmapn_vd_vd_vd_vd(vdouble x, vdouble y,
                                         vdouble z) { // x * y - z
  return svnmsb_f64_x(ptrue, x, y, z);
}
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y,
                                         vdouble z) { // z = x * y - z
  return svnmsb_f64_x(ptrue, x, y, z);
}

// Float comparison
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) {
  return svcmplt_f64(ptrue, x, y);
}
static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) {
  return svcmpeq_f64(ptrue, x, y);
}
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) {
  return svcmpgt_f64(ptrue, x, y);
}
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) {
  return svcmpge_f64(ptrue, x, y);
}
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) {
  return svcmpne_f64(ptrue, x, y);
}
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) {
  return svcmple_f64(ptrue, x, y);
}

// predicates
static INLINE vopmask visnan_vo_vd(vdouble vd) {
  return svcmpne_f64(ptrue, vd, vd);
}
static INLINE vopmask visinf_vo_vd(vdouble vd) {
  return svcmpeq_n_f64(ptrue, svabs_f64_x(ptrue, vd), SLEEF_INFINITY);
}
static INLINE vopmask vispinf_vo_vd(vdouble vd) {
  return svcmpeq_n_f64(ptrue, vd, SLEEF_INFINITY);
}
static INLINE vopmask visminf_vo_vd(vdouble vd) {
  return svcmpeq_n_f64(ptrue, vd, -SLEEF_INFINITY);
}

// Comparing bit masks
static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) {
  return svcmpeq_s64(ptrue, svreinterpret_s64_s32(x), svreinterpret_s64_s32(y));
}

// pure predicate operations
static INLINE vopmask vcast_vo32_vo64(vopmask o) { return o; }
static INLINE vopmask vcast_vo64_vo32(vopmask o) { return o; }

// logical integer operations
static INLINE vint vand_vi_vo_vi(vopmask x, vint y) {
  // This needs to be a zeroing instruction because we need to make
  // sure that the inactive elements for the unpacked integers vector
  // are zero.
  return svand_s32_z(x, y, y);
}

static INLINE vint vandnot_vi_vo_vi(vopmask x, vint y) {
  return svsel_s32(x, ALL_FALSE_MASK, y);
}
#define vsra_vi_vi_i(x, c) svasr_n_s32_x(ptrue, x, c)
#define vsll_vi_vi_i(x, c) svlsl_n_s32_x(ptrue, x, c)
#define vsrl_vi_vi_i(x, c) svlsr_n_s32_x(ptrue, x, c)

static INLINE vint vand_vi_vi_vi(vint x, vint y) {
  return svand_s32_x(ptrue, x, y);
}
static INLINE vint vxor_vi_vi_vi(vint x, vint y) {
  return sveor_s32_x(ptrue, x, y);
}

// integer math
static INLINE vint vadd_vi_vi_vi(vint x, vint y) {
  return svadd_s32_x(ptrue, x, y);
}
static INLINE vint vsub_vi_vi_vi(vint x, vint y) {
  return svsub_s32_x(ptrue, x, y);
}
static INLINE vint vneg_vi_vi(vint x) { return svneg_s32_x(ptrue, x); }

// integer comparison
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) {
  return svcmpgt_s32(ptrue, x, y);
}
static INLINE vopmask veq_vo_vi_vi(vint x, vint y) {
  return svcmpeq_s32(ptrue, x, y);
}

// Splat
static INLINE vint vcast_vi_i(int i) { return svdup_n_s32(i); }

// bitmask logical operations
static INLINE vmask vand_vm_vo64_vm(vopmask x, vmask y) {
  // This needs to be a zeroing instruction because we need to make
  // sure that the inactive elements for the unpacked integers vector
  // are zero.
  return svreinterpret_s32_s64(
      svand_s64_z(x, svreinterpret_s64_s32(y), svreinterpret_s64_s32(y)));
}
static INLINE vmask vandnot_vm_vo64_vm(vopmask x, vmask y) {
  return svreinterpret_s32_s64(svsel_s64(
      x, svreinterpret_s64_s32(ALL_FALSE_MASK), svreinterpret_s64_s32(y)));
}
static INLINE vmask vor_vm_vo64_vm(vopmask x, vmask y) {
  return svreinterpret_s32_s64(svsel_s64(
      x, svreinterpret_s64_s32(ALL_TRUE_MASK), svreinterpret_s64_s32(y)));
}

static INLINE vfloat vrev21_vf_vf(vfloat vf) {
  return svreinterpret_f32_u64(svrevw_u64_x(ptrue, svreinterpret_u64_f32(vf)));
}

// Comparison returning integer
static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) {
  return svsel_s32(svcmpeq_s32(ptrue, x, y), ALL_TRUE_MASK, ALL_FALSE_MASK);
}


// Operations for DFT

static INLINE vdouble vposneg_vd_vd(vdouble d) {
  vmask pnmask = svreinterpret_s32_u64(svlsl_n_u64_x(ptrue, svindex_u64(0, 1), 63));
  return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), pnmask));
}

static INLINE vdouble vnegpos_vd_vd(vdouble d) {
  vmask pnmask = svreinterpret_s32_u64(svlsl_n_u64_x(ptrue, svindex_u64(1, 1), 63));
  return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), pnmask));
}

static INLINE vfloat vposneg_vf_vf(vfloat d) {
  vmask pnmask = svreinterpret_s32_u32(svlsl_n_u32_x(ptrue, svindex_u32(0, 1), 31));
  return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), pnmask));
}

static INLINE vfloat vnegpos_vf_vf(vfloat d) {
  vmask pnmask = svreinterpret_s32_u32(svlsl_n_u32_x(ptrue, svindex_u32(1, 1), 31));
  return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), pnmask));
}

static INLINE vdouble vsubadd_vd_vd_vd(vdouble x, vdouble y) { return vadd_vd_vd_vd(x, vnegpos_vd_vd(y)); }
static INLINE vfloat vsubadd_vf_vf_vf(vfloat d0, vfloat d1) { return vadd_vf_vf_vf(d0, vnegpos_vf_vf(d1)); }
static INLINE vdouble vmlsubadd_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vfma_vd_vd_vd_vd(x, y, vnegpos_vd_vd(z)); }
static INLINE vfloat vmlsubadd_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vfma_vf_vf_vf_vf(x, y, vnegpos_vf_vf(z)); }

//

static INLINE vdouble vrev21_vd_vd(vdouble x) { return svzip1_f64(svuzp2_f64(x, x), svuzp1_f64(x, x)); }

static INLINE vdouble vreva2_vd_vd(vdouble vd) {
  svint64_t x = svindex_s64((VECTLENDP-1), -1);
  x = svzip1_s64(svuzp2_s64(x, x), svuzp1_s64(x, x));
  return svtbl_f64(vd, svreinterpret_u64_s64(x));
}

static INLINE vfloat vreva2_vf_vf(vfloat vf) {
  svint32_t x = svindex_s32((VECTLENSP-1), -1);
  x = svzip1_s32(svuzp2_s32(x, x), svuzp1_s32(x, x));
  return svtbl_f32(vf, svreinterpret_u32_s32(x));
}

//

static INLINE void vscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) {
  svst1_scatter_u64index_f64(ptrue, ptr + offset*2, svzip1_u64(svindex_u64(0, step*2), svindex_u64(1, step*2)), v);
}

static INLINE void vscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) {
  svst1_scatter_u32index_f32(ptrue, ptr + offset*2, svzip1_u32(svindex_u32(0, step*2), svindex_u32(1, step*2)), v);
}

static INLINE void vstore_v_p_vd(double *ptr, vdouble v) { vstoreu_v_p_vd(ptr, v); }
static INLINE void vstream_v_p_vd(double *ptr, vdouble v) { vstore_v_p_vd(ptr, v); }
static INLINE void vstore_v_p_vf(float *ptr, vfloat v) { vstoreu_v_p_vf(ptr, v); }
static INLINE void vstream_v_p_vf(float *ptr, vfloat v) { vstore_v_p_vf(ptr, v); }
static INLINE void vsscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) { vscatter2_v_p_i_i_vd(ptr, offset, step, v); }
static INLINE void vsscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) { vscatter2_v_p_i_i_vf(ptr, offset, step, v); }
