#ifndef __ARM_NEON__
#error Please specify -mfpu=neon.
#endif

#ifdef __aarch64__
#warning This implementation is for AARCH32.
#endif

#include <arm_neon.h>
#include <stdint.h>

typedef uint32x4_t vmask;
typedef uint32x4_t vopmask;

typedef float32x4_t vfloat;
typedef int32x4_t vint2;

//

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return vandq_u32(x, y); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return vbicq_u32(y, x); }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return vorrq_u32(x, y); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return veorq_u32(x, y); }

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) { return vandq_u32(x, y); }
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) { return vbicq_u32(y, x); }
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) { return vorrq_u32(x, y); }
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) { return veorq_u32(x, y); }

static INLINE vmask vand_vm_vo64_vm(vopmask x, vmask y) { return vandq_u32(x, y); }
static INLINE vmask vandnot_vm_vo64_vm(vopmask x, vmask y) { return vbicq_u32(y, x); }
static INLINE vmask vor_vm_vo64_vm(vopmask x, vmask y) { return vorrq_u32(x, y); }
static INLINE vmask vxor_vm_vo64_vm(vopmask x, vmask y) { return veorq_u32(x, y); }

static INLINE vmask vand_vm_vo32_vm(vopmask x, vmask y) { return vandq_u32(x, y); }
static INLINE vmask vandnot_vm_vo32_vm(vopmask x, vmask y) { return vbicq_u32(y, x); }
static INLINE vmask vor_vm_vo32_vm(vopmask x, vmask y) { return vorrq_u32(x, y); }
static INLINE vmask vxor_vm_vo32_vm(vopmask x, vmask y) { return veorq_u32(x, y); }

static INLINE vopmask vcast_vo32_vo64(vopmask m) { return vuzpq_u32(m, m).val[0]; }
static INLINE vopmask vcast_vo64_vo32(vopmask m) { return vzipq_u32(m, m).val[0]; }

//

static INLINE vmask vcast_vm_i_i(int i0, int i1) { return (vmask)vdupq_n_u64((uint64_t)i0 | (((uint64_t)i1) << 32)); }
static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) {
  uint32x4_t t = vceqq_u32(x, y);
  return vandq_u32(t, vrev64q_u32(t));
}

//

static INLINE vint2 vcast_vi2_vm(vmask vm) { return (vint2)vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return (vmask)vi; }
static INLINE vint2 vrint_vi2_vf(vfloat d) {
  return vcvtq_s32_f32(vaddq_f32(d, (float32x4_t)vorrq_u32(vandq_u32((uint32x4_t)d, (uint32x4_t)vdupq_n_f32(-0.0f)), (uint32x4_t)vdupq_n_f32(0.5f))));
}
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return vcvtq_s32_f32(vf); }
static INLINE vfloat vcast_vf_vi2(vint2 vi) { return vcvtq_f32_s32(vi); }
static INLINE vfloat vcast_vf_f(float f) { return vdupq_n_f32(f); }
static INLINE vint2 vcast_vi2_i(int i) { return vdupq_n_s32(i); }
static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return (vmask)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return (vfloat)vm; }
static INLINE vfloat vreinterpret_vf_vi2(vint2 vm) { return (vfloat)vm; }
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) { return (vint2)vf; }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return vaddq_f32(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return vsubq_f32(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return vmulq_f32(x, y); }

static INLINE vfloat vdiv_vf_vf_vf(vfloat n, vfloat d) {
  float32x4_t x = vrecpeq_f32(d);
  x = vmulq_f32(x, vrecpsq_f32(d, x));
  float32x4_t t = vmulq_f32(n, x);
  return vmlsq_f32(vaddq_f32(t, t), vmulq_f32(t, x), d);
}

static INLINE vfloat vrec_vf_vf(vfloat d) {
  float32x4_t x = vrecpeq_f32(d);
  x = vmulq_f32(x, vrecpsq_f32(d, x));
  return vmlsq_f32(vaddq_f32(x, x), vmulq_f32(x, x), d);
}

static INLINE vfloat vsqrt_vf_vf(vfloat d) {
  float32x4_t x = vrsqrteq_f32(d);
  x = vmulq_f32(x, vrsqrtsq_f32(d, vmulq_f32(x, x)));
  float32x4_t u = vmulq_f32(x, d);
  u = vmlaq_f32(u, vmlsq_f32(d, u, u), vmulq_f32(x, vdupq_n_f32(0.5)));
  return (float32x4_t)vbicq_u32((uint32x4_t)u, vceqq_f32(d, vdupq_n_f32(0.0f)));
}

static INLINE vfloat vrecsqrt_vf_vf(vfloat d) {
  float32x4_t x = vrsqrteq_f32(d);
  x = vmulq_f32(x, vrsqrtsq_f32(d, vmulq_f32(x, x)));
  return vmlaq_f32(x, vmlsq_f32(vdupq_n_f32(1), x, vmulq_f32(x, d)), vmulq_f32(x, vdupq_n_f32(0.5)));
}

#define ENABLE_RECSQRT_SP

static INLINE vfloat vabs_vf_vf(vfloat f) { return vabsq_f32(f); }
static INLINE vfloat vneg_vf_vf(vfloat f) { return vnegq_f32(f); }
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vmlaq_f32(z, x, y); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vmlsq_f32(z, x, y); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return vmaxq_f32(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return vminq_f32(x, y); }

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) { return vceqq_f32(x, y); }
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) { return vmvnq_u32(vceqq_f32(x, y)); }
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return vcltq_f32(x, y); }
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) { return vcleq_f32(x, y); }
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) { return vcgtq_f32(x, y); }
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) { return vcgeq_f32(x, y); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return vaddq_s32(x, y); }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return vsubq_s32(x, y); }
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return vnegq_s32(e); }

static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return vandq_s32(x, y); }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return vbicq_s32(y, x); }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return vorrq_s32(x, y); }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return veorq_s32(x, y); }

static INLINE vint2 vand_vi2_vo_vi2(vopmask x, vint2 y) { return (vint2)vandq_u32(x, (vopmask)y); }
static INLINE vint2 vandnot_vi2_vo_vi2(vopmask x, vint2 y) { return (vint2)vbicq_u32((vopmask)y, x); }

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return (int32x4_t) vshlq_n_u32((uint32x4_t)x, c); }
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return (int32x4_t) vshrq_n_u32((uint32x4_t)x, c); }
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return vshrq_n_s32(x, c); }

static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) { return vceqq_s32(x, y); }
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) { return vcgeq_s32(x, y); }
static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) { return (vint2)vceqq_s32(x, y); }
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) { return (vint2)vcgeq_s32(x, y); }

static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask m, vint2 x, vint2 y) { return (vint2)vbslq_u32(m, (vmask)x, (vmask)y); }

static INLINE vfloat vsel_vf_vo_vf_vf(vopmask mask, vfloat x, vfloat y) {
  return (vfloat)vbslq_u32(mask, (vmask)x, (vmask)y);
}

static INLINE vopmask visinf_vo_vf(vfloat d) { return veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(INFINITYf)); }
static INLINE vopmask vispinf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(INFINITYf)); }
static INLINE vopmask visminf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(-INFINITYf)); }
static INLINE vopmask visnan_vo_vf(vfloat d) { return vneq_vo_vf_vf(d, d); }

static INLINE float vcast_f_vf(vfloat v) {
  float p[4];
  vst1q_f32 (p, v);
  return p[0];
}
