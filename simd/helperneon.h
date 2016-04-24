#ifndef __ARM_NEON__
#error Please specify -mfpu=neon.
#endif

#include <arm_neon.h>
#include <stdint.h>

typedef int32x4_t vint;
typedef uint32x4_t vmask;

typedef float32x4_t vfloat;
typedef int32x4_t vint2;

//

static INLINE vint vcast_vi_i(int i) { return vdupq_n_s32(i); }

static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return (vmask)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return (vfloat)vm; }

//

static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return vaddq_s32(x, y); }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return vsubq_s32(x, y); }
static INLINE vint vneg_vi_vi(vint e) { return vnegq_s32(e); }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return vandq_s32(x, y); }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return vbicq_s32(y, x); }
static INLINE vint vor_vi_vi_vi(vint x, vint y) { return vorrq_s32(x, y); }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return veorq_s32(x, y); }

static INLINE vint vsll_vi_vi_i(vint x, int c) { return (int32x4_t) vshlq_n_u32((uint32x4_t)x, c); }
static INLINE vint vsrl_vi_vi_i(vint x, int c) { return (int32x4_t) vshrq_n_u32((uint32x4_t)x, c); }
static INLINE vint vsra_vi_vi_i(vint x, int c) { return vshrq_n_s32(x, c); }

//

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return vandq_u32(x, y); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return vbicq_u32(y, x); }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return vorrq_u32(x, y); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return veorq_u32(x, y); }

static INLINE vmask veq_vm_vf_vf(vfloat x, vfloat y) { return vceqq_f32(x, y); }
static INLINE vmask vneq_vm_vf_vf(vfloat x, vfloat y) { return vmvnq_u32(vceqq_f32(x, y)); }
static INLINE vmask vlt_vm_vf_vf(vfloat x, vfloat y) { return vcltq_f32(x, y); }
static INLINE vmask vle_vm_vf_vf(vfloat x, vfloat y) { return vcleq_f32(x, y); }
static INLINE vmask vgt_vm_vf_vf(vfloat x, vfloat y) { return vcgtq_f32(x, y); }
static INLINE vmask vge_vm_vf_vf(vfloat x, vfloat y) { return vcgeq_f32(x, y); }

//

static INLINE vfloat vcast_vf_f(float f) { return vdupq_n_f32(f); }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return vaddq_f32(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return vsubq_f32(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return vmulq_f32(x, y); }
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vmlaq_f32(z, x, y); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vmlsq_f32(z, x, y); }

static INLINE vfloat vabs_vf_vf(vfloat f) { return vabsq_f32(f); }
static INLINE vfloat vneg_vf_vf(vfloat f) { return vnegq_f32(f); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return vmaxq_f32(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return vminq_f32(x, y); }

static INLINE vfloat vsel_vf_vm_vf_vf(vmask mask, vfloat x, vfloat y) {
  return (vfloat)vbslq_u32(mask, (vmask)x, (vmask)y);
}

static INLINE vfloat vrec_vf_vf(vfloat d) {
  float32x4_t x = vrecpeq_f32(d);
  x = vmulq_f32(x, vrecpsq_f32(d, x));
  return vmlsq_f32(vaddq_f32(x, x), vmulq_f32(x, x), d);
}

static INLINE vfloat vdiv_vf_vf_vf(vfloat n, vfloat d) {
  float32x4_t x = vrecpeq_f32(d);
  x = vmulq_f32(x, vrecpsq_f32(d, x));
  float32x4_t t = vmulq_f32(n, x);
  return vmlsq_f32(vaddq_f32(t, t), vmulq_f32(t, x), d);
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

//

static INLINE vmask veq_vm_vi_vi(vint x, vint y) { return vceqq_s32(x, y); }

//

static INLINE vint2 vcast_vi2_vm(vmask vm) { return (vint2)vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return (vmask)vi; }

static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return vcvtq_s32_f32(vf); }

static INLINE vint2 vrint_vi2_vf(vfloat d) {
  //return vcvtq_s32_f32(vrndqn_f32(d));
  return vcvtq_s32_f32(vaddq_f32(d, (float32x4_t)vorrq_u32(vandq_u32((uint32x4_t)d, (uint32x4_t)vdupq_n_f32(-0.0f)), (uint32x4_t)vdupq_n_f32(0.5f))));
}

static INLINE vfloat vcast_vf_vi2(vint2 vi) { return vcvtq_f32_s32(vi); }
static INLINE vint2 vcast_vi2_i(int i) { return vdupq_n_s32(i); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return vadd_vi_vi_vi(x, y); }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return vsub_vi_vi_vi(x, y); }
static INLINE vint vneg_vi2_vi2(vint2 e) { return vneg_vi_vi(e); }

static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return vand_vi_vi_vi(x, y); }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return vandnot_vi_vi_vi(x, y); }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return vor_vi_vi_vi(x, y); }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return vxor_vi_vi_vi(x, y); }

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return vsll_vi_vi_i(x, c); }
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return vsrl_vi_vi_i(x, c); }
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return vsra_vi_vi_i(x, c); }

static INLINE vmask veq_vm_vi2_vi2(vint2 x, vint2 y) { return vceqq_s32(x, y); }
static INLINE vmask vgt_vm_vi2_vi2(vint2 x, vint2 y) { return vcgeq_s32(x, y); }
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) { return (vint2)vcgeq_s32(x, y); }
static INLINE vint2 vsel_vi2_vm_vi2_vi2(vmask m, vint2 x, vint2 y) { return (vint2)vbslq_u32(m, (vmask)x, (vmask)y); }

//

static INLINE float vcast_f_vf(vfloat v) {
  float p[4];
  vst1q_f32 (p, v);
  return p[0];
}

static INLINE vfloat vupper_vf_vf(vfloat d) {
  return (vfloat)vandq_s32((vint)d, vdupq_n_s32(0xfffff000));
}
