#ifndef __SSE2__
#error Please specify -msse2.
#endif

#include <immintrin.h>
#include <stdint.h>

typedef __m128i vmask;
typedef __m128i vopmask;

typedef __m128d vdouble;
typedef __m128i vint;

typedef __m128  vfloat;
typedef __m128i vint2;

//

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return _mm_and_si128(x, y); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return _mm_andnot_si128(x, y); }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return _mm_or_si128(x, y); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return _mm_xor_si128(x, y); }

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) { return _mm_and_si128(x, y); }
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) { return _mm_andnot_si128(x, y); }
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) { return _mm_or_si128(x, y); }
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) { return _mm_xor_si128(x, y); }

static INLINE vmask vand_vm_vo64_vm(vopmask x, vmask y) { return _mm_and_si128(x, y); }
static INLINE vmask vor_vm_vo64_vm(vopmask x, vmask y) { return _mm_or_si128(x, y); }
static INLINE vmask vandnot_vm_vo64_vm(vmask x, vmask y) { return _mm_andnot_si128(x, y); }
static INLINE vmask vxor_vm_vo64_vm(vmask x, vmask y) { return _mm_xor_si128(x, y); }

static INLINE vmask vand_vm_vo32_vm(vopmask x, vmask y) { return _mm_and_si128(x, y); }
static INLINE vmask vor_vm_vo32_vm(vopmask x, vmask y) { return _mm_or_si128(x, y); }
static INLINE vmask vandnot_vm_vo32_vm(vmask x, vmask y) { return _mm_andnot_si128(x, y); }
static INLINE vmask vxor_vm_vo32_vm(vmask x, vmask y) { return _mm_xor_si128(x, y); }

static INLINE vopmask vcast_vo32_vo64(vopmask m) { return _mm_shuffle_epi32(m, 0x08); }
static INLINE vopmask vcast_vo64_vo32(vopmask m) { return _mm_shuffle_epi32(m, 0x50); }

//

static INLINE vint vrint_vi_vd(vdouble vd) { return _mm_cvtpd_epi32(vd); }
static INLINE vint vtruncate_vi_vd(vdouble vd) { return _mm_cvttpd_epi32(vd); }
static INLINE vdouble vcast_vd_vi(vint vi) { return _mm_cvtepi32_pd(vi); }
static INLINE vint vcast_vi_i(int i) { return _mm_set_epi32(0, 0, i, i); }
static INLINE vint2 vcastu_vi2_vi(vint vi) { return _mm_shuffle_epi32(vi, 0x73); }
static INLINE vint vcastu_vi_vi2(vint2 vi) { return _mm_shuffle_epi32(vi, 0x0d); }

static INLINE vmask vcast_vm_i_i(int i0, int i1) { return _mm_set_epi32(i0, i1, i0, i1); }

static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) {
  vmask t = _mm_cmpeq_epi32(x, y);
  return vand_vm_vm_vm(t, _mm_shuffle_epi32(t, 0xb1));
}

//

static INLINE vdouble vcast_vd_d(double d) { return _mm_set1_pd(d); }
static INLINE vmask vreinterpret_vm_vd(vdouble vd) { return (__m128i)vd; }
static INLINE vint2 vreinterpret_vi2_vd(vdouble vd) { return (__m128i)vd; }
static INLINE vdouble vreinterpret_vd_vi2(vint2 vi) { return (__m128d)vi; }
static INLINE vdouble vreinterpret_vd_vm(vmask vm) { return (__m128d)vm; }

static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) { return _mm_add_pd(x, y); }
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) { return _mm_sub_pd(x, y); }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) { return _mm_mul_pd(x, y); }
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) { return _mm_div_pd(x, y); }
static INLINE vdouble vrec_vd_vd(vdouble x) { return _mm_div_pd(_mm_set1_pd(1), x); }
static INLINE vdouble vsqrt_vd_vd(vdouble x) { return _mm_sqrt_pd(x); }
static INLINE vdouble vabs_vd_vd(vdouble d) { return (__m128d)_mm_andnot_pd(_mm_set1_pd(-0.0), d); }
static INLINE vdouble vneg_vd_vd(vdouble d) { return (__m128d)_mm_xor_pd(_mm_set1_pd(-0.0), d); }
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vsub_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) { return _mm_max_pd(x, y); }
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) { return _mm_min_pd(x, y); }

static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmpeq_pd(x, y); }
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmpneq_pd(x, y); }
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmplt_pd(x, y); }
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmple_pd(x, y); }
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmpgt_pd(x, y); }
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmpge_pd(x, y); }

static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return _mm_add_epi32(x, y); }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return _mm_sub_epi32(x, y); }
static INLINE vint vneg_vi_vi(vint e) { return vsub_vi_vi_vi(vcast_vi_i(0), e); }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return _mm_and_si128(x, y); }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return _mm_andnot_si128(x, y); }
static INLINE vint vor_vi_vi_vi(vint x, vint y) { return _mm_or_si128(x, y); }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return _mm_xor_si128(x, y); }

static INLINE vint vand_vi_vo_vi(vopmask x, vint y) { return _mm_and_si128(x, y); }
static INLINE vint vandnot_vi_vo_vi(vopmask x, vint y) { return _mm_andnot_si128(x, y); }

static INLINE vint vsll_vi_vi_i(vint x, int c) { return _mm_slli_epi32(x, c); }
static INLINE vint vsrl_vi_vi_i(vint x, int c) { return _mm_srli_epi32(x, c); }
static INLINE vint vsra_vi_vi_i(vint x, int c) { return _mm_srai_epi32(x, c); }

static INLINE vint veq_vi_vi_vi(vint x, vint y) { return _mm_cmpeq_epi32(x, y); }
static INLINE vint vgt_vi_vi_vi(vint x, vint y) { return _mm_cmpgt_epi32(x, y); }

static INLINE vopmask veq_vo_vi_vi(vint x, vint y) { return _mm_cmpeq_epi32(x, y); }
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) { return _mm_cmpgt_epi32(x, y); }

static INLINE vint vsel_vi_vo_vi_vi(vopmask m, vint x, vint y) { return vor_vm_vm_vm(vand_vm_vm_vm(m, x), vandnot_vm_vm_vm(m, y)); }

static INLINE vdouble vsel_vd_vo_vd_vd(vopmask opmask, vdouble x, vdouble y) {
  return (__m128d)vor_vm_vm_vm(vand_vm_vm_vm(opmask, (__m128i)x), vandnot_vm_vm_vm(opmask, (__m128i)y));
}

static INLINE vopmask visinf_vo_vd(vdouble d) {
  return (vopmask)_mm_cmpeq_pd(vabs_vd_vd(d), _mm_set1_pd(INFINITY));
}

static INLINE vopmask vispinf_vo_vd(vdouble d) {
  return (vopmask)_mm_cmpeq_pd(d, _mm_set1_pd(INFINITY));
}

static INLINE vopmask visminf_vo_vd(vdouble d) {
  return (vopmask)_mm_cmpeq_pd(d, _mm_set1_pd(-INFINITY));
}

static INLINE vopmask visnan_vo_vd(vdouble d) {
  return (vopmask)_mm_cmpneq_pd(d, d);
}

//

static INLINE double vcast_d_vd(vdouble v) {
  double s[2];
  _mm_storeu_pd(s, v);
  return s[0];
}

//

static INLINE vint2 vcast_vi2_vm(vmask vm) { return (vint2)vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return (vmask)vi; }
static INLINE vint2 vrint_vi2_vf(vfloat vf) { return _mm_cvtps_epi32(vf); }
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return _mm_cvttps_epi32(vf); }
static INLINE vfloat vcast_vf_vi2(vint2 vi) { return _mm_cvtepi32_ps(vcast_vm_vi2(vi)); }
static INLINE vfloat vcast_vf_f(float f) { return _mm_set1_ps(f); }
static INLINE vint2 vcast_vi2_i(int i) { return _mm_set1_epi32(i); }
static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return (__m128i)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return (__m128)vm; }
static INLINE vfloat vreinterpret_vf_vi2(vint2 vm) { return (__m128)vm; }
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) { return (__m128i)vf; }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return _mm_add_ps(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return _mm_sub_ps(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return _mm_mul_ps(x, y); }
static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return _mm_div_ps(x, y); }
static INLINE vfloat vrec_vf_vf(vfloat x) { return vdiv_vf_vf_vf(vcast_vf_f(1.0f), x); }
static INLINE vfloat vsqrt_vf_vf(vfloat x) { return _mm_sqrt_ps(x); }
static INLINE vfloat vabs_vf_vf(vfloat f) { return (vfloat)vandnot_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)f); }
static INLINE vfloat vneg_vf_vf(vfloat d) { return (vfloat)vxor_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)d); }
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vsub_vf_vf_vf(z, vmul_vf_vf_vf(x, y)); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return _mm_max_ps(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return _mm_min_ps(x, y); }

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmpeq_ps(x, y); }
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmpneq_ps(x, y); }
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmplt_ps(x, y); }
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmple_ps(x, y); }
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmpgt_ps(x, y); }
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmpge_ps(x, y); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return vadd_vi_vi_vi(x, y); }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return vsub_vi_vi_vi(x, y); }
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return vsub_vi2_vi2_vi2(vcast_vi2_i(0), e); }

static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return vand_vi_vi_vi(x, y); }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return vandnot_vi_vi_vi(x, y); }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return vor_vi_vi_vi(x, y); }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return vxor_vi_vi_vi(x, y); }

static INLINE vint2 vand_vi2_vo_vi2(vopmask x, vint2 y) { return vand_vi_vo_vi(x, y); }
static INLINE vint2 vandnot_vi2_vo_vi2(vopmask x, vint2 y) { return vandnot_vi_vo_vi(x, y); }

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return vsll_vi_vi_i(x, c); }
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return vsrl_vi_vi_i(x, c); }
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return vsra_vi_vi_i(x, c); }

static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) { return _mm_cmpeq_epi32(x, y); }
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) { return _mm_cmpgt_epi32(x, y); }
static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm_cmpeq_epi32(x, y); }
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm_cmpgt_epi32(x, y); }

static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask m, vint2 x, vint2 y) {
  return vor_vi2_vi2_vi2(vand_vi2_vi2_vi2(m, x), vandnot_vi2_vi2_vi2(m, y));
}

static INLINE vfloat vsel_vf_vo_vf_vf(vopmask mask, vfloat x, vfloat y) {
  return (vfloat)vor_vm_vm_vm(vand_vm_vm_vm(mask, (vmask)x), vandnot_vm_vm_vm(mask, (vmask)y));
}

static INLINE vopmask visinf_vo_vf(vfloat d) { return veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(INFINITYf)); }
static INLINE vopmask vispinf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(INFINITYf)); }
static INLINE vopmask visminf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(-INFINITYf)); }
static INLINE vopmask visnan_vo_vf(vfloat d) { return vneq_vo_vf_vf(d, d); }

static INLINE float vcast_f_vf(vfloat v) {
  float s[4];
  _mm_storeu_ps(s, v);
  return s[0];
}
