#ifndef __AVX512F__
#error Please specify -mavx512f.
#endif

#include <immintrin.h>
#include <stdint.h>

typedef __m512i vmask;
typedef __mmask16 vopmask;

typedef __m512d vdouble;
typedef __m256i vint;

typedef __m512 vfloat;
typedef __m512i vint2;

#define ENABLE_FMA_DP
#define ENABLE_FMA_SP
#define ENABLE_GETEXP_DP
#define ENABLE_GETEXP_SP

//

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return _mm512_and_si512(x, y); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return _mm512_andnot_si512(x, y); }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return _mm512_or_si512(x, y); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return _mm512_xor_si512(x, y); }

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) { return _mm512_kand(x, y); }
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) { return _mm512_kandn(x, y); }
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) { return _mm512_kor(x, y); }
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) { return _mm512_kxor(x, y); }

static INLINE vmask vand_vm_vo64_vm(vopmask o, vmask m) { return _mm512_mask_and_epi64(_mm512_set1_epi32(0), o, m, m); }
static INLINE vmask vandnot_vm_vo64_vm(vopmask o, vmask m) { return _mm512_mask_and_epi64(m, o, _mm512_set1_epi32(0), _mm512_set1_epi32(0)); }
static INLINE vmask vor_vm_vo64_vm(vopmask o, vmask m) { return _mm512_mask_or_epi64(m, o, _mm512_set1_epi32(-1), _mm512_set1_epi32(-1)); }

static INLINE vmask vand_vm_vo32_vm(vopmask o, vmask m) { return _mm512_mask_and_epi32(_mm512_set1_epi32(0), o, m, m); }
static INLINE vmask vandnot_vm_vo32_vm(vopmask o, vmask m) { return _mm512_mask_and_epi32(m, o, _mm512_set1_epi32(0), _mm512_set1_epi32(0)); }
static INLINE vmask vor_vm_vo32_vm(vopmask o, vmask m) { return _mm512_mask_or_epi32(m, o, _mm512_set1_epi32(-1), _mm512_set1_epi32(-1)); }

static INLINE vopmask vcast_vo32_vo64(vopmask o) { return o; }
static INLINE vopmask vcast_vo64_vo32(vopmask o) { return o; }

//

static INLINE vint vrint_vi_vd(vdouble vd) {
  return _mm512_cvt_roundpd_epi32(vd, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
}

static INLINE vint vtruncate_vi_vd(vdouble vd) {
  return _mm512_cvt_roundpd_epi32(vd, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);
}

static INLINE vdouble vcast_vd_vi(vint vi) { return _mm512_cvtepi32_pd(vi); }
static INLINE vint vcast_vi_i(int i) { return _mm256_set1_epi32(i); }

static INLINE vdouble vtruncate_vd_vd(vdouble vd) { return vcast_vd_vi(vtruncate_vi_vd(vd)); }
static INLINE vdouble vrint_vd_vd(vdouble vd) { return vcast_vd_vi(vrint_vi_vd(vd)); }

static INLINE vint2 vcastu_vi2_vi(vint vi) {
  return _mm512_maskz_permutexvar_epi32(0xaaaa, _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), _mm512_castsi256_si512(vi));
}

static INLINE vint vcastu_vi_vi2(vint2 vi) {
  return _mm512_castsi512_si256(_mm512_maskz_permutexvar_epi32(0x00ff, _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1), vi));
}

static INLINE vmask vcast_vm_i_i(int i0, int i1) { return _mm512_set_epi32(i0, i1, i0, i1, i0, i1, i0, i1, i0, i1, i0, i1, i0, i1, i0, i1); }

static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) { return _mm512_cmp_epi64_mask(x, y, _MM_CMPINT_EQ); }

//

static INLINE vdouble vcast_vd_d(double d) { return _mm512_set1_pd(d); }
static INLINE vmask vreinterpret_vm_vd(vdouble vd) { return (__m512i)vd; }
static INLINE vdouble vreinterpret_vd_vm(vmask vm) { return (__m512d)vm; }
static INLINE vint2 vreinterpret_vi2_vd(vdouble vd) { return (__m512i)vd; }
static INLINE vdouble vreinterpret_vd_vi2(vint2 vi) { return (__m512d)vi; }

static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) { return _mm512_add_pd(x, y); }
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) { return _mm512_sub_pd(x, y); }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) { return _mm512_mul_pd(x, y); }
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) { return _mm512_div_pd(x, y); }
static INLINE vdouble vrec_vd_vd(vdouble x) { return _mm512_div_pd(_mm512_set1_pd(1), x); }
static INLINE vdouble vsqrt_vd_vd(vdouble x) { return _mm512_sqrt_pd(x); }
static INLINE vdouble vabs_vd_vd(vdouble d) { return (__m512d)_mm512_andnot_si512((__m512i)_mm512_set1_pd(-0.0), (__m512i)d); }
static INLINE vdouble vneg_vd_vd(vdouble d) { return (__m512d)_mm512_xor_si512((__m512i)_mm512_set1_pd(-0.0), (__m512i)d); }
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmadd_pd(x, y, z); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmsub_pd(x, y, z); }
static INLINE vdouble vmlanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fnmadd_pd(x, y, z); }
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) { return _mm512_max_pd(x, y); }
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) { return _mm512_min_pd(x, y); }

static INLINE vdouble vfma_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmadd_pd(x, y, z); }
static INLINE vdouble vfmapp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmadd_pd(x, y, z); }
static INLINE vdouble vfmapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmsub_pd(x, y, z); }
static INLINE vdouble vfmanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fnmadd_pd(x, y, z); }
static INLINE vdouble vfmann_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fnmsub_pd(x, y, z); }

static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ); }
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_NEQ_UQ); }
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_LT_OQ); }
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_LE_OQ); }
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_GT_OQ); }
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_GE_OQ); }

//

static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return _mm256_add_epi32(x, y); }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return _mm256_sub_epi32(x, y); }
static INLINE vint vneg_vi_vi(vint e) { return vsub_vi_vi_vi(vcast_vi_i(0), e); }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return _mm256_and_si256(x, y); }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return _mm256_andnot_si256(x, y); }

static INLINE vint vandnot_vi_vo_vi(vopmask o, vint y) {
  return _mm512_castsi512_si256(_mm512_mask_and_epi32(_mm512_castsi256_si512(y), o, _mm512_set1_epi32(0), _mm512_set1_epi32(0)));
}
static INLINE vint vand_vi_vo_vi(vopmask o, vint y) {
  return _mm512_castsi512_si256(_mm512_mask_and_epi32(_mm512_set1_epi32(0), o, _mm512_castsi256_si512(y), _mm512_castsi256_si512(y)));
}

static INLINE vint vor_vi_vi_vi(vint x, vint y) { return _mm256_or_si256(x, y); }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return _mm256_xor_si256(x, y); }
static INLINE vint vsll_vi_vi_i(vint x, int c) { return _mm256_slli_epi32(x, c); }
static INLINE vint vsrl_vi_vi_i(vint x, int c) { return _mm256_srli_epi32(x, c); }
static INLINE vint vsra_vi_vi_i(vint x, int c) { return _mm256_srai_epi32(x, c); }

static INLINE vint veq_vi_vi_vi(vint x, vint y) { return _mm256_cmpeq_epi32(x, y); }
static INLINE vint vgt_vi_vi_vi(vint x, vint y) { return _mm256_cmpgt_epi32(x, y); }

static INLINE vopmask veq_vo_vi_vi(vint x, vint y) {
  return _mm512_cmp_epi32_mask(_mm512_castsi256_si512(x), _mm512_castsi256_si512(y), _MM_CMPINT_EQ);
}
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) {
  return _mm512_cmp_epi32_mask(_mm512_castsi256_si512(y), _mm512_castsi256_si512(x), _MM_CMPINT_LT);
}

static INLINE vdouble vsel_vd_vo_vd_vd(vopmask mask, vdouble x, vdouble y) {
  return (__m512d)_mm512_mask_and_epi64((__m512i)y, mask, (__m512i)x, (__m512i)x);
}

static INLINE vopmask visinf_vo_vd(vdouble d) {
  return _mm512_cmp_pd_mask(vabs_vd_vd(d), _mm512_set1_pd(INFINITY), _CMP_EQ_OQ);
}

static INLINE vopmask vispinf_vo_vd(vdouble d) {
  return _mm512_cmp_pd_mask(d, _mm512_set1_pd(INFINITY), _CMP_EQ_OQ);
}

static INLINE vopmask visminf_vo_vd(vdouble d) {
  return _mm512_cmp_pd_mask(d, _mm512_set1_pd(-INFINITY), _CMP_EQ_OQ);
}

static INLINE vopmask visnan_vo_vd(vdouble d) {
  return _mm512_cmp_pd_mask(d, d, _CMP_NEQ_UQ);
}

static INLINE vint vilogbk_vi_vd(vdouble d) { return vrint_vi_vd(_mm512_getexp_pd(d)); }

static INLINE double vcast_d_vd(vdouble v) {
  double s[4];
  _mm512_storeu_pd(s, v);
  return s[0];
}

//

static INLINE vint vsel_vi_vo_vi_vi(vopmask m, vint x, vint y) {
  //return vor_vi_vi_vi(vand_vi_vi_vi(m, x), vandnot_vi_vi_vi(m, y));
  return _mm512_castsi512_si256(_mm512_mask_and_epi32(_mm512_castsi256_si512(y), m, _mm512_castsi256_si512(x), _mm512_castsi256_si512(x)));
}

//

static INLINE vint2 vcast_vi2_vm(vmask vm) { return vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return vi; }

static INLINE vint2 vrint_vi2_vf(vfloat vf) { return vcast_vi2_vm((vmask)_mm512_cvtps_epi32(vf)); }
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return vcast_vi2_vm((vmask)_mm512_cvttps_epi32(vf)); }
static INLINE vfloat vcast_vf_vi2(vint2 vi) { return _mm512_cvtepi32_ps((vmask)vcast_vm_vi2(vi)); }
static INLINE vfloat vcast_vf_f(float f) { return _mm512_set1_ps(f); }
static INLINE vint2 vcast_vi2_i(int i) { return _mm512_set1_epi32(i); }
static INLINE vfloat vtruncate_vf_vf(vfloat vd) { return vcast_vf_vi2(vtruncate_vi2_vf(vd)); }
static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return (__m512i)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return (__m512)vm; }

static INLINE vfloat vreinterpret_vf_vi2(vint2 vi) { return (vfloat)vcast_vm_vi2(vi); }
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) { return vcast_vi2_vm((vmask)vf); }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return _mm512_add_ps(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return _mm512_sub_ps(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return _mm512_mul_ps(x, y); }
static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return _mm512_div_ps(x, y); }
static INLINE vfloat vrec_vf_vf(vfloat x) { return vdiv_vf_vf_vf(vcast_vf_f(1.0f), x); }
static INLINE vfloat vsqrt_vf_vf(vfloat x) { return _mm512_sqrt_ps(x); }
static INLINE vfloat vabs_vf_vf(vfloat f) { return (vfloat)vandnot_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)f); }
static INLINE vfloat vneg_vf_vf(vfloat d) { return (vfloat)vxor_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)d); }
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmadd_ps(x, y, z); }
static INLINE vfloat vmlapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmsub_ps(x, y, z); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fnmadd_ps(x, y, z); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return _mm512_max_ps(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return _mm512_min_ps(x, y); }

static INLINE vfloat vfma_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmadd_ps(x, y, z); }
static INLINE vfloat vfmapp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmadd_ps(x, y, z); }
static INLINE vfloat vfmapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmsub_ps(x, y, z); }
static INLINE vfloat vfmanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fnmadd_ps(x, y, z); }
static INLINE vfloat vfmann_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fnmsub_ps(x, y, z); }

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ); }
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ); }
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ); }
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_LE_OQ); }
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ); }
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_add_epi32(x, y); }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_sub_epi32(x, y); }
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return vsub_vi2_vi2_vi2(vcast_vi2_i(0), e); }
static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_and_si512(x, y); }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_andnot_si512(x, y); }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_or_si512(x, y); }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_xor_si512(x, y); }

static INLINE vint2 vand_vi2_vo_vi2(vopmask o, vint2 m) {
  return _mm512_mask_and_epi32(_mm512_set1_epi32(0), o, m, m);
}

static INLINE vint2 vandnot_vi2_vo_vi2(vopmask o, vint2 m) {
  return _mm512_mask_and_epi32(m, o, _mm512_set1_epi32(0), _mm512_set1_epi32(0));
}

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return _mm512_slli_epi32(x, c); }
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return _mm512_srli_epi32(x, c); }
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return _mm512_srai_epi32(x, c); }
static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) { return _mm512_cmpeq_epi32_mask(x, y); }
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) { return _mm512_cmpgt_epi32_mask(x, y); }

static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) {
  __mmask16 m = _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_EQ);
  return _mm512_mask_and_epi32(_mm512_set1_epi32(0), m, _mm512_set1_epi32(-1), _mm512_set1_epi32(-1));
}
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) {
  __mmask16 m = _mm512_cmp_epi32_mask(y, x, _MM_CMPINT_LT);
  return _mm512_mask_and_epi32(_mm512_set1_epi32(0), m, _mm512_set1_epi32(-1), _mm512_set1_epi32(-1));
}

static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask m, vint2 x, vint2 y) {
  return _mm512_mask_and_epi32(y, m, x, x);
}

static INLINE vfloat vsel_vf_vo_vf_vf(vopmask m, vfloat x, vfloat y) {
  return (__m512)_mm512_mask_and_epi32((__m512i)y, m, (__m512i)x, (__m512i)x);
}

static INLINE vopmask visinf_vo_vf(vfloat d) { return veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(INFINITYf)); }
static INLINE vopmask vispinf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(INFINITYf)); }
static INLINE vopmask visminf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(-INFINITYf)); }
static INLINE vopmask visnan_vo_vf(vfloat d) { return vneq_vo_vf_vf(d, d); }

static INLINE vint2 vilogbk_vi2_vf(vfloat d) { return vrint_vi2_vf(_mm512_getexp_ps(d)); }

static INLINE float vcast_f_vf(vfloat v) {
  float s[8];
  _mm512_storeu_ps(s, v);
  return s[0];
}
