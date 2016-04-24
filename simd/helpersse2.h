#ifndef __SSE2__
#error Please specify -msse2.
#endif

#include <immintrin.h>
#include <stdint.h>

typedef __m128d vdouble;
typedef __m128i vint;
typedef __m128i vmask;

typedef __m128 vfloat;
typedef __m128i vint2;

//

static INLINE vint vrint_vi_vd(vdouble vd) { return _mm_cvtpd_epi32(vd); }
static INLINE vint vtruncate_vi_vd(vdouble vd) { return _mm_cvttpd_epi32(vd); }
static INLINE vdouble vcast_vd_vi(vint vi) { return _mm_cvtepi32_pd(vi); }
static INLINE vdouble vcast_vd_d(double d) { return _mm_set_pd(d, d); }
static INLINE vint vcast_vi_i(int i) { return _mm_set_epi32(0, 0, i, i); }

static INLINE vmask vreinterpret_vm_vd(vdouble vd) { return (__m128i)vd; }
static INLINE vdouble vreinterpret_vd_vm(vint vm) { return (__m128d)vm; }

static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return (__m128i)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return (__m128)vm; }

//

static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return _mm_add_epi32(x, y); }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return _mm_sub_epi32(x, y); }
static INLINE vint vneg_vi_vi(vint e) { return vsub_vi_vi_vi(vcast_vi_i(0), e); }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return _mm_and_si128(x, y); }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return _mm_andnot_si128(x, y); }
static INLINE vint vor_vi_vi_vi(vint x, vint y) { return _mm_or_si128(x, y); }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return _mm_xor_si128(x, y); }

static INLINE vint vsll_vi_vi_i(vint x, int c) { return _mm_slli_epi32(x, c); }
static INLINE vint vsrl_vi_vi_i(vint x, int c) { return _mm_srli_epi32(x, c); }
static INLINE vint vsra_vi_vi_i(vint x, int c) { return _mm_srai_epi32(x, c); }

//

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return _mm_and_si128(x, y); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return _mm_andnot_si128(x, y); }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return _mm_or_si128(x, y); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return _mm_xor_si128(x, y); }

static INLINE vmask veq_vm_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmpeq_pd(x, y); }
static INLINE vmask vneq_vm_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmpneq_pd(x, y); }
static INLINE vmask vlt_vm_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmplt_pd(x, y); }
static INLINE vmask vle_vm_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmple_pd(x, y); }
static INLINE vmask vgt_vm_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmpgt_pd(x, y); }
static INLINE vmask vge_vm_vd_vd(vdouble x, vdouble y) { return (__m128i)_mm_cmpge_pd(x, y); }

static INLINE vmask veq_vm_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmpeq_ps(x, y); }
static INLINE vmask vneq_vm_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmpneq_ps(x, y); }
static INLINE vmask vlt_vm_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmplt_ps(x, y); }
static INLINE vmask vle_vm_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmple_ps(x, y); }
static INLINE vmask vgt_vm_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmpgt_ps(x, y); }
static INLINE vmask vge_vm_vf_vf(vfloat x, vfloat y) { return (__m128i)_mm_cmpge_ps(x, y); }

//

static INLINE vfloat vcast_vf_f(float f) { return _mm_set_ps(f, f, f, f); }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return _mm_add_ps(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return _mm_sub_ps(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return _mm_mul_ps(x, y); }
static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return _mm_div_ps(x, y); }
static INLINE vfloat vrec_vf_vf(vfloat x) { return vdiv_vf_vf_vf(vcast_vf_f(1.0f), x); }
static INLINE vfloat vsqrt_vf_vf(vfloat x) { return _mm_sqrt_ps(x); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return _mm_max_ps(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return _mm_min_ps(x, y); }

static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vsub_vf_vf_vf(z, vmul_vf_vf_vf(x, y)); }
static INLINE vfloat vabs_vf_vf(vfloat f) { return (vfloat)vandnot_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)f); }
static INLINE vfloat vneg_vf_vf(vfloat d) { return (vfloat)vxor_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)d); }

//

static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) { return _mm_add_pd(x, y); }
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) { return _mm_sub_pd(x, y); }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) { return _mm_mul_pd(x, y); }
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) { return _mm_div_pd(x, y); }
static INLINE vdouble vrec_vd_vd(vdouble x) { return _mm_div_pd(_mm_set_pd(1, 1), x); }
static INLINE vdouble vsqrt_vd_vd(vdouble x) { return _mm_sqrt_pd(x); }
static INLINE vdouble vabs_vd_vd(vdouble d) { return (__m128d)_mm_andnot_pd(_mm_set_pd(-0.0,-0.0), d); }
static INLINE vdouble vneg_vd_vd(vdouble d) { return (__m128d)_mm_xor_pd(_mm_set_pd(-0.0,-0.0), d); }
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vsub_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }

static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) { return _mm_max_pd(x, y); }
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) { return _mm_min_pd(x, y); }

//

static INLINE vmask veq_vm_vi_vi(vint x, vint y) {
  __m128 s = (__m128)_mm_cmpeq_epi32(x, y);
  return (__m128i)_mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 1, 0, 0));
}

static INLINE vdouble vsel_vd_vm_vd_vd(vmask mask, vdouble x, vdouble y) {
  return (__m128d)vor_vm_vm_vm(vand_vm_vm_vm(mask, (__m128i)x), vandnot_vm_vm_vm(mask, (__m128i)y));
}

static INLINE vfloat vsel_vf_vm_vf_vf(vmask mask, vfloat x, vfloat y) {
  return (vfloat)vor_vm_vm_vm(vand_vm_vm_vm(mask, (vmask)x), vandnot_vm_vm_vm(mask, (vmask)y));
}

static INLINE vint vsel_vi_vd_vd_vi_vi(vdouble d0, vdouble d1, vint x, vint y) {
  vmask mask = (vmask)_mm_cmpeq_ps(_mm_cvtpd_ps((vdouble)vlt_vm_vd_vd(d0, d1)), _mm_set_ps(0, 0, 0, 0));
  return vor_vi_vi_vi(vandnot_vi_vi_vi(mask, x), vand_vi_vi_vi(mask, y));
}

//

static INLINE vint2 vcast_vi2_vm(vmask vm) { return (vint2)vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return (vmask)vi; }

static INLINE vint2 vrint_vi2_vf(vfloat vf) { return _mm_cvtps_epi32(vf); }
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return _mm_cvttps_epi32(vf); }
static INLINE vfloat vcast_vf_vi2(vint2 vi) { return _mm_cvtepi32_ps(vcast_vm_vi2(vi)); }
static INLINE vint2 vcast_vi2_i(int i) { return _mm_set_epi32(i, i, i, i); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return vadd_vi_vi_vi(x, y); }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return vsub_vi_vi_vi(x, y); }
static INLINE vint vneg_vi2_vi2(vint2 e) { return vsub_vi2_vi2_vi2(vcast_vi2_i(0), e); }

static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return vand_vi_vi_vi(x, y); }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return vandnot_vi_vi_vi(x, y); }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return vor_vi_vi_vi(x, y); }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return vxor_vi_vi_vi(x, y); }

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return vsll_vi_vi_i(x, c); }
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return vsrl_vi_vi_i(x, c); }
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return vsra_vi_vi_i(x, c); }

static INLINE vmask veq_vm_vi2_vi2(vint2 x, vint2 y) { return _mm_cmpeq_epi32(x, y); }
static INLINE vmask vgt_vm_vi2_vi2(vint2 x, vint2 y) { return _mm_cmpgt_epi32(x, y); }
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm_cmpgt_epi32(x, y); }
static INLINE vint2 vsel_vi2_vm_vi2_vi2(vmask m, vint2 x, vint2 y) { return vor_vm_vm_vm(vand_vm_vm_vm(m, x), vandnot_vm_vm_vm(m, y)); }

//

static INLINE double vcast_d_vd(vdouble v) {
  double s[2];
  _mm_storeu_pd(s, v);
  return s[0];
}

static INLINE float vcast_f_vf(vfloat v) {
  float s[4];
  _mm_storeu_ps(s, v);
  return s[0];
}

static INLINE vmask vsignbit_vm_vd(vdouble d) {
  return _mm_and_si128((__m128i)d, _mm_set_epi32(0x80000000, 0x0, 0x80000000, 0x0));
}

static INLINE vdouble vsign_vd_vd(vdouble d) {
  return (__m128d)_mm_or_si128((__m128i)_mm_set_pd(1, 1), _mm_and_si128((__m128i)d, _mm_set_epi32(0x80000000, 0x0, 0x80000000, 0x0)));
}

static INLINE vdouble vmulsign_vd_vd_vd(vdouble x, vdouble y) {
  return (__m128d)vxor_vi_vi_vi((__m128i)x, vsignbit_vm_vd(y));
}

static INLINE vmask visinf_vm_vd(vdouble d) {
  return (vmask)_mm_cmpeq_pd(vabs_vd_vd(d), _mm_set_pd(INFINITY, INFINITY));
}

static INLINE vmask vispinf_vm_vd(vdouble d) {
  return (vmask)_mm_cmpeq_pd(d, _mm_set_pd(INFINITY, INFINITY));
}

static INLINE vmask visminf_vm_vd(vdouble d) {
  return (vmask)_mm_cmpeq_pd(d, _mm_set_pd(-INFINITY, -INFINITY));
}

static INLINE vmask visnan_vm_vd(vdouble d) {
  return (vmask)_mm_cmpneq_pd(d, d);
}

static INLINE vdouble visinf(vdouble d) {
  return (__m128d)_mm_and_si128(visinf_vm_vd(d), _mm_or_si128(vsignbit_vm_vd(d), (__m128i)_mm_set_pd(1, 1)));
}

static INLINE vdouble visinf2(vdouble d, vdouble m) {
  return (__m128d)_mm_and_si128(visinf_vm_vd(d), _mm_or_si128(vsignbit_vm_vd(d), (__m128i)m));
}

//

static INLINE vdouble vpow2i_vd_vi(vint q) {
  q = _mm_add_epi32(_mm_set_epi32(0x0, 0x0, 0x3ff, 0x3ff), q);
  q = (__m128i)_mm_shuffle_ps((__m128)q, (__m128)q, _MM_SHUFFLE(1,3,0,3));
  return (__m128d)_mm_slli_epi32(q, 20);
}

static INLINE vdouble vldexp_vd_vd_vi(vdouble x, vint q) {
  vint m = _mm_srai_epi32(q, 31);
  m = _mm_slli_epi32(_mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(m, q), 9), m), 7);
  q = _mm_sub_epi32(q, _mm_slli_epi32(m, 2));
  m = _mm_add_epi32(_mm_set_epi32(0x0, 0x0, 0x3ff, 0x3ff), m);
  m = _mm_andnot_si128(_mm_cmplt_epi32(m, _mm_set_epi32(0, 0, 0, 0)), m);
  vint n = _mm_cmpgt_epi32(m, _mm_set_epi32(0x0, 0x0, 0x7ff, 0x7ff));
  m = _mm_or_si128(_mm_andnot_si128(n, m), _mm_and_si128(n, _mm_set_epi32(0x0, 0x0, 0x7ff, 0x7ff)));
  m = (__m128i)_mm_shuffle_ps((__m128)m, (__m128)m, _MM_SHUFFLE(1,3,0,3));
  vdouble y = (__m128d)_mm_slli_epi32(m, 20);
  return vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(x, y), y), y), y), vpow2i_vd_vi(q));
}

static INLINE vint vilogbp1_vi_vd(vdouble d) {
  vint m = vlt_vm_vd_vd(d, vcast_vd_d(4.9090934652977266E-91));
  d = vsel_vd_vm_vd_vd(m, vmul_vd_vd_vd(vcast_vd_d(2.037035976334486E90), d), d);
  __m128i q = _mm_and_si128((__m128i)d, _mm_set_epi32(((1 << 12)-1) << 20, 0, ((1 << 12)-1) << 20, 0));
  q = _mm_srli_epi32(q, 20);
  q = vor_vm_vm_vm(vand_vm_vm_vm   (m, _mm_sub_epi32(q, _mm_set_epi32(300 + 0x3fe, 0, 300 + 0x3fe, 0))),
		   vandnot_vm_vm_vm(m, _mm_sub_epi32(q, _mm_set_epi32(      0x3fe, 0,       0x3fe, 0))));
  q = (__m128i)_mm_shuffle_ps((__m128)q, (__m128)q, _MM_SHUFFLE(0,0,3,1));
  return q;
}

static INLINE vdouble vupper_vd_vd(vdouble d) {
  return (__m128d)_mm_and_si128((__m128i)d, _mm_set_epi32(0xffffffff, 0xf8000000, 0xffffffff, 0xf8000000));
}

static INLINE vfloat vupper_vf_vf(vfloat d) {
  return (__m128)_mm_and_si128((__m128i)d, _mm_set_epi32(0xfffff000, 0xfffff000, 0xfffff000, 0xfffff000));
}
