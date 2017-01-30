//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdint.h>

// When you change VECTLENDP, you also need to change the same macro in sleefsimd.h

#ifndef VECTLENDP
#define VECTLENDP 8
#endif

#ifndef VECTLENSP
#define VECTLENSP (VECTLENDP*2)
#endif

typedef uint32_t vmask __attribute__((ext_vector_type(VECTLENDP*2)));
typedef uint32_t vopmask __attribute__((ext_vector_type(VECTLENDP*2)));

typedef double vdouble __attribute__((ext_vector_type(VECTLENDP)));
typedef int32_t vint __attribute__((ext_vector_type(VECTLENDP)));

typedef float vfloat __attribute__((ext_vector_type(VECTLENDP*2)));
typedef int32_t vint2 __attribute__((ext_vector_type(VECTLENDP*2)));

//

#if VECTLENDP == 2
static INLINE vopmask vcast_vo32_vo64(vopmask m) { return (vopmask){ m[1], m[3], 0, 0 }; }
static INLINE vopmask vcast_vo64_vo32(vopmask m) { return (vopmask){ m[0], m[0], m[1], m[1] }; }

static INLINE vmask vcast_vm_i_i(int h, int l) { return (vmask){ l, h, l, h }; }
static INLINE vint2 vcastu_vi2_vi(vint vi) { return (vint2){ 0, vi[0], 0, vi[1] }; }
static INLINE vint vcastu_vi_vi2(vint2 vi2) { return (vint){ vi2[1], vi2[3] }; }

static INLINE vint vreinterpretFirstHalf_vi_vi2(vint2 vi2) { return (vint){ vi2[0], vi2[1] }; }
static INLINE vint2 vreinterpretFirstHalf_vi2_vi(vint vi) { return (vint2){ vi[0], vi[1], 0, 0 }; }
#elif VECTLENDP == 4
static INLINE vopmask vcast_vo32_vo64(vopmask m) { return (vopmask){ m[1], m[3], m[5], m[7], 0, 0, 0, 0 }; }
static INLINE vopmask vcast_vo64_vo32(vopmask m) { return (vopmask){ m[0], m[0], m[1], m[1], m[2], m[2], m[3], m[3] }; }

static INLINE vmask vcast_vm_i_i(int h, int l) { return (vmask){ l, h, l, h, l, h, l, h }; }
static INLINE vint2 vcastu_vi2_vi(vint vi) { return (vint2){ 0, vi[0], 0, vi[1], 0, vi[2], 0, vi[3] }; }
static INLINE vint vcastu_vi_vi2(vint2 vi2) { return (vint){ vi2[1], vi2[3], vi2[5], vi2[7] }; }

static INLINE vint vreinterpretFirstHalf_vi_vi2(vint2 vi2) { return (vint){ vi2[0], vi2[1], vi2[2], vi2[3] }; }
static INLINE vint2 vreinterpretFirstHalf_vi2_vi(vint vi) { return (vint2){ vi[0], vi[1], vi[2], vi[3], 0, 0, 0, 0 }; }
#elif VECTLENDP == 8
static INLINE vopmask vcast_vo32_vo64(vopmask m) { return (vopmask){ m[1], m[3], m[5], m[7], m[9], m[11], m[13], m[15], 0, 0, 0, 0, 0, 0, 0, 0 }; }
static INLINE vopmask vcast_vo64_vo32(vopmask m) { return (vopmask){ m[0], m[0], m[1], m[1], m[2], m[2], m[3], m[3], m[4], m[4], m[5], m[5], m[6], m[6], m[7], m[7] }; }

static INLINE vmask vcast_vm_i_i(int h, int l) { return (vmask){ l, h, l, h, l, h, l, h, l, h, l, h, l, h, l, h }; }
static INLINE vint2 vcastu_vi2_vi(vint vi) { return (vint2){ 0, vi[0], 0, vi[1], 0, vi[2], 0, vi[3], 0, vi[4], 0, vi[5], 0, vi[6], 0, vi[7] }; }
static INLINE vint vcastu_vi_vi2(vint2 vi2) { return (vint){ vi2[1], vi2[3], vi2[5], vi2[7], vi2[9], vi2[11], vi2[13], vi2[15] }; }

static INLINE vint vreinterpretFirstHalf_vi_vi2(vint2 vi2) { return (vint){ vi2[0], vi2[1], vi2[2], vi2[3], vi2[4], vi2[5], vi2[6], vi2[7] }; }
static INLINE vint2 vreinterpretFirstHalf_vi2_vi(vint vi) { return (vint2){ vi[0], vi[1], vi[2], vi[3], vi[4], vi[5], vi[6], vi[7], 0, 0, 0, 0, 0, 0, 0, 0 }; }
#endif

//

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return x & y; }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return y & ~x; }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return x | y; }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return x ^ y; }

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) { return x & y; }
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) { return y & ~x; }
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) { return x | y; }
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) { return x ^ y; }

static INLINE vmask vand_vm_vo64_vm(vopmask x, vmask y) { return x & y; }
static INLINE vmask vandnot_vm_vo64_vm(vopmask x, vmask y) { return y & ~x; }
static INLINE vmask vor_vm_vo64_vm(vopmask x, vmask y) { return x | y; }
static INLINE vmask vxor_vm_vo64_vm(vopmask x, vmask y) { return x ^ y; }

static INLINE vmask vand_vm_vo32_vm(vopmask x, vmask y) { return x & y; }
static INLINE vmask vandnot_vm_vo32_vm(vopmask x, vmask y) { return y & ~x; }
static INLINE vmask vor_vm_vo32_vm(vopmask x, vmask y) { return x | y; }
static INLINE vmask vxor_vm_vo32_vm(vopmask x, vmask y) { return x ^ y; }

//

static INLINE vdouble vsel_vd_vo_vd_vd(vopmask o, vdouble x, vdouble y) { return (vdouble)(((vmask)o & (vmask)x) | ((vmask)y & ~(vmask)o)); }
static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask o, vint2 x, vint2 y) { return (vint2)(((vmask)o & (vmask)x) | ((vmask)y & ~(vmask)o)); }

static INLINE vdouble vcast_vd_vi(vint vi) { return __builtin_convertvector(vi, vdouble); }
static INLINE vint vtruncate_vi_vd(vdouble vd) { return __builtin_convertvector(vd, vint); }
static INLINE vint vrint_vi_vd(vdouble vd) { return vtruncate_vi_vd(vsel_vd_vo_vd_vd((vopmask)(vd < 0.0), vd - 0.5, vd + 0.5)); }
static INLINE vdouble vtruncate_vd_vd(vdouble vd) { return vcast_vd_vi(vtruncate_vi_vd(vd)); }
static INLINE vdouble vrint_vd_vd(vdouble vd) { return vcast_vd_vi(vrint_vi_vd(vd)); }
static INLINE vint vcast_vi_i(int i) { return (vint)(i); }

static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) {
  typedef int64_t vi64 __attribute__((ext_vector_type(VECTLENDP)));
  return (vopmask)((vi64)x == (vi64)y);
}

//

static INLINE vdouble vcast_vd_d(double d) { return (vdouble)(d); }
static INLINE vmask vreinterpret_vm_vd(vdouble vd) { return (vmask)vd; }
static INLINE vint2 vreinterpret_vi2_vd(vdouble vd) { return (vint2)vd; }
static INLINE vdouble vreinterpret_vd_vi2(vint2 vi) { return (vdouble)vi; }
static INLINE vdouble vreinterpret_vd_vm(vmask vm) { return (vdouble)vm; }

static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) { return x + y; }
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) { return x - y; }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) { return x * y; }
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) { return x / y; }
static INLINE vdouble vrec_vd_vd(vdouble x) { return 1.0 / x; }

static INLINE vdouble vabs_vd_vd(vdouble d) { return (vdouble)((vmask)d & ~(vmask)(vdouble)(-0.0)); }
static INLINE vdouble vneg_vd_vd(vdouble d) { return -d; }
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return x * y + z; }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return x * y - z; }
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) { return vsel_vd_vo_vd_vd((vopmask)(x > y), x, y); }
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) { return vsel_vd_vo_vd_vd((vopmask)(x < y), x, y); }

static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)(x == y); }
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)(x != y); }
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)(x < y); }
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)(x <= y); }
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)(x > y); }
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)(x >= y); }

static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return x + y; }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return x - y; }
static INLINE vint vneg_vi_vi(vint e) { return -e; }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return x & y; }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return y & ~x; }
static INLINE vint vor_vi_vi_vi(vint x, vint y) { return x | y; }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return x ^ y; }

static INLINE vint vand_vi_vo_vi(vopmask x, vint y) { return vreinterpretFirstHalf_vi_vi2((vint2)x) & y; }
static INLINE vint vandnot_vi_vo_vi(vopmask x, vint y) { return y & ~vreinterpretFirstHalf_vi_vi2((vint2)x); }

static INLINE vint vsll_vi_vi_i(vint x, int c) {
  typedef uint32_t vu __attribute__((ext_vector_type(VECTLENDP)));
  return (vint)(((vu)x) << c);
}
static INLINE vint vsrl_vi_vi_i(vint x, int c) {
  typedef uint32_t vu __attribute__((ext_vector_type(VECTLENDP)));
  return (vint)(((vu)x) >> c);
}
static INLINE vint vsra_vi_vi_i(vint x, int c) { return x >> c; }

static INLINE vint veq_vi_vi_vi(vint x, vint y) { return x == y; }
static INLINE vint vgt_vi_vi_vi(vint x, vint y) { return x > y; }

static INLINE vopmask veq_vo_vi_vi(vint x, vint y) { return (vopmask)vreinterpretFirstHalf_vi2_vi(x == y); }
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) { return (vopmask)vreinterpretFirstHalf_vi2_vi(x > y);}

static INLINE vint vsel_vi_vo_vi_vi(vopmask m, vint x, vint y) {
  return vor_vi_vi_vi(vand_vi_vi_vi(vreinterpretFirstHalf_vi_vi2((vint2)m), x),
		      vandnot_vi_vi_vi(vreinterpretFirstHalf_vi_vi2((vint2)m), y));
}

static INLINE vopmask visinf_vo_vd(vdouble d) { return (vopmask)(vabs_vd_vd(d) == INFINITY); }
static INLINE vopmask vispinf_vo_vd(vdouble d) { return (vopmask)(d == INFINITY); }
static INLINE vopmask visminf_vo_vd(vdouble d) { return (vopmask)(d == -INFINITY); }
static INLINE vopmask visnan_vo_vd(vdouble d) { return (vopmask)(d != d); }

static INLINE vdouble vsqrt_vd_vd(vdouble d) {
  typedef int64_t vi64 __attribute__((ext_vector_type(VECTLENDP)));

  vdouble q = 1;

  vopmask o = (vopmask)(d < 8.636168555094445E-78);
  d = (vdouble)((o & (vmask)(d * 1.157920892373162E77)) | (~o & (vmask)d));
  q = (vdouble)((o & (vmask)(vdouble)(2.9387358770557188E-39)) | (~o & (vmask)(vdouble)(1)));
  q = (vdouble)vor_vm_vm_vm(vlt_vo_vd_vd(d, 0), (vmask)q);
  
  vdouble x = (vdouble)(0x5fe6ec85e7de30daLL - ((vi64)(d + 1e-320) >> 1));
  x = x * (  3 - d * x * x);
  x = x * ( 12 - d * x * x);
  x = x * (768 - d * x * x);
  x *= 1.0 / (1 << 13);
  x = (d - (d * x) * (d * x)) * (x * 0.5) + d * x;

  return x * q;
}

static INLINE double vcast_d_vd(vdouble v) { return v[0]; }

//

static INLINE vfloat vsel_vf_vo_vf_vf(vopmask o, vfloat x, vfloat y) { return (vfloat)(((vmask)o & (vmask)x) | (~(vmask)o & (vmask)y)); }

static INLINE vint2 vcast_vi2_vm(vmask vm) { return (vint2)vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return (vmask)vi; }

static INLINE vfloat vcast_vf_vi2(vint2 vi) { return __builtin_convertvector(vi, vfloat); }
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return __builtin_convertvector(vf, vint2); }
static INLINE vint2 vrint_vi2_vf(vfloat vf) { return vtruncate_vi2_vf(vsel_vf_vo_vf_vf((vopmask)(vf < 0), vf - 0.5f, vf + 0.5)); }
static INLINE vint2 vcast_vi2_i(int i) { return (vint2)(i); }
static INLINE vfloat vtruncate_vf_vf(vfloat vd) { return vcast_vf_vi2(vtruncate_vi2_vf(vd)); }

static INLINE vfloat vcast_vf_f(float f) { return (vfloat)(f); }
static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return (vmask)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return (vfloat)vm; }
static INLINE vfloat vreinterpret_vf_vi2(vint2 vi) { return (vfloat)vi; }
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) { return (vint2)vf; }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return x + y; }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return x - y; }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return x * y; }
static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return x / y; }
static INLINE vfloat vrec_vf_vf(vfloat x) { return 1.0f / x; }

static INLINE vfloat vabs_vf_vf(vfloat f) { return (vfloat)vandnot_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)f); }
static INLINE vfloat vneg_vf_vf(vfloat d) { return -d; }
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return x*y+z; }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return z-x*y; }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return vsel_vf_vo_vf_vf((vopmask)(x > y), x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return vsel_vf_vo_vf_vf((vopmask)(x < y), x, y); }

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)(x == y); }
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)(x != y); }
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)(x < y); }
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)(x <= y); }
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)(x > y); }
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)(x >= y); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return x + y; }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return x - y; }
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return -e; }

static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return x & y; }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return  y & ~x; }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return x | y; }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return x ^ y; }

static INLINE vint2 vand_vi2_vo_vi2(vopmask x, vint2 y) { return (vint2)x & y; }
static INLINE vint2 vandnot_vi2_vo_vi2(vopmask x, vint2 y) { return y & ~(vint2)x; }

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) {
  typedef uint32_t vu __attribute__((ext_vector_type(VECTLENDP*2)));
  return (vint2)(((vu)x) << c);
}
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) {
  typedef uint32_t vu __attribute__((ext_vector_type(VECTLENDP*2)));
  return (vint2)(((vu)x) >> c);
}
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return x >> c; }

static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) { return (vopmask)(x == y); }
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) { return (vopmask)(x > y); }
static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) { return x == y; }
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) { return x > y; }

static INLINE vopmask visinf_vo_vf(vfloat d) { return (vopmask)(vabs_vf_vf(d) == INFINITYf); }
static INLINE vopmask vispinf_vo_vf(vfloat d) { return (vopmask)(d == INFINITYf); }
static INLINE vopmask visminf_vo_vf(vfloat d) { return (vopmask)(d == -INFINITYf); }
static INLINE vopmask visnan_vo_vf(vfloat d) { return (vopmask)(d != d); }

static INLINE vfloat vsqrt_vf_vf(vfloat d) {
  vfloat q = 1;

  vopmask o = (vopmask)(d < 5.4210108624275221700372640043497e-20f); // 2^-64
  d = (vfloat)((o & (vmask)(d * 18446744073709551616.0f)) | (~o & (vmask)d)); // 2^64
  q = (vfloat)((o & (vmask)(vfloat)(0.00000000023283064365386962890625f)) | (~o & (vmask)(vfloat)(1))); // 2^-32
  q = (vfloat)vor_vm_vm_vm(vlt_vo_vf_vf(d, 0), (vmask)q);
  
  vfloat x = (vfloat)(0x5f330de2 - (((vint2)d) >> 1));
  x = x * ( 3.0f - d * x * x);
  x = x * (12.0f - d * x * x);
  x *= 0.0625f;
  x = (d - (d * x) * (d * x)) * (x * 0.5) + d * x;

  return x * q;
}

static INLINE float vcast_f_vf(vfloat v) { return v[0]; }
