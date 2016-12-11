// Always use -ffp-contract=off option to compile SLEEF.

#include <assert.h>
#include <math.h>
#include <limits.h>

#if defined (__GNUC__) || defined (__INTEL_COMPILER) || defined (__clang__)
#define INLINE __attribute__((always_inline))
#else
#define INLINE inline
#endif

#include "nonnumber.h"

#ifdef ENABLE_SSE2
#include "helpersse2.h"
#endif

#ifdef ENABLE_AVX
#include "helperavx.h"
#endif

#ifdef ENABLE_AVX2
#include "helperavx2.h"
#endif

#ifdef ENABLE_AVX512F
#include "helperavx512f.h"
#endif

#ifdef ENABLE_FMA4
#include "helperfma4.h"
#endif

#ifdef ENABLE_NEON64
#include "helperneon64.h"
#endif

#ifdef ENABLE_CLANGVEC
#include "helperclangvec.h"
#endif

//

#include "dd.h"

//

#define PI_A 3.1415926218032836914
#define PI_B 3.1786509424591713469e-08
#define PI_C 1.2246467864107188502e-16
#define PI_D 1.2736634327021899816e-24

#define PI4_A 0.78539816290140151978
#define PI4_B 4.9604678871439933374e-10
#define PI4_C 1.1258708853173288931e-18
#define PI4_D 1.7607799325916000908e-27

#define M_2_PI_H 0.63661977236758138243
#define M_2_PI_L -3.9357353350364971764e-17

#define TRIGRANGEMAX 1e+14
#define SQRT_DBL_MAX 1.3407807929942596355e+154

#define M_4_PI 1.273239544735162542821171882678754627704620361328125

#define L2U .69314718055966295651160180568695068359375
#define L2L .28235290563031577122588448175013436025525412068e-12
#define R_LN2 1.442695040888963407359924681001892137426645954152985934135449406931

//

static INLINE vopmask vsignbit_vo_vd(vdouble d) {
  return veq64_vo_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
}

// return d0 < d1 ? x : y
static INLINE vint vsel_vi_vd_vd_vi_vi(vdouble d0, vdouble d1, vint x, vint y) { return vsel_vi_vo_vi_vi(vcast_vo32_vo64(vlt_vo_vd_vd(d0, d1)), x, y); } 

// return d0 < 0 ? x : 0
static INLINE vint vsel_vi_vd_vi(vdouble d, vint x) { return vand_vi_vo_vi(vcast_vo32_vo64(vsignbit_vo_vd(d)), x); }

static INLINE vopmask visnegzero_vo_vd(vdouble d) {
  return veq64_vo_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
}

static INLINE vmask vsignbit_vm_vd(vdouble d) {
  return vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
}

static INLINE vdouble vmulsign_vd_vd_vd(vdouble x, vdouble y) {
  return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(x), vsignbit_vm_vd(y)));
}

static INLINE vdouble vsign_vd_vd(vdouble d) {
  return vmulsign_vd_vd_vd(vcast_vd_d(1.0), d);
}

static INLINE vdouble vpow2i_vd_vi(vint q) {
  q = vadd_vi_vi_vi(vcast_vi_i(0x3ff), q);
  vint2 r = vcastu_vi2_vi(q);
  return vreinterpret_vd_vi2(vsll_vi2_vi2_i(r, 20));
}

static INLINE vdouble vldexp_vd_vd_vi(vdouble x, vint q) {
  vint m = vsra_vi_vi_i(q, 31);
  m = vsll_vi_vi_i(vsub_vi_vi_vi(vsra_vi_vi_i(vadd_vi_vi_vi(m, q), 9), m), 7);
  q = vsub_vi_vi_vi(q, vsll_vi_vi_i(m, 2));
  m = vadd_vi_vi_vi(vcast_vi_i(0x3ff), m);
  m = vandnot_vi_vo_vi(vgt_vo_vi_vi(vcast_vi_i(0), m), m);
  m = vsel_vi_vo_vi_vi(vgt_vo_vi_vi(m, vcast_vi_i(0x7ff)), vcast_vi_i(0x7ff), m);
  vint2 r = vcastu_vi2_vi(m);
  vdouble y = vreinterpret_vd_vi2(vsll_vi2_vi2_i(r, 20));
  return vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(x, y), y), y), y), vpow2i_vd_vi(q));
}

#ifndef ENABLE_GETEXP_DP
static INLINE vint vilogbk_vi_vd(vdouble d) {
  vopmask o = vlt_vo_vd_vd(d, vcast_vd_d(4.9090934652977266E-91));
  d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(vcast_vd_d(2.037035976334486E90), d), d);
  vint q = vcastu_vi_vi2(vreinterpret_vi2_vd(d));
  q = vand_vi_vi_vi(q, vcast_vi_i(((1 << 12)-1) << 20));
  q = vsrl_vi_vi_i(q, 20);
  q = vsub_vi_vi_vi(q, vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vcast_vi_i(300 + 0x3ff), vcast_vi_i(0x3ff)));
  return q;
}
#endif

static INLINE vopmask visint(vdouble d) {
  vdouble x = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0 / (1 << 31))));
  x = vmla_vd_vd_vd_vd(vcast_vd_d(-(double)(1 << 31)), x, d);
  return vor_vo_vo_vo(veq_vo_vd_vd(vtruncate_vd_vd(x), x),
		      vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1LL << 52)));
}

//

vdouble xldexp(vdouble x, vint q) { return vldexp_vd_vd_vi(x, q); }

vint xilogb(vdouble d) {
  vdouble e = vcast_vd_vi(vilogbk_vi_vd(vabs_vd_vd(d)));
  e = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(FP_ILOGB0), e);
  e = vsel_vd_vo_vd_vd(visnan_vo_vd(d), vcast_vd_d(FP_ILOGBNAN), e);
  e = vsel_vd_vo_vd_vd(visinf_vo_vd(d), vcast_vd_d(INT_MAX), e);
  return vrint_vi_vd(e);
}

vdouble xsin(vdouble d) {
  vdouble u, s, r = d;
#if 0
  vint ql = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)));

  u = vcast_vd_vi(ql);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_A*4), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_B*4), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_C*4), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_D*4), d);
#else
  vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 24))));
  vint ql = vrint_vi_vd(vsub_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)),
				      vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24))));
  vdouble dql = vcast_vd_vi(ql);

  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * (1 << 24)), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A            ), d);
  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_B * (1 << 24)), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_B            ), d);
  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_C * (1 << 24)), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_C            ), d);
  d = vmla_vd_vd_vd_vd(vmla_vd_vd_vd_vd(dqh, vcast_vd_d(1 << 24), dql),
		       vcast_vd_d(-PI_D), d);
#endif
  
  s = vmul_vd_vd_vd(d, d);

  d = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1))), (vmask)vcast_vd_d(-0.0)), (vmask)d));

  u = vcast_vd_d(-7.97255955009037868891952e-18);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.81009972710863200091251e-15));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-7.64712219118158833288484e-13));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(1.60590430605664501629054e-10));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.50521083763502045810755e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573192239198747630416e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000198412698412696162806809));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00833333333333332974823815));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.166666666666666657414808));

  u = vadd_vd_vd_vd(vmul_vd_vd_vd(s, vmul_vd_vd_vd(u, d)), d);

  u = vsel_vd_vo_vd_vd(vandnot_vo_vo_vo(visinf_vo_vd(r),
					vor_vo_vo_vo(visnegzero_vo_vd(r),
						     vgt_vo_vd_vd(vabs_vd_vd(r), vcast_vd_d(TRIGRANGEMAX)))),
		       vcast_vd_d(-0.0), u);
  
  return u;
}

vdouble xsin_u1(vdouble d) {
  vdouble u;
  vdouble2 s, t, x;
#if 0
  vint ql = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)));
  u = vcast_vd_vi(ql);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_A*4)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_B*4)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_C*4)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_D*4)));
#else
  vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 24))));
  vint ql = vrint_vi_vd(vsub_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)),
				      vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24))));
  vdouble dql = vcast_vd_vi(ql);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_A * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(vmla_vd_vd_vd_vd(dqh, vcast_vd_d(1 << 24), dql),
					 vcast_vd_d(-PI_D)));
#endif
  
  t = s;
  s = ddsqu_vd2_vd2(s);

  u = vcast_vd_d(2.72052416138529567917983e-15);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-7.6429259411395447190023e-13));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(1.60589370117277896211623e-10));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.5052106814843123359368e-08));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.75573192104428224777379e-06));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.000198412698412046454654947));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00833333333333318056201922));

  x = ddadd_vd2_vd_vd2(vcast_vd_d(1), ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(-0.166666666666666657414808), vmul_vd_vd_vd(u, s.x)), s));

  x = ddmul_vd2_vd2_vd2(t, x);
  u = vadd_vd_vd_vd(x.x, x.y);

  u = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1))), (vmask)vcast_vd_d(-0.0)), (vmask)u));
  u = vsel_vd_vo_vd_vd(vandnot_vo_vo_vo(visinf_vo_vd(d), vor_vo_vo_vo(visnegzero_vo_vd(d),
								      vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX)))),
		       vcast_vd_d(-0.0), u);

  return u;
}

vdouble xcos(vdouble d) {
  vdouble u, s, r = d;
#if 0
  vint ql = vrint_vi_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI), vcast_vd_d(-0.5)));
  ql = vadd_vi_vi_vi(vadd_vi_vi_vi(ql, ql), vcast_vi_i(1));

  u = vcast_vd_vi(ql);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_A*2), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_B*2), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_C*2), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_D*2), d);
#else
  vdouble dqh = vtruncate_vd_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 23)), vcast_vd_d(-0.5 * M_1_PI / (1 << 23))));
  vint ql = vrint_vi_vd(vadd_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)),
				      vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-(1 << 23)), vcast_vd_d(-0.5))));
  ql = vadd_vi_vi_vi(vadd_vi_vi_vi(ql, ql), vcast_vi_i(1));
  vdouble dql = vcast_vd_vi(ql);

  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5 * (1 << 24)), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A * 0.5            ), d);
  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_B * 0.5 * (1 << 24)), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_B * 0.5            ), d);
  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_C * 0.5 * (1 << 24)), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_C * 0.5            ), d);
  d = vmla_vd_vd_vd_vd(vmla_vd_vd_vd_vd(dqh, vcast_vd_d(1 << 24), dql),
		       vcast_vd_d(-PI_D * 0.5), d);
#endif
  s = vmul_vd_vd_vd(d, d);

  d = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(2)), vcast_vi_i(0))), (vmask)vcast_vd_d(-0.0)), (vmask)d));

  u = vcast_vd_d(-7.97255955009037868891952e-18);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.81009972710863200091251e-15));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-7.64712219118158833288484e-13));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(1.60590430605664501629054e-10));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.50521083763502045810755e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573192239198747630416e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000198412698412696162806809));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00833333333333332974823815));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.166666666666666657414808));

  u = vadd_vd_vd_vd(vmul_vd_vd_vd(s, vmul_vd_vd_vd(u, d)), d);

  u = (vdouble)vandnot_vm_vo64_vm(vandnot_vo_vo_vo(visinf_vo_vd(r), vgt_vo_vd_vd(vabs_vd_vd(r), vcast_vd_d(TRIGRANGEMAX))),
				  (vmask)u);
  
  return u;
}

vdouble xcos_u1(vdouble d) {
  vdouble u;
  vdouble2 s, t, x;
#if 0
  vint ql = vrint_vi_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI), vcast_vd_d(-0.5)));
  ql = vadd_vi_vi_vi(vadd_vi_vi_vi(ql, ql), vcast_vi_i(1));
  u = vcast_vd_vi(ql);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_A*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_B*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_C*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_D*2)));
#else
  vdouble dqh = vtruncate_vd_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 23)), vcast_vd_d(-0.5 * M_1_PI / (1 << 23))));
  vint ql = vrint_vi_vd(vadd_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)),
				      vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-(1 << 23)), vcast_vd_d(-0.5))));
  ql = vadd_vi_vi_vi(vadd_vi_vi_vi(ql, ql), vcast_vi_i(1));
  vdouble dql = vcast_vd_vi(ql);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_A*0.5 * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A*0.5            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B*0.5 * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B*0.5            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C*0.5 * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C*0.5            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(vmla_vd_vd_vd_vd(dqh, vcast_vd_d(1 << 24), dql),
					 vcast_vd_d(-PI_D*0.5)));
#endif
  
  t = s;
  s = ddsqu_vd2_vd2(s);

  u = vcast_vd_d(2.72052416138529567917983e-15);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-7.6429259411395447190023e-13));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(1.60589370117277896211623e-10));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.5052106814843123359368e-08));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.75573192104428224777379e-06));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.000198412698412046454654947));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00833333333333318056201922));

  x = ddadd_vd2_vd_vd2(vcast_vd_d(1), ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(-0.166666666666666657414808), vmul_vd_vd_vd(u, s.x)), s));

  x = ddmul_vd2_vd2_vd2(t, x);
  u = vadd_vd_vd_vd(x.x, x.y);

  u = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(2)), vcast_vi_i(0))), (vmask)vcast_vd_d(-0.0)), (vmask)u));

  u = (vdouble)vandnot_vm_vo64_vm(vandnot_vo_vo_vo(visinf_vo_vd(d), vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX))),
				  (vmask)u);
  
  return u;
}

vdouble2 xsincos(vdouble d) {
  vopmask o;
  vdouble u, s, t, rx, ry;
  vdouble2 r;

  s = d;
#if 0
  vint ql = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_2_PI)));

  u = vcast_vd_vi(ql);
  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_A*2), s);
  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_B*2), s);
  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_C*2), s);
  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_D*2), s);
#else
  vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI / (1 << 24))));
  vint ql = vrint_vi_vd(vsub_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI)),
				      vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24))));
  vdouble dql = vcast_vd_vi(ql);

  s = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5 * (1 << 24)), s);
  s = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A * 0.5            ), s);
  s = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_B * 0.5 * (1 << 24)), s);
  s = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_B * 0.5            ), s);
  s = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_C * 0.5 * (1 << 24)), s);
  s = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_C * 0.5            ), s);
  s = vmla_vd_vd_vd_vd(vmla_vd_vd_vd_vd(dqh, vcast_vd_d(1 << 24), dql),
		       vcast_vd_d(-PI_D * 0.5), s);
#endif
  
  t = s;

  s = vmul_vd_vd_vd(s, s);

  u = vcast_vd_d(1.58938307283228937328511e-10);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.50506943502539773349318e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573131776846360512547e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000198412698278911770864914));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0083333333333191845961746));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.166666666666666130709393));
  u = vmul_vd_vd_vd(vmul_vd_vd_vd(u, s), t);

  rx = vadd_vd_vd_vd(t, u);
  rx = vsel_vd_vo_vd_vd(visnegzero_vo_vd(d), vcast_vd_d(-0.0), rx);

  u = vcast_vd_d(-1.13615350239097429531523e-11);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.08757471207040055479366e-09));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.75573144028847567498567e-07));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.48015872890001867311915e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.00138888888888714019282329));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0416666666666665519592062));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.5));

  ry = vmla_vd_vd_vd_vd(s, u, vcast_vd_d(1));

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(0)));
  r.x = vsel_vd_vo_vd_vd(o, rx, ry);
  r.y = vsel_vd_vo_vd_vd(o, ry, rx);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(2)), vcast_vi_i(2)));
  r.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.x)));

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vadd_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(2)), vcast_vi_i(2)));
  r.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.y)));

  o = vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX));
  r.x = (vdouble)vandnot_vm_vo64_vm(o, (vmask)r.x);
  r.y = (vdouble)vandnot_vm_vo64_vm(o, (vmask)r.y);
  
  o = visinf_vo_vd(d);
  r.x = (vdouble)vor_vm_vo64_vm(o, (vmask)r.x);
  r.y = (vdouble)vor_vm_vo64_vm(o, (vmask)r.y);

  return r;
}

vdouble2 xsincos_u1(vdouble d) {
  vopmask o;
  vdouble u, rx, ry;
  vdouble2 r, s, t, x;
#if 0
  vint ql = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(2 * M_1_PI)));
  u = vcast_vd_vi(ql);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_A*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_B*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_C*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_D*2)));
#else
  vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI / (1 << 24))));
  vint ql = vrint_vi_vd(vsub_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI)),
				      vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24))));
  vdouble dql = vcast_vd_vi(ql);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_A*0.5 * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A*0.5            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B*0.5 * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B*0.5            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C*0.5 * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C*0.5            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(vmla_vd_vd_vd_vd(dqh, vcast_vd_d(1 << 24), dql),
					 vcast_vd_d(-PI_D*0.5)));
#endif
  
  t = s;

  s = ddsqu_vd2_vd2(s);
  s.x = vadd_vd_vd_vd(s.x, s.y);

  u = vcast_vd_d(1.58938307283228937328511e-10);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.50506943502539773349318e-08));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.75573131776846360512547e-06));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.000198412698278911770864914));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0083333333333191845961746));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.166666666666666130709393));

  u = vmul_vd_vd_vd(u, vmul_vd_vd_vd(s.x, t.x));

  x = ddadd_vd2_vd2_vd(t, u);
  rx = vadd_vd_vd_vd(x.x, x.y);

  rx = vsel_vd_vo_vd_vd(visnegzero_vo_vd(d), vcast_vd_d(-0.0), rx);
  
  u = vcast_vd_d(-1.13615350239097429531523e-11);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.08757471207040055479366e-09));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.75573144028847567498567e-07));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.48015872890001867311915e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.00138888888888714019282329));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0416666666666665519592062));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.5));

  x = ddadd_vd2_vd_vd2(vcast_vd_d(1), ddmul_vd2_vd_vd(s.x, u));
  ry = vadd_vd_vd_vd(x.x, x.y);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(0)));
  r.x = vsel_vd_vo_vd_vd(o, rx, ry);
  r.y = vsel_vd_vo_vd_vd(o, ry, rx);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(2)), vcast_vi_i(2)));
  r.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.x)));

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vadd_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(2)), vcast_vi_i(2)));
  r.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.y)));

  o = vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX));
  r.x = (vdouble)vandnot_vm_vo64_vm(o, (vmask)r.x);
  r.y = (vdouble)vandnot_vm_vo64_vm(o, (vmask)r.y);

  o = visinf_vo_vd(d);
  r.x = (vdouble)vor_vm_vo64_vm(o, (vmask)r.x);
  r.y = (vdouble)vor_vm_vo64_vm(o, (vmask)r.y);

  return r;
}

vdouble xtan(vdouble d) {
  vdouble u, s, x;
  vopmask o;
#if 0
  vint ql = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_2_PI)));

  u = vcast_vd_vi(ql);
  x = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_A*2), d);
  x = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_B*2), x);
  x = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_C*2), x);
  x = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_D*2), x);
#else
  vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI / (1 << 24))));
  vint ql = vrint_vi_vd(vsub_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI)),
				      vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24))));
  vdouble dql = vcast_vd_vi(ql);

  x = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5 * (1 << 24)), d);
  x = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A * 0.5            ), x);
  x = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_B * 0.5 * (1 << 24)), x);
  x = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_B * 0.5            ), x);
  x = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_C * 0.5 * (1 << 24)), x);
  x = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_C * 0.5            ), x);
  x = vmla_vd_vd_vd_vd(vmla_vd_vd_vd_vd(dqh, vcast_vd_d(1 << 24), dql),
		       vcast_vd_d(-PI_D * 0.5), x);
#endif
  
  s = vmul_vd_vd_vd(x, x);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1)));
  x = (vdouble)vxor_vm_vm_vm(vand_vm_vo64_vm(o, (vmask)vcast_vd_d(-0.0)), (vmask)x);

  u = vcast_vd_d(9.99583485362149960784268e-06);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-4.31184585467324750724175e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000103573238391744000389851));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000137892809714281708733524));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000157624358465342784274554));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-6.07500301486087879295969e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000148898734751616411290179));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000219040550724571513561967));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000595799595197098359744547));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00145461240472358871965441));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0035923150771440177410343));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00886321546662684547901456));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0218694899718446938985394));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0539682539049961967903002));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.133333333334818976423364));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.333333333333320047664472));

  u = vmla_vd_vd_vd_vd(s, vmul_vd_vd_vd(u, x), x);

  u = vsel_vd_vo_vd_vd(o, vrec_vd_vd(u), u);

  u = (vdouble)vor_vm_vo64_vm(visinf_vo_vd(d), (vmask)u);
  u = vsel_vd_vo_vd_vd(visnegzero_vo_vd(d), vcast_vd_d(-0.0), u);

  return u;
}

vdouble xtan_u1(vdouble d) {
  vdouble u;
  vdouble2 s, t, x;
  vopmask o;
#if 0
  vint ql = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_2_PI)));

  u = vcast_vd_vi(ql);
  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_A*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_B*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_C*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_D*2)));
#else
  vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI / (1 << 24))));
  s = ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd(vcast_vd2_d_d(M_2_PI_H, M_2_PI_L), d),
			vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-(double)(1 << 24)),
					 vsel_vd_vo_vd_vd(vlt_vo_vd_vd(d, vcast_vd_d(0)),
							  vcast_vd_d(-0.5), vcast_vd_d(0.5))));
  vint ql = vtruncate_vi_vd(vadd_vd_vd_vd(s.x, s.y));
  vdouble dql = vcast_vd_vi(ql);
  
  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_A*0.5 * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A*0.5            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B*0.5 * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B*0.5            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C*0.5 * (1 << 24))));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C*0.5            )));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(vmla_vd_vd_vd_vd(dqh, vcast_vd_d(1 << 24), dql),
					 vcast_vd_d(-PI_D*0.5)));
#endif
  
  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1)));
  vmask n = vand_vm_vo64_vm(o, (vmask)vcast_vd_d(-0.0));
  s.x = (vdouble)vxor_vm_vm_vm((vmask)s.x, n);
  s.y = (vdouble)vxor_vm_vm_vm((vmask)s.y, n);

  t = s;
  s = ddsqu_vd2_vd2(s);

  u = vcast_vd_d(1.01419718511083373224408e-05);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.59519791585924697698614e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(5.23388081915899855325186e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-3.05033014433946488225616e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(7.14707504084242744267497e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(8.09674518280159187045078e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.000244884931879331847054404));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.000588505168743587154904506));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00145612788922812427978848));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00359208743836906619142924));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00886323944362401618113356));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0218694882853846389592078));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0539682539781298417636002));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.133333333333125941821962));

  x = ddadd_vd2_vd_vd2(vcast_vd_d(1), ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(0.333333333333334980164153), vmul_vd_vd_vd(u, s.x)), s));
  x = ddmul_vd2_vd2_vd2(t, x);

  x = vsel_vd2_vo_vd2_vd2(o, ddrec_vd2_vd2(x), x);

  u = vadd_vd_vd_vd(x.x, x.y);

  u = vsel_vd_vo_vd_vd(vandnot_vo_vo_vo(visinf_vo_vd(d),
					vor_vo_vo_vo(visnegzero_vo_vd(d),
						     vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX)))),
		       vcast_vd_d(-0.0), u);

  return u;
}

static INLINE vdouble atan2k(vdouble y, vdouble x) {
  vdouble s, t, u;
  vint q;
  vopmask p;

  q = vsel_vi_vd_vi(x, vcast_vi_i(-2));
  x = vabs_vd_vd(x);

  q = vsel_vi_vd_vd_vi_vi(x, y, vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
  p = vlt_vo_vd_vd(x, y);
  s = vsel_vd_vo_vd_vd(p, vneg_vd_vd(x), y);
  t = vmax_vd_vd_vd(x, y);

  s = vdiv_vd_vd_vd(s, t);
  t = vmul_vd_vd_vd(s, s);

  u = vcast_vd_d(-1.88796008463073496563746e-05);
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.000209850076645816976906797));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.00110611831486672482563471));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.00370026744188713119232403));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.00889896195887655491740809));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.016599329773529201970117));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0254517624932312641616861));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0337852580001353069993897));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0407629191276836500001934));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0466667150077840625632675));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0523674852303482457616113));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0587666392926673580854313));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0666573579361080525984562));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0769219538311769618355029));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.090908995008245008229153));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.111111105648261418443745));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.14285714266771329383765));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.199999999996591265594148));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.333333333333311110369124));

  t = vmla_vd_vd_vd_vd(s, vmul_vd_vd_vd(t, u), s);
  t = vmla_vd_vd_vd_vd(vcast_vd_vi(q), vcast_vd_d(M_PI/2), t);

  return t;
}

static INLINE vdouble2 atan2k_u1(vdouble2 y, vdouble2 x) {
  vdouble u;
  vdouble2 s, t;
  vint q;
  vopmask p;

  q = vsel_vi_vd_vi(x.x, vcast_vi_i(-2));
  p = vlt_vo_vd_vd(x.x, vcast_vd_d(0));
  vmask b = vand_vm_vo64_vm(p, (vmask)vcast_vd_d(-0.0));
  x.x = (vdouble)vxor_vm_vm_vm(b, (vmask)x.x);
  x.y = (vdouble)vxor_vm_vm_vm(b, (vmask)x.y);

  q = vsel_vi_vd_vd_vi_vi(x.x, y.x, vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
  p = vlt_vo_vd_vd(x.x, y.x);
  s = vsel_vd2_vo_vd2_vd2(p, ddneg_vd2_vd2(x), y);
  t = vsel_vd2_vo_vd2_vd2(p, y, x);

  s = dddiv_vd2_vd2_vd2(s, t);
  t = ddsqu_vd2_vd2(s);
  t = ddnormalize_vd2_vd2(t);

  u = vcast_vd_d(1.06298484191448746607415e-05);
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.000125620649967286867384336));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.00070557664296393412389774));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.00251865614498713360352999));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.00646262899036991172313504));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0128281333663399031014274));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0208024799924145797902497));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0289002344784740315686289));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0359785005035104590853656));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.041848579703592507506027));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0470843011653283988193763));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0524914210588448421068719));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0587946590969581003860434));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0666620884778795497194182));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0769225330296203768654095));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0909090442773387574781907));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.111111108376896236538123));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.142857142756268568062339));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.199999999997977351284817));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.333333333333317605173818));

  t = ddmul_vd2_vd2_vd(t, u);
  t = ddmul_vd2_vd2_vd2(s, ddadd_vd2_vd_vd2(vcast_vd_d(1), t));
  t = ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_d_d(1.570796326794896557998982, 6.12323399573676603586882e-17), vcast_vd_vi(q)), t);

  return t;
}

static INLINE vdouble visinf2_vd_vd_vd(vdouble d, vdouble m) {
  return vreinterpret_vd_vm(vand_vm_vo64_vm(visinf_vo_vd(d), vor_vm_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(m))));
}

vdouble xatan2(vdouble y, vdouble x) {
  vdouble r = atan2k(vabs_vd_vd(y), x);

  r = vmulsign_vd_vd_vd(r, x);
  r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, vcast_vd_d(0))), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/2), x))), r);
  r = vsel_vd_vo_vd_vd(visinf_vo_vd(y), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/4), x))), r);
  r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(y, vcast_vd_d(0.0)), (vdouble)vand_vm_vo64_vm(vsignbit_vo_vd(x), (vmask)vcast_vd_d(M_PI)), r);

  r = (vdouble)vor_vm_vo64_vm(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), (vmask)vmulsign_vd_vd_vd(r, y));
  return r;
}

vdouble xatan2_u1(vdouble y, vdouble x) {
  vdouble2 d = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(y), vcast_vd_d(0)), vcast_vd2_vd_vd(x, vcast_vd_d(0)));
  vdouble r = vadd_vd_vd_vd(d.x, d.y);

  r = vmulsign_vd_vd_vd(r, x);
  r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, vcast_vd_d(0))), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/2), x))), r);
  r = vsel_vd_vo_vd_vd(visinf_vo_vd(y), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/4), x))), r);
  r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(y, vcast_vd_d(0.0)), (vdouble)vand_vm_vo64_vm(vsignbit_vo_vd(x), (vmask)vcast_vd_d(M_PI)), r);

  r = (vdouble)vor_vm_vo64_vm(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), (vmask)vmulsign_vd_vd_vd(r, y));
  return r;
}

vdouble xasin(vdouble d) {
  vdouble x, y;
  x = vadd_vd_vd_vd(vcast_vd_d(1), d);
  y = vsub_vd_vd_vd(vcast_vd_d(1), d);
  x = vmul_vd_vd_vd(x, y);
  x = vsqrt_vd_vd(x);
  x = (vdouble)vor_vm_vo64_vm(visnan_vo_vd(x), (vmask)atan2k(vabs_vd_vd(d), x));
  return vmulsign_vd_vd_vd(x, d);
}

vdouble xasin_u1(vdouble d) {
  vdouble2 d2 = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0)), ddsqrt_vd2_vd2(ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(1), d), ddsub_vd2_vd_vd(vcast_vd_d(1), d))));
  vdouble r = vadd_vd_vd_vd(d2.x, d2.y);
  r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1)), vcast_vd_d(1.570796326794896557998982), r);
  return vmulsign_vd_vd_vd(r, d);
}

vdouble xacos(vdouble d) {
  vdouble x, y;
  x = vadd_vd_vd_vd(vcast_vd_d(1), d);
  y = vsub_vd_vd_vd(vcast_vd_d(1), d);
  x = vmul_vd_vd_vd(x, y);
  x = vsqrt_vd_vd(x);
  x = vmulsign_vd_vd_vd(atan2k(x, vabs_vd_vd(d)), d);
  y = (vdouble)vand_vm_vo64_vm(vsignbit_vo_vd(d), (vmask)vcast_vd_d(M_PI));
  x = vadd_vd_vd_vd(x, y);
  return x;
}

vdouble xacos_u1(vdouble d) {
  vdouble2 d2 = atan2k_u1(ddsqrt_vd2_vd2(ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(1), d), ddsub_vd2_vd_vd(vcast_vd_d(1), d))), vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0)));
  d2 = ddscale_vd2_vd2_vd(d2, vmulsign_vd_vd_vd(vcast_vd_d(1), d));

  vopmask o;
  o = vneq_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1));
  d2.x = (vdouble)vand_vm_vo64_vm(o, (vmask)d2.x);
  d2.y = (vdouble)vand_vm_vo64_vm(o, (vmask)d2.y);
  o = vsignbit_vo_vd(d);
  d2 = vsel_vd2_vo_vd2_vd2(o, ddadd_vd2_vd2_vd2(vcast_vd2_d_d(3.141592653589793116, 1.2246467991473532072e-16), d2), d2);

  return vadd_vd_vd_vd(d2.x, d2.y);
}

vdouble xatan_u1(vdouble d) {
  vdouble2 d2 = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0)), vcast_vd2_d_d(1, 0));
  vdouble r = vadd_vd_vd_vd(d2.x, d2.y);
  r = vsel_vd_vo_vd_vd(visinf_vo_vd(d), vcast_vd_d(1.570796326794896557998982), r);
  return vmulsign_vd_vd_vd(r, d);
}

vdouble xatan(vdouble s) {
  vdouble t, u;
  vint q;

  q = vsel_vi_vd_vi(s, vcast_vi_i(2));
  s = vabs_vd_vd(s);

  q = vsel_vi_vd_vd_vi_vi(vcast_vd_d(1), s, vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
  s = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(vcast_vd_d(1), s), vrec_vd_vd(s), s);

  t = vmul_vd_vd_vd(s, s);

  u = vcast_vd_d(-1.88796008463073496563746e-05);
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.000209850076645816976906797));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.00110611831486672482563471));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.00370026744188713119232403));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.00889896195887655491740809));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.016599329773529201970117));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0254517624932312641616861));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0337852580001353069993897));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0407629191276836500001934));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0466667150077840625632675));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0523674852303482457616113));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0587666392926673580854313));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0666573579361080525984562));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0769219538311769618355029));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.090908995008245008229153));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.111111105648261418443745));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.14285714266771329383765));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.199999999996591265594148));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.333333333333311110369124));

  t = vmla_vd_vd_vd_vd(s, vmul_vd_vd_vd(t, u), s);

  t = vsel_vd_vo_vd_vd(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1))), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), t), t);
  t = (vdouble)vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(2))), (vmask)vcast_vd_d(-0.0)), (vmask)t);

  return t;
}

vdouble xlog(vdouble d) {
  vdouble x, x2;
  vdouble t, m;

#ifndef ENABLE_AVX512F
  vint e = vilogbk_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  m = vldexp_vd_vd_vi(d, vneg_vi_vi(e));
#else
  vdouble e = _mm512_getexp_pd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  e = vsel_vd_vo_vd_vd(vispinf_vo_vd(e), vcast_vd_d(1024.0), e);
  m = _mm512_getmant_pd(d, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan);
#endif
  
  x = vdiv_vd_vd_vd(vadd_vd_vd_vd(vcast_vd_d(-1), m), vadd_vd_vd_vd(vcast_vd_d(1), m));
  x2 = vmul_vd_vd_vd(x, x);

  t = vcast_vd_d(0.153487338491425068243146);
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.152519917006351951593857));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.181863266251982985677316));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.222221366518767365905163));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.285714294746548025383248));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.399999999950799600689777));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.6666666666667778740063));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(2));

#ifndef ENABLE_AVX512F
  x = vmla_vd_vd_vd_vd(x, t, vmul_vd_vd_vd(vcast_vd_d(0.693147180559945286226764), vcast_vd_vi(e)));
  x = (vdouble)vor_vm_vo64_vm(vgt_vo_vd_vd(vcast_vd_d(0), d), (vmask)x);
  x = vsel_vd_vo_vd_vd(vispinf_vo_vd(d), vcast_vd_d(INFINITY), x);
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(-INFINITY), x);
#else
  x = vmla_vd_vd_vd_vd(x, t, vmul_vd_vd_vd(vcast_vd_d(0.693147180559945286226764), e));
  x = vsel_vd_vo_vd_vd(vispinf_vo_vd(d), vcast_vd_d(INFINITY), x);
#endif

  return x;
}

vdouble xexp(vdouble d) {
  vint q = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(R_LN2)));
  vdouble s, u;

  s = vmla_vd_vd_vd_vd(vcast_vd_vi(q), vcast_vd_d(-L2U), d);
  s = vmla_vd_vd_vd_vd(vcast_vd_vi(q), vcast_vd_d(-L2L), s);

  u = vcast_vd_d(2.08860621107283687536341e-09);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.51112930892876518610661e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573911234900471893338e-07));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75572362911928827629423e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.4801587159235472998791e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000198412698960509205564975));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00138888888889774492207962));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00833333333331652721664984));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0416666666666665047591422));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.166666666666666851703837));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.5));

  u = vadd_vd_vd_vd(vcast_vd_d(1), vmla_vd_vd_vd_vd(vmul_vd_vd_vd(s, s), u, s));

  u = vldexp_vd_vd_vi(u, q);

  u = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(d, vcast_vd_d(709.78271114955742909217217426)), vcast_vd_d(INFINITY), u);
  u = (vdouble)vandnot_vm_vo64_vm(vlt_vo_vd_vd(d, vcast_vd_d(-1000)), (vmask)u);

  return u;
}

static INLINE vdouble2 logk(vdouble d) {
  vdouble2 x, x2;
  vdouble t, m;

#ifndef ENABLE_AVX512F
  vint e = vilogbk_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  m = vldexp_vd_vd_vi(d, vneg_vi_vi(e));
#else
  vdouble e = _mm512_getexp_pd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  e = vsel_vd_vo_vd_vd(vispinf_vo_vd(e), vcast_vd_d(1024.0), e);
  m = _mm512_getmant_pd(d, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan);
#endif

  x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(-1), m), ddadd2_vd2_vd_vd(vcast_vd_d(1), m));
  x2 = ddsqu_vd2_vd2(x);

  t = vcast_vd_d(0.116255524079935043668677);
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.103239680901072952701192));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.117754809412463995466069));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.13332981086846273921509));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.153846227114512262845736));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.181818180850050775676507));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.222222222230083560345903));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.285714285714249172087875));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.400000000000000077715612));
  vdouble2 c = vcast_vd2_d_d(0.666666666666666629659233, 3.80554962542412056336616e-17);

#ifndef ENABLE_AVX512F
  return ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.693147180559945286226764, 2.319046813846299558417771e-17), vcast_vd_vi(e)),
			    ddadd2_vd2_vd2_vd2(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)),
					       ddmul_vd2_vd2_vd2(ddmul_vd2_vd2_vd2(x2, x),
								 ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(x2, t), c))));
#else
  return ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(0.693147180559945286226764), vcast_vd_d(2.319046813846299558417771e-17)), e),
			    ddadd2_vd2_vd2_vd2(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)),
					       ddmul_vd2_vd2_vd2(ddmul_vd2_vd2_vd2(x2, x),
								 ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(x2, t), c))));
#endif
}

vdouble xlog_u1(vdouble d) {
  vdouble2 s = logk(d);
  vdouble x = vadd_vd_vd_vd(s.x, s.y);

  x = vsel_vd_vo_vd_vd(vispinf_vo_vd(d), vcast_vd_d(INFINITY), x);
  x = (vdouble)vor_vm_vo64_vm(vgt_vo_vd_vd(vcast_vd_d(0), d), (vmask)x);
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(-INFINITY), x);

  return x;
}

static INLINE vdouble expk(vdouble2 d) {
  vdouble u = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(R_LN2));
  vint q = vrint_vi_vd(u);
  vdouble2 s, t;

  s = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(vcast_vd_vi(q), vcast_vd_d(-L2U)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(vcast_vd_vi(q), vcast_vd_d(-L2L)));

  s = ddnormalize_vd2_vd2(s);

  u = vcast_vd_d(2.51069683420950419527139e-08);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.76286166770270649116855e-07));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.75572496725023574143864e-06));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.48014973989819794114153e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.000198412698809069797676111));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0013888888939977128960529));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00833333333332371417601081));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0416666666665409524128449));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.166666666666666740681535));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.500000000000000999200722));

  t = ddadd_vd2_vd2_vd2(s, ddmul_vd2_vd2_vd(ddsqu_vd2_vd2(s), u));

  t = ddadd_vd2_vd_vd2(vcast_vd_d(1), t);
  u = vadd_vd_vd_vd(t.x, t.y);
  u = vldexp_vd_vd_vi(u, q);

  u = (vdouble)vandnot_vm_vo64_vm(vlt_vo_vd_vd(d.x, vcast_vd_d(-1000)), (vmask)u);
  
  return u;
}

vdouble xpow(vdouble x, vdouble y) {
#if 1
  vopmask yisint = visint(y);
  vopmask yisodd = vand_vo_vo_vo(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vtruncate_vi_vd(y), vcast_vi_i(1)), vcast_vi_i(1))), yisint);

  vdouble2 d = ddmul_vd2_vd2_vd(logk(vabs_vd_vd(x)), y);
  vdouble result = expk(d);
  result = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(d.x, vcast_vd_d(709.78271114955742909217217426)), vcast_vd_d(INFINITY), result);

  result = vmul_vd_vd_vd(result,
			 vsel_vd_vo_vd_vd(vgt_vo_vd_vd(x, vcast_vd_d(0)),
					  vcast_vd_d(1),
					  vsel_vd_vo_vd_vd(yisint, vsel_vd_vo_vd_vd(yisodd, vcast_vd_d(-1.0), vcast_vd_d(1)), vcast_vd_d(NAN))));

  vdouble efx = vmulsign_vd_vd_vd(vsub_vd_vd_vd(vabs_vd_vd(x), vcast_vd_d(1)), y);

  result = vsel_vd_vo_vd_vd(visinf_vo_vd(y),
			    (vdouble)vandnot_vm_vo64_vm(vlt_vo_vd_vd(efx, vcast_vd_d(0.0)),
							(vmask)vsel_vd_vo_vd_vd(veq_vo_vd_vd(efx, vcast_vd_d(0.0)),
										vcast_vd_d(1.0),
										vcast_vd_d(INFINITY))),
			    result);

  result = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, vcast_vd_d(0.0))),
			    vmul_vd_vd_vd(vsel_vd_vo_vd_vd(yisodd, vsign_vd_vd(x), vcast_vd_d(1.0)),
					  (vdouble)vandnot_vm_vo64_vm(vlt_vo_vd_vd(vsel_vd_vo_vd_vd(veq_vo_vd_vd(x, vcast_vd_d(0.0)), vneg_vd_vd(y), y), vcast_vd_d(0.0)),
								      (vmask)vcast_vd_d(INFINITY))),
			    result);

  result = (vdouble)vor_vm_vo64_vm(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), (vmask)result);

  result = vsel_vd_vo_vd_vd(vor_vo_vo_vo(veq_vo_vd_vd(y, vcast_vd_d(0)), veq_vo_vd_vd(x, vcast_vd_d(1))), vcast_vd_d(1), result);

  return result;
#else
  return expk(ddmul_vd2_vd2_vd(logk(x), y));
#endif
}

static INLINE vdouble2 expk2(vdouble2 d) {
  vdouble u = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(R_LN2));
  vint q = vrint_vi_vd(u);
  vdouble2 s, t;

  s = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(vcast_vd_vi(q), vcast_vd_d(-L2U)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(vcast_vd_vi(q), vcast_vd_d(-L2L)));

  u = vcast_vd_d(2.51069683420950419527139e-08);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.76286166770270649116855e-07));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.75572496725023574143864e-06));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.48014973989819794114153e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.000198412698809069797676111));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0013888888939977128960529));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00833333333332371417601081));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0416666666665409524128449));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.166666666666666740681535));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.500000000000000999200722));

  t = ddadd_vd2_vd2_vd2(s, ddmul_vd2_vd2_vd(ddsqu_vd2_vd2(s), u));

  t = ddadd_vd2_vd_vd2(vcast_vd_d(1), t);

  return ddscale_vd2_vd2_vd(ddscale_vd2_vd2_vd(t, vcast_vd_d(2)), vpow2i_vd_vi(vsub_vi_vi_vi(q, vcast_vi_i(1))));
}

vdouble xsinh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  d = ddsub_vd2_vd2_vd2(d, ddrec_vd2_vd2(d));
  y = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(0.5));

  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(710)), visnan_vo_vd(y)), vcast_vd_d(INFINITY), y);
  y = vmulsign_vd_vd_vd(y, x);
  y = (vdouble)vor_vm_vo64_vm(visnan_vo_vd(x), (vmask)y);

  return y;
}

vdouble xcosh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  d = ddadd_vd2_vd2_vd2(d, ddrec_vd2_vd2(d));
  y = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(0.5));

  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(710)), visnan_vo_vd(y)), vcast_vd_d(INFINITY), y);
  y = (vdouble)vor_vm_vo64_vm(visnan_vo_vd(x), (vmask)y);

  return y;
}

vdouble xtanh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  vdouble2 e = ddrec_vd2_vd2(d);
  d = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd2_vd2(d, ddneg_vd2_vd2(e)), ddadd2_vd2_vd2_vd2(d, e));
  y = vadd_vd_vd_vd(d.x, d.y);

  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(18.714973875)), visnan_vo_vd(y)), vcast_vd_d(1.0), y);
  y = vmulsign_vd_vd_vd(y, x);
  y = (vdouble)vor_vm_vo64_vm(visnan_vo_vd(x), (vmask)y);

  return y;
}

static INLINE vdouble2 logk2(vdouble2 d) {
  vdouble2 x, x2, m;
  vdouble t;
  vint e;
  
#ifndef ENABLE_AVX512F
  e = vilogbk_vi_vd(vmul_vd_vd_vd(d.x, vcast_vd_d(1.0/0.75)));
#else
  e = vrint_vi_vd(_mm512_getexp_pd(vmul_vd_vd_vd(d.x, vcast_vd_d(1.0/0.75))));
#endif
  m = ddscale_vd2_vd2_vd(d, vpow2i_vd_vi(vneg_vi_vi(e)));

  x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd2_vd(m, vcast_vd_d(-1)), ddadd2_vd2_vd2_vd(m, vcast_vd_d(1)));
  x2 = ddsqu_vd2_vd2(x);

  t = vcast_vd_d(0.13860436390467167910856);
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.131699838841615374240845));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.153914168346271945653214));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.181816523941564611721589));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.22222224632662035403996));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.285714285511134091777308));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.400000000000914013309483));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.666666666666664853302393));

  return ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(0.693147180559945286226764), vcast_vd_d(2.319046813846299558417771e-17)),
		       vcast_vd_vi(e)),
		ddadd2_vd2_vd2_vd2(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)), ddmul_vd2_vd2_vd(ddmul_vd2_vd2_vd2(x2, x), t)));
}

vdouble xasinh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vopmask o = vgt_vo_vd_vd(y, vcast_vd_d(1));
  vdouble2 d;
  
  d = vsel_vd2_vo_vd2_vd2(o, ddrec_vd2_vd(x), vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  d = ddsqrt_vd2_vd2(ddadd2_vd2_vd2_vd(ddsqu_vd2_vd2(d), vcast_vd_d(1)));
  d = vsel_vd2_vo_vd2_vd2(o, ddmul_vd2_vd2_vd(d, y), d);

  d = logk2(ddnormalize_vd2_vd2(ddadd2_vd2_vd2_vd(d, x)));
  y = vadd_vd_vd_vd(d.x, d.y);
  
  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(SQRT_DBL_MAX)),
				    visnan_vo_vd(y)),
		       vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), x), y);
  y = (vdouble)vor_vm_vo64_vm(visnan_vo_vd(x), (vmask)y);
  y = vsel_vd_vo_vd_vd(visnegzero_vo_vd(x), vcast_vd_d(-0.0), y);

  return y;
}

vdouble xacosh(vdouble x) {
  vdouble2 d = logk2(ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd2(ddsqrt_vd2_vd2(ddadd2_vd2_vd_vd(x, vcast_vd_d(1))), ddsqrt_vd2_vd2(ddadd2_vd2_vd_vd(x, vcast_vd_d(-1)))), x));
  vdouble y = vadd_vd_vd_vd(d.x, d.y);

  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(SQRT_DBL_MAX)),
				    visnan_vo_vd(y)),
		       vcast_vd_d(INFINITY), y);
  y = (vdouble)vandnot_vm_vo64_vm(veq_vo_vd_vd(x, vcast_vd_d(1.0)), (vmask)y);

  y = (vdouble)vor_vm_vo64_vm(vlt_vo_vd_vd(x, vcast_vd_d(1.0)), (vmask)y);
  y = (vdouble)vor_vm_vo64_vm(visnan_vo_vd(x), (vmask)y);

  return y;
}

vdouble xatanh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = logk2(dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(1), y), ddadd2_vd2_vd_vd(vcast_vd_d(1), vneg_vd_vd(y))));
  y = (vdouble)vor_vm_vo64_vm(vgt_vo_vd_vd(y, vcast_vd_d(1.0)), (vmask)vsel_vd_vo_vd_vd(veq_vo_vd_vd(y, vcast_vd_d(1.0)), vcast_vd_d(INFINITY), vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(0.5))));

  y = (vdouble)vor_vm_vo64_vm(vor_vo_vo_vo(visinf_vo_vd(x), visnan_vo_vd(y)), (vmask)y);
  y = vmulsign_vd_vd_vd(y, x);
  y = (vdouble)vor_vm_vo64_vm(visnan_vo_vd(x), (vmask)y);

  return y;
}

vdouble xcbrt(vdouble d) {
  vdouble x, y, q = vcast_vd_d(1.0);
  vint e, qu, re;
  vdouble t;

#ifdef ENABLE_GETEXP_DP
  vdouble s = d;
#endif
  e = vadd_vi_vi_vi(vilogbk_vi_vd(vabs_vd_vd(d)), vcast_vi_i(1));
  d = vldexp_vd_vd_vi(d, vneg_vi_vi(e));

  t = vadd_vd_vd_vd(vcast_vd_vi(e), vcast_vd_d(6144));
  qu = vtruncate_vi_vd(vmul_vd_vd_vd(t, vcast_vd_d(1.0/3.0)));
  re = vtruncate_vi_vd(vsub_vd_vd_vd(t, vmul_vd_vd_vd(vcast_vd_vi(qu), vcast_vd_d(3))));

  q = vsel_vd_vo_vd_vd(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(1))), vcast_vd_d(1.2599210498948731647672106), q);
  q = vsel_vd_vo_vd_vd(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(2))), vcast_vd_d(1.5874010519681994747517056), q);
  q = vldexp_vd_vd_vi(q, vsub_vi_vi_vi(qu, vcast_vi_i(2048)));

  q = vmulsign_vd_vd_vd(q, d);

  d = vabs_vd_vd(d);

  x = vcast_vd_d(-0.640245898480692909870982);
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.96155103020039511818595));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-5.73353060922947843636166));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(6.03990368989458747961407));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-3.85841935510444988821632));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.2307275302496609725722));

  y = vmul_vd_vd_vd(x, x); y = vmul_vd_vd_vd(y, y); x = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vmlapn_vd_vd_vd_vd(d, y, x), vcast_vd_d(1.0 / 3.0)));
  y = vmul_vd_vd_vd(vmul_vd_vd_vd(d, x), x);
  y = vmul_vd_vd_vd(vsub_vd_vd_vd(y, vmul_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_d(2.0 / 3.0), y), vmla_vd_vd_vd_vd(y, x, vcast_vd_d(-1.0)))), q);

#ifdef ENABLE_GETEXP_DP
  y = vsel_vd_vo_vd_vd(visinf_vo_vd(s), vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), s), y);
  y = vsel_vd_vo_vd_vd(veq_vo_vd_vd(s, vcast_vd_d(0)), vmulsign_vd_vd_vd(vcast_vd_d(0), s), y);
#endif
  
  return y;
}

vdouble xcbrt_u1(vdouble d) {
  vdouble x, y, z, t;
  vdouble2 q2 = vcast_vd2_d_d(1, 0), u, v;
  vint e, qu, re;

#ifdef ENABLE_GETEXP_DP
  vdouble s = d;
#endif
  e = vadd_vi_vi_vi(vilogbk_vi_vd(vabs_vd_vd(d)), vcast_vi_i(1));
  d = vldexp_vd_vd_vi(d, vneg_vi_vi(e));

  t = vadd_vd_vd_vd(vcast_vd_vi(e), vcast_vd_d(6144));
  qu = vtruncate_vi_vd(vmul_vd_vd_vd(t, vcast_vd_d(1.0/3.0)));
  re = vtruncate_vi_vd(vsub_vd_vd_vd(t, vmul_vd_vd_vd(vcast_vd_vi(qu), vcast_vd_d(3))));

  q2 = vsel_vd2_vo_vd2_vd2(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(1))), vcast_vd2_d_d(1.2599210498948731907, -2.5899333753005069177e-17), q2);
  q2 = vsel_vd2_vo_vd2_vd2(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(2))), vcast_vd2_d_d(1.5874010519681995834, -1.0869008194197822986e-16), q2);

  q2.x = vmulsign_vd_vd_vd(q2.x, d); q2.y = vmulsign_vd_vd_vd(q2.y, d);
  d = vabs_vd_vd(d);

  x = vcast_vd_d(-0.640245898480692909870982);
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.96155103020039511818595));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-5.73353060922947843636166));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(6.03990368989458747961407));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-3.85841935510444988821632));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.2307275302496609725722));

  y = vmul_vd_vd_vd(x, x); y = vmul_vd_vd_vd(y, y); x = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vmlapn_vd_vd_vd_vd(d, y, x), vcast_vd_d(1.0 / 3.0)));

  z = x;

  u = ddmul_vd2_vd_vd(x, x);
  u = ddmul_vd2_vd2_vd2(u, u);
  u = ddmul_vd2_vd2_vd(u, d);
  u = ddadd2_vd2_vd2_vd(u, vneg_vd_vd(x));
  y = vadd_vd_vd_vd(u.x, u.y);

  y = vmul_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_d(-2.0 / 3.0), y), z);
  v = ddadd2_vd2_vd2_vd(ddmul_vd2_vd_vd(z, z), y);
  v = ddmul_vd2_vd2_vd(v, d);
  v = ddmul_vd2_vd2_vd2(v, q2);
  z = vldexp_vd_vd_vi(vadd_vd_vd_vd(v.x, v.y), vsub_vi_vi_vi(qu, vcast_vi_i(2048)));

  z = vsel_vd_vo_vd_vd(visinf_vo_vd(d), vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), q2.x), z);
  z = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0)), (vdouble)vsignbit_vm_vd(q2.x), z);

#ifdef ENABLE_GETEXP_DP
  z = vsel_vd_vo_vd_vd(visinf_vo_vd(s), vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), s), z);
  z = vsel_vd_vo_vd_vd(veq_vo_vd_vd(s, vcast_vd_d(0)), vmulsign_vd_vd_vd(vcast_vd_d(0), s), z);
#endif
  
  return z;
}

vdouble xexp2(vdouble a) {
  vdouble u = expk(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(0.69314718055994528623), vcast_vd_d(2.3190468138462995584e-17)), a));
  u = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(a, vcast_vd_d(1024)), vcast_vd_d(INFINITY), u);
  u = (vdouble)vandnot_vm_vo64_vm(visminf_vo_vd(a), (vmask)u);
  return u;
}

vdouble xexp10(vdouble a) {
  vdouble u = expk(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(2.3025850929940459011), vcast_vd_d(-2.1707562233822493508e-16)), a));
  u = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(a, vcast_vd_d(308.254715559916743850652254)), vcast_vd_d(INFINITY), u);
  u = (vdouble)vandnot_vm_vo64_vm(visminf_vo_vd(a), (vmask)u);
  return u;
}

vdouble xexpm1(vdouble a) {
  vdouble2 d = ddadd2_vd2_vd2_vd(expk2(vcast_vd2_vd_vd(a, vcast_vd_d(0))), vcast_vd_d(-1.0));
  vdouble x = vadd_vd_vd_vd(d.x, d.y);
  x = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(a, vcast_vd_d(709.782712893383996732223)), vcast_vd_d(INFINITY), x);
  x = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(a, vcast_vd_d(-36.736800569677101399113302437)), vcast_vd_d(-1), x);
  x = vsel_vd_vo_vd_vd(visnegzero_vo_vd(a), vcast_vd_d(-0.0), x);
  return x;
}

vdouble xlog10(vdouble a) {
  vdouble2 d = ddmul_vd2_vd2_vd2(logk(a), vcast_vd2_vd_vd(vcast_vd_d(0.43429448190325176116), vcast_vd_d(6.6494347733425473126e-17)));
  vdouble x = vadd_vd_vd_vd(d.x, d.y);

  x = vsel_vd_vo_vd_vd(vispinf_vo_vd(a), vcast_vd_d(INFINITY), x);
  x = (vdouble)vor_vm_vo64_vm(vgt_vo_vd_vd(vcast_vd_d(0), a), (vmask)x);
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(a, vcast_vd_d(0)), vcast_vd_d(-INFINITY), x);

  return x;
}

vdouble xlog1p(vdouble a) {
  vdouble2 d = logk2(ddadd2_vd2_vd_vd(a, vcast_vd_d(1)));
  vdouble x = vadd_vd_vd_vd(d.x, d.y);

  x = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(a, vcast_vd_d(1e+307)), vcast_vd_d(INFINITY), x);
  x = (vdouble)vor_vm_vo64_vm(vgt_vo_vd_vd(vcast_vd_d(-1.0), a), (vmask)x);
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(a, vcast_vd_d(-1)), vcast_vd_d(-INFINITY), x);
  x = vsel_vd_vo_vd_vd(visnegzero_vo_vd(a), vcast_vd_d(-0.0), x);

  return x;
}
