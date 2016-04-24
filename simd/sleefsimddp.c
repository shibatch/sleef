#include <assert.h>
#include <math.h>

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

#ifdef ENABLE_FMA4
#include "helperfma4.h"
#endif

//

#include "dd.h"

//

#define PI4_A 0.78539816290140151978
#define PI4_B 4.9604678871439933374e-10
#define PI4_C 1.1258708853173288931e-18
#define PI4_D 1.7607799325916000908e-27

#define M_4_PI 1.273239544735162542821171882678754627704620361328125

#define L2U .69314718055966295651160180568695068359375
#define L2L .28235290563031577122588448175013436025525412068e-12
#define R_LN2 1.442695040888963407359924681001892137426645954152985934135449406931

//

#define PI4_Af 0.78515625f
#define PI4_Bf 0.00024187564849853515625f
#define PI4_Cf 3.7747668102383613586e-08f
#define PI4_Df 1.2816720341285448015e-12f

#define L2Uf 0.693145751953125f
#define L2Lf 1.428606765330187045e-06f
#define R_LN2f 1.442695040888963407359924681001892137426645954152985934135449406931f

//

vdouble xldexp(vdouble x, vint q) { return vldexp_vd_vd_vi(x, q); }

vint xilogb(vdouble d) {
  vdouble e = vcast_vd_vi(vsub_vi_vi_vi(vilogbp1_vi_vd(vabs_vd_vd(d)), vcast_vi_i(1)));
  e = vsel_vd_vm_vd_vd(veq_vm_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(-2147483648.0), e);
  e = vsel_vd_vm_vd_vd(veq_vm_vd_vd(vabs_vd_vd(d), vcast_vd_d(INFINITY)), vcast_vd_d(2147483647), e);
  return vrint_vi_vd(e);
}

vdouble xsin(vdouble d) {
  vint q;
  vdouble u, s;

  q = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)));

  u = vcast_vd_vi(q);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_A*4), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_B*4), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_C*4), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_D*4), d);

  s = vmul_vd_vd_vd(d, d);

  d = (vdouble)vxor_vm_vm_vm(vand_vm_vm_vm(veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1)), (vmask)vcast_vd_d(-0.0)), (vmask)d);

  u = vcast_vd_d(-7.97255955009037868891952e-18);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.81009972710863200091251e-15));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-7.64712219118158833288484e-13));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(1.60590430605664501629054e-10));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.50521083763502045810755e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573192239198747630416e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000198412698412696162806809));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00833333333333332974823815));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.166666666666666657414808));

  u = vmla_vd_vd_vd_vd(s, vmul_vd_vd_vd(u, d), d);

  return u;
}

vdouble xsin_u1(vdouble d) {
  vint q;
  vdouble u;
  vdouble2 s, t, x;

  q = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)));
  u = vcast_vd_vi(q);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_A*4)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_B*4)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_C*4)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_D*4)));

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

  u = (vdouble)vxor_vm_vm_vm(vand_vm_vm_vm(veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1)), (vmask)vcast_vd_d(-0.0)), (vmask)u);

  return u;
}

vdouble xcos(vdouble d) {
  vint q;
  vdouble u, s;

  q = vrint_vi_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI), vcast_vd_d(-0.5)));
  q = vadd_vi_vi_vi(vadd_vi_vi_vi(q, q), vcast_vi_i(1));

  u = vcast_vd_vi(q);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_A*2), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_B*2), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_C*2), d);
  d = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_D*2), d);

  s = vmul_vd_vd_vd(d, d);

  d = (vdouble)vxor_vm_vm_vm(vand_vm_vm_vm(veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(0)), (vmask)vcast_vd_d(-0.0)), (vmask)d);

  u = vcast_vd_d(-7.97255955009037868891952e-18);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.81009972710863200091251e-15));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-7.64712219118158833288484e-13));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(1.60590430605664501629054e-10));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.50521083763502045810755e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573192239198747630416e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000198412698412696162806809));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00833333333333332974823815));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.166666666666666657414808));

  u = vmla_vd_vd_vd_vd(s, vmul_vd_vd_vd(u, d), d);

  return u;
}

vdouble xcos_u1(vdouble d) {
  vint q;
  vdouble u;
  vdouble2 s, t, x;

  q = vrint_vi_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI), vcast_vd_d(-0.5)));
  q = vadd_vi_vi_vi(vadd_vi_vi_vi(q, q), vcast_vi_i(1));
  u = vcast_vd_vi(q);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_A*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_B*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_C*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_D*2)));

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

  u = (vdouble)vxor_vm_vm_vm(vand_vm_vm_vm(veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(0)), (vmask)vcast_vd_d(-0.0)), (vmask)u);

  return u;
}

vdouble2 xsincos(vdouble d) {
  vint q;
  vmask m;
  vdouble u, s, t, rx, ry;
  vdouble2 r;

  q = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_2_PI)));

  s = d;

  u = vcast_vd_vi(q);
  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_A*2), s);
  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_B*2), s);
  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_C*2), s);
  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_D*2), s);

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

  u = vcast_vd_d(-1.13615350239097429531523e-11);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.08757471207040055479366e-09));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.75573144028847567498567e-07));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.48015872890001867311915e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.00138888888888714019282329));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0416666666666665519592062));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.5));

  ry = vmla_vd_vd_vd_vd(s, u, vcast_vd_d(1));

  m = veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(0));
  r.x = vsel_vd_vm_vd_vd(m, rx, ry);
  r.y = vsel_vd_vm_vd_vd(m, ry, rx);

  m = veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(2));
  r.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vm_vm(m, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.x)));

  m = veq_vm_vi_vi(vand_vi_vi_vi(vadd_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(2)), vcast_vi_i(2));
  r.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vm_vm(m, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.y)));

  m = visinf_vm_vd(d);
  r.x = (vdouble)vor_vm_vm_vm(m, (vmask)r.x);
  r.y = (vdouble)vor_vm_vm_vm(m, (vmask)r.y);

  return r;
}

vdouble2 xsincos_u1(vdouble d) {
  vint q;
  vmask m;
  vdouble u, rx, ry;
  vdouble2 r, s, t, x;

  q = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(2 * M_1_PI)));
  u = vcast_vd_vi(q);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_A*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_B*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_C*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_D*2)));

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

  u = vcast_vd_d(-1.13615350239097429531523e-11);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.08757471207040055479366e-09));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.75573144028847567498567e-07));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.48015872890001867311915e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.00138888888888714019282329));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0416666666666665519592062));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.5));

  x = ddadd_vd2_vd_vd2(vcast_vd_d(1), ddmul_vd2_vd_vd(s.x, u));
  ry = vadd_vd_vd_vd(x.x, x.y);

  m = veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(0));
  r.x = vsel_vd_vm_vd_vd(m, rx, ry);
  r.y = vsel_vd_vm_vd_vd(m, ry, rx);

  m = veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(2));
  r.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vm_vm(m, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.x)));

  m = veq_vm_vi_vi(vand_vi_vi_vi(vadd_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(2)), vcast_vi_i(2));
  r.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vm_vm(m, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.y)));

  m = visinf_vm_vd(d);
  r.x = (vdouble)vor_vm_vm_vm(m, (vmask)r.x);
  r.y = (vdouble)vor_vm_vm_vm(m, (vmask)r.y);

  return r;
}

vdouble xtan(vdouble d) {
  vint q;
  vdouble u, s, x;
  vmask m;

  q = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_2_PI)));

  u = vcast_vd_vi(q);
  x = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_A*2), d);
  x = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_B*2), x);
  x = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_C*2), x);
  x = vmla_vd_vd_vd_vd(u, vcast_vd_d(-PI4_D*2), x);

  s = vmul_vd_vd_vd(x, x);

  m = veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1));
  x = (vdouble)vxor_vm_vm_vm(vand_vm_vm_vm(m, (vmask)vcast_vd_d(-0.0)), (vmask)x);

  u = vcast_vd_d(1.01419718511083373224408e-05);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.59519791585924697698614e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(5.23388081915899855325186e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-3.05033014433946488225616e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(7.14707504084242744267497e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(8.09674518280159187045078e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000244884931879331847054404));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000588505168743587154904506));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00145612788922812427978848));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00359208743836906619142924));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00886323944362401618113356));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0218694882853846389592078));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0539682539781298417636002));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.133333333333125941821962));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.333333333333334980164153));

  u = vmla_vd_vd_vd_vd(s, vmul_vd_vd_vd(u, x), x);

  u = vsel_vd_vm_vd_vd(m, vrec_vd_vd(u), u);

  u = (vdouble)vor_vm_vm_vm(visinf_vm_vd(d), (vmask)u);

  return u;
}

vdouble xtan_u1(vdouble d) {
  vint q;
  vdouble u;
  vdouble2 s, t, x;
  vmask m;

  q = vrint_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_2_PI)));
  u = vcast_vd_vi(q);

  s = ddadd2_vd2_vd_vd (d, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_A*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_B*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_C*2)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(u, vcast_vd_d(-PI4_D*2)));

  m = veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1));
  vmask n = vand_vm_vm_vm(m, (vmask)vcast_vd_d(-0.0));
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

  x = vsel_vd2_vm_vd2_vd2(m, ddrec_vd2_vd2(x), x);

  u = vadd_vd_vd_vd(x.x, x.y);

  return u;
}

static INLINE vdouble atan2k(vdouble y, vdouble x) {
  vdouble s, t, u;
  vint q;
  vmask p;

  q = vsel_vi_vd_vd_vi_vi(x, vcast_vd_d(0), vcast_vi_i(-2), vcast_vi_i(0));
  x = vabs_vd_vd(x);

  q = vsel_vi_vd_vd_vi_vi(x, y, vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
  p = vlt_vm_vd_vd(x, y);
  s = vsel_vd_vm_vd_vd(p, vneg_vd_vd(x), y);
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
  vmask p;

  q = vsel_vi_vd_vd_vi_vi(x.x, vcast_vd_d(0), vcast_vi_i(-2), vcast_vi_i(0));
  p = vlt_vm_vd_vd(x.x, vcast_vd_d(0));
  p = vand_vm_vm_vm(p, (vmask)vcast_vd_d(-0.0));
  x.x = (vdouble)vxor_vm_vm_vm((vmask)x.x, p);
  x.y = (vdouble)vxor_vm_vm_vm((vmask)x.y, p);

  q = vsel_vi_vd_vd_vi_vi(x.x, y.x, vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
  p = vlt_vm_vd_vd(x.x, y.x);
  s = vsel_vd2_vm_vd2_vd2(p, ddneg_vd2_vd2(x), y);
  t = vsel_vd2_vm_vd2_vd2(p, y, x);

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

vdouble xatan2(vdouble y, vdouble x) {
  vdouble r = atan2k(vabs_vd_vd(y), x);

  r = vmulsign_vd_vd_vd(r, x);
  r = vsel_vd_vm_vd_vd(vor_vm_vm_vm(visinf_vm_vd(x), veq_vm_vd_vd(x, vcast_vd_d(0))), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/2), x))), r);
  r = vsel_vd_vm_vd_vd(visinf_vm_vd(y), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/4), x))), r);
  r = vsel_vd_vm_vd_vd(veq_vm_vd_vd(y, vcast_vd_d(0.0)), (vdouble)vand_vm_vm_vm(veq_vm_vd_vd(vsign_vd_vd(x), vcast_vd_d(-1.0)), (vmask)vcast_vd_d(M_PI)), r);

  r = (vdouble)vor_vm_vm_vm(vor_vm_vm_vm(visnan_vm_vd(x), visnan_vm_vd(y)), (vmask)vmulsign_vd_vd_vd(r, y));
  return r;
}

vdouble xatan2_u1(vdouble y, vdouble x) {
  vdouble2 d = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(y), vcast_vd_d(0)), vcast_vd2_vd_vd(x, vcast_vd_d(0)));
  vdouble r = vadd_vd_vd_vd(d.x, d.y);

  r = vmulsign_vd_vd_vd(r, x);
  r = vsel_vd_vm_vd_vd(vor_vm_vm_vm(visinf_vm_vd(x), veq_vm_vd_vd(x, vcast_vd_d(0))), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/2), x))), r);
  r = vsel_vd_vm_vd_vd(visinf_vm_vd(y), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/4), x))), r);
  r = vsel_vd_vm_vd_vd(veq_vm_vd_vd(y, vcast_vd_d(0.0)), (vdouble)vand_vm_vm_vm(veq_vm_vd_vd(vsign_vd_vd(x), vcast_vd_d(-1.0)), (vmask)vcast_vd_d(M_PI)), r);

  r = (vdouble)vor_vm_vm_vm(vor_vm_vm_vm(visnan_vm_vd(x), visnan_vm_vd(y)), (vmask)vmulsign_vd_vd_vd(r, y));
  return r;
}

vdouble xasin(vdouble d) {
  vdouble x, y;
  x = vadd_vd_vd_vd(vcast_vd_d(1), d);
  y = vsub_vd_vd_vd(vcast_vd_d(1), d);
  x = vmul_vd_vd_vd(x, y);
  x = vsqrt_vd_vd(x);
  x = (vdouble)vor_vm_vm_vm(visnan_vm_vd(x), (vmask)atan2k(vabs_vd_vd(d), x));
  return vmulsign_vd_vd_vd(x, d);
}

vdouble xasin_u1(vdouble d) {
  vdouble2 d2 = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0)), ddsqrt_vd2_vd2(ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(1), d), ddsub_vd2_vd_vd(vcast_vd_d(1), d))));
  vdouble r = vadd_vd_vd_vd(d2.x, d2.y);
  r = vsel_vd_vm_vd_vd(veq_vm_vd_vd(vabs_vd_vd(d), vcast_vd_d(1)), vcast_vd_d(1.570796326794896557998982), r);
  return vmulsign_vd_vd_vd(r, d);
}

vdouble xacos(vdouble d) {
  vdouble x, y;
  x = vadd_vd_vd_vd(vcast_vd_d(1), d);
  y = vsub_vd_vd_vd(vcast_vd_d(1), d);
  x = vmul_vd_vd_vd(x, y);
  x = vsqrt_vd_vd(x);
  x = vmulsign_vd_vd_vd(atan2k(x, vabs_vd_vd(d)), d);
  y = (vdouble)vand_vm_vm_vm(vlt_vm_vd_vd(d, vcast_vd_d(0)), (vmask)vcast_vd_d(M_PI));
  x = vadd_vd_vd_vd(x, y);
  return x;
}

vdouble xacos_u1(vdouble d) {
  vdouble2 d2 = atan2k_u1(ddsqrt_vd2_vd2(ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(1), d), ddsub_vd2_vd_vd(vcast_vd_d(1), d))), vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0)));
  d2 = ddscale_vd2_vd2_vd(d2, vmulsign_vd_vd_vd(vcast_vd_d(1), d));

  vmask m;
  m = vneq_vm_vd_vd(vabs_vd_vd(d), vcast_vd_d(1));
  d2.x = (vdouble)vand_vm_vm_vm(m, (vmask)d2.x);
  d2.y = (vdouble)vand_vm_vm_vm(m, (vmask)d2.y);
  m = vlt_vm_vd_vd(d, vcast_vd_d(0));
  d2 = vsel_vd2_vm_vd2_vd2(m, ddadd_vd2_vd2_vd2(vcast_vd2_d_d(3.141592653589793116, 1.2246467991473532072e-16), d2), d2);

  return vadd_vd_vd_vd(d2.x, d2.y);
}

vdouble xatan_u1(vdouble d) {
  vdouble2 d2 = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0)), vcast_vd2_d_d(1, 0));
  vdouble r = vadd_vd_vd_vd(d2.x, d2.y);
  r = vsel_vd_vm_vd_vd(visinf_vm_vd(d), vcast_vd_d(1.570796326794896557998982), r);
  return vmulsign_vd_vd_vd(r, d);
}

vdouble xatan(vdouble s) {
  vdouble t, u;
  vint q;

  q = vsel_vi_vd_vd_vi_vi(s, vcast_vd_d(0), vcast_vi_i(2), vcast_vi_i(0));
  s = vabs_vd_vd(s);

  q = vsel_vi_vd_vd_vi_vi(vcast_vd_d(1), s, vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
  s = vsel_vd_vm_vd_vd(vlt_vm_vd_vd(vcast_vd_d(1), s), vrec_vd_vd(s), s);

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

  t = vsel_vd_vm_vd_vd(veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1)), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), t), t);
  t = (vdouble)vxor_vm_vm_vm(vand_vm_vm_vm(veq_vm_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(2)), (vmask)vcast_vd_d(-0.0)), (vmask)t);

  return t;
}

vdouble xlog(vdouble d) {
  vdouble x, x2;
  vdouble t, m;
  vint e;

  e = vilogbp1_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(0.7071)));
  m = vldexp_vd_vd_vi(d, vneg_vi_vi(e));

  x = vdiv_vd_vd_vd(vadd_vd_vd_vd(vcast_vd_d(-1), m), vadd_vd_vd_vd(vcast_vd_d(1), m));
  x2 = vmul_vd_vd_vd(x, x);

  t = vcast_vd_d(0.148197055177935105296783);
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.153108178020442575739679));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.181837339521549679055568));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.22222194152736701733275));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.285714288030134544449368));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.399999999989941956712869));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.666666666666685503450651));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(2));

  x = vmla_vd_vd_vd_vd(x, t, vmul_vd_vd_vd(vcast_vd_d(0.693147180559945286226764), vcast_vd_vi(e)));

  x = vsel_vd_vm_vd_vd(vispinf_vm_vd(d), vcast_vd_d(INFINITY), x);
  x = (vdouble)vor_vm_vm_vm(vgt_vm_vd_vd(vcast_vd_d(0), d), (vmask)x);
  x = vsel_vd_vm_vd_vd(veq_vm_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(-INFINITY), x);

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

  u = (vdouble)vandnot_vm_vm_vm(visminf_vm_vd(d), (vmask)u);

  return u;
}

static INLINE vdouble2 logk(vdouble d) {
  vdouble2 x, x2;
  vdouble t, m;
  vint e;

  e = vilogbp1_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(0.7071)));
  m = vldexp_vd_vd_vi(d, vneg_vi_vi(e));

  x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(-1), m), ddadd2_vd2_vd_vd(vcast_vd_d(1), m));
  x2 = ddsqu_vd2_vd2(x);

  t = vcast_vd_d(0.134601987501262130076155);
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.132248509032032670243288));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.153883458318096079652524));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.181817427573705403298686));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.222222231326187414840781));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.285714285651261412873718));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.400000000000222439910458));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.666666666666666371239645));

  return ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(0.693147180559945286226764), vcast_vd_d(2.319046813846299558417771e-17)),
		       vcast_vd_vi(e)),
		ddadd2_vd2_vd2_vd2(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)), ddmul_vd2_vd2_vd(ddmul_vd2_vd2_vd2(x2, x), t)));
}

vdouble xlog_u1(vdouble d) {
  vdouble2 s = logk(d);
  vdouble x = vadd_vd_vd_vd(s.x, s.y);

  x = vsel_vd_vm_vd_vd(vispinf_vm_vd(d), vcast_vd_d(INFINITY), x);
  x = (vdouble)vor_vm_vm_vm(vgt_vm_vd_vd(vcast_vd_d(0), d), (vmask)x);
  x = vsel_vd_vm_vd_vd(veq_vm_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(-INFINITY), x);

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

  return u;
}

vdouble xpow(vdouble x, vdouble y) {
#if 1
  vmask yisnint = vneq_vm_vd_vd(vcast_vd_vi(vrint_vi_vd(y)), y);
  vmask yisodd = vandnot_vm_vm_vm(yisnint, veq_vm_vi_vi(vand_vi_vi_vi(vrint_vi_vd(y), vcast_vi_i(1)), vcast_vi_i(1)));

  vdouble result = expk(ddmul_vd2_vd2_vd(logk(vabs_vd_vd(x)), y));

  result = vmul_vd_vd_vd(result,
			 vsel_vd_vm_vd_vd(vgt_vm_vd_vd(x, vcast_vd_d(0)),
					  vcast_vd_d(1),
					  (vdouble)vor_vm_vm_vm(yisnint, (vmask)vsel_vd_vm_vd_vd(yisodd, vcast_vd_d(-1.0), vcast_vd_d(1)))));

  vdouble efx = (vdouble)vxor_vm_vm_vm((vmask)vsub_vd_vd_vd(vabs_vd_vd(x), vcast_vd_d(1)), vsignbit_vm_vd(y));

  result = vsel_vd_vm_vd_vd(visinf_vm_vd(y),
			    (vdouble)vandnot_vm_vm_vm(vlt_vm_vd_vd(efx, vcast_vd_d(0.0)),
						      (vmask)vsel_vd_vm_vd_vd(veq_vm_vd_vd(efx, vcast_vd_d(0.0)),
									      vcast_vd_d(1.0),
									      vcast_vd_d(INFINITY))),
			    result);

  result = vsel_vd_vm_vd_vd(vor_vm_vm_vm(visinf_vm_vd(x), veq_vm_vd_vd(x, vcast_vd_d(0.0))),
			    vmul_vd_vd_vd(vsel_vd_vm_vd_vd(yisodd, vsign_vd_vd(x), vcast_vd_d(1.0)),
					  (vdouble)vandnot_vm_vm_vm(vlt_vm_vd_vd(vsel_vd_vm_vd_vd(veq_vm_vd_vd(x, vcast_vd_d(0.0)), vneg_vd_vd(y), y), vcast_vd_d(0.0)),
								   (vmask)vcast_vd_d(INFINITY))),
			    result);

  result = (vdouble)vor_vm_vm_vm(vor_vm_vm_vm(visnan_vm_vd(x), visnan_vm_vd(y)), (vmask)result);

  result = vsel_vd_vm_vd_vd(vor_vm_vm_vm(veq_vm_vd_vd(y, vcast_vd_d(0)), veq_vm_vd_vd(x, vcast_vd_d(1))), vcast_vd_d(1), result);

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

  return ddscale_vd2_vd2_vd(t, vpow2i_vd_vi(q));
}

vdouble xsinh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  d = ddsub_vd2_vd2_vd2(d, ddrec_vd2_vd2(d));
  y = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(0.5));

  y = vsel_vd_vm_vd_vd(vor_vm_vm_vm(vgt_vm_vd_vd(vabs_vd_vd(x), vcast_vd_d(710)), visnan_vm_vd(y)), vcast_vd_d(INFINITY), y);
  y = vmulsign_vd_vd_vd(y, x);
  y = (vdouble)vor_vm_vm_vm(visnan_vm_vd(x), (vmask)y);

  return y;
}

vdouble xcosh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  d = ddadd_vd2_vd2_vd2(d, ddrec_vd2_vd2(d));
  y = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(0.5));

  y = vsel_vd_vm_vd_vd(vor_vm_vm_vm(vgt_vm_vd_vd(vabs_vd_vd(x), vcast_vd_d(710)), visnan_vm_vd(y)), vcast_vd_d(INFINITY), y);
  y = (vdouble)vor_vm_vm_vm(visnan_vm_vd(x), (vmask)y);

  return y;
}

vdouble xtanh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  vdouble2 e = ddrec_vd2_vd2(d);
  d = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd2_vd2(d, ddneg_vd2_vd2(e)), ddadd2_vd2_vd2_vd2(d, e));
  y = vadd_vd_vd_vd(d.x, d.y);

  y = vsel_vd_vm_vd_vd(vor_vm_vm_vm(vgt_vm_vd_vd(vabs_vd_vd(x), vcast_vd_d(18.714973875)), visnan_vm_vd(y)), vcast_vd_d(1.0), y);
  y = vmulsign_vd_vd_vd(y, x);
  y = (vdouble)vor_vm_vm_vm(visnan_vm_vd(x), (vmask)y);

  return y;
}

static INLINE vdouble2 logk2(vdouble2 d) {
  vdouble2 x, x2, m;
  vdouble t;
  vint e;

  e = vilogbp1_vi_vd(vmul_vd_vd_vd(d.x, vcast_vd_d(0.7071)));
  m = ddscale_vd2_vd2_vd(d, vpow2i_vd_vi(vneg_vi_vi(e)));

  x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd2_vd(m, vcast_vd_d(-1)), ddadd2_vd2_vd2_vd(m, vcast_vd_d(1)));
  x2 = ddsqu_vd2_vd2(x);

  t = vcast_vd_d(0.134601987501262130076155);
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.132248509032032670243288));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.153883458318096079652524));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.181817427573705403298686));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.222222231326187414840781));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.285714285651261412873718));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.400000000000222439910458));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.666666666666666371239645));

  return ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(0.693147180559945286226764), vcast_vd_d(2.319046813846299558417771e-17)),
		       vcast_vd_vi(e)),
		ddadd2_vd2_vd2_vd2(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)), ddmul_vd2_vd2_vd(ddmul_vd2_vd2_vd2(x2, x), t)));
}

vdouble xasinh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = logk2(ddadd2_vd2_vd2_vd(ddsqrt_vd2_vd2(ddadd2_vd2_vd2_vd(ddmul_vd2_vd_vd(y, y),  vcast_vd_d(1))), y));
  y = vadd_vd_vd_vd(d.x, d.y);

  y = vsel_vd_vm_vd_vd(vor_vm_vm_vm(visinf_vm_vd(x), visnan_vm_vd(y)), vcast_vd_d(INFINITY), y);
  y = vmulsign_vd_vd_vd(y, x);
  y = (vdouble)vor_vm_vm_vm(visnan_vm_vd(x), (vmask)y);

  return y;
}

vdouble xacosh(vdouble x) {
  vdouble2 d = logk2(ddadd2_vd2_vd2_vd(ddsqrt_vd2_vd2(ddadd2_vd2_vd2_vd(ddmul_vd2_vd_vd(x, x), vcast_vd_d(-1))), x));
  vdouble y = vadd_vd_vd_vd(d.x, d.y);

  y = vsel_vd_vm_vd_vd(vor_vm_vm_vm(visinf_vm_vd(x), visnan_vm_vd(y)), vcast_vd_d(INFINITY), y);
  y = (vdouble)vandnot_vm_vm_vm(veq_vm_vd_vd(x, vcast_vd_d(1.0)), (vmask)y);

  y = (vdouble)vor_vm_vm_vm(vlt_vm_vd_vd(x, vcast_vd_d(1.0)), (vmask)y);
  y = (vdouble)vor_vm_vm_vm(visnan_vm_vd(x), (vmask)y);

  return y;
}

vdouble xatanh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = logk2(dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(1), y), ddadd2_vd2_vd_vd(vcast_vd_d(1), vneg_vd_vd(y))));
  y = (vdouble)vor_vm_vm_vm(vgt_vm_vd_vd(y, vcast_vd_d(1.0)), (vmask)vsel_vd_vm_vd_vd(veq_vm_vd_vd(y, vcast_vd_d(1.0)), vcast_vd_d(INFINITY), vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(0.5))));

  y = (vdouble)vor_vm_vm_vm(vor_vm_vm_vm(visinf_vm_vd(x), visnan_vm_vd(y)), (vmask)y);

  y = vmulsign_vd_vd_vd(y, x);
  y = (vdouble)vor_vm_vm_vm(visnan_vm_vd(x), (vmask)y);

  return y;
}

vdouble xcbrt(vdouble d) {
  vdouble x, y, q = vcast_vd_d(1.0);
  vint e, qu, re;
  vdouble t;

  e = vilogbp1_vi_vd(vabs_vd_vd(d));
  d = vldexp_vd_vd_vi(d, vneg_vi_vi(e));

  t = vadd_vd_vd_vd(vcast_vd_vi(e), vcast_vd_d(6144));
  qu = vtruncate_vi_vd(vmul_vd_vd_vd(t, vcast_vd_d(1.0/3.0)));
  re = vtruncate_vi_vd(vsub_vd_vd_vd(t, vmul_vd_vd_vd(vcast_vd_vi(qu), vcast_vd_d(3))));

  q = vsel_vd_vm_vd_vd(veq_vm_vi_vi(re, vcast_vi_i(1)), vcast_vd_d(1.2599210498948731647672106), q);
  q = vsel_vd_vm_vd_vd(veq_vm_vi_vi(re, vcast_vi_i(2)), vcast_vd_d(1.5874010519681994747517056), q);
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

  return y;
}

vdouble xcbrt_u1(vdouble d) {
  vdouble x, y, z, t;
  vdouble2 q2 = vcast_vd2_d_d(1, 0), u, v;
  vint e, qu, re;

  e = vilogbp1_vi_vd(vabs_vd_vd(d));
  d = vldexp_vd_vd_vi(d, vneg_vi_vi(e));

  t = vadd_vd_vd_vd(vcast_vd_vi(e), vcast_vd_d(6144));
  qu = vtruncate_vi_vd(vmul_vd_vd_vd(t, vcast_vd_d(1.0/3.0)));
  re = vtruncate_vi_vd(vsub_vd_vd_vd(t, vmul_vd_vd_vd(vcast_vd_vi(qu), vcast_vd_d(3))));

  q2 = vsel_vd2_vm_vd2_vd2(veq_vm_vi_vi(re, vcast_vi_i(1)), vcast_vd2_d_d(1.2599210498948731907, -2.5899333753005069177e-17), q2);
  q2 = vsel_vd2_vm_vd2_vd2(veq_vm_vi_vi(re, vcast_vi_i(2)), vcast_vd2_d_d(1.5874010519681995834, -1.0869008194197822986e-16), q2);

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

  z = vsel_vd_vm_vd_vd(visinf_vm_vd(d), vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), q2.x), z);
  z = vsel_vd_vm_vd_vd(veq_vm_vd_vd(d, vcast_vd_d(0)), (vdouble)vsignbit_vm_vd(q2.x), z);

  return z;
}

vdouble xexp2(vdouble a) {
  vdouble u = expk(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(0.69314718055994528623), vcast_vd_d(2.3190468138462995584e-17)), a));
  u = vsel_vd_vm_vd_vd(vgt_vm_vd_vd(a, vcast_vd_d(1023)), vcast_vd_d(INFINITY), u);
  u = (vdouble)vandnot_vm_vm_vm(visminf_vm_vd(a), (vmask)u);
  return u;
}

vdouble xexp10(vdouble a) {
  vdouble u = expk(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(2.3025850929940459011), vcast_vd_d(-2.1707562233822493508e-16)), a));
  u = vsel_vd_vm_vd_vd(vgt_vm_vd_vd(a, vcast_vd_d(308)), vcast_vd_d(INFINITY), u);
  u = (vdouble)vandnot_vm_vm_vm(visminf_vm_vd(a), (vmask)u);
  return u;
}

vdouble xexpm1(vdouble a) {
  vdouble2 d = ddadd2_vd2_vd2_vd(expk2(vcast_vd2_vd_vd(a, vcast_vd_d(0))), vcast_vd_d(-1.0));
  vdouble x = vadd_vd_vd_vd(d.x, d.y);
  x = vsel_vd_vm_vd_vd(vgt_vm_vd_vd(a, vcast_vd_d(700)), vcast_vd_d(INFINITY), x);
  x = vsel_vd_vm_vd_vd(vlt_vm_vd_vd(a, vcast_vd_d(-0.36043653389117156089696070315825181539851971360337e+2)), vcast_vd_d(-1), x);
  return x;
}

vdouble xlog10(vdouble a) {
  vdouble2 d = ddmul_vd2_vd2_vd2(logk(a), vcast_vd2_vd_vd(vcast_vd_d(0.43429448190325176116), vcast_vd_d(6.6494347733425473126e-17)));
  vdouble x = vadd_vd_vd_vd(d.x, d.y);

  x = vsel_vd_vm_vd_vd(vispinf_vm_vd(a), vcast_vd_d(INFINITY), x);
  x = (vdouble)vor_vm_vm_vm(vgt_vm_vd_vd(vcast_vd_d(0), a), (vmask)x);
  x = vsel_vd_vm_vd_vd(veq_vm_vd_vd(a, vcast_vd_d(0)), vcast_vd_d(-INFINITY), x);

  return x;
}

vdouble xlog1p(vdouble a) {
  vdouble2 d = logk2(ddadd2_vd2_vd_vd(a, vcast_vd_d(1)));
  vdouble x = vadd_vd_vd_vd(d.x, d.y);

  x = vsel_vd_vm_vd_vd(vispinf_vm_vd(a), vcast_vd_d(INFINITY), x);
  x = (vdouble)vor_vm_vm_vm(vgt_vm_vd_vd(vcast_vd_d(-1.0), a), (vmask)x);
  x = vsel_vd_vm_vd_vd(veq_vm_vd_vd(a, vcast_vd_d(-1)), vcast_vd_d(-INFINITY), x);

  return x;
}
