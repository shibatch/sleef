//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Always use -ffp-contract=off option to compile SLEEF.

#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#include "misc.h"

#ifdef DORENAME
#include "rename.h"
#endif

#if (defined(_MSC_VER))
#pragma fp_contract (off)
#endif

#define PI_A 3.1415926218032836914
#define PI_B 3.1786509424591713469e-08
#define PI_C 1.2246467864107188502e-16
#define PI_D 1.2736634327021899816e-24

#define M_2_PI_H 0.63661977236758138243
#define M_2_PI_L -3.9357353350364971764e-17

#define TRIGRANGEMAX 1e+14
#define SQRT_DBL_MAX 1.3407807929942596355e+154

#define M_4_PI 1.273239544735162542821171882678754627704620361328125

#define L2U .69314718055966295651160180568695068359375
#define L2L .28235290563031577122588448175013436025525412068e-12
#define R_LN2 1.442695040888963407359924681001892137426645954152985934135449406931

static INLINE CONST int64_t doubleToRawLongBits(double d) {
  union {
    double f;
    int64_t i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

static INLINE CONST double longBitsToDouble(int64_t i) {
  union {
    double f;
    int64_t i;
  } tmp;
  tmp.i = i;
  return tmp.f;
}

static INLINE CONST double fabsk(double x) {
  return longBitsToDouble(0x7fffffffffffffffLL & doubleToRawLongBits(x));
}

static INLINE CONST double mulsign(double x, double y) {
  return longBitsToDouble(doubleToRawLongBits(x) ^ (doubleToRawLongBits(y) & (1LL << 63)));
}

static INLINE CONST double copysignk(double x, double y) {
  return longBitsToDouble((doubleToRawLongBits(x) & ~(1LL << 63)) ^ (doubleToRawLongBits(y) & (1LL << 63)));
}

static INLINE CONST double sign(double d) { return mulsign(1, d); }
static INLINE CONST double mla(double x, double y, double z) { return x * y + z; }
static INLINE CONST double rintk(double x) { return x < 0 ? (int)(x - 0.5) : (int)(x + 0.5); }
static INLINE CONST int ceilk(double x) { return (int)x + (x < 0 ? 0 : 1); }
static INLINE CONST double trunck(double x) { return (double)(int)x; }
static INLINE CONST double fmink(double x, double y) { return x < y ? x : y; }
static INLINE CONST double fmaxk(double x, double y) { return x > y ? x : y; }

static INLINE CONST int xisnan(double x) { return x != x; }
static INLINE CONST int xisinf(double x) { return x == INFINITY || x == -INFINITY; }
static INLINE CONST int xisminf(double x) { return x == -INFINITY; }
static INLINE CONST int xispinf(double x) { return x == INFINITY; }
static INLINE CONST int xisnegzero(double x) { return doubleToRawLongBits(x) == doubleToRawLongBits(-0.0); }

static INLINE CONST int xisint(double d) {
  double x = d - (double)(1 << 31) * (int)(d * (1.0 / (1 << 31)));
  return (x == (int)x) || (fabsk(d) >= (double)(1LL << 52));
}

static INLINE CONST int xisodd(double d) {
  double x = d - (double)(1 << 31) * (int)(d * (1.0 / (1 << 31)));
  return (1 & (int)x) != 0;
}

static INLINE CONST double pow2i(int q) {
  return longBitsToDouble(((int64_t)(q + 0x3ff)) << 52);
}

static INLINE CONST double ldexpk(double x, int q) {
  double u;
  int m;
  m = q >> 31;
  m = (((m + q) >> 9) - m) << 7;
  q = q - (m << 2);
  m += 0x3ff;
  m = m < 0     ? 0     : m;
  m = m > 0x7ff ? 0x7ff : m;
  u = longBitsToDouble(((int64_t)m) << 52);
  x = x * u * u * u * u;
  u = longBitsToDouble(((int64_t)(q + 0x3ff)) << 52);
  return x * u;
}

EXPORT CONST double xldexp(double x, int exp) {
  if (exp >  2100) exp =  2100;
  if (exp < -2100) exp = -2100;
  
  int e0 = exp >> 2;
  if (exp < 0) e0++;
  if (-100 < exp && exp < 100) e0 = 0;
  int e1 = exp - (e0 << 2);

  double p = pow2i(e0);
  double ret = x * pow2i(e1) * p * p * p * p;
  
  return ret;
}

static INLINE CONST int ilogbk(double d) {
  int m = d < 4.9090934652977266E-91;
  d = m ? 2.037035976334486E90 * d : d;
  int q = (doubleToRawLongBits(d) >> 52) & 0x7ff;
  q = m ? q - (300 + 0x03ff) : q - 0x03ff;
  return q;
}

EXPORT CONST int xilogb(double d) {
  int e = ilogbk(fabsk(d));
  e = d == 0.0  ? FP_ILOGB0 : e;
  e = xisnan(d) ? FP_ILOGBNAN : e;
  e = xisinf(d) ? INT_MAX : e;
  return e;
}

//

#ifndef NDEBUG
static int checkfp(double x) {
  if (xisinf(x) || xisnan(x)) return 1;
  return 0;
}
#endif

static INLINE CONST double upper(double d) {
  return longBitsToDouble(doubleToRawLongBits(d) & 0xfffffffff8000000LL);
}

static INLINE CONST Sleef_double2 dd(double h, double l) {
  Sleef_double2 ret;
  ret.x = h; ret.y = l;
  return ret;
}

static INLINE CONST Sleef_double2 ddnormalize_d2_d2(Sleef_double2 t) {
  Sleef_double2 s;

  s.x = t.x + t.y;
  s.y = t.x - s.x + t.y;

  return s;
}

static INLINE CONST Sleef_double2 ddscale_d2_d2_d(Sleef_double2 d, double s) {
  Sleef_double2 r;

  r.x = d.x * s;
  r.y = d.y * s;

  return r;
}

static INLINE CONST Sleef_double2 ddneg_d2_d2(Sleef_double2 d) {
  Sleef_double2 r;

  r.x = -d.x;
  r.y = -d.y;

  return r;
}

static INLINE CONST Sleef_double2 ddadd_d2_d_d(double x, double y) {
  // |x| >= |y|

  Sleef_double2 r;

#ifndef NDEBUG
  if (!(checkfp(x) || checkfp(y) || fabsk(x) >= fabsk(y))) {
    fprintf(stderr, "[ddadd_d2_d_d : %g, %g]\n", x, y);
    fflush(stderr);
  }
#endif

  r.x = x + y;
  r.y = x - r.x + y;

  return r;
}

static INLINE CONST Sleef_double2 ddadd2_d2_d_d(double x, double y) {
  Sleef_double2 r;

  r.x = x + y;
  double v = r.x - x;
  r.y = (x - (r.x - v)) + (y - v);

  return r;
}

static INLINE CONST Sleef_double2 ddadd_d2_d2_d(Sleef_double2 x, double y) {
  // |x| >= |y|

  Sleef_double2 r;

#ifndef NDEBUG
  if (!(checkfp(x.x) || checkfp(y) || fabsk(x.x) >= fabsk(y))) {
    fprintf(stderr, "[ddadd_d2_d2_d : %g %g]\n", x.x, y);
    fflush(stderr);
  }
#endif

  r.x = x.x + y;
  r.y = x.x - r.x + y + x.y;

  return r;
}

static INLINE CONST Sleef_double2 ddadd2_d2_d2_d(Sleef_double2 x, double y) {
  // |x| >= |y|

  Sleef_double2 r;

  r.x  = x.x + y;
  double v = r.x - x.x;
  r.y = (x.x - (r.x - v)) + (y - v);
  r.y += x.y;

  return r;
}

static INLINE CONST Sleef_double2 ddadd_d2_d_d2(double x, Sleef_double2 y) {
  // |x| >= |y|

  Sleef_double2 r;

#ifndef NDEBUG
  if (!(checkfp(x) || checkfp(y.x) || fabsk(x) >= fabsk(y.x))) {
    fprintf(stderr, "[ddadd_d2_d_d2 : %g %g]\n", x, y.x);
    fflush(stderr);
  }
#endif

  r.x = x + y.x;
  r.y = x - r.x + y.x + y.y;

  return r;
}

static INLINE CONST Sleef_double2 ddadd2_d2_d_d2(double x, Sleef_double2 y) {
  Sleef_double2 r;

  r.x  = x + y.x;
  double v = r.x - x;
  r.y = (x - (r.x - v)) + (y.x - v) + y.y;

  return r;
}

static INLINE CONST Sleef_double2 ddadd_d2_d2_d2(Sleef_double2 x, Sleef_double2 y) {
  // |x| >= |y|

  Sleef_double2 r;

#ifndef NDEBUG
  if (!(checkfp(x.x) || checkfp(y.x) || fabsk(x.x) >= fabsk(y.x))) {
    fprintf(stderr, "[ddadd_d2_d2_d2 : %g %g]\n", x.x, y.x);
    fflush(stderr);
  }
#endif

  r.x = x.x + y.x;
  r.y = x.x - r.x + y.x + x.y + y.y;

  return r;
}

static INLINE CONST Sleef_double2 ddadd2_d2_d2_d2(Sleef_double2 x, Sleef_double2 y) {
  Sleef_double2 r;

  r.x  = x.x + y.x;
  double v = r.x - x.x;
  r.y = (x.x - (r.x - v)) + (y.x - v);
  r.y += x.y + y.y;

  return r;
}

static INLINE CONST Sleef_double2 ddsub_d2_d2_d2(Sleef_double2 x, Sleef_double2 y) {
  // |x| >= |y|

  Sleef_double2 r;

#ifndef NDEBUG
  if (!(checkfp(x.x) || checkfp(y.x) || fabsk(x.x) >= fabsk(y.x))) {
    fprintf(stderr, "[ddsub_d2_d2_d2 : %g %g]\n", x.x, y.x);
    fflush(stderr);
  }
#endif

  r.x = x.x - y.x;
  r.y = x.x - r.x - y.x + x.y - y.y;

  return r;
}

static INLINE CONST Sleef_double2 dddiv_d2_d2_d2(Sleef_double2 n, Sleef_double2 d) {
  double t = 1.0 / d.x;
  double dh  = upper(d.x), dl  = d.x - dh;
  double th  = upper(t  ), tl  = t   - th;
  double nhh = upper(n.x), nhl = n.x - nhh;

  Sleef_double2 q;

  q.x = n.x * t;

  double u = -q.x + nhh * th + nhh * tl + nhl * th + nhl * tl +
    q.x * (1 - dh * th - dh * tl - dl * th - dl * tl);

  q.y = t * (n.y - q.x * d.y) + u;

  return q;
}

static INLINE CONST Sleef_double2 ddmul_d2_d_d(double x, double y) {
  double xh = upper(x), xl = x - xh;
  double yh = upper(y), yl = y - yh;
  Sleef_double2 r;

  r.x = x * y;
  r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl;

  return r;
}

static INLINE CONST Sleef_double2 ddmul_d2_d2_d(Sleef_double2 x, double y) {
  double xh = upper(x.x), xl = x.x - xh;
  double yh = upper(y  ), yl = y   - yh;
  Sleef_double2 r;

  r.x = x.x * y;
  r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.y * y;

  return r;
}

static INLINE CONST Sleef_double2 ddmul_d2_d2_d2(Sleef_double2 x, Sleef_double2 y) {
  double xh = upper(x.x), xl = x.x - xh;
  double yh = upper(y.x), yl = y.x - yh;
  Sleef_double2 r;

  r.x = x.x * y.x;
  r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.x * y.y + x.y * y.x;

  return r;
}

static INLINE CONST Sleef_double2 ddsqu_d2_d2(Sleef_double2 x) {
  double xh = upper(x.x), xl = x.x - xh;
  Sleef_double2 r;

  r.x = x.x * x.x;
  r.y = xh * xh - r.x + (xh + xh) * xl + xl * xl + x.x * (x.y + x.y);

  return r;
}

static INLINE CONST Sleef_double2 ddrec_d2_d(double d) {
  double t = 1.0 / d;
  double dh = upper(d), dl = d - dh;
  double th = upper(t), tl = t - th;
  Sleef_double2 q;

  q.x = t;
  q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl);

  return q;
}

static INLINE CONST Sleef_double2 ddrec_d2_d2(Sleef_double2 d) {
  double t = 1.0 / d.x;
  double dh = upper(d.x), dl = d.x - dh;
  double th = upper(t  ), tl = t   - th;
  Sleef_double2 q;

  q.x = t;
  q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl - d.y * t);

  return q;
}

static INLINE CONST Sleef_double2 ddsqrt_d2_d2(Sleef_double2 d) {
  double t = sqrt(d.x + d.y);
  return ddscale_d2_d2_d(ddmul_d2_d2_d2(ddadd2_d2_d2_d2(d, ddmul_d2_d_d(t, t)), ddrec_d2_d(t)), 0.5);
}

//

static INLINE CONST double atan2k(double y, double x) {
  double s, t, u;
  int q = 0;

  if (x < 0) { x = -x; q = -2; }
  if (y > x) { t = x; x = y; y = -t; q += 1; }

  s = y / x;
  t = s * s;

  u = -1.88796008463073496563746e-05;
  u = mla(u, t, 0.000209850076645816976906797);
  u = mla(u, t, -0.00110611831486672482563471);
  u = mla(u, t, 0.00370026744188713119232403);
  u = mla(u, t, -0.00889896195887655491740809);
  u = mla(u, t, 0.016599329773529201970117);
  u = mla(u, t, -0.0254517624932312641616861);
  u = mla(u, t, 0.0337852580001353069993897);
  u = mla(u, t, -0.0407629191276836500001934);
  u = mla(u, t, 0.0466667150077840625632675);
  u = mla(u, t, -0.0523674852303482457616113);
  u = mla(u, t, 0.0587666392926673580854313);
  u = mla(u, t, -0.0666573579361080525984562);
  u = mla(u, t, 0.0769219538311769618355029);
  u = mla(u, t, -0.090908995008245008229153);
  u = mla(u, t, 0.111111105648261418443745);
  u = mla(u, t, -0.14285714266771329383765);
  u = mla(u, t, 0.199999999996591265594148);
  u = mla(u, t, -0.333333333333311110369124);

  t = u * t * s + s;
  t = q * (M_PI/2) + t;

  return t;
}

EXPORT CONST double xatan2(double y, double x) {
  double r = atan2k(fabsk(y), x);

  r = mulsign(r, x);
  if (xisinf(x) || x == 0) r = M_PI/2 - (xisinf(x) ? (sign(x) * (M_PI  /2)) : 0);
  if (xisinf(y)          ) r = M_PI/2 - (xisinf(x) ? (sign(x) * (M_PI*1/4)) : 0);
  if (             y == 0) r = (sign(x) == -1 ? M_PI : 0);

  return xisnan(x) || xisnan(y) ? NAN : mulsign(r, y);
}

EXPORT CONST double xasin(double d) {
  return mulsign(atan2k(fabsk(d), sqrt((1+d)*(1-d))), d);
}

EXPORT CONST double xacos(double d) {
  return mulsign(atan2k(sqrt((1+d)*(1-d)), fabsk(d)), d) + (sign(d) == -1 ? M_PI : 0);
}

EXPORT CONST double xatan(double s) {
  double t, u;
  int q = 0;

  if (sign(s) == -1) { s = -s; q = 2; }
  if (s > 1) { s = 1.0 / s; q |= 1; }

  t = s * s;

  u = -1.88796008463073496563746e-05;
  u = mla(u, t, 0.000209850076645816976906797);
  u = mla(u, t, -0.00110611831486672482563471);
  u = mla(u, t, 0.00370026744188713119232403);
  u = mla(u, t, -0.00889896195887655491740809);
  u = mla(u, t, 0.016599329773529201970117);
  u = mla(u, t, -0.0254517624932312641616861);
  u = mla(u, t, 0.0337852580001353069993897);
  u = mla(u, t, -0.0407629191276836500001934);
  u = mla(u, t, 0.0466667150077840625632675);
  u = mla(u, t, -0.0523674852303482457616113);
  u = mla(u, t, 0.0587666392926673580854313);
  u = mla(u, t, -0.0666573579361080525984562);
  u = mla(u, t, 0.0769219538311769618355029);
  u = mla(u, t, -0.090908995008245008229153);
  u = mla(u, t, 0.111111105648261418443745);
  u = mla(u, t, -0.14285714266771329383765);
  u = mla(u, t, 0.199999999996591265594148);
  u = mla(u, t, -0.333333333333311110369124);

  t = s + s * (t * u);

  if ((q & 1) != 0) t = 1.570796326794896557998982 - t;
  if ((q & 2) != 0) t = -t;

  return t;
}

static Sleef_double2 atan2k_u1(Sleef_double2 y, Sleef_double2 x) {
  double u;
  Sleef_double2 s, t;
  int q = 0;

  if (x.x < 0) { x.x = -x.x; x.y = -x.y; q = -2; }
  if (y.x > x.x) { t = x; x = y; y.x = -t.x; y.y = -t.y; q += 1; }

  s = dddiv_d2_d2_d2(y, x);
  t = ddsqu_d2_d2(s);
  t = ddnormalize_d2_d2(t);

  u = 1.06298484191448746607415e-05;
  u = mla(u, t.x, -0.000125620649967286867384336);
  u = mla(u, t.x, 0.00070557664296393412389774);
  u = mla(u, t.x, -0.00251865614498713360352999);
  u = mla(u, t.x, 0.00646262899036991172313504);
  u = mla(u, t.x, -0.0128281333663399031014274);
  u = mla(u, t.x, 0.0208024799924145797902497);
  u = mla(u, t.x, -0.0289002344784740315686289);
  u = mla(u, t.x, 0.0359785005035104590853656);
  u = mla(u, t.x, -0.041848579703592507506027);
  u = mla(u, t.x, 0.0470843011653283988193763);
  u = mla(u, t.x, -0.0524914210588448421068719);
  u = mla(u, t.x, 0.0587946590969581003860434);
  u = mla(u, t.x, -0.0666620884778795497194182);
  u = mla(u, t.x, 0.0769225330296203768654095);
  u = mla(u, t.x, -0.0909090442773387574781907);
  u = mla(u, t.x, 0.111111108376896236538123);
  u = mla(u, t.x, -0.142857142756268568062339);
  u = mla(u, t.x, 0.199999999997977351284817);
  u = mla(u, t.x, -0.333333333333317605173818);

  t = ddmul_d2_d2_d(t, u);
  t = ddmul_d2_d2_d2(s, ddadd_d2_d_d2(1, t));
  if (fabsk(s.x) < 1e-200) t = s;
  t = ddadd2_d2_d2_d2(ddmul_d2_d2_d(dd(1.570796326794896557998982, 6.12323399573676603586882e-17), q), t);
  
  return t;
}

EXPORT CONST double xatan2_u1(double y, double x) {
  if (fabsk(x) < 5.5626846462680083984e-309) { y *= (1ULL << 53); x *= (1ULL << 53); } // nexttoward((1.0 / DBL_MAX), 1)
  Sleef_double2 d = atan2k_u1(dd(fabsk(y), 0), dd(x, 0));
  double r = d.x + d.y;

  r = mulsign(r, x);
  if (xisinf(x) || x == 0) r = M_PI/2 - (xisinf(x) ? (sign(x) * (M_PI  /2)) : 0);
  if (xisinf(y)          ) r = M_PI/2 - (xisinf(x) ? (sign(x) * (M_PI*1/4)) : 0);
  if (             y == 0) r = (sign(x) == -1 ? M_PI : 0);

  return xisnan(x) || xisnan(y) ? NAN : mulsign(r, y);
}

EXPORT CONST double xasin_u1(double d) {
  Sleef_double2 d2 = atan2k_u1(dd(fabsk(d), 0), ddsqrt_d2_d2(ddmul_d2_d2_d2(ddadd_d2_d_d(1, d), ddadd_d2_d_d(1,-d))));
  double r = d2.x + d2.y;
  if (fabsk(d) == 1) r = 1.570796326794896557998982;
  return mulsign(r, d);
}

EXPORT CONST double xacos_u1(double d) {
  Sleef_double2 d2 = atan2k_u1(ddsqrt_d2_d2(ddmul_d2_d2_d2(ddadd_d2_d_d(1, d), ddadd_d2_d_d(1,-d))), dd(fabsk(d), 0));
  d2 = ddscale_d2_d2_d(d2, mulsign(1, d));
  if (fabsk(d) == 1) d2 = dd(0, 0);
  if (sign(d) == -1) d2 = ddadd_d2_d2_d2(dd(3.141592653589793116, 1.2246467991473532072e-16), d2);
  return d2.x + d2.y;
}

EXPORT CONST double xatan_u1(double d) {
  Sleef_double2 d2 = atan2k_u1(dd(fabsk(d), 0), dd(1, 0));
  double r = d2.x + d2.y;
  if (xisinf(d)) r = 1.570796326794896557998982;
  return mulsign(r, d);
}

EXPORT CONST double xsin(double d) {
  double u, s, t = d;

  int qh = trunck(d * (M_1_PI / (1 << 24)));
  int ql = rintk(d * M_1_PI - qh * (double)(1 << 24));

  d = mla(qh, -PI_A * (1 << 24), d);
  d = mla(ql, -PI_A,             d);
  d = mla(qh, -PI_B * (1 << 24), d);
  d = mla(ql, -PI_B,             d);
  d = mla(qh, -PI_C * (1 << 24), d);
  d = mla(ql, -PI_C,             d);
  d = mla((double)qh * (1 << 24) + ql, -PI_D, d);
  
  s = d * d;

  if ((ql & 1) != 0) d = -d;

  u = -7.97255955009037868891952e-18;
  u = mla(u, s, 2.81009972710863200091251e-15);
  u = mla(u, s, -7.64712219118158833288484e-13);
  u = mla(u, s, 1.60590430605664501629054e-10);
  u = mla(u, s, -2.50521083763502045810755e-08);
  u = mla(u, s, 2.75573192239198747630416e-06);
  u = mla(u, s, -0.000198412698412696162806809);
  u = mla(u, s, 0.00833333333333332974823815);
  u = mla(u, s, -0.166666666666666657414808);

  u = mla(s, u * d, d);

  if (!xisinf(t) && (xisnegzero(t) || fabsk(t) > TRIGRANGEMAX)) u = -0.0;

  return u;
}

EXPORT CONST double xsin_u1(double d) {
  double u;
  Sleef_double2 s, t, x;

  int qh = trunck(d * (M_1_PI / (1 << 24)));
  int ql = rintk(d * M_1_PI - qh * (double)(1 << 24));

  s = ddadd2_d2_d_d (d, qh * (-PI_A * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_A            ));
  s = ddadd2_d2_d2_d(s, qh * (-PI_B * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_B            ));
  s = ddadd2_d2_d2_d(s, qh * (-PI_C * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_C            ));
  s = ddadd2_d2_d2_d(s, ((double)qh * (1 << 24) + ql) * -PI_D);
  
  t = s;
  s = ddsqu_d2_d2(s);

  u = 2.72052416138529567917983e-15;
  u = mla(u, s.x, -7.6429259411395447190023e-13);
  u = mla(u, s.x, 1.60589370117277896211623e-10);
  u = mla(u, s.x, -2.5052106814843123359368e-08);
  u = mla(u, s.x, 2.75573192104428224777379e-06);
  u = mla(u, s.x, -0.000198412698412046454654947);
  u = mla(u, s.x, 0.00833333333333318056201922);

  x = ddadd_d2_d_d2(1, ddmul_d2_d2_d2(ddadd_d2_d_d(-0.166666666666666657414808, u * s.x), s));
  
  x = ddmul_d2_d2_d2(t, x);
  u = x.x + x.y;

  if ((ql & 1) != 0) u = -u;
  if (!xisinf(d) && (xisnegzero(d) || fabsk(d) > TRIGRANGEMAX)) u = -0.0;

  return u;
}

EXPORT CONST double xcos(double d) {
  double u, s, t = d;

  int qh = trunck(d * (M_1_PI / (1LL << 23)) - 0.5 * (M_1_PI / (1LL << 23)));
  int ql = 2*rintk(d * M_1_PI - 0.5 - qh * (double)(1LL << 23))+1;

  d = mla(qh, -PI_A*0.5*(1LL << 24), d);
  d = mla(ql, -PI_A*0.5,             d);
  d = mla(qh, -PI_B*0.5*(1LL << 24), d);
  d = mla(ql, -PI_B*0.5,             d);
  d = mla(qh, -PI_C*0.5*(1LL << 24), d);
  d = mla(ql, -PI_C*0.5,             d);
  d = mla((double)qh*(1LL << 24) + ql , -PI_D*0.5, d);
  
  s = d * d;

  if ((ql & 2) == 0) d = -d;

  u = -7.97255955009037868891952e-18;
  u = mla(u, s, 2.81009972710863200091251e-15);
  u = mla(u, s, -7.64712219118158833288484e-13);
  u = mla(u, s, 1.60590430605664501629054e-10);
  u = mla(u, s, -2.50521083763502045810755e-08);
  u = mla(u, s, 2.75573192239198747630416e-06);
  u = mla(u, s, -0.000198412698412696162806809);
  u = mla(u, s, 0.00833333333333332974823815);
  u = mla(u, s, -0.166666666666666657414808);

  u = mla(s, u * d, d);

  if (!xisinf(t) && fabsk(t) > TRIGRANGEMAX) u = 0.0;

  return u;
}

EXPORT CONST double xcos_u1(double d) {
  double u;
  Sleef_double2 s, t, x;

  d = fabsk(d);

  int qh = trunck(d * (M_1_PI / (1LL << (23))) - 0.5 * (M_1_PI / (1LL << (23))));
  int ql = 2*rintk(d * M_1_PI - 0.5 - qh * (double)(1LL << (23)))+1;

  s = ddadd2_d2_d_d (d, qh * (-PI_A*0.5 * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_A*0.5            ));
  s = ddadd2_d2_d2_d(s, qh * (-PI_B*0.5 * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_B*0.5            ));
  s = ddadd2_d2_d2_d(s, qh * (-PI_C*0.5 * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_C*0.5            ));
  s = ddadd2_d2_d2_d(s, ((double)qh * (1 << 24) + ql) * (-PI_D*0.5));
  
  t = s;
  s = ddsqu_d2_d2(s);

  u = 2.72052416138529567917983e-15;
  u = mla(u, s.x, -7.6429259411395447190023e-13);
  u = mla(u, s.x, 1.60589370117277896211623e-10);
  u = mla(u, s.x, -2.5052106814843123359368e-08);
  u = mla(u, s.x, 2.75573192104428224777379e-06);
  u = mla(u, s.x, -0.000198412698412046454654947);
  u = mla(u, s.x, 0.00833333333333318056201922);

  x = ddadd_d2_d_d2(1, ddmul_d2_d2_d2(ddadd_d2_d_d(-0.166666666666666657414808, u * s.x), s));

  x = ddmul_d2_d2_d2(t, x);

  u = x.x + x.y;

  if ((((int)ql) & 2) == 0) u = -u;
  if (!xisinf(d) && d > TRIGRANGEMAX) u = 0.0;

  return u;
}

EXPORT CONST Sleef_double2 xsincos(double d) {
  double u, s, t;
  Sleef_double2 r;

  s = d;

  int qh = trunck(d * ((2 * M_1_PI) / (1 << 24)));
  int ql = rintk(d * (2 * M_1_PI) - qh * (double)(1 << 24));

  s = mla(qh, -PI_A * 0.5 * (1 << 24), s);
  s = mla(ql, -PI_A * 0.5,             s);
  s = mla(qh, -PI_B * 0.5 * (1 << 24), s);
  s = mla(ql, -PI_B * 0.5,             s);
  s = mla(qh, -PI_C * 0.5 * (1 << 24), s);
  s = mla(ql, -PI_C * 0.5,             s);
  s = mla((double)qh * (1 << 24) + ql, -PI_D * 0.5, s);
  
  t = s;

  s = s * s;
  
  u = 1.58938307283228937328511e-10;
  u = mla(u, s, -2.50506943502539773349318e-08);
  u = mla(u, s, 2.75573131776846360512547e-06);
  u = mla(u, s, -0.000198412698278911770864914);
  u = mla(u, s, 0.0083333333333191845961746);
  u = mla(u, s, -0.166666666666666130709393);
  u = u * s * t;

  r.x = t + u;

  if (xisnegzero(d)) r.x = -0.0;

  u = -1.13615350239097429531523e-11;
  u = mla(u, s, 2.08757471207040055479366e-09);
  u = mla(u, s, -2.75573144028847567498567e-07);
  u = mla(u, s, 2.48015872890001867311915e-05);
  u = mla(u, s, -0.00138888888888714019282329);
  u = mla(u, s, 0.0416666666666665519592062);
  u = mla(u, s, -0.5);

  r.y = u * s + 1;

  if ((ql & 1) != 0) { s = r.y; r.y = r.x; r.x = s; }
  if ((ql & 2) != 0) { r.x = -r.x; }
  if (((ql+1) & 2) != 0) { r.y = -r.y; }

  if (fabsk(d) > TRIGRANGEMAX) { r.x = r.y = 0; }
  if (xisinf(d)) { r.x = r.y = NAN; }

  return r;
}

EXPORT CONST Sleef_double2 xsincos_u1(double d) {
  double u;
  Sleef_double2 r, s, t, x;

  int qh = trunck(d * ((2 * M_1_PI) / (1 << 24)));
  int ql = rintk(d * (2 * M_1_PI) - qh * (double)(1 << 24));

  s = ddadd2_d2_d_d (d, qh * (-PI_A*0.5 * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_A*0.5            ));
  s = ddadd2_d2_d2_d(s, qh * (-PI_B*0.5 * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_B*0.5            ));
  s = ddadd2_d2_d2_d(s, qh * (-PI_C*0.5 * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_C*0.5            ));
  s = ddadd2_d2_d2_d(s, ((double)qh * (1 << 24) + ql) * (-PI_D*0.5));
  
  t = s;
  s = ddsqu_d2_d2(s);
  s.x = s.x + s.y;

  u = 1.58938307283228937328511e-10;
  u = mla(u, s.x, -2.50506943502539773349318e-08);
  u = mla(u, s.x, 2.75573131776846360512547e-06);
  u = mla(u, s.x, -0.000198412698278911770864914);
  u = mla(u, s.x, 0.0083333333333191845961746);
  u = mla(u, s.x, -0.166666666666666130709393);

  u *= s.x * t.x;

  x = ddadd_d2_d2_d(t, u);
  r.x = x.x + x.y;

  if (xisnegzero(d)) r.x = -0.0;

  u = -1.13615350239097429531523e-11;
  u = mla(u, s.x, 2.08757471207040055479366e-09);
  u = mla(u, s.x, -2.75573144028847567498567e-07);
  u = mla(u, s.x, 2.48015872890001867311915e-05);
  u = mla(u, s.x, -0.00138888888888714019282329);
  u = mla(u, s.x, 0.0416666666666665519592062);
  u = mla(u, s.x, -0.5);

  x = ddadd_d2_d_d2(1, ddmul_d2_d_d(s.x, u));
  r.y = x.x + x.y;

  if ((ql & 1) != 0) { u = r.y; r.y = r.x; r.x = u; }
  if ((ql & 2) != 0) { r.x = -r.x; }
  if (((ql+1) & 2) != 0) { r.y = -r.y; }

  if (fabsk(d) > TRIGRANGEMAX) { r.x = r.y = 0; }
  if (xisinf(d)) { r.x = r.y = NAN; }

  return r;
}

#define TRIGRANGEMAX2 1e+9

EXPORT CONST Sleef_double2 xsincospi_u05(double d) {
  double u, s, t;
  Sleef_double2 r, x, s2;

  u = d * 4;
  int q = ceilk(u) & ~(int)1;
  
  s = u - (double)q;
  t = s;
  s = s * s;
  s2 = ddmul_d2_d_d(t, t);
  
  //
  
  u = -2.02461120785182399295868e-14;
  u = mla(u, s, 6.94821830580179461327784e-12);
  u = mla(u, s, -1.75724749952853179952664e-09);
  u = mla(u, s, 3.13361688966868392878422e-07);
  u = mla(u, s, -3.6576204182161551920361e-05);
  u = mla(u, s, 0.00249039457019271850274356);
  x = ddadd2_d2_d_d2(u * s, dd(-0.0807455121882807852484731, 3.61852475067037104849987e-18));
  x = ddadd2_d2_d2_d2(ddmul_d2_d2_d2(s2, x), dd(0.785398163397448278999491, 3.06287113727155002607105e-17));

  x = ddmul_d2_d2_d(x, t);
  r.x = x.x + x.y;

  if (xisnegzero(d)) r.x = -0.0;
  
  //

  u = 9.94480387626843774090208e-16;
  u = mla(u, s, -3.89796226062932799164047e-13);
  u = mla(u, s, 1.15011582539996035266901e-10);
  u = mla(u, s, -2.4611369501044697495359e-08);
  u = mla(u, s, 3.59086044859052754005062e-06);
  u = mla(u, s, -0.000325991886927389905997954);
  x = ddadd2_d2_d_d2(u * s, dd(0.0158543442438155018914259, -1.04693272280631521908845e-18));
  x = ddadd2_d2_d2_d2(ddmul_d2_d2_d2(s2, x), dd(-0.308425137534042437259529, -1.95698492133633550338345e-17));

  x = ddadd2_d2_d2_d(ddmul_d2_d2_d2(x, s2), 1);
  r.y = x.x + x.y;

  //

  if ((q & 2) != 0) { s = r.y; r.y = r.x; r.x = s; }
  if ((q & 4) != 0) { r.x = -r.x; }
  if (((q+2) & 4) != 0) { r.y = -r.y; }

  if (fabsk(d) > TRIGRANGEMAX2/4) { r.x = r.y = 0; }
  if (xisinf(d)) { r.x = r.y = NAN; }

  return r;
}

EXPORT CONST Sleef_double2 xsincospi_u35(double d) {
  double u, s, t;
  Sleef_double2 r;

  u = d * 4;
  int q = ceilk(u) & ~(int)1;
  
  s = u - (double)q;
  t = s;
  s = s * s;
  
  //
  
  u = +0.6880638894766060136e-11;
  u = mla(u, s, -0.1757159564542310199e-8);
  u = mla(u, s, +0.3133616327257867311e-6);
  u = mla(u, s, -0.3657620416388486452e-4);
  u = mla(u, s, +0.2490394570189932103e-2);
  u = mla(u, s, -0.8074551218828056320e-1);
  u = mla(u, s, +0.7853981633974482790e+0);
  
  r.x = u * t;

  //

  u = -0.3860141213683794352e-12;
  u = mla(u, s, +0.1150057888029681415e-9);
  u = mla(u, s, -0.2461136493006663553e-7);
  u = mla(u, s, +0.3590860446623516713e-5);
  u = mla(u, s, -0.3259918869269435942e-3);
  u = mla(u, s, +0.1585434424381541169e-1);
  u = mla(u, s, -0.3084251375340424373e+0);
  u = mla(u, s, 1);

  r.y = u;

  //
  
  if ((q & 2) != 0) { s = r.y; r.y = r.x; r.x = s; }
  if ((q & 4) != 0) { r.x = -r.x; }
  if (((q+2) & 4) != 0) { r.y = -r.y; }

  if (fabsk(d) > TRIGRANGEMAX2/4) { r.x = r.y = 0; }
  if (xisinf(d)) { r.x = r.y = NAN; }

  return r;
}

EXPORT CONST double xtan(double d) {
  double u, s, x;

  int qh = trunck(d * ((2 * M_1_PI) / (1 << 24)));
  int ql = rintk(d * (2 * M_1_PI) - qh * (double)(1 << 24));

  x = mla(qh, -PI_A * 0.5 * (1 << 24), d);
  x = mla(ql, -PI_A * 0.5,             x);
  x = mla(qh, -PI_B * 0.5 * (1 << 24), x);
  x = mla(ql, -PI_B * 0.5,             x);
  x = mla(qh, -PI_C * 0.5 * (1 << 24), x);
  x = mla(ql, -PI_C * 0.5,             x);
  x = mla((double)qh * (1 << 24) + ql, -PI_D * 0.5, x);
  
  s = x * x;

  if ((ql & 1) != 0) x = -x;

  u = 9.99583485362149960784268e-06;
  u = mla(u, s, -4.31184585467324750724175e-05);
  u = mla(u, s, 0.000103573238391744000389851);
  u = mla(u, s, -0.000137892809714281708733524);
  u = mla(u, s, 0.000157624358465342784274554);
  u = mla(u, s, -6.07500301486087879295969e-05);
  u = mla(u, s, 0.000148898734751616411290179);
  u = mla(u, s, 0.000219040550724571513561967);
  u = mla(u, s, 0.000595799595197098359744547);
  u = mla(u, s, 0.00145461240472358871965441);
  u = mla(u, s, 0.0035923150771440177410343);
  u = mla(u, s, 0.00886321546662684547901456);
  u = mla(u, s, 0.0218694899718446938985394);
  u = mla(u, s, 0.0539682539049961967903002);
  u = mla(u, s, 0.133333333334818976423364);
  u = mla(u, s, 0.333333333333320047664472);
  
  u = mla(s, u * x, x);

  if ((ql & 1) != 0) u = 1.0 / u;

  if (xisinf(d)) u = NAN;

  return u;
}

EXPORT CONST double xtan_u1(double d) {
  double u;
  Sleef_double2 s, t, x;

  int qh = trunck(d * (M_2_PI / (1 << 24)));
  s = ddadd2_d2_d2_d(ddmul_d2_d2_d(dd(M_2_PI_H, M_2_PI_L), d), (d < 0 ? -0.5 : 0.5) - qh * (double)(1 << 24));
  int ql = s.x + s.y;
  
  s = ddadd2_d2_d_d (d, qh * (-PI_A*0.5 * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_A*0.5            ));
  s = ddadd2_d2_d2_d(s, qh * (-PI_B*0.5 * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_B*0.5            ));
  s = ddadd2_d2_d2_d(s, qh * (-PI_C*0.5 * (1 << 24)));
  s = ddadd2_d2_d2_d(s, ql * (-PI_C*0.5            ));
  s = ddadd2_d2_d2_d(s, ((double)qh * (1 << 24) + ql) * (-PI_D*0.5));
  
  if ((ql & 1) != 0) s = ddneg_d2_d2(s);

  t = s;
  s = ddsqu_d2_d2(s);

  u = 1.01419718511083373224408e-05;
  u = mla(u, s.x, -2.59519791585924697698614e-05);
  u = mla(u, s.x, 5.23388081915899855325186e-05);
  u = mla(u, s.x, -3.05033014433946488225616e-05);
  u = mla(u, s.x, 7.14707504084242744267497e-05);
  u = mla(u, s.x, 8.09674518280159187045078e-05);
  u = mla(u, s.x, 0.000244884931879331847054404);
  u = mla(u, s.x, 0.000588505168743587154904506);
  u = mla(u, s.x, 0.00145612788922812427978848);
  u = mla(u, s.x, 0.00359208743836906619142924);
  u = mla(u, s.x, 0.00886323944362401618113356);
  u = mla(u, s.x, 0.0218694882853846389592078);
  u = mla(u, s.x, 0.0539682539781298417636002);
  u = mla(u, s.x, 0.133333333333125941821962);

  x = ddadd_d2_d_d2(1, ddmul_d2_d2_d2(ddadd_d2_d_d(0.333333333333334980164153, u * s.x), s));
  x = ddmul_d2_d2_d2(t, x);

  if ((ql & 1) != 0) x = ddrec_d2_d2(x);

  u = x.x + x.y;

  if (!xisinf(d) && (xisnegzero(d) || fabsk(d) > TRIGRANGEMAX)) u = -0.0;
  
  return u;
}

EXPORT CONST double xlog(double d) {
  double x, x2, t, m;
  int e;

  e = ilogbk(d * (1.0/0.75));
  m = ldexpk(d, -e);

  x = (m-1) / (m+1);
  x2 = x * x;

  t = 0.153487338491425068243146;
  t = mla(t, x2, 0.152519917006351951593857);
  t = mla(t, x2, 0.181863266251982985677316);
  t = mla(t, x2, 0.222221366518767365905163);
  t = mla(t, x2, 0.285714294746548025383248);
  t = mla(t, x2, 0.399999999950799600689777);
  t = mla(t, x2, 0.6666666666667778740063);
  t = mla(t, x2, 2);
  
  x = x * t + 0.693147180559945286226764 * e;
  
  if (xisinf(d)) x = INFINITY;
  if (d < 0) x = NAN;
  if (d == 0) x = -INFINITY;

  return x;
}

EXPORT CONST double xexp(double d) {
  int q = (int)rintk(d * R_LN2);
  double s, u;

  s = mla(q, -L2U, d);
  s = mla(q, -L2L, s);

  u = 2.08860621107283687536341e-09;
  u = mla(u, s, 2.51112930892876518610661e-08);
  u = mla(u, s, 2.75573911234900471893338e-07);
  u = mla(u, s, 2.75572362911928827629423e-06);
  u = mla(u, s, 2.4801587159235472998791e-05);
  u = mla(u, s, 0.000198412698960509205564975);
  u = mla(u, s, 0.00138888888889774492207962);
  u = mla(u, s, 0.00833333333331652721664984);
  u = mla(u, s, 0.0416666666666665047591422);
  u = mla(u, s, 0.166666666666666851703837);
  u = mla(u, s, 0.5);

  u = s * s * u + s + 1;
  u = ldexpk(u, q);

  if (d < -1000) u = 0;
  
  return u;
}

static INLINE CONST Sleef_double2 logk(double d) {
  Sleef_double2 x, x2;
  double m, t;
  int e;

  e = ilogbk(d * (1.0/0.75));
  m = ldexpk(d, -e);

  x = dddiv_d2_d2_d2(ddadd2_d2_d_d(-1, m), ddadd2_d2_d_d(1, m));
  x2 = ddsqu_d2_d2(x);

  t = 0.116255524079935043668677;
  t = mla(t, x2.x, 0.103239680901072952701192);
  t = mla(t, x2.x, 0.117754809412463995466069);
  t = mla(t, x2.x, 0.13332981086846273921509);
  t = mla(t, x2.x, 0.153846227114512262845736);
  t = mla(t, x2.x, 0.181818180850050775676507);
  t = mla(t, x2.x, 0.222222222230083560345903);
  t = mla(t, x2.x, 0.285714285714249172087875);
  t = mla(t, x2.x, 0.400000000000000077715612);
  Sleef_double2 c = dd(0.666666666666666629659233, 3.80554962542412056336616e-17);

  return ddadd2_d2_d2_d2(ddmul_d2_d2_d(dd(0.693147180559945286226764, 2.319046813846299558417771e-17), e),
			 ddadd2_d2_d2_d2(ddscale_d2_d2_d(x, 2),
					 ddmul_d2_d2_d2(ddmul_d2_d2_d2(x2, x),
							ddadd2_d2_d2_d2(ddmul_d2_d2_d(x2, t), c))));
}

EXPORT CONST double xlog_u1(double d) {
  Sleef_double2 s = logk(d);
  double x = s.x + s.y;

  if (xisinf(d)) x = INFINITY;
  if (d < 0) x = NAN;
  if (d == 0) x = -INFINITY;

  return x;
}

static INLINE CONST double expk(Sleef_double2 d) {
  int q = (int)rintk((d.x + d.y) * R_LN2);
  Sleef_double2 s, t;
  double u;

  s = ddadd2_d2_d2_d(d, q * -L2U);
  s = ddadd2_d2_d2_d(s, q * -L2L);

  s = ddnormalize_d2_d2(s);

  u = 2.51069683420950419527139e-08;
  u = mla(u, s.x, 2.76286166770270649116855e-07);
  u = mla(u, s.x, 2.75572496725023574143864e-06);
  u = mla(u, s.x, 2.48014973989819794114153e-05);
  u = mla(u, s.x, 0.000198412698809069797676111);
  u = mla(u, s.x, 0.0013888888939977128960529);
  u = mla(u, s.x, 0.00833333333332371417601081);
  u = mla(u, s.x, 0.0416666666665409524128449);
  u = mla(u, s.x, 0.166666666666666740681535);
  u = mla(u, s.x, 0.500000000000000999200722);

  t = ddadd_d2_d2_d2(s, ddmul_d2_d2_d(ddsqu_d2_d2(s), u));

  t = ddadd_d2_d_d2(1, t);

  u = ldexpk(t.x + t.y, q);

  if (d.x < -1000) u = 0;

  return u;
}

EXPORT CONST double xpow(double x, double y) {
  int yisint = xisint(y);
  int yisodd = yisint && xisodd(y);

  double result = expk(ddmul_d2_d2_d(logk(fabsk(x)), y));

  result = xisnan(result) ? INFINITY : result;
  result *= (x > 0 ? 1 : (!yisint ? NAN : (yisodd ? -1 : 1)));

  double efx = mulsign(fabsk(x) - 1, y);
  if (xisinf(y)) result = efx < 0 ? 0.0 : (efx == 0 ? 1.0 : INFINITY);
  if (xisinf(x) || x == 0) result = (yisodd ? sign(x) : 1) * ((x == 0 ? -y : y) < 0 ? 0 : INFINITY);
  if (xisnan(x) || xisnan(y)) result = NAN;
  if (y == 0 || x == 1) result = 1;

  return result;
}

static INLINE CONST Sleef_double2 expk2(Sleef_double2 d) {
  int q = (int)rintk((d.x + d.y) * R_LN2);
  Sleef_double2 s, t;
  double u;

  s = ddadd2_d2_d2_d(d, q * -L2U);
  s = ddadd2_d2_d2_d(s, q * -L2L);

  u = 2.51069683420950419527139e-08;
  u = mla(u, s.x, 2.76286166770270649116855e-07);
  u = mla(u, s.x, 2.75572496725023574143864e-06);
  u = mla(u, s.x, 2.48014973989819794114153e-05);
  u = mla(u, s.x, 0.000198412698809069797676111);
  u = mla(u, s.x, 0.0013888888939977128960529);
  u = mla(u, s.x, 0.00833333333332371417601081);
  u = mla(u, s.x, 0.0416666666665409524128449);
  u = mla(u, s.x, 0.166666666666666740681535);
  u = mla(u, s.x, 0.500000000000000999200722);

  t = ddadd_d2_d2_d2(s, ddmul_d2_d2_d(ddsqu_d2_d2(s), u));

  t = ddadd_d2_d_d2(1, t);
  return ddscale_d2_d2_d(ddscale_d2_d2_d(t, 2), pow2i(q-1));
}

EXPORT CONST double xsinh(double x) {
  double y = fabsk(x);
  Sleef_double2 d = expk2(dd(y, 0));
  d = ddsub_d2_d2_d2(d, ddrec_d2_d2(d));
  y = (d.x + d.y) * 0.5;

  y = fabsk(x) > 710 ? INFINITY : y;
  y = xisnan(y) ? INFINITY : y;
  y = mulsign(y, x);
  y = xisnan(x) ? NAN : y;

  return y;
}

EXPORT CONST double xcosh(double x) {
  double y = fabsk(x);
  Sleef_double2 d = expk2(dd(y, 0));
  d = ddadd_d2_d2_d2(d, ddrec_d2_d2(d));
  y = (d.x + d.y) * 0.5;

  y = fabsk(x) > 710 ? INFINITY : y;
  y = xisnan(y) ? INFINITY : y;
  y = xisnan(x) ? NAN : y;

  return y;
}

EXPORT CONST double xtanh(double x) {
  double y = fabsk(x);
  Sleef_double2 d = expk2(dd(y, 0));
  Sleef_double2 e = ddrec_d2_d2(d);
  d = dddiv_d2_d2_d2(ddsub_d2_d2_d2(d, e), ddadd_d2_d2_d2(d, e));
  y = d.x + d.y;

  y = fabsk(x) > 18.714973875 ? 1.0 : y;
  y = xisnan(y) ? 1.0 : y;
  y = mulsign(y, x);
  y = xisnan(x) ? NAN : y;

  return y;
}

static INLINE CONST Sleef_double2 logk2(Sleef_double2 d) {
  Sleef_double2 x, x2, m;
  double t;
  int e;
  
  e = ilogbk(d.x * (1.0/0.75));
  m = ddscale_d2_d2_d(d, pow2i(-e));

  x = dddiv_d2_d2_d2(ddadd2_d2_d2_d(m, -1), ddadd2_d2_d2_d(m, 1));
  x2 = ddsqu_d2_d2(x);
  
  t = 0.13860436390467167910856;
  t = mla(t, x2.x, 0.131699838841615374240845);
  t = mla(t, x2.x, 0.153914168346271945653214);
  t = mla(t, x2.x, 0.181816523941564611721589);
  t = mla(t, x2.x, 0.22222224632662035403996);
  t = mla(t, x2.x, 0.285714285511134091777308);
  t = mla(t, x2.x, 0.400000000000914013309483);
  t = mla(t, x2.x, 0.666666666666664853302393);

  return ddadd2_d2_d2_d2(ddmul_d2_d2_d(dd(0.693147180559945286226764, 2.319046813846299558417771e-17), e),
			 ddadd2_d2_d2_d2(ddscale_d2_d2_d(x, 2), ddmul_d2_d2_d(ddmul_d2_d2_d2(x2, x), t)));
}

EXPORT CONST double xasinh(double x) {
  double y = fabsk(x);
  Sleef_double2 d;

  d = y > 1 ? ddrec_d2_d(x) : dd(y, 0);
  d = ddsqrt_d2_d2(ddadd2_d2_d2_d(ddsqu_d2_d2(d), 1));
  d = y > 1 ? ddmul_d2_d2_d(d, y) : d;
  
  d = logk2(ddnormalize_d2_d2(ddadd_d2_d2_d(d, x)));
  y = d.x + d.y;

  y = (fabsk(x) > SQRT_DBL_MAX || xisnan(y)) ? mulsign(INFINITY, x) : y;
  y = xisnan(x) ? NAN : y;
  y = xisnegzero(x) ? -0.0 : y;
  
  return y;
}

EXPORT CONST double xacosh(double x) {
  Sleef_double2 d = logk2(ddadd2_d2_d2_d(ddmul_d2_d2_d2(ddsqrt_d2_d2(ddadd2_d2_d_d(x, 1)), ddsqrt_d2_d2(ddadd2_d2_d_d(x, -1))), x));
  double y = d.x + d.y;

  y = (x > SQRT_DBL_MAX || xisnan(y)) ? INFINITY : y;
  y = x == 1.0 ? 0.0 : y;
  y = x < 1.0 ? NAN : y;
  y = xisnan(x) ? NAN : y;

  return y;
}

EXPORT CONST double xatanh(double x) {
  double y = fabsk(x);
  Sleef_double2 d = logk2(dddiv_d2_d2_d2(ddadd2_d2_d_d(1, y), ddadd2_d2_d_d(1, -y)));
  y = y > 1.0 ? NAN : (y == 1.0 ? INFINITY : (d.x + d.y) * 0.5);

  y = mulsign(y, x);
  y = (xisinf(x) || xisnan(y)) ? NAN : y;

  return y;
}

//

EXPORT CONST double xcbrt(double d) { // max error : 2 ulps
  double x, y, q = 1.0;
  int e, r;

  e = ilogbk(fabsk(d))+1;
  d = ldexpk(d, -e);
  r = (e + 6144) % 3;
  q = (r == 1) ? 1.2599210498948731647672106 : q;
  q = (r == 2) ? 1.5874010519681994747517056 : q;
  q = ldexpk(q, (e + 6144) / 3 - 2048);

  q = mulsign(q, d);
  d = fabsk(d);

  x = -0.640245898480692909870982;
  x = mla(x, d, 2.96155103020039511818595);
  x = mla(x, d, -5.73353060922947843636166);
  x = mla(x, d, 6.03990368989458747961407);
  x = mla(x, d, -3.85841935510444988821632);
  x = mla(x, d, 2.2307275302496609725722);

  y = x * x; y = y * y; x -= (d * y - x) * (1.0 / 3.0);
  y = d * x * x;
  y = (y - (2.0 / 3.0) * y * (y * x - 1)) * q;

  return y;
}

EXPORT CONST double xcbrt_u1(double d) {
  double x, y, z;
  Sleef_double2 q2 = dd(1, 0), u, v;
  int e, r;

  e = ilogbk(fabsk(d))+1;
  d = ldexpk(d, -e);
  r = (e + 6144) % 3;
  q2 = (r == 1) ? dd(1.2599210498948731907, -2.5899333753005069177e-17) : q2;
  q2 = (r == 2) ? dd(1.5874010519681995834, -1.0869008194197822986e-16) : q2;

  q2.x = mulsign(q2.x, d); q2.y = mulsign(q2.y, d);
  d = fabsk(d);

  x = -0.640245898480692909870982;
  x = mla(x, d, 2.96155103020039511818595);
  x = mla(x, d, -5.73353060922947843636166);
  x = mla(x, d, 6.03990368989458747961407);
  x = mla(x, d, -3.85841935510444988821632);
  x = mla(x, d, 2.2307275302496609725722);

  y = x * x; y = y * y; x -= (d * y - x) * (1.0 / 3.0);

  z = x;

  u = ddmul_d2_d_d(x, x);
  u = ddmul_d2_d2_d2(u, u);
  u = ddmul_d2_d2_d(u, d);
  u = ddadd2_d2_d2_d(u, -x);
  y = u.x + u.y;

  y = -2.0 / 3.0 * y * z;
  v = ddadd2_d2_d2_d(ddmul_d2_d_d(z, z), y);
  v = ddmul_d2_d2_d(v, d);
  v = ddmul_d2_d2_d2(v, q2);
  z = ldexpk(v.x + v.y, (e + 6144) / 3 - 2048);

  if (xisinf(d)) { z = mulsign(INFINITY, q2.x); }
  if (d == 0) { z = mulsign(0, q2.x); }

  return z;
}

EXPORT CONST double xexp2(double a) {
  double u = expk(ddmul_d2_d2_d(dd(0.69314718055994528623, 2.3190468138462995584e-17), a));
  if (a > 1024) u = INFINITY; // log2(DBL_MAX)
  if (xisminf(a)) u = 0;
  return u;
}

EXPORT CONST double xexp10(double a) {
  double u = expk(ddmul_d2_d2_d(dd(2.3025850929940459011, -2.1707562233822493508e-16), a));
  if (a > 308.254715559916743850652254) u = INFINITY; // log10(DBL_MAX)
  if (xisminf(a)) u = 0;
  return u;
}

EXPORT CONST double xexpm1(double a) {
  Sleef_double2 d = ddadd2_d2_d2_d(expk2(dd(a, 0)), -1.0);
  double x = d.x + d.y;
  if (a > 709.782712893383996732223) x = INFINITY; // log(DBL_MAX)
  if (a < -36.736800569677101399113302437) x = -1; // log(1 - nexttoward(1, 0))
  if (xisnegzero(a)) x = -0.0;
  return x;
}

EXPORT CONST double xlog10(double a) {
  Sleef_double2 d = ddmul_d2_d2_d2(logk(a), dd(0.43429448190325176116, 6.6494347733425473126e-17));
  double x = d.x + d.y;

  if (xisinf(a)) x = INFINITY;
  if (a < 0) x = NAN;
  if (a == 0) x = -INFINITY;

  return x;
}

EXPORT CONST double xlog1p(double a) {
  Sleef_double2 d = logk2(ddadd2_d2_d_d(a, 1));
  double x = d.x + d.y;

  if (a > 1e+307) x = INFINITY;
  if (a < -1) x = NAN;
  if (a == -1) x = -INFINITY;
  if (xisnegzero(a)) x = -0.0;

  return x;
}

//

EXPORT CONST double xfma(double x, double y, double z) {
  double h2 = x * y + z, q = 1;
  if (fabsk(h2) < 1e-300) {
    const double c0 = 1ULL << 54, c1 = c0 * c0, c2 = c1 * c1;
    x *= c1;
    y *= c1;
    z *= c2;
    q = 1.0 / c2;
  }
  Sleef_double2 d = ddmul_d2_d_d(x, y);
  d = ddadd2_d2_d2_d(d, z);
  if (xisinf(z) && !xisinf(x) && !xisnan(x) && !xisinf(y) && !xisnan(y)) h2 = z;
  return (xisinf(h2) || xisnan(h2)) ? h2 : (d.x + d.y)*q;
}

EXPORT CONST double xsqrt(double d) {
  double q = 1;

  d = d < 0 ? NAN : d;
  
  if (d < 8.636168555094445E-78) {
    d *= 1.157920892373162E77;
    q = 2.9387358770557188E-39;
  }

  if (d > 1.3407807929942597e+154) {
    d *= 7.4583407312002070e-155;
    q = 1.1579208923731620e+77;
  }
  
  // http://en.wikipedia.org/wiki/Fast_inverse_square_root
  double x = longBitsToDouble(0x5fe6ec85e7de30da - (doubleToRawLongBits(d + 1e-320) >> 1));

  x = x * (1.5 - 0.5 * d * x * x);
  x = x * (1.5 - 0.5 * d * x * x);
  x = x * (1.5 - 0.5 * d * x * x) * d;

  Sleef_double2 d2 = ddmul_d2_d2_d2(ddadd2_d2_d_d2(d, ddmul_d2_d_d(x, x)), ddrec_d2_d(x));

  return d == INFINITY ? INFINITY : (d2.x + d2.y) * (0.5 * q);
}

EXPORT CONST double xfabs(double x) { return fabsk(x); }

EXPORT CONST double xcopysign(double x, double y) { return copysignk(x, y); }

EXPORT CONST double xfmax(double x, double y) {
  return y != y ? x : (x > y ? x : y);
}

EXPORT CONST double xfmin(double x, double y) {
  return y != y ? x : (x < y ? x : y);
}

EXPORT CONST double xfdim(double x, double y) {
  double ret = x - y;
  if (ret < 0 || x == y) ret = 0;
  return ret;
}

EXPORT CONST double xtrunc(double x) {
  double fr = x - (double)(1 << 31) * (int32_t)(x * (1.0 / (1 << 31)));
  fr = fr - (int32_t)fr;
  return (xisinf(x) || fabsk(x) > (double)(1LL << 52)) ? x : copysignk(x - fr, x);
}

EXPORT CONST double xfloor(double x) {
  double fr = x - (double)(1 << 31) * (int32_t)(x * (1.0 / (1 << 31)));
  fr = fr - (int32_t)fr;
  fr = fr < 0 ? fr+1.0 : fr;
  return (xisinf(x) || fabsk(x) > (double)(1LL << 52)) ? x : copysignk(x - fr, x);
}

EXPORT CONST double xceil(double x) {
  double fr = x - (double)(1 << 31) * (int32_t)(x * (1.0 / (1 << 31)));
  fr = fr - (int32_t)fr;
  fr = fr <= 0 ? fr : fr-1.0;
  return (xisinf(x) || fabsk(x) > (double)(1LL << 52)) ? x : copysignk(x - fr, x);
}

EXPORT CONST double xround(double d) {
  double x = d + 0.5;
  double fr = x - (double)(1 << 31) * (int32_t)(x * (1.0 / (1 << 31)));
  fr = fr - (int32_t)fr;
  if (fr == 0 && x <= 0) x--;
  fr = fr < 0 ? fr+1.0 : fr;
  return (xisinf(x) || fabsk(x) > (double)(1LL << 52)) ? d : copysignk(x - fr, d);
}

EXPORT CONST double xrint(double d) {
  double x = d + 0.5;
  double fr = x - (double)(1 << 31) * (int32_t)(x * (1.0 / (1 << 31)));
  int32_t isodd = (1 & (int32_t)fr) != 0;
  fr = fr - (int32_t)fr;
  fr = (fr < 0 || (fr == 0 && isodd)) ? fr+1.0 : fr;
  return (xisinf(x) || fabsk(x) > (double)(1LL << 52)) ? d : copysignk(x - fr, d);
}

EXPORT CONST double xhypot_u05(double x, double y) {
  x = fabsk(x);
  y = fabsk(y);
  double min = fmink(x, y), n = min;
  double max = fmaxk(x, y), d = max;

  if (max < DBL_MIN) { n *= 1ULL << 54; d *= 1ULL << 54; }
  Sleef_double2 t = dddiv_d2_d2_d2(dd(n, 0), dd(d, 0));
  t = ddmul_d2_d2_d(ddsqrt_d2_d2(ddadd2_d2_d2_d(ddsqu_d2_d2(t), 1)), max);
  double ret = t.x + t.y;
  if (xisnan(ret)) ret = INFINITY;
  if (min == 0) ret = max;
  if (xisnan(x) || xisnan(y)) ret = NAN;
  if (x == INFINITY || y == INFINITY) ret = INFINITY;
  return ret;
}

EXPORT CONST double xhypot_u35(double x, double y) {
  x = fabsk(x);
  y = fabsk(y);
  double min = fmink(x, y);
  double max = fmaxk(x, y);

  double t = min / max;
  double ret = max * sqrt(1 + t*t);
  if (min == 0) ret = max;
  if (xisnan(x) || xisnan(y)) ret = NAN;
  if (x == INFINITY || y == INFINITY) ret = INFINITY;
  return ret;
}

EXPORT CONST double xnextafter(double x, double y) {
  union {
    double f;
    int64_t i;
  } cx;

  cx.f = x;
  int c = (cx.i < 0) == (y < x);
  if (c) cx.i = -(cx.i ^ (1ULL << 63));

  if (x != y) cx.i--;

  if (c) cx.i = -(cx.i ^ (1ULL << 63));

  if (cx.f == 0 && x != 0) cx.f = mulsign(0, x);
  if (x == 0 && y == 0) cx.f = y;
  if (xisnan(x) || xisnan(y)) cx.f = NAN;
  
  return cx.f;
}

EXPORT CONST double xfrfrexp(double x) {
  union {
    double f;
    uint64_t u;
  } cx;

  if (fabsk(x) < DBL_MIN) x *= (1ULL << 63);
  
  cx.f = x;
  cx.u &= ~0x7ff0000000000000ULL;
  cx.u |=  0x3fe0000000000000ULL;

  if (xisinf(x)) cx.f = mulsign(INFINITY, x);
  if (x == 0) cx.f = x;
  
  return cx.f;
}

EXPORT CONST int xexpfrexp(double x) {
  union {
    double f;
    uint64_t u;
  } cx;

  int ret = 0;
  
  if (fabsk(x) < DBL_MIN) { x *= (1ULL << 63); ret = -63; }
  
  cx.f = x;
  ret += (int32_t)(((cx.u >> 52) & 0x7ff)) - 0x3fe;

  if (x == 0 || xisnan(x) || xisinf(x)) ret = 0;
  
  return ret;
}

static INLINE CONST double nexttoward0(double x) {
  union {
    double f;
    uint64_t u;
  } cx;
  cx.f = x;
  cx.u--;
  return x == 0 ? 0 : cx.f;
}

static INLINE CONST double upper2(double d) {
  return longBitsToDouble(doubleToRawLongBits(d) & 0xfffffffffffffffeLL);
}

EXPORT CONST double xfmod(double x, double y) {
  double nu = fabsk(x), de = fabsk(y), s = 1;
  if (de < DBL_MIN) { nu *= 1ULL << 54; de *= 1ULL << 54; s = 1.0 / (1ULL << 54); }
  Sleef_double2 q, r = dd(nu, 0);

  for(int i=0;i<4;i++) {
    q = ddnormalize_d2_d2(dddiv_d2_d2_d2(r, dd(de, 0)));
    r = ddnormalize_d2_d2(ddadd2_d2_d2_d2(r, ddmul_d2_d_d(-upper2(xtrunc(q.y < 0 ? nexttoward0(q.x) : q.x)), de)));
  }
  
  double ret = (r.x + r.y) * s;
  if (r.x + r.y == de) ret = 0;
  ret = mulsign(ret, x);
  if (fabsk(x) < fabsk(y)) ret = x;

  return ret;
}

EXPORT CONST Sleef_double2 xmodf(double x) {
  double fr = x - (double)(1 << 31) * (int32_t)(x * (1.0 / (1 << 31)));
  fr = fr - (int32_t)fr;
  fr = fabsk(x) >= (double)(1LL << 52) ? 0 : fr;
  Sleef_double2 ret = { copysignk(fr, x), copysignk(x - fr, x) };
  return ret;
}

#if 0
// gcc -I../common sleefdp.c -lm
#include <stdlib.h>
int main(int argc, char **argv) {
  double d1 = atof(argv[1]);
  printf("arg1 = %.20g\n", d1);
  double d2 = atof(argv[2]);
  printf("arg2 = %.20g\n", d2);
#if 0
  double d3 = atof(argv[3]);
  printf("arg3 = %.20g\n", d3);
#endif
  double r = xhypot_u35(d1, d2);
  //double r = xfma(d1, d2, d3);
  printf("test = %.20g\n", r);
  printf("corr = %.20g\n", hypot(d1, d2));
  //printf("%.20g %.20g\n", xround(d1), xrint(d1));
  //Sleef_double2 r = xsincospi_u35(d);
  //printf("%g, %g\n", (double)r.x, (double)r.y);
}
#endif
