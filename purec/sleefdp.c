#include <stdio.h>

#include <assert.h>
#include <stdint.h>
#include <math.h>

#include "nonnumber.h"

#define PI4_A 0.78539816290140151978
#define PI4_B 4.9604678871439933374e-10
#define PI4_C 1.1258708853173288931e-18
#define PI4_D 1.7607799325916000908e-27

#define M_4_PI 1.273239544735162542821171882678754627704620361328125

#define L2U .69314718055966295651160180568695068359375
#define L2L .28235290563031577122588448175013436025525412068e-12
#define R_LN2 1.442695040888963407359924681001892137426645954152985934135449406931

static inline int64_t doubleToRawLongBits(double d) {
  union {
    double f;
    int64_t i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

static inline double longBitsToDouble(int64_t i) {
  union {
    double f;
    int64_t i;
  } tmp;
  tmp.i = i;
  return tmp.f;
}

static inline double xfabs(double x) {
  return longBitsToDouble(0x7fffffffffffffffLL & doubleToRawLongBits(x));
}

static inline double mulsign(double x, double y) {
  return longBitsToDouble(doubleToRawLongBits(x) ^ (doubleToRawLongBits(y) & (1LL << 63)));
}

static inline double sign(double d) { return mulsign(1, d); }
static inline double mla(double x, double y, double z) { return x * y + z; }
static inline double xrint(double x) { return x < 0 ? (int)(x - 0.5) : (int)(x + 0.5); }

static inline int xisnan(double x) { return x != x; }
static inline int xisinf(double x) { return x == INFINITY || x == -INFINITY; }
static inline int xisminf(double x) { return x == -INFINITY; }
static inline int xispinf(double x) { return x == INFINITY; }

static inline double pow2i(int q) {
  return longBitsToDouble(((int64_t)(q + 0x3ff)) << 52);
}

static inline double ldexpk(double x, int q) {
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

double xldexp(double x, int q) { return ldexpk(x, q); }

static inline int ilogbp1(double d) {
  int m = d < 4.9090934652977266E-91;
  d = m ? 2.037035976334486E90 * d : d;
  int q = (doubleToRawLongBits(d) >> 52) & 0x7ff;
  q = m ? q - (300 + 0x03fe) : q - 0x03fe;
  return q;
}

int xilogb(double d) {
  int e = ilogbp1(xfabs(d)) - 1;
  e = d == 0 ? -2147483648 : e;
  e = d == INFINITY || d == -INFINITY ? 2147483647 : e;
  return e;
}

//

typedef struct {
  double x, y;
} double2;

#ifndef NDEBUG
static int checkfp(double x) {
  if (xisinf(x) || xisnan(x)) return 1;
  return 0;
}
#endif

static inline double upper(double d) {
  return longBitsToDouble(doubleToRawLongBits(d) & 0xfffffffff8000000LL);
}

static inline double2 dd(double h, double l) {
  double2 ret;
  ret.x = h; ret.y = l;
  return ret;
}

static inline double2 ddnormalize_d2_d2(double2 t) {
  double2 s;

  s.x = t.x + t.y;
  s.y = t.x - s.x + t.y;

  return s;
}

static inline double2 ddscale_d2_d2_d(double2 d, double s) {
  double2 r;

  r.x = d.x * s;
  r.y = d.y * s;

  return r;
}

static inline double2 ddneg_d2_d2(double2 d) {
  double2 r;

  r.x = -d.x;
  r.y = -d.y;

  return r;
}

static inline double2 ddadd_d2_d_d(double x, double y) {
  // |x| >= |y|

  double2 r;

#ifndef NDEBUG
  if (!(checkfp(x) || checkfp(y) || xfabs(x) >= xfabs(y))) fprintf(stderr, "[ddadd_d2_d_d : %g, %g]", x, y);
#endif

  r.x = x + y;
  r.y = x - r.x + y;

  return r;
}

static inline double2 ddadd2_d2_d_d(double x, double y) {
  double2 r;

  r.x = x + y;
  double v = r.x - x;
  r.y = (x - (r.x - v)) + (y - v);

  return r;
}

static inline double2 ddadd_d2_d2_d(double2 x, double y) {
  // |x| >= |y|

  double2 r;

#ifndef NDEBUG
  if (!(checkfp(x.x) || checkfp(y) || xfabs(x.x) >= xfabs(y))) fprintf(stderr, "[ddadd_d2_d2_d : %g %g]", x.x, y);
#endif

  r.x = x.x + y;
  r.y = x.x - r.x + y + x.y;

  return r;
}

static inline double2 ddadd2_d2_d2_d(double2 x, double y) {
  // |x| >= |y|

  double2 r;

  r.x  = x.x + y;
  double v = r.x - x.x;
  r.y = (x.x - (r.x - v)) + (y - v);
  r.y += x.y;

  return r;
}

static inline double2 ddadd_d2_d_d2(double x, double2 y) {
  // |x| >= |y|

  double2 r;

#ifndef NDEBUG
  if (!(checkfp(x) || checkfp(y.x) || xfabs(x) >= xfabs(y.x))) fprintf(stderr, "[ddadd_d2_d_d2 : %g %g]", x, y.x);
#endif

  r.x = x + y.x;
  r.y = x - r.x + y.x + y.y;

  return r;
}

static inline double2 ddadd2_d2_d_d2(double x, double2 y) {
  double2 r;

  r.x  = x + y.x;
  double v = r.x - x;
  r.y = (x - (r.x - v)) + (y.x - v) + y.y;

  return r;
}

static inline double2 ddadd_d2_d2_d2(double2 x, double2 y) {
  // |x| >= |y|

  double2 r;

#ifndef NDEBUG
  if (!(checkfp(x.x) || checkfp(y.x) || xfabs(x.x) >= xfabs(y.x))) fprintf(stderr, "[ddadd_d2_d2_d2 : %g %g]", x.x, y.x);
#endif

  r.x = x.x + y.x;
  r.y = x.x - r.x + y.x + x.y + y.y;

  return r;
}

static inline double2 ddadd2_d2_d2_d2(double2 x, double2 y) {
  double2 r;

  r.x  = x.x + y.x;
  double v = r.x - x.x;
  r.y = (x.x - (r.x - v)) + (y.x - v);
  r.y += x.y + y.y;

  return r;
}

static inline double2 ddsub_d2_d2_d2(double2 x, double2 y) {
  // |x| >= |y|

  double2 r;

#ifndef NDEBUG
  if (!(checkfp(x.x) || checkfp(y.x) || xfabs(x.x) >= xfabs(y.x))) fprintf(stderr, "[ddsub_d2_d2_d2 : %g %g]", x.x, y.x);
#endif

  r.x = x.x - y.x;
  r.y = x.x - r.x - y.x + x.y - y.y;

  return r;
}

static inline double2 dddiv_d2_d2_d2(double2 n, double2 d) {
  double t = 1.0 / d.x;
  double dh  = upper(d.x), dl  = d.x - dh;
  double th  = upper(t  ), tl  = t   - th;
  double nhh = upper(n.x), nhl = n.x - nhh;

  double2 q;

  q.x = n.x * t;

  double u = -q.x + nhh * th + nhh * tl + nhl * th + nhl * tl +
    q.x * (1 - dh * th - dh * tl - dl * th - dl * tl);

  q.y = t * (n.y - q.x * d.y) + u;

  return q;
}

static inline double2 ddmul_d2_d_d(double x, double y) {
  double xh = upper(x), xl = x - xh;
  double yh = upper(y), yl = y - yh;
  double2 r;

  r.x = x * y;
  r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl;

  return r;
}

static inline double2 ddmul_d2_d2_d(double2 x, double y) {
  double xh = upper(x.x), xl = x.x - xh;
  double yh = upper(y  ), yl = y   - yh;
  double2 r;

  r.x = x.x * y;
  r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.y * y;

  return r;
}

static inline double2 ddmul_d2_d2_d2(double2 x, double2 y) {
  double xh = upper(x.x), xl = x.x - xh;
  double yh = upper(y.x), yl = y.x - yh;
  double2 r;

  r.x = x.x * y.x;
  r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.x * y.y + x.y * y.x;

  return r;
}

static inline double2 ddsqu_d2_d2(double2 x) {
  double xh = upper(x.x), xl = x.x - xh;
  double2 r;

  r.x = x.x * x.x;
  r.y = xh * xh - r.x + (xh + xh) * xl + xl * xl + x.x * (x.y + x.y);

  return r;
}

static inline double2 ddrec_d2_d(double d) {
  double t = 1.0 / d;
  double dh = upper(d), dl = d - dh;
  double th = upper(t), tl = t - th;
  double2 q;

  q.x = t;
  q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl);

  return q;
}

static inline double2 ddrec_d2_d2(double2 d) {
  double t = 1.0 / d.x;
  double dh = upper(d.x), dl = d.x - dh;
  double th = upper(t  ), tl = t   - th;
  double2 q;

  q.x = t;
  q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl - d.y * t);

  return q;
}

static inline double2 ddsqrt_d2_d2(double2 d) {
  double t = sqrt(d.x + d.y);
  return ddscale_d2_d2_d(ddmul_d2_d2_d2(ddadd2_d2_d2_d2(d, ddmul_d2_d_d(t, t)), ddrec_d2_d(t)), 0.5);
}

//

static inline double atan2k(double y, double x) {
  double s, t, u;
  int q = 0;

  if (x < 0) { x = -x; q = -2; }
  if (y > x) { t = x; x = y; y = -t; q += 1; }

  s = y / x;
  t = s * s;

  u = -1.88796008463073496563746e-05;
  u = u * t + (0.000209850076645816976906797);
  u = u * t + (-0.00110611831486672482563471);
  u = u * t + (0.00370026744188713119232403);
  u = u * t + (-0.00889896195887655491740809);
  u = u * t + (0.016599329773529201970117);
  u = u * t + (-0.0254517624932312641616861);
  u = u * t + (0.0337852580001353069993897);
  u = u * t + (-0.0407629191276836500001934);
  u = u * t + (0.0466667150077840625632675);
  u = u * t + (-0.0523674852303482457616113);
  u = u * t + (0.0587666392926673580854313);
  u = u * t + (-0.0666573579361080525984562);
  u = u * t + (0.0769219538311769618355029);
  u = u * t + (-0.090908995008245008229153);
  u = u * t + (0.111111105648261418443745);
  u = u * t + (-0.14285714266771329383765);
  u = u * t + (0.199999999996591265594148);
  u = u * t + (-0.333333333333311110369124);

  t = u * t * s + s;
  t = q * (M_PI/2) + t;

  return t;
}

double xatan2(double y, double x) {
  double r = atan2k(xfabs(y), x);

  r = mulsign(r, x);
  if (xisinf(x) || x == 0) r = M_PI/2 - (xisinf(x) ? (sign(x) * (M_PI  /2)) : 0);
  if (xisinf(y)          ) r = M_PI/2 - (xisinf(x) ? (sign(x) * (M_PI*1/4)) : 0);
  if (             y == 0) r = (sign(x) == -1 ? M_PI : 0);

  return xisnan(x) || xisnan(y) ? NAN : mulsign(r, y);
}

double xasin(double d) {
  return mulsign(atan2k(xfabs(d), sqrt((1+d)*(1-d))), d);
}

double xacos(double d) {
  return mulsign(atan2k(sqrt((1+d)*(1-d)), xfabs(d)), d) + (d < 0 ? M_PI : 0);
}

double xatan(double s) {
  double t, u;
  int q = 0;

  if (s < 0) { s = -s; q = 2; }
  if (s > 1) { s = 1.0 / s; q |= 1; }

  t = s * s;

  u = -1.88796008463073496563746e-05;
  u = u * t + (0.000209850076645816976906797);
  u = u * t + (-0.00110611831486672482563471);
  u = u * t + (0.00370026744188713119232403);
  u = u * t + (-0.00889896195887655491740809);
  u = u * t + (0.016599329773529201970117);
  u = u * t + (-0.0254517624932312641616861);
  u = u * t + (0.0337852580001353069993897);
  u = u * t + (-0.0407629191276836500001934);
  u = u * t + (0.0466667150077840625632675);
  u = u * t + (-0.0523674852303482457616113);
  u = u * t + (0.0587666392926673580854313);
  u = u * t + (-0.0666573579361080525984562);
  u = u * t + (0.0769219538311769618355029);
  u = u * t + (-0.090908995008245008229153);
  u = u * t + (0.111111105648261418443745);
  u = u * t + (-0.14285714266771329383765);
  u = u * t + (0.199999999996591265594148);
  u = u * t + (-0.333333333333311110369124);

  t = s + s * (t * u);

  if ((q & 1) != 0) t = 1.570796326794896557998982 - t;
  if ((q & 2) != 0) t = -t;

  return t;
}

static double2 atan2k_u1(double2 y, double2 x) {
  double u;
  double2 s, t;
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
  t = ddadd2_d2_d2_d2(ddmul_d2_d2_d(dd(1.570796326794896557998982, 6.12323399573676603586882e-17), q), t);

  return t;
}

double xatan2_u1(double y, double x) {
  double2 d = atan2k_u1(dd(xfabs(y), 0), dd(x, 0));
  double r = d.x + d.y;

  r = mulsign(r, x);
  if (xisinf(x) || x == 0) r = M_PI/2 - (xisinf(x) ? (sign(x) * (M_PI  /2)) : 0);
  if (xisinf(y)          ) r = M_PI/2 - (xisinf(x) ? (sign(x) * (M_PI*1/4)) : 0);
  if (             y == 0) r = (sign(x) == -1 ? M_PI : 0);

  return xisnan(x) || xisnan(y) ? NAN : mulsign(r, y);
}

double xasin_u1(double d) {
  double2 d2 = atan2k_u1(dd(xfabs(d), 0), ddsqrt_d2_d2(ddmul_d2_d2_d2(ddadd_d2_d_d(1, d), ddadd_d2_d_d(1,-d))));
  double r = d2.x + d2.y;
  if (xfabs(d) == 1) r = 1.570796326794896557998982;
  return mulsign(r, d);
}

double xacos_u1(double d) {
  double2 d2 = atan2k_u1(ddsqrt_d2_d2(ddmul_d2_d2_d2(ddadd_d2_d_d(1, d), ddadd_d2_d_d(1,-d))), dd(xfabs(d), 0));
  d2 = ddscale_d2_d2_d(d2, mulsign(1, d));
  if (xfabs(d) == 1) d2 = dd(0, 0);
  if (d < 0) d2 = ddadd_d2_d2_d2(dd(3.141592653589793116, 1.2246467991473532072e-16), d2);
  return d2.x + d2.y;
}

double xatan_u1(double d) {
  double2 d2 = atan2k_u1(dd(xfabs(d), 0), dd(1, 0));
  double r = d2.x + d2.y;
  if (xisinf(d)) r = 1.570796326794896557998982;
  return mulsign(r, d);
}

double xsin(double d) {
  int q;
  double u, s;

  q = (int)xrint(d * M_1_PI);

  d = mla(q, -PI4_A*4, d);
  d = mla(q, -PI4_B*4, d);
  d = mla(q, -PI4_C*4, d);
  d = mla(q, -PI4_D*4, d);

  s = d * d;

  if ((q & 1) != 0) d = -d;

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

  return u;
}

double xsin_u1(double d) {
  int q;
  double u;
  double2 s, t, x;

  q = (int)xrint(d * M_1_PI);

  s = ddadd2_d2_d_d(d, q * (-PI4_A*4));
  s = ddadd2_d2_d2_d(s, q * (-PI4_B*4));
  s = ddadd2_d2_d2_d(s, q * (-PI4_C*4));
  s = ddadd2_d2_d2_d(s, q * (-PI4_D*4));

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

  if ((q & 1) != 0) u = -u;

  return u;
}

double xcos(double d) {
  int q;
  double u, s;

  q = 1 + 2*(int)xrint(d * M_1_PI - 0.5);

  d = mla(q, -PI4_A*2, d);
  d = mla(q, -PI4_B*2, d);
  d = mla(q, -PI4_C*2, d);
  d = mla(q, -PI4_D*2, d);

  s = d * d;

  if ((q & 2) == 0) d = -d;

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

  return u;
}

double xcos_u1(double d) {
  double u, q;
  double2 s, t, x;

  d = fabs(d);

  q = mla(2, xrint(d * M_1_PI - 0.5), 1);

  s = ddadd2_d2_d_d(d, q * (-PI4_A*2));
  s = ddadd2_d2_d2_d(s, q * (-PI4_B*2));
  s = ddadd2_d2_d2_d(s, q * (-PI4_C*2));
  s = ddadd2_d2_d2_d(s, q * (-PI4_D*2));

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

  if ((((int)q) & 2) == 0) u = -u;

  return u;
}

double2 xsincos(double d) {
  int q;
  double u, s, t;
  double2 r;

  q = (int)xrint(d * (2 * M_1_PI));

  s = d;

  s = mla(-q, PI4_A*2, s);
  s = mla(-q, PI4_B*2, s);
  s = mla(-q, PI4_C*2, s);
  s = mla(-q, PI4_D*2, s);

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

  u = -1.13615350239097429531523e-11;
  u = mla(u, s, 2.08757471207040055479366e-09);
  u = mla(u, s, -2.75573144028847567498567e-07);
  u = mla(u, s, 2.48015872890001867311915e-05);
  u = mla(u, s, -0.00138888888888714019282329);
  u = mla(u, s, 0.0416666666666665519592062);
  u = mla(u, s, -0.5);

  r.y = u * s + 1;

  if ((q & 1) != 0) { s = r.y; r.y = r.x; r.x = s; }
  if ((q & 2) != 0) { r.x = -r.x; }
  if (((q+1) & 2) != 0) { r.y = -r.y; }

  if (xisinf(d)) { r.x = r.y = NAN; }

  return r;
}

double2 xsincos_u1(double d) {
  int q;
  double u;
  double2 r, s, t, x;

  q = (int)xrint(d * (2 * M_1_PI));

  s = ddadd2_d2_d_d(d, q * (-PI4_A*2));
  s = ddadd2_d2_d2_d(s, q * (-PI4_B*2));
  s = ddadd2_d2_d2_d(s, q * (-PI4_C*2));
  s = ddadd2_d2_d2_d(s, q * (-PI4_D*2));

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

  u = -1.13615350239097429531523e-11;
  u = mla(u, s.x, 2.08757471207040055479366e-09);
  u = mla(u, s.x, -2.75573144028847567498567e-07);
  u = mla(u, s.x, 2.48015872890001867311915e-05);
  u = mla(u, s.x, -0.00138888888888714019282329);
  u = mla(u, s.x, 0.0416666666666665519592062);
  u = mla(u, s.x, -0.5);

  x = ddadd_d2_d_d2(1, ddmul_d2_d_d(s.x, u));
  r.y = x.x + x.y;

  if ((q & 1) != 0) { u = r.y; r.y = r.x; r.x = u; }
  if ((q & 2) != 0) { r.x = -r.x; }
  if (((q+1) & 2) != 0) { r.y = -r.y; }

  if (xisinf(d)) { r.x = r.y = NAN; }

  return r;
}

double xtan(double d) {
  int q;
  double u, s, x;

  q = (int)xrint(d * (2 * M_1_PI));

  x = mla(q, -PI4_A*2, d);
  x = mla(q, -PI4_B*2, x);
  x = mla(q, -PI4_C*2, x);
  x = mla(q, -PI4_D*2, x);

  s = x * x;

  if ((q & 1) != 0) x = -x;

  u = 1.01419718511083373224408e-05;
  u = mla(u, s, -2.59519791585924697698614e-05);
  u = mla(u, s, 5.23388081915899855325186e-05);
  u = mla(u, s, -3.05033014433946488225616e-05);
  u = mla(u, s, 7.14707504084242744267497e-05);
  u = mla(u, s, 8.09674518280159187045078e-05);
  u = mla(u, s, 0.000244884931879331847054404);
  u = mla(u, s, 0.000588505168743587154904506);
  u = mla(u, s, 0.00145612788922812427978848);
  u = mla(u, s, 0.00359208743836906619142924);
  u = mla(u, s, 0.00886323944362401618113356);
  u = mla(u, s, 0.0218694882853846389592078);
  u = mla(u, s, 0.0539682539781298417636002);
  u = mla(u, s, 0.133333333333125941821962);
  u = mla(u, s, 0.333333333333334980164153);

  u = mla(s, u * x, x);

  if ((q & 1) != 0) u = 1.0 / u;

  if (xisinf(d)) u = NAN;

  return u;
}

double xtan_u1(double d) {
  int q;
  double u;
  double2 s, t, x;

  q = (int)xrint(d * M_2_PI);

  s = ddadd2_d2_d_d(d, q * (-PI4_A*2));
  s = ddadd2_d2_d2_d(s, q * (-PI4_B*2));
  s = ddadd2_d2_d2_d(s, q * (-PI4_C*2));
  s = ddadd2_d2_d2_d(s, q * (-PI4_D*2));

  if ((q & 1) != 0) s = ddneg_d2_d2(s);

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

  if ((q & 1) != 0) x = ddrec_d2_d2(x);

  u = x.x + x.y;

  return u;
}

double xlog(double d) {
  double x, x2, t, m;
  int e;

  e = ilogbp1(d * 0.7071);
  m = ldexpk(d, -e);

  x = (m-1) / (m+1);
  x2 = x * x;

  t = 0.148197055177935105296783;
  t = mla(t, x2, 0.153108178020442575739679);
  t = mla(t, x2, 0.181837339521549679055568);
  t = mla(t, x2, 0.22222194152736701733275);
  t = mla(t, x2, 0.285714288030134544449368);
  t = mla(t, x2, 0.399999999989941956712869);
  t = mla(t, x2, 0.666666666666685503450651);
  t = mla(t, x2, 2);

  x = x * t + 0.693147180559945286226764 * e;

  if (xisinf(d)) x = INFINITY;
  if (d < 0) x = NAN;
  if (d == 0) x = -INFINITY;

  return x;
}

double xexp(double d) {
  int q = (int)xrint(d * R_LN2);
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

  if (xisminf(d)) u = 0;

  return u;
}

static inline double2 logk(double d) {
  double2 x, x2;
  double m, t;
  int e;

  e = ilogbp1(d * 0.7071);
  m = ldexpk(d, -e);

  x = dddiv_d2_d2_d2(ddadd2_d2_d_d(-1, m), ddadd2_d2_d_d(1, m));
  x2 = ddsqu_d2_d2(x);

  t = 0.134601987501262130076155;
  t = mla(t, x2.x, 0.132248509032032670243288);
  t = mla(t, x2.x, 0.153883458318096079652524);
  t = mla(t, x2.x, 0.181817427573705403298686);
  t = mla(t, x2.x, 0.222222231326187414840781);
  t = mla(t, x2.x, 0.285714285651261412873718);
  t = mla(t, x2.x, 0.400000000000222439910458);
  t = mla(t, x2.x, 0.666666666666666371239645);

  return ddadd2_d2_d2_d2(ddmul_d2_d2_d(dd(0.693147180559945286226764, 2.319046813846299558417771e-17), e),
			 ddadd2_d2_d2_d2(ddscale_d2_d2_d(x, 2), ddmul_d2_d2_d(ddmul_d2_d2_d2(x2, x), t)));
}

double xlog_u1(double d) {
  double2 s = logk(d);
  double x = s.x + s.y;

  if (xisinf(d)) x = INFINITY;
  if (d < 0) x = NAN;
  if (d == 0) x = -INFINITY;

  return x;
}

static inline double expk(double2 d) {
  int q = (int)xrint((d.x + d.y) * R_LN2);
  double2 s, t;
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
  return ldexpk(t.x + t.y, q);
}

double xpow(double x, double y) {
  int yisint = (int)y == y;
  int yisodd = (1 & (int)y) != 0 && yisint;

  double result = expk(ddmul_d2_d2_d(logk(xfabs(x)), y));

  result = xisnan(result) ? INFINITY : result;
  result *=  (x >= 0 ? 1 : (!yisint ? NAN : (yisodd ? -1 : 1)));

  double efx = mulsign(xfabs(x) - 1, y);
  if (xisinf(y)) result = efx < 0 ? 0.0 : (efx == 0 ? 1.0 : INFINITY);
  if (xisinf(x) || x == 0) result = (yisodd ? sign(x) : 1) * ((x == 0 ? -y : y) < 0 ? 0 : INFINITY);
  if (xisnan(x) || xisnan(y)) result = NAN;
  if (y == 0 || x == 1) result = 1;

  return result;
}

static inline double2 expk2(double2 d) {
  int q = (int)xrint((d.x + d.y) * R_LN2);
  double2 s, t;
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
  return ddscale_d2_d2_d(t, pow2i(q));
}

double xsinh(double x) {
  double y = xfabs(x);
  double2 d = expk2(dd(y, 0));
  d = ddsub_d2_d2_d2(d, ddrec_d2_d2(d));
  y = (d.x + d.y) * 0.5;

  y = xfabs(x) > 710 ? INFINITY : y;
  y = xisnan(y) ? INFINITY : y;
  y = mulsign(y, x);
  y = xisnan(x) ? NAN : y;

  return y;
}

double xcosh(double x) {
  double y = xfabs(x);
  double2 d = expk2(dd(y, 0));
  d = ddadd_d2_d2_d2(d, ddrec_d2_d2(d));
  y = (d.x + d.y) * 0.5;

  y = xfabs(x) > 710 ? INFINITY : y;
  y = xisnan(y) ? INFINITY : y;
  y = xisnan(x) ? NAN : y;

  return y;
}

double xtanh(double x) {
  double y = xfabs(x);
  double2 d = expk2(dd(y, 0));
  double2 e = ddrec_d2_d2(d);
  d = dddiv_d2_d2_d2(ddsub_d2_d2_d2(d, e), ddadd_d2_d2_d2(d, e));
  y = d.x + d.y;

  y = xfabs(x) > 18.714973875 ? 1.0 : y;
  y = xisnan(y) ? 1.0 : y;
  y = mulsign(y, x);
  y = xisnan(x) ? NAN : y;

  return y;
}

static inline double2 logk2(double2 d) {
  double2 x, x2, m;
  double t;
  int e;

  e = ilogbp1(d.x * 0.7071);
  m = ddscale_d2_d2_d(d, pow2i(-e));

  x = dddiv_d2_d2_d2(ddadd2_d2_d2_d(m, -1), ddadd2_d2_d2_d(m, 1));
  x2 = ddsqu_d2_d2(x);

  t = 0.134601987501262130076155;
  t = mla(t, x2.x, 0.132248509032032670243288);
  t = mla(t, x2.x, 0.153883458318096079652524);
  t = mla(t, x2.x, 0.181817427573705403298686);
  t = mla(t, x2.x, 0.222222231326187414840781);
  t = mla(t, x2.x, 0.285714285651261412873718);
  t = mla(t, x2.x, 0.400000000000222439910458);
  t = mla(t, x2.x, 0.666666666666666371239645);

  return ddadd2_d2_d2_d2(ddmul_d2_d2_d(dd(0.693147180559945286226764, 2.319046813846299558417771e-17), e),
			 ddadd2_d2_d2_d2(ddscale_d2_d2_d(x, 2), ddmul_d2_d2_d(ddmul_d2_d2_d2(x2, x), t)));
}

double xasinh(double x) {
  double y = xfabs(x);
  double2 d = logk2(ddadd_d2_d2_d(ddsqrt_d2_d2(ddadd2_d2_d2_d(ddmul_d2_d_d(y, y),  1)), y));
  y = d.x + d.y;

  y = xisinf(x) || xisnan(y) ? INFINITY : y;
  y = mulsign(y, x);
  y = xisnan(x) ? NAN : y;

  return y;
}

double xacosh(double x) {
  double2 d = logk2(ddadd2_d2_d2_d(ddsqrt_d2_d2(ddadd2_d2_d2_d(ddmul_d2_d_d(x, x), -1)), x));
  double y = d.x + d.y;

  y = xisinf(x) || xisnan(y) ? INFINITY : y;
  y = x == 1.0 ? 0.0 : y;
  y = x < 1.0 ? NAN : y;
  y = xisnan(x) ? NAN : y;

  return y;
}

double xatanh(double x) {
  double y = xfabs(x);
  double2 d = logk2(dddiv_d2_d2_d2(ddadd2_d2_d_d(1, y), ddadd2_d2_d_d(1, -y)));
  y = y > 1.0 ? NAN : (y == 1.0 ? INFINITY : (d.x + d.y) * 0.5);

  y = xisinf(x) || xisnan(y) ? NAN : y;
  y = mulsign(y, x);
  y = xisnan(x) ? NAN : y;

  return y;
}

//

double xfma(double x, double y, double z) {
  union {
    double f;
    long long int i;
  } tmp;

  tmp.f = x;
  tmp.i = (tmp.i + 0x4000000) & 0xfffffffff8000000LL;
  double xh = tmp.f, xl = x - xh;

  tmp.f = y;
  tmp.i = (tmp.i + 0x4000000) & 0xfffffffff8000000LL;
  double yh = tmp.f, yl = y - yh;

  double h = x * y;
  double l = xh * yh - h + xl * yh + xh * yl + xl * yl;

  double h2, l2, v;

  h2 = h + z;
  v = h2 - h;
  l2 = (h - (h2 - v)) + (z - v) + l;

  return h2 + l2;
}

double xsqrt(double d) { // max error : 0.5 ulp
  double q = 1;

  if (d < 8.636168555094445E-78) {
    d *= 1.157920892373162E77;
    q = 2.9387358770557188E-39;
  }

  // http://en.wikipedia.org/wiki/Fast_inverse_square_root
  double x = longBitsToDouble(0x5fe6ec85e7de30da - (doubleToRawLongBits(d + 1e-320) >> 1));

  x = x * (1.5 - 0.5 * d * x * x);
  x = x * (1.5 - 0.5 * d * x * x);
  x = x * (1.5 - 0.5 * d * x * x);

  // You can change xfma to fma if fma is correctly implemented
  x = xfma(d * x, d * x, -d) * (x * -0.5) + d * x;

  return d == INFINITY ? INFINITY : x * q;
}

double xcbrt(double d) { // max error : 2 ulps
  double x, y, q = 1.0;
  int e, r;

  e = ilogbp1(d);
  d = ldexpk(d, -e);
  r = (e + 6144) % 3;
  q = (r == 1) ? 1.2599210498948731647672106 : q;
  q = (r == 2) ? 1.5874010519681994747517056 : q;
  q = ldexpk(q, (e + 6144) / 3 - 2048);

  q = mulsign(q, d);
  d = xfabs(d);

  x = -0.640245898480692909870982;
  x = x * d + 2.96155103020039511818595;
  x = x * d + -5.73353060922947843636166;
  x = x * d + 6.03990368989458747961407;
  x = x * d + -3.85841935510444988821632;
  x = x * d + 2.2307275302496609725722;

  y = x * x; y = y * y; x -= (d * y - x) * (1.0 / 3.0);
  y = d * x * x;
  y = (y - (2.0 / 3.0) * y * (y * x - 1)) * q;

  return y;
}

double xcbrt_u1(double d) {
  double x, y, z;
  double2 q2 = dd(1, 0), u, v;
  int e, r;

  e = ilogbp1(d);
  d = ldexpk(d, -e);
  r = (e + 6144) % 3;
  q2 = (r == 1) ? dd(1.2599210498948731907, -2.5899333753005069177e-17) : q2;
  q2 = (r == 2) ? dd(1.5874010519681995834, -1.0869008194197822986e-16) : q2;

  q2.x = mulsign(q2.x, d); q2.y = mulsign(q2.y, d);
  d = xfabs(d);

  x = -0.640245898480692909870982;
  x = x * d + 2.96155103020039511818595;
  x = x * d + -5.73353060922947843636166;
  x = x * d + 6.03990368989458747961407;
  x = x * d + -3.85841935510444988821632;
  x = x * d + 2.2307275302496609725722;

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
  z = ldexp(v.x + v.y, (e + 6144) / 3 - 2048);

  if (xisinf(d)) { z = mulsign(INFINITY, q2.x); }
  if (d == 0) { z = mulsign(0, q2.x); }

  return z;
}

double xexp2(double a) {
  double u = expk(ddmul_d2_d2_d(dd(0.69314718055994528623, 2.3190468138462995584e-17), a));
  if (a > 1023) u = INFINITY;
  if (xisminf(a)) u = 0;
  return u;
}

double xexp10(double a) {
  double u = expk(ddmul_d2_d2_d(dd(2.3025850929940459011, -2.1707562233822493508e-16), a));
  if (a > 308) u = INFINITY;
  if (xisminf(a)) u = 0;
  return u;
}

double xexpm1(double a) {
  double2 d = ddadd2_d2_d2_d(expk2(dd(a, 0)), -1.0);
  double x = d.x + d.y;
  if (a > 700) x = INFINITY;
  if (a < -0.36043653389117156089696070315825181539851971360337e+2) x = -1;
  return x;
}

double xlog10(double a) {
  double2 d = ddmul_d2_d2_d2(logk(a), dd(0.43429448190325176116, 6.6494347733425473126e-17));
  double x = d.x + d.y;

  if (xisinf(a)) x = INFINITY;
  if (a < 0) x = NAN;
  if (a == 0) x = -INFINITY;

  return x;
}

double xlog1p(double a) {
  double2 d = logk2(ddadd2_d2_d_d(a, 1));
  double x = d.x + d.y;

  if (xisinf(a)) x = INFINITY;
  if (a < -1) x = NAN;
  if (a == -1) x = -INFINITY;

  return x;
}
