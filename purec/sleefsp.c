#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>

#include "nonnumber.h"

#define PI4_Af 0.78515625f
#define PI4_Bf 0.00024187564849853515625f
#define PI4_Cf 3.7747668102383613586e-08f
#define PI4_Df 1.2816720341285448015e-12f

#define L2Uf 0.693145751953125f
#define L2Lf 1.428606765330187045e-06f

#define R_LN2f 1.442695040888963407359924681001892137426645954152985934135449406931f
#define M_PIf ((float)M_PI)

static inline int32_t floatToRawIntBits(float d) {
  union {
    float f;
    int32_t i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

static inline float intBitsToFloat(int32_t i) {
  union {
    float f;
    int32_t i;
  } tmp;
  tmp.i = i;
  return tmp.f;
}

static inline float xfabsf(float x) {
  return intBitsToFloat(0x7fffffffL & floatToRawIntBits(x));
}

static inline float mulsignf(float x, float y) {
  return intBitsToFloat(floatToRawIntBits(x) ^ (floatToRawIntBits(y) & (1 << 31)));
}

static inline float signf(float d) { return mulsignf(1, d); }
static inline float mlaf(float x, float y, float z) { return x * y + z; }
static inline float xrintf(float x) { return x < 0 ? (int)(x - 0.5f) : (int)(x + 0.5f); }

static inline int xisnanf(float x) { return x != x; }
static inline int xisinff(float x) { return x == INFINITYf || x == -INFINITYf; }
static inline int xisminff(float x) { return x == -INFINITYf; }
static inline int xispinff(float x) { return x == INFINITYf; }

static inline int ilogbp1f(float d) {
  int m = d < 5.421010862427522E-20f;
  d = m ? 1.8446744073709552E19f * d : d;
  int q = (floatToRawIntBits(d) >> 23) & 0xff;
  q = m ? q - (64 + 0x7e) : q - 0x7e;
  return q;
}

static inline float pow2if(int q) {
  return intBitsToFloat(((int32_t)(q + 0x7f)) << 23);
}

static inline float ldexpkf(float x, int q) {
  float u;
  int m;
  m = q >> 31;
  m = (((m + q) >> 6) - m) << 4;
  q = q - (m << 2);
  m += 127;
  m = m <   0 ?   0 : m;
  m = m > 255 ? 255 : m;
  u = intBitsToFloat(((int32_t)m) << 23);
  x = x * u * u * u * u;
  u = intBitsToFloat(((int32_t)(q + 0x7f)) << 23);
  return x * u;
}

float xldexpf(float x, int q) { return ldexpkf(x, q); }

//

typedef struct {
  float x, y;
} float2;

#ifndef NDEBUG
static int checkfp(float x) {
  if (xisinff(x) || xisnanf(x)) return 1;
  return 0;
}
#endif

static inline float upperf(float d) {
  return intBitsToFloat(floatToRawIntBits(d) & 0xfffff000);
}

static inline float2 df(float h, float l) {
  float2 ret;
  ret.x = h; ret.y = l;
  return ret;
}

static inline float2 dfnormalize_f2_f2(float2 t) {
  float2 s;

  s.x = t.x + t.y;
  s.y = t.x - s.x + t.y;

  return s;
}

static inline float2 dfscale_f2_f2_f(float2 d, float s) {
  float2 r;

  r.x = d.x * s;
  r.y = d.y * s;

  return r;
}

static inline float2 dfneg_f2_f2(float2 d) {
  float2 r;

  r.x = -d.x;
  r.y = -d.y;

  return r;
}

static inline float2 dfadd_f2_f_f(float x, float y) {
  // |x| >= |y|

  float2 r;

#ifndef NDEBUG
  if (!(checkfp(x) || checkfp(y) || xfabsf(x) >= xfabsf(y))) fprintf(stderr, "[dfadd_f2_f_f : %g, %g]", x, y);
#endif

  r.x = x + y;
  r.y = x - r.x + y;

  return r;
}

static inline float2 dfadd2_f2_f_f(float x, float y) {
  float2 r;

  r.x = x + y;
  float v = r.x - x;
  r.y = (x - (r.x - v)) + (y - v);

  return r;
}

static inline float2 dfadd_f2_f2_f(float2 x, float y) {
  // |x| >= |y|

  float2 r;

#ifndef NDEBUG
  if (!(checkfp(x.x) || checkfp(y) || xfabsf(x.x) >= xfabsf(y))) fprintf(stderr, "[dfadd_f2_f2_f : %g %g]", x.x, y);
#endif

  r.x = x.x + y;
  r.y = x.x - r.x + y + x.y;

  return r;
}

static inline float2 dfadd2_f2_f2_f(float2 x, float y) {
  // |x| >= |y|

  float2 r;

  r.x  = x.x + y;
  float v = r.x - x.x;
  r.y = (x.x - (r.x - v)) + (y - v);
  r.y += x.y;

  return r;
}

static inline float2 dfadd_f2_f_f2(float x, float2 y) {
  // |x| >= |y|

  float2 r;

#ifndef NDEBUG
  if (!(checkfp(x) || checkfp(y.x) || xfabsf(x) >= xfabsf(y.x))) fprintf(stderr, "[dfadd_df_f_f2 : %g %g]", x, y.x);
#endif

  r.x = x + y.x;
  r.y = x - r.x + y.x + y.y;

  return r;
}

static inline float2 dfadd_f2_f2_f2(float2 x, float2 y) {
  // |x| >= |y|

  float2 r;

#ifndef NDEBUG
  if (!(checkfp(x.x) || checkfp(y.x) || xfabsf(x.x) >= xfabsf(y.x))) fprintf(stderr, "[dfadd_f2_f2_f2 : %g %g]", x.x, y.x);
#endif

  r.x = x.x + y.x;
  r.y = x.x - r.x + y.x + x.y + y.y;

  return r;
}

static inline float2 dfadd2_f2_f2_f2(float2 x, float2 y) {
  float2 r;

  r.x  = x.x + y.x;
  float v = r.x - x.x;
  r.y = (x.x - (r.x - v)) + (y.x - v);
  r.y += x.y + y.y;

  return r;
}

static inline float2 dfsub_f2_f2_f2(float2 x, float2 y) {
  // |x| >= |y|

  float2 r;

#ifndef NDEBUG
  if (!(checkfp(x.x) || checkfp(y.x) || xfabsf(x.x) >= xfabsf(y.x))) fprintf(stderr, "[dfsub_f2_f2_f2 : %g %g]", x.x, y.x);
#endif

  r.x = x.x - y.x;
  r.y = x.x - r.x - y.x + x.y - y.y;

  return r;
}

static inline float2 dfdiv_f2_f2_f2(float2 n, float2 d) {
  float t = 1.0f / d.x;
  float dh  = upperf(d.x), dl  = d.x - dh;
  float th  = upperf(t  ), tl  = t   - th;
  float nhh = upperf(n.x), nhl = n.x - nhh;

  float2 q;

  q.x = n.x * t;

  float u = -q.x + nhh * th + nhh * tl + nhl * th + nhl * tl +
    q.x * (1 - dh * th - dh * tl - dl * th - dl * tl);

  q.y = t * (n.y - q.x * d.y) + u;

  return q;
}

static inline float2 dfmul_f2_f_f(float x, float y) {
  float xh = upperf(x), xl = x - xh;
  float yh = upperf(y), yl = y - yh;
  float2 r;

  r.x = x * y;
  r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl;

  return r;
}

static inline float2 dfmul_f2_f2_f(float2 x, float y) {
  float xh = upperf(x.x), xl = x.x - xh;
  float yh = upperf(y  ), yl = y   - yh;
  float2 r;

  r.x = x.x * y;
  r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.y * y;

  return r;
}

static inline float2 dfmul_f2_f2_f2(float2 x, float2 y) {
  float xh = upperf(x.x), xl = x.x - xh;
  float yh = upperf(y.x), yl = y.x - yh;
  float2 r;

  r.x = x.x * y.x;
  r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.x * y.y + x.y * y.x;

  return r;
}

static inline float2 dfsqu_f2_f2(float2 x) {
  float xh = upperf(x.x), xl = x.x - xh;
  float2 r;

  r.x = x.x * x.x;
  r.y = xh * xh - r.x + (xh + xh) * xl + xl * xl + x.x * (x.y + x.y);

  return r;
}

static inline float2 dfrec_f2_f(float d) {
  float t = 1.0f / d;
  float dh = upperf(d), dl = d - dh;
  float th = upperf(t), tl = t - th;
  float2 q;

  q.x = t;
  q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl);

  return q;
}

static inline float2 dfrec_f2_f2(float2 d) {
  float t = 1.0f / d.x;
  float dh = upperf(d.x), dl = d.x - dh;
  float th = upperf(t  ), tl = t   - th;
  float2 q;

  q.x = t;
  q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl - d.y * t);

  return q;
}

static inline float2 dfsqrt_f2_f2(float2 d) {
  float t = sqrtf(d.x + d.y);
  return dfscale_f2_f2_f(dfmul_f2_f2_f2(dfadd2_f2_f2_f2(d, dfmul_f2_f_f(t, t)), dfrec_f2_f(t)), 0.5f);
}

//

float xsinf(float d) {
  int q;
  float u, s;

  q = (int)xrintf(d * (float)M_1_PI);

  d = mlaf(q, -PI4_Af*4, d);
  d = mlaf(q, -PI4_Bf*4, d);
  d = mlaf(q, -PI4_Cf*4, d);
  d = mlaf(q, -PI4_Df*4, d);

  s = d * d;

  if ((q & 1) != 0) d = -d;

  u = 2.6083159809786593541503e-06f;
  u = mlaf(u, s, -0.0001981069071916863322258f);
  u = mlaf(u, s, 0.00833307858556509017944336f);
  u = mlaf(u, s, -0.166666597127914428710938f);

  u = mlaf(s, u * d, d);

  if (xisinff(d)) u = NANf;

  return u;
}

float xsinf_u1(float d) {
  int q;
  float u;
  float2 s, t, x;

  q = (int)xrintf(d * (float)M_1_PI);

  s = dfadd2_f2_f_f(d, q * (-PI4_Af*4));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Bf*4));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Cf*4));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Df*4));

  t = s;
  s = dfsqu_f2_f2(s);

  u = 2.6083159809786593541503e-06f;
  u = mlaf(u, s.x, -0.0001981069071916863322258f);
  u = mlaf(u, s.x, 0.00833307858556509017944336f);

  x = dfadd_f2_f_f2(1, dfmul_f2_f2_f2(dfadd_f2_f_f(-0.166666597127914428710938f, u * s.x), s));

  x = dfmul_f2_f2_f2(t, x);
  u = x.x + x.y;

  if ((q & 1) != 0) u = -u;

  return u;
}

float xcosf(float d) {
  int q;
  float u, s;

  q = 1 + 2*(int)xrintf(d * (float)M_1_PI - 0.5f);

  d = mlaf(q, -PI4_Af*2, d);
  d = mlaf(q, -PI4_Bf*2, d);
  d = mlaf(q, -PI4_Cf*2, d);
  d = mlaf(q, -PI4_Df*2, d);

  s = d * d;

  if ((q & 2) == 0) d = -d;

  u = 2.6083159809786593541503e-06f;
  u = mlaf(u, s, -0.0001981069071916863322258f);
  u = mlaf(u, s, 0.00833307858556509017944336f);
  u = mlaf(u, s, -0.166666597127914428710938f);

  u = mlaf(s, u * d, d);

  if (xisinff(d)) u = NANf;

  return u;
}

float xcosf_u1(float d) {
  float u, q;
  float2 s, t, x;

  d = fabsf(d);

  q = 1 + 2*(int)xrintf(d * (float)M_1_PI - 0.5f);

  s = dfadd2_f2_f_f(d, q * (-PI4_Af*2));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Bf*2));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Cf*2));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Df*2));

  t = s;
  s = dfsqu_f2_f2(s);

  u = 2.6083159809786593541503e-06f;
  u = mlaf(u, s.x, -0.0001981069071916863322258f);
  u = mlaf(u, s.x, 0.00833307858556509017944336f);

  x = dfadd_f2_f_f2(1, dfmul_f2_f2_f2(dfadd_f2_f_f(-0.166666597127914428710938f, u * s.x), s));

  x = dfmul_f2_f2_f2(t, x);
  u = x.x + x.y;

  if ((((int)q) & 2) == 0) u = -u;

  return u;
}

float2 xsincosf(float d) {
  int q;
  float u, s, t;
  float2 r;

  q = (int)xrintf(d * ((float)(2 * M_1_PI)));

  s = d;

  s = mlaf(q, -PI4_Af*2, s);
  s = mlaf(q, -PI4_Bf*2, s);
  s = mlaf(q, -PI4_Cf*2, s);
  s = mlaf(q, -PI4_Df*2, s);

  t = s;

  s = s * s;

  u = -0.000195169282960705459117889f;
  u = mlaf(u, s, 0.00833215750753879547119141f);
  u = mlaf(u, s, -0.166666537523269653320312f);
  u = u * s * t;

  r.x = t + u;

  u = -2.71811842367242206819355e-07f;
  u = mlaf(u, s, 2.47990446951007470488548e-05f);
  u = mlaf(u, s, -0.00138888787478208541870117f);
  u = mlaf(u, s, 0.0416666641831398010253906f);
  u = mlaf(u, s, -0.5f);

  r.y = u * s + 1;

  if ((q & 1) != 0) { s = r.y; r.y = r.x; r.x = s; }
  if ((q & 2) != 0) { r.x = -r.x; }
  if (((q+1) & 2) != 0) { r.y = -r.y; }

  if (xisinff(d)) { r.x = r.y = NANf; }

  return r;
}

float2 xsincosf_u1(float d) {
  int q;
  float u;
  float2 r, s, t, x;

  q = (int)xrintf(d * (float)(2 * M_1_PI));

  s = dfadd2_f2_f_f(d, q * (-PI4_Af*2));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Bf*2));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Cf*2));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Df*2));

  t = s;
  s = dfsqu_f2_f2(s);
  s.x = s.x + s.y;

  u = -0.000195169282960705459117889f;
  u = mlaf(u, s.x, 0.00833215750753879547119141f);
  u = mlaf(u, s.x, -0.166666537523269653320312f);

  u *= s.x * t.x;

  x = dfadd_f2_f2_f(t, u);
  r.x = x.x + x.y;

  u = -2.71811842367242206819355e-07f;
  u = mlaf(u, s.x, 2.47990446951007470488548e-05f);
  u = mlaf(u, s.x, -0.00138888787478208541870117f);
  u = mlaf(u, s.x, 0.0416666641831398010253906f);
  u = mlaf(u, s.x, -0.5f);

  x = dfadd_f2_f_f2(1, dfmul_f2_f_f(s.x, u));
  r.y = x.x + x.y;

  if ((q & 1) != 0) { u = r.y; r.y = r.x; r.x = u; }
  if ((q & 2) != 0) { r.x = -r.x; }
  if (((q+1) & 2) != 0) { r.y = -r.y; }

  if (xisinff(d)) { r.x = r.y = NAN; }

  return r;
}

float xtanf(float d) {
  int q;
  float u, s, x;

  q = (int)xrintf(d * (float)(2 * M_1_PI));

  x = d;

  x = mlaf(q, -PI4_Af*2, x);
  x = mlaf(q, -PI4_Bf*2, x);
  x = mlaf(q, -PI4_Cf*2, x);
  x = mlaf(q, -PI4_Df*2, x);

  s = x * x;

  if ((q & 1) != 0) x = -x;

  u = 0.00927245803177356719970703f;
  u = mlaf(u, s, 0.00331984995864331722259521f);
  u = mlaf(u, s, 0.0242998078465461730957031f);
  u = mlaf(u, s, 0.0534495301544666290283203f);
  u = mlaf(u, s, 0.133383005857467651367188f);
  u = mlaf(u, s, 0.333331853151321411132812f);

  u = mlaf(s, u * x, x);

  if ((q & 1) != 0) u = 1.0f / u;

  if (xisinff(d)) u = NANf;

  return u;
}

float xtanf_u1(float d) {
  int q;
  float u;
  float2 s, t, x;

  q = (int)xrintf(d * (float)(2 * M_1_PI));

  s = dfadd2_f2_f_f(d, q * (-PI4_Af*2));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Bf*2));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Cf*2));
  s = dfadd2_f2_f2_f(s, q * (-PI4_Df*2));

  if ((q & 1) != 0) s = dfneg_f2_f2(s);

  t = s;
  s = dfsqu_f2_f2(s);
  s = dfnormalize_f2_f2(s);

  u = 0.00446636462584137916564941f;
  u = mlaf(u, s.x, -8.3920182078145444393158e-05f);
  u = mlaf(u, s.x, 0.0109639242291450500488281f);
  u = mlaf(u, s.x, 0.0212360303848981857299805f);
  u = mlaf(u, s.x, 0.0540687143802642822265625f);

  x = dfadd_f2_f_f(0.133325666189193725585938f, u * s.x);
  x = dfadd_f2_f_f2(1, dfmul_f2_f2_f2(dfadd_f2_f_f2(0.33333361148834228515625f, dfmul_f2_f2_f2(s, x)), s));
  x = dfmul_f2_f2_f2(t, x);

  if ((q & 1) != 0) x = dfrec_f2_f2(x);

  u = x.x + x.y;

  return u;
}

float xatanf(float s) {
  float t, u;
  int q = 0;

  if (s < 0) { s = -s; q = 2; }
  if (s > 1) { s = 1.0f / s; q |= 1; }

  t = s * s;

  u = 0.00282363896258175373077393f;
  u = mlaf(u, t, -0.0159569028764963150024414f);
  u = mlaf(u, t, 0.0425049886107444763183594f);
  u = mlaf(u, t, -0.0748900920152664184570312f);
  u = mlaf(u, t, 0.106347933411598205566406f);
  u = mlaf(u, t, -0.142027363181114196777344f);
  u = mlaf(u, t, 0.199926957488059997558594f);
  u = mlaf(u, t, -0.333331018686294555664062f);

  t = s + s * (t * u);

  if ((q & 1) != 0) t = 1.570796326794896557998982f - t;
  if ((q & 2) != 0) t = -t;

  return t;
}

static inline float atan2kf(float y, float x) {
  float s, t, u;
  int q = 0;

  if (x < 0) { x = -x; q = -2; }
  if (y > x) { t = x; x = y; y = -t; q += 1; }

  s = y / x;
  t = s * s;

  u = 0.00282363896258175373077393f;
  u = mlaf(u, t, -0.0159569028764963150024414f);
  u = mlaf(u, t, 0.0425049886107444763183594f);
  u = mlaf(u, t, -0.0748900920152664184570312f);
  u = mlaf(u, t, 0.106347933411598205566406f);
  u = mlaf(u, t, -0.142027363181114196777344f);
  u = mlaf(u, t, 0.199926957488059997558594f);
  u = mlaf(u, t, -0.333331018686294555664062f);

  t = u * t * s + s;
  t = q * (float)(M_PI/2) + t;

  return t;
}

float xatan2f(float y, float x) {
  float r = atan2kf(xfabsf(y), x);

  r = mulsignf(r, x);
  if (xisinff(x) || x == 0) r = M_PIf/2 - (xisinff(x) ? (signf(x) * (float)(M_PI  /2)) : 0);
  if (xisinff(y)          ) r = M_PIf/2 - (xisinff(x) ? (signf(x) * (float)(M_PI*1/4)) : 0);
  if (              y == 0) r = (signf(x) == -1 ? M_PIf : 0);

  return xisnanf(x) || xisnanf(y) ? NANf : mulsignf(r, y);
}

float xasinf(float d) {
  return mulsignf(atan2kf(fabsf(d), sqrtf((1.0f+d)*(1.0f-d))), d);
}

float xacosf(float d) {
  return mulsignf(atan2kf(sqrtf((1.0f+d)*(1.0f-d)), fabsf(d)), d) + (d < 0 ? (float)M_PI : 0.0f);
}

static float2 atan2kf_u1(float2 y, float2 x) {
  float u;
  float2 s, t;
  int q = 0;

  if (x.x < 0) { x.x = -x.x; x.y = -x.y; q = -2; }
  if (y.x > x.x) { t = x; x = y; y.x = -t.x; y.y = -t.y; q += 1; }

  s = dfdiv_f2_f2_f2(y, x);
  t = dfsqu_f2_f2(s);
  t = dfnormalize_f2_f2(t);

  u = -0.00176397908944636583328247f;
  u = mlaf(u, t.x, 0.0107900900766253471374512f);
  u = mlaf(u, t.x, -0.0309564601629972457885742f);
  u = mlaf(u, t.x, 0.0577365085482597351074219f);
  u = mlaf(u, t.x, -0.0838950723409652709960938f);
  u = mlaf(u, t.x, 0.109463557600975036621094f);
  u = mlaf(u, t.x, -0.142626821994781494140625f);
  u = mlaf(u, t.x, 0.199983194470405578613281f);

  //u = mlaf(u, t.x, -0.333332866430282592773438f);
  //t = dfmul_f2_f2_f(t, u);

  t = dfmul_f2_f2_f2(t, dfadd_f2_f_f(-0.333332866430282592773438f, u * t.x));
  t = dfmul_f2_f2_f2(s, dfadd_f2_f_f2(1, t));
  t = dfadd2_f2_f2_f2(dfmul_f2_f2_f(df(1.5707963705062866211f, -4.3711388286737928865e-08f), q), t);

  return t;
}

float xatan2f_u1(float y, float x) {
  float2 d = atan2kf_u1(df(xfabsf(y), 0), df(x, 0));
  float r = d.x + d.y;

  r = mulsignf(r, x);
  if (xisinff(x) || x == 0) r = (float)M_PI/2 - (xisinff(x) ? (signf(x) * (float)(M_PI  /2)) : 0.0f);
  if (xisinff(y)          ) r = (float)M_PI/2 - (xisinff(x) ? (signf(x) * (float)(M_PI*1/4)) : 0.0f);
  if (              y == 0) r = (signf(x) == -1 ? (float)M_PI : 0.0f);

  return xisnanf(x) || xisnanf(y) ? NANf : mulsignf(r, y);
}

float xasinf_u1(float d) {
  float2 d2 = atan2kf_u1(df(xfabsf(d), 0), dfsqrt_f2_f2(dfmul_f2_f2_f2(dfadd_f2_f_f(1, d), dfadd_f2_f_f(1,-d))));
  float r = d2.x + d2.y;
  if (xfabsf(d) == 1) r = 1.570796326794896557998982f;
  return mulsignf(r, d);
}

float xacosf_u1(float d) {
  float2 d2 = atan2kf_u1(dfsqrt_f2_f2(dfmul_f2_f2_f2(dfadd_f2_f_f(1, d), dfadd_f2_f_f(1,-d))), df(xfabsf(d), 0));
  d2 = dfscale_f2_f2_f(d2, mulsignf(1.0f, d));
  if (xfabsf(d) == 1) d2 = df(0.0f, 0.0f);
  if (d < 0) d2 = dfadd_f2_f2_f2(df(3.1415927410125732422f,-8.7422776573475857731e-08f), d2);
  return d2.x + d2.y;
}

float xatanf_u1(float d) {
  float2 d2 = atan2kf_u1(df(xfabsf(d), 0.0f), df(1.0f, 0.0f));
  float r = d2.x + d2.y;
  if (xisinff(d)) r = 1.570796326794896557998982f;
  return mulsignf(r, d);
}

float xlogf(float d) {
  float x, x2, t, m;
  int e;

  e = ilogbp1f(d * 0.7071f);
  m = ldexpkf(d, -e);

  x = (m-1.0f) / (m+1.0f);
  x2 = x * x;

  t = 0.2371599674224853515625f;
  t = mlaf(t, x2, 0.285279005765914916992188f);
  t = mlaf(t, x2, 0.400005519390106201171875f);
  t = mlaf(t, x2, 0.666666567325592041015625f);
  t = mlaf(t, x2, 2.0f);

  x = x * t + 0.693147180559945286226764f * e;

  if (xisinff(d)) x = INFINITYf;
  if (d < 0) x = NANf;
  if (d == 0) x = -INFINITYf;

  return x;
}

float xexpf(float d) {
  int q = (int)xrintf(d * R_LN2f);
  float s, u;

  s = mlaf(q, -L2Uf, d);
  s = mlaf(q, -L2Lf, s);

  u = 0.00136324646882712841033936f;
  u = mlaf(u, s, 0.00836596917361021041870117f);
  u = mlaf(u, s, 0.0416710823774337768554688f);
  u = mlaf(u, s, 0.166665524244308471679688f);
  u = mlaf(u, s, 0.499999850988388061523438f);

  u = s * s * u + s + 1.0f;
  u = ldexpkf(u, q);

  if (xisminff(d)) u = 0;

  return u;
}

//#define L2Af 0.693145751953125
//#define L2Bf 1.4285906217992305756e-06
//#define L2Cf 1.619850954759360917e-11

static inline float expkf(float2 d) {
  int q = (int)xrintf((d.x + d.y) * R_LN2f);
  float2 s, t;
  float u;

  s = dfadd2_f2_f2_f(d, q * -L2Uf);
  s = dfadd2_f2_f2_f(s, q * -L2Lf);

  //s = dfadd2_f2_f2_f(d, q * -L2Af);
  //s = dfadd2_f2_f2_f(s, q * -L2Bf);
  //s = dfadd2_f2_f2_f(s, q * -L2Cf);

  s = dfnormalize_f2_f2(s);

  u = 0.00136324646882712841033936f;
  u = mlaf(u, s.x, 0.00836596917361021041870117f);
  u = mlaf(u, s.x, 0.0416710823774337768554688f);
  u = mlaf(u, s.x, 0.166665524244308471679688f);
  u = mlaf(u, s.x, 0.499999850988388061523438f);

  t = dfadd_f2_f2_f2(s, dfmul_f2_f2_f(dfsqu_f2_f2(s), u));

  t = dfadd_f2_f_f2(1, t);
  return ldexpkf(t.x + t.y, q);
}

static inline float2 logkf(float d) {
  float2 x, x2;
  float m, t;
  int e;

  e = ilogbp1f(d * 0.7071f);
  m = ldexpkf(d, -e);

  x = dfdiv_f2_f2_f2(dfadd2_f2_f_f(-1, m), dfadd2_f2_f_f(1, m));
  x2 = dfsqu_f2_f2(x);

  t = 0.2371599674224853515625f;
  t = mlaf(t, x2.x, 0.285279005765914916992188f);
  t = mlaf(t, x2.x, 0.400005519390106201171875f);
  t = mlaf(t, x2.x, 0.666666567325592041015625f);

  return dfadd2_f2_f2_f2(dfmul_f2_f2_f(df(0.69314718246459960938f, -1.904654323148236017e-09f), e),
			 dfadd2_f2_f2_f2(dfscale_f2_f2_f(x, 2), dfmul_f2_f2_f(dfmul_f2_f2_f2(x2, x), t)));
}

float xlogf_u1(float d) {
  float2 s = logkf(d);
  float x = s.x + s.y;

  if (xisinff(d)) x = INFINITYf;
  if (d < 0) x = NANf;
  if (d == 0) x = -INFINITYf;

  return x;
}

static inline float2 expk2f(float2 d) {
  int q = (int)xrintf((d.x + d.y) * R_LN2f);
  float2 s, t;
  float u;

  s = dfadd2_f2_f2_f(d, q * -L2Uf);
  s = dfadd2_f2_f2_f(s, q * -L2Lf);

  u = 0.00136324646882712841033936f;
  u = mlaf(u, s.x, 0.00836596917361021041870117f);
  u = mlaf(u, s.x, 0.0416710823774337768554688f);
  u = mlaf(u, s.x, 0.166665524244308471679688f);
  u = mlaf(u, s.x, 0.499999850988388061523438f);

  t = dfadd_f2_f2_f2(s, dfmul_f2_f2_f(dfsqu_f2_f2(s), u));

  t = dfadd_f2_f_f2(1, t);
  return dfscale_f2_f2_f(t, pow2if(q));
}

float xpowf(float x, float y) {
  int yisint = (int)y == y;
  int yisodd = (1 & (int)y) != 0 && yisint;

  float result = expkf(dfmul_f2_f2_f(logkf(xfabsf(x)), y));

  result = xisnanf(result) ? INFINITYf : result;
  result *=  (x >= 0 ? 1 : (!yisint ? NANf : (yisodd ? -1 : 1)));

  float efx = mulsignf(xfabsf(x) - 1, y);
  if (xisinff(y)) result = efx < 0 ? 0.0f : (efx == 0 ? 1.0f : INFINITYf);
  if (xisinff(x) || x == 0) result = (yisodd ? signf(x) : 1) * ((x == 0 ? -y : y) < 0 ? 0 : INFINITYf);
  if (xisnanf(x) || xisnanf(y)) result = NANf;
  if (y == 0 || x == 1) result = 1;

  return result;
}

float xsinhf(float x) {
  float y = xfabsf(x);
  float2 d = expk2f(df(y, 0));
  d = dfsub_f2_f2_f2(d, dfrec_f2_f2(d));
  y = (d.x + d.y) * 0.5f;

  y = xfabsf(x) > 89 ? INFINITY : y;
  y = xisnanf(y) ? INFINITYf : y;
  y = mulsignf(y, x);
  y = xisnanf(x) ? NANf : y;

  return y;
}

float xcoshf(float x) {
  float y = xfabsf(x);
  float2 d = expk2f(df(y, 0));
  d = dfadd_f2_f2_f2(d, dfrec_f2_f2(d));
  y = (d.x + d.y) * 0.5f;

  y = xfabsf(x) > 89 ? INFINITY : y;
  y = xisnanf(y) ? INFINITYf : y;
  y = xisnanf(x) ? NANf : y;

  return y;
}

float xtanhf(float x) {
  float y = xfabsf(x);
  float2 d = expk2f(df(y, 0));
  float2 e = dfrec_f2_f2(d);
  d = dfdiv_f2_f2_f2(dfsub_f2_f2_f2(d, e), dfadd_f2_f2_f2(d, e));
  y = d.x + d.y;

  y = xfabsf(x) > 8.664339742f ? 1.0f : y;
  y = xisnanf(y) ? 1.0f : y;
  y = mulsignf(y, x);
  y = xisnanf(x) ? NANf : y;

  return y;
}

static inline float2 logk2f(float2 d) {
  float2 x, x2, m;
  float t;
  int e;

  e = ilogbp1f(d.x * 0.7071f);
  m = dfscale_f2_f2_f(d, pow2if(-e));

  x = dfdiv_f2_f2_f2(dfadd2_f2_f2_f(m, -1), dfadd2_f2_f2_f(m, 1));
  x2 = dfsqu_f2_f2(x);

  t = 0.2371599674224853515625f;
  t = mlaf(t, x2.x, 0.285279005765914916992188f);
  t = mlaf(t, x2.x, 0.400005519390106201171875f);
  t = mlaf(t, x2.x, 0.666666567325592041015625f);

  return dfadd2_f2_f2_f2(dfmul_f2_f2_f(df(0.69314718246459960938f, -1.904654323148236017e-09f), e),
			 dfadd2_f2_f2_f2(dfscale_f2_f2_f(x, 2), dfmul_f2_f2_f(dfmul_f2_f2_f2(x2, x), t)));
}

float xasinhf(float x) {
  float y = xfabsf(x);
  float2 d = logk2f(dfadd2_f2_f2_f(dfsqrt_f2_f2(dfadd2_f2_f2_f(dfmul_f2_f_f(y, y),  1)), y));
  y = d.x + d.y;

  y = xisinff(x) || xisnanf(y) ? INFINITYf : y;
  y = mulsignf(y, x);
  y = xisnanf(x) ? NANf : y;

  return y;
}

float xacoshf(float x) {
  float2 d = logk2f(dfadd2_f2_f2_f(dfsqrt_f2_f2(dfadd2_f2_f2_f(dfmul_f2_f_f(x, x), -1)), x));
  float y = d.x + d.y;

  y = xisinff(x) || xisnanf(y) ? INFINITYf : y;
  y = x == 1.0f ? 0.0f : y;
  y = x < 1.0f ? NANf : y;
  y = xisnanf(x) ? NANf : y;

  return y;
}

float xatanhf(float x) {
  float y = xfabsf(x);
  float2 d = logk2f(dfdiv_f2_f2_f2(dfadd2_f2_f_f(1, y), dfadd2_f2_f_f(1, -y)));
  y = y > 1.0 ? NANf : (y == 1.0 ? INFINITYf : (d.x + d.y) * 0.5f);

  y = xisinff(x) || xisnanf(y) ? NANf : y;
  y = mulsignf(y, x);
  y = xisnanf(x) ? NANf : y;

  return y;
}

float xexp2f(float a) {
  float u = expkf(dfmul_f2_f2_f(df(0.69314718246459960938f, -1.904654323148236017e-09f), a));
  if (xispinff(a)) u = INFINITYf;
  if (xisminff(a)) u = 0;
  return u;
}

float xexp10f(float a) {
  float u = expkf(dfmul_f2_f2_f(df(2.3025851249694824219f, -3.1975436520781386207e-08f), a));
  if (xispinff(a)) u = INFINITYf;
  if (xisminff(a)) u = 0;
  return u;
}

float xexpm1f(float a) {
  float2 d = dfadd2_f2_f2_f(expk2f(df(a, 0)), -1.0f);
  float x = d.x + d.y;
  if (a > 88.0f) x = INFINITYf;
  if (a < -0.15942385152878742116596338793538061065739925620174e+2f) x = -1;
  return x;
}

float xlog10f(float a) {
  float2 d = dfmul_f2_f2_f2(logkf(a), df(0.43429449200630187988f, -1.0103050118726031315e-08f));
  float x = d.x + d.y;

  if (xisinff(a)) x = INFINITYf;
  if (a < 0) x = NANf;
  if (a == 0) x = -INFINITYf;

  return x;
}

float xlog1pf(float a) {
  float2 d = logk2f(dfadd2_f2_f_f(a, 1));
  float x = d.x + d.y;

  if (xisinff(a)) x = INFINITYf;
  if (a < -1) x = NANf;
  if (a == -1) x = -INFINITYf;

  return x;
}

float xsqrtf(float f) { return sqrtf(f); }

float xcbrtf(float d) {
  float x, y, q = 1.0f;
  int e, r;

  e = ilogbp1f(d);
  d = ldexpkf(d, -e);
  r = (e + 6144) % 3;
  q = (r == 1) ? 1.2599210498948731647672106f : q;
  q = (r == 2) ? 1.5874010519681994747517056f : q;
  q = ldexpkf(q, (e + 6144) / 3 - 2048);

  q = mulsignf(q, d);
  d = xfabsf(d);

  x = -0.601564466953277587890625f;
  x = mlaf(x, d, 2.8208892345428466796875f);
  x = mlaf(x, d, -5.532182216644287109375f);
  x = mlaf(x, d, 5.898262500762939453125f);
  x = mlaf(x, d, -3.8095417022705078125f);
  x = mlaf(x, d, 2.2241256237030029296875f);

  y = d * x * x;
  y = (y - (2.0f / 3.0f) * y * (y * x - 1.0f)) * q;

  return y;
}

float xcbrtf_u1(float d) {
  float x, y, z;
  float2 q2 = df(1, 0), u, v;
  int e, r;

  e = ilogbp1f(d);
  d = ldexpkf(d, -e);
  r = (e + 6144) % 3;
  q2 = (r == 1) ? df(1.2599210739135742188, -2.4018701694217270415e-08) : q2;
  q2 = (r == 2) ? df(1.5874010324478149414,  1.9520385308169352356e-08) : q2;

  q2.x = mulsignf(q2.x, d); q2.y = mulsignf(q2.y, d);
  d = xfabsf(d);

  x = -0.601564466953277587890625f;
  x = mlaf(x, d, 2.8208892345428466796875f);
  x = mlaf(x, d, -5.532182216644287109375f);
  x = mlaf(x, d, 5.898262500762939453125f);
  x = mlaf(x, d, -3.8095417022705078125f);
  x = mlaf(x, d, 2.2241256237030029296875f);

  y = x * x; y = y * y; x -= (d * y - x) * (1.0 / 3.0f);

  z = x;

  u = dfmul_f2_f_f(x, x);
  u = dfmul_f2_f2_f2(u, u);
  u = dfmul_f2_f2_f(u, d);
  u = dfadd2_f2_f2_f(u, -x);
  y = u.x + u.y;

  y = -2.0 / 3.0 * y * z;
  v = dfadd2_f2_f2_f(dfmul_f2_f_f(z, z), y);
  v = dfmul_f2_f2_f(v, d);
  v = dfmul_f2_f2_f2(v, q2);
  z = ldexpf(v.x + v.y, (e + 6144) / 3 - 2048);

  if (xisinff(d)) { z = mulsignf(INFINITYf, q2.x); }
  if (d == 0) { z = mulsignf(0, q2.x); }

  return z;
}
