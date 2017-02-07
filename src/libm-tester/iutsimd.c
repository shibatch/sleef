//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <setjmp.h>
#include <inttypes.h>
#include <math.h>
#include <assert.h>

#if defined(_MSC_VER)
#define STDIN_FILENO 0
#else
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#endif

#include "misc.h"
#include "sleef.h"

#define DORENAME

#ifdef ENABLE_SSE2
#define CONFIG 2
#include "helpersse2.h"
#include "renamesse2.h"
typedef Sleef___m128d_2 vdouble2;
typedef Sleef___m128_2 vfloat2;
#endif

#ifdef ENABLE_AVX
#define CONFIG 1
#include "helperavx.h"
#include "renameavx.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_FMA4
#define CONFIG 4
#include "helperavx.h"
#include "renamefma4.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_AVX2
#define CONFIG 1
#include "helperavx2.h"
#include "renameavx2.h"
typedef Sleef___m256d_2 vdouble2;
typedef Sleef___m256_2 vfloat2;
#endif

#ifdef ENABLE_AVX512F
#define CONFIG 1
#include "helperavx512f.h"
#include "renameavx512f.h"
typedef Sleef___m512d_2 vdouble2;
typedef Sleef___m512_2 vfloat2;
#endif

#ifdef ENABLE_VECEXT
#define CONFIG 1
#include "helpervecext.h"
#include "norename.h"
#endif

#ifdef ENABLE_PUREC
#define CONFIG 1
#include "helperpurec.h"
#include "norename.h"
#endif

#ifdef ENABLE_NEON32
#define CONFIG 1
#include "helperneon32.h"
#include "renameneon32.h"
typedef Sleef_float32x4_t_2 vfloat2;
#endif

static jmp_buf sigjmp;

static void sighandler(int signum) {
  longjmp(sigjmp, 1);
}

int detectFeature() {
  signal(SIGILL, sighandler);

  if (setjmp(sigjmp) == 0) {
#ifdef ENABLE_DP
    double s[VECTLENDP];
    int i;
    for(i=0;i<VECTLENDP;i++) {
      s[i] = 1.0;
    }
    vdouble a = vloadu_vd_p(s);
    a = xpow(a, a);
    vstoreu_v_p_vd(s, a);
#elif defined(ENABLE_SP)
    float s[VECTLENSP];
    int i;
    for(i=0;i<VECTLENSP;i++) {
      s[i] = 1.0;
    }
    vfloat a = vloadu_vf_p(s);
    a = xpowf(a, a);
    vstoreu_v_p_vf(s, a);
#endif
    signal(SIGILL, SIG_DFL);
    return 1;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}

int readln(int fd, char *buf, int cnt) {
  int i, rcnt = 0;

  if (cnt < 1) return -1;

  while(cnt >= 2) {
    i = read(fd, buf, 1);
    if (i != 1) return i;

    if (*buf == '\n') break;

    rcnt++;
    buf++;
    cnt--;
  }

  *++buf = '\0';
  rcnt++;
  return rcnt;
}

int startsWith(char *str, char *prefix) {
  return strncmp(str, prefix, strlen(prefix)) == 0;
}

double u2d(uint64_t u) {
  union {
    double f;
    uint64_t i;
  } tmp;
  tmp.i = u;
  return tmp.f;
}

uint64_t d2u(double d) {
  union {
    double f;
    uint64_t i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

float u2f(uint32_t u) {
  union {
    float f;
    uint32_t i;
  } tmp;
  tmp.i = u;
  return tmp.f;
}

uint32_t f2u(float d) {
  union {
    float f;
    uint32_t i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

#ifdef ENABLE_DP

double xxsin(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xsin(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxcos(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xcos(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxtan(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xtan(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxasin(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xasin(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxacos(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xacos(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxatan(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xatan(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxlog(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xlog(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxexp(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xexp(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

Sleef_double2 xxsincos(double d) {
  double s[VECTLENDP], t[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
    t[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);

  s[idx] = d;

  vdouble2 v;

  vdouble a = vloadu_vd_p(s);
  v = xsincos(a);
  vstoreu_v_p_vd(s, v.x);
  vstoreu_v_p_vd(t, v.y);

  Sleef_double2 d2;
  d2.x = s[idx];
  d2.y = t[idx];

  return d2;
}

Sleef_double2 xxsincospi_u05(double d) {
  double s[VECTLENDP], t[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
    t[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);

  s[idx] = d;

  vdouble2 v;

  vdouble a = vloadu_vd_p(s);
  v = xsincospi_u05(a);
  vstoreu_v_p_vd(s, v.x);
  vstoreu_v_p_vd(t, v.y);

  Sleef_double2 d2;
  d2.x = s[idx];
  d2.y = t[idx];

  return d2;
}

Sleef_double2 xxsincospi_u35(double d) {
  double s[VECTLENDP], t[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
    t[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);

  s[idx] = d;

  vdouble2 v;

  vdouble a = vloadu_vd_p(s);
  v = xsincospi_u35(a);
  vstoreu_v_p_vd(s, v.x);
  vstoreu_v_p_vd(t, v.y);

  Sleef_double2 d2;
  d2.x = s[idx];
  d2.y = t[idx];

  return d2;
}

double xxsinh(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xsinh(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxcosh(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xcosh(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxtanh(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xtanh(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxasinh(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xasinh(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxacosh(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xacosh(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxatanh(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xatanh(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

#if 0
double xxsqrt(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xsqrt(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}
#endif

double xxcbrt(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xcbrt(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxexp2(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xexp2(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxexp10(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xexp10(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxexpm1(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xexpm1(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxlog10(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xlog10(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxlog1p(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xlog1p(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxpow(double x, double y) {
  double s[VECTLENDP], t[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
    t[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);

  s[idx] = x;
  t[idx] = y;

  vdouble a, b;

  a = vloadu_vd_p(s);
  b = vloadu_vd_p(t);
  a = xpow(a, b);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxatan2(double y, double x) {
  double s[VECTLENDP], t[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
    t[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);

  s[idx] = y;
  t[idx] = x;

  vdouble a, b;

  a = vloadu_vd_p(s);
  b = vloadu_vd_p(t);
  a = xatan2(a, b);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxldexp(double x, int q) {
  union {
    double s[VECTLENDP];
    vdouble vd;
  } ud;
  union {
    int t[VECTLENDP*2];
    vint vi;
  } ui;

  int i;
  for(i=0;i<VECTLENDP;i++) {
    ud.s[i] = rand()/(double)RAND_MAX*20000-10000;
    ui.t[i*2+0] = ui.t[i*2+1] = (int)(rand()/(double)RAND_MAX*20000-10000);
  }

  int idx = rand() & (VECTLENDP-1);

  ud.s[idx] = x;
  ui.t[idx] = q;

  ud.vd = xldexp(ud.vd, ui.vi);

  return ud.s[idx];
}

double xxsin_u1(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xsin_u1(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxcos_u1(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xcos_u1(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxtan_u1(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xtan_u1(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxasin_u1(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xasin_u1(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxacos_u1(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xacos_u1(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxatan_u1(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xatan_u1(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxatan2_u1(double y, double x) {
  double s[VECTLENDP], t[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
    t[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);

  s[idx] = y;
  t[idx] = x;

  vdouble a, b;

  a = vloadu_vd_p(s);
  b = vloadu_vd_p(t);
  a = xatan2_u1(a, b);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

double xxlog_u1(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xlog_u1(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

Sleef_double2 xxsincos_u1(double d) {
  double s[VECTLENDP], t[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
    t[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);

  s[idx] = d;

  vdouble2 v;

  vdouble a = vloadu_vd_p(s);
  v = xsincos_u1(a);
  vstoreu_v_p_vd(s, v.x);
  vstoreu_v_p_vd(t, v.y);

  Sleef_double2 d2;
  d2.x = s[idx];
  d2.y = t[idx];

  return d2;
}

double xxcbrt_u1(double d) {
  double s[VECTLENDP];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  a = xcbrt_u1(a);
  vstoreu_v_p_vd(s, a);

  return s[idx];
}

int xxilogb(double d) {
  double s[VECTLENDP];
  int t[VECTLENDP*2];
  int i;
  for(i=0;i<VECTLENDP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENDP-1);
  s[idx] = d;

  vdouble a = vloadu_vd_p(s);
  vint vi = xilogb(a);
  vstoreu_v_p_vi(t, vi);

  return t[idx];
}
#endif

//

#ifdef ENABLE_SP

float xxsinf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xsinf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxcosf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xcosf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxtanf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xtanf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxasinf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xasinf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxacosf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xacosf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxatanf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xatanf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxatan2f(float y, float x) {
  float s[VECTLENSP], t[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
    t[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);

  s[idx] = y;
  t[idx] = x;

  vfloat a, b;

  a = vloadu_vf_p(s);
  b = vloadu_vf_p(t);
  a = xatan2f(a, b);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxlogf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xlogf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxexpf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xexpf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxsqrtf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xsqrtf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxcbrtf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xcbrtf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

Sleef_float2 xxsincosf(float d) {
  float s[VECTLENSP], t[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
    t[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);

  s[idx] = d;

  vfloat2 v;

  vfloat a = vloadu_vf_p(s);
  v = xsincosf(a);
  vstoreu_v_p_vf(s, v.x);
  vstoreu_v_p_vf(t, v.y);

  Sleef_float2 d2;
  d2.x = s[idx];
  d2.y = t[idx];

  return d2;
}

Sleef_float2 xxsincospif_u05(float d) {
  float s[VECTLENSP], t[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
    t[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);

  s[idx] = d;

  vfloat2 v;

  vfloat a = vloadu_vf_p(s);
  v = xsincospif_u05(a);
  vstoreu_v_p_vf(s, v.x);
  vstoreu_v_p_vf(t, v.y);

  Sleef_float2 d2;
  d2.x = s[idx];
  d2.y = t[idx];

  return d2;
}

Sleef_float2 xxsincospif_u35(float d) {
  float s[VECTLENSP], t[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
    t[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);

  s[idx] = d;

  vfloat2 v;

  vfloat a = vloadu_vf_p(s);
  v = xsincospif_u35(a);
  vstoreu_v_p_vf(s, v.x);
  vstoreu_v_p_vf(t, v.y);

  Sleef_float2 d2;
  d2.x = s[idx];
  d2.y = t[idx];

  return d2;
}

float xxsinhf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xsinhf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxcoshf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xcoshf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxtanhf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xtanhf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxasinhf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xasinhf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxacoshf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xacoshf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxatanhf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xatanhf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxexp2f(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xexp2f(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxexp10f(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xexp10f(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxexpm1f(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xexpm1f(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxlog10f(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xlog10f(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxlog1pf(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xlog1pf(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxpowf(float x, float y) {
  float s[VECTLENSP], t[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
    t[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);

  s[idx] = x;
  t[idx] = y;

  vfloat a, b;

  a = vloadu_vf_p(s);
  b = vloadu_vf_p(t);
  a = xpowf(a, b);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

#if 0
float xxldexpf(float x, int q) {
  float s[VECTLENSP];
  int t[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(double)RAND_MAX*20000-10000;
    t[i] = (int)(rand()/(double)RAND_MAX*20000-10000);
  }

  int idx = rand() & (VECTLENSP-1);

  s[idx] = x;
  t[idx] = q;

  vfloat a;
  vint2 b;

  a = vloadu_vf_p(s);
  b = vloadu_vi2_p(t);
  a = xldexpf(a, b);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}
#endif

float xxsinf_u1(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xsinf_u1(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxcosf_u1(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xcosf_u1(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxtanf_u1(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xtanf_u1(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxasinf_u1(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xasinf_u1(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxacosf_u1(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xacosf_u1(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxatanf_u1(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xatanf_u1(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxatan2f_u1(float y, float x) {
  float s[VECTLENSP], t[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
    t[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);

  s[idx] = y;
  t[idx] = x;

  vfloat a, b;

  a = vloadu_vf_p(s);
  b = vloadu_vf_p(t);
  a = xatan2f_u1(a, b);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxlogf_u1(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xlogf_u1(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

float xxcbrtf_u1(float d) {
  float s[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  a = xcbrtf_u1(a);
  vstoreu_v_p_vf(s, a);

  return s[idx];
}

Sleef_float2 xxsincosf_u1(float d) {
  float s[VECTLENSP], t[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
    t[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);

  s[idx] = d;

  vfloat2 v;

  vfloat a = vloadu_vf_p(s);
  v = xsincosf_u1(a);
  vstoreu_v_p_vf(s, v.x);
  vstoreu_v_p_vf(t, v.y);

  Sleef_float2 d2;
  d2.x = s[idx];
  d2.y = t[idx];

  return d2;
}

#if 0
int xxilogbf(float d) {
  float s[VECTLENSP];
  int t[VECTLENSP];
  int i;
  for(i=0;i<VECTLENSP;i++) {
    s[i] = rand()/(float)RAND_MAX*20000-10000;
  }
  int idx = rand() & (VECTLENSP-1);
  s[idx] = d;

  vfloat a = vloadu_vf_p(s);
  vint2 vi = xilogbf(a);
  vstoreu_v_p_vi2(t, vi);

  return t[idx];
}
#endif
#endif

//

#define BUFSIZE 1024

int main(int argc, char **argv) {
  srand(time(NULL));

  if (!detectFeature()) {
    fprintf(stderr, "\n\n***** This host does not support the necessary CPU features to execute this program *****\n\n\n");
    printf("0\n");
    fclose(stdout);
    exit(-1);
  }

  {
    int k = 0;
#ifdef ENABLE_DP
    k += 1;
#endif
#ifdef ENABLE_SP
    k += 2;
#endif
    printf("%d\n", k);
    fflush(stdout);
  }
  
  char buf[BUFSIZE];

  //fprintf(stderr, "IUT start\n");

  for(;;) {
    if (readln(STDIN_FILENO, buf, BUFSIZE-1) < 1) break;

    //fprintf(stderr, "iut: got %s\n", buf);

#ifdef ENABLE_DP
    if (startsWith(buf, "sin ")) {
      uint64_t u;
      sscanf(buf, "sin %" PRIx64, &u);
      u = d2u(xxsin(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cos ")) {
      uint64_t u;
      sscanf(buf, "cos %" PRIx64, &u);
      u = d2u(xxcos(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sincos ")) {
      uint64_t u;
      sscanf(buf, "sincos %" PRIx64, &u);
      Sleef_double2 x = xxsincos(u2d(u));
      printf("%" PRIx64 " %" PRIx64 "\n", d2u(x.x), d2u(x.y));
    } else if (startsWith(buf, "tan ")) {
      uint64_t u;
      sscanf(buf, "tan %" PRIx64, &u);
      u = d2u(xxtan(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sincospi_u05 ")) {
      uint64_t u;
      sscanf(buf, "sincospi_u05 %" PRIx64, &u);
      Sleef_double2 x = xxsincospi_u05(u2d(u));
      printf("%" PRIx64 " %" PRIx64 "\n", d2u(x.x), d2u(x.y));
    } else if (startsWith(buf, "sincospi_u35 ")) {
      uint64_t u;
      sscanf(buf, "sincospi_u35 %" PRIx64, &u);
      Sleef_double2 x = xxsincospi_u35(u2d(u));
      printf("%" PRIx64 " %" PRIx64 "\n", d2u(x.x), d2u(x.y));
    } else if (startsWith(buf, "asin ")) {
      uint64_t u;
      sscanf(buf, "asin %" PRIx64, &u);
      u = d2u(xxasin(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "acos ")) {
      uint64_t u;
      sscanf(buf, "acos %" PRIx64, &u);
      u = d2u(xxacos(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atan ")) {
      uint64_t u;
      sscanf(buf, "atan %" PRIx64, &u);
      u = d2u(xxatan(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "log ")) {
      uint64_t u;
      sscanf(buf, "log %" PRIx64, &u);
      u = d2u(xxlog(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "exp ")) {
      uint64_t u;
      sscanf(buf, "exp %" PRIx64, &u);
      u = d2u(xxexp(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atan2 ")) {
      uint64_t u, v;
      sscanf(buf, "atan2 %" PRIx64 " %" PRIx64, &u, &v);
      u = d2u(xxatan2(u2d(u), u2d(v)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "pow ")) {
      uint64_t u, v;
      sscanf(buf, "pow %" PRIx64 " %" PRIx64, &u, &v);
      u = d2u(xxpow(u2d(u), u2d(v)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sinh ")) {
      uint64_t u;
      sscanf(buf, "sinh %" PRIx64, &u);
      u = d2u(xxsinh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cosh ")) {
      uint64_t u;
      sscanf(buf, "cosh %" PRIx64, &u);
      u = d2u(xxcosh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "tanh ")) {
      uint64_t u;
      sscanf(buf, "tanh %" PRIx64, &u);
      u = d2u(xxtanh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "asinh ")) {
      uint64_t u;
      sscanf(buf, "asinh %" PRIx64, &u);
      u = d2u(xxasinh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "acosh ")) {
      uint64_t u;
      sscanf(buf, "acosh %" PRIx64, &u);
      u = d2u(xxacosh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atanh ")) {
      uint64_t u;
      sscanf(buf, "atanh %" PRIx64, &u);
      u = d2u(xxatanh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sqrt ")) {
      uint64_t u;
      sscanf(buf, "sqrt %" PRIx64, &u);
      u = d2u(sqrt(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cbrt ")) {
      uint64_t u;
      sscanf(buf, "cbrt %" PRIx64, &u);
      u = d2u(xxcbrt(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "exp2 ")) {
      uint64_t u;
      sscanf(buf, "exp2 %" PRIx64, &u);
      u = d2u(xxexp2(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "exp10 ")) {
      uint64_t u;
      sscanf(buf, "exp10 %" PRIx64, &u);
      u = d2u(xxexp10(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "expm1 ")) {
      uint64_t u;
      sscanf(buf, "expm1 %" PRIx64, &u);
      u = d2u(xxexpm1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "log10 ")) {
      uint64_t u;
      sscanf(buf, "log10 %" PRIx64, &u);
      u = d2u(xxlog10(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "ldexp ")) {
      uint64_t u, v;
      sscanf(buf, "ldexp %" PRIx64 " %" PRIx64, &u, &v);
      u = d2u(xxldexp(u2d(u), (int)u2d(v)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "log1p ")) {
      uint64_t u;
      sscanf(buf, "log1p %" PRIx64, &u);
      u = d2u(xxlog1p(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sin_u1 ")) {
      uint64_t u;
      sscanf(buf, "sin_u1 %" PRIx64, &u);
      u = d2u(xxsin_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cos_u1 ")) {
      uint64_t u;
      sscanf(buf, "cos_u1 %" PRIx64, &u);
      u = d2u(xxcos_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sincos_u1 ")) {
      uint64_t u;
      sscanf(buf, "sincos_u1 %" PRIx64, &u);
      Sleef_double2 x = xxsincos_u1(u2d(u));
      printf("%" PRIx64 " %" PRIx64 "\n", d2u(x.x), d2u(x.y));
    } else if (startsWith(buf, "tan_u1 ")) {
      uint64_t u;
      sscanf(buf, "tan_u1 %" PRIx64, &u);
      u = d2u(xxtan_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "asin_u1 ")) {
      uint64_t u;
      sscanf(buf, "asin_u1 %" PRIx64, &u);
      u = d2u(xxasin_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "acos_u1 ")) {
      uint64_t u;
      sscanf(buf, "acos_u1 %" PRIx64, &u);
      u = d2u(xxacos_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atan_u1 ")) {
      uint64_t u;
      sscanf(buf, "atan_u1 %" PRIx64, &u);
      u = d2u(xxatan_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atan2_u1 ")) {
      uint64_t u, v;
      sscanf(buf, "atan2_u1 %" PRIx64 " %" PRIx64, &u, &v);
      u = d2u(xxatan2_u1(u2d(u), u2d(v)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "log_u1 ")) {
      uint64_t u;
      sscanf(buf, "log_u1 %" PRIx64, &u);
      u = d2u(xxlog_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cbrt_u1 ")) {
      uint64_t u;
      sscanf(buf, "cbrt_u1 %" PRIx64, &u);
      u = d2u(xxcbrt_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "ilogb ")) {
      uint64_t u;
      int i;
      sscanf(buf, "ilogb %" PRIx64, &u);
      i = xxilogb(u2d(u));
      printf("%d\n", i);
    }
#ifdef ENABLE_SP
    else 
#endif
#endif

#ifdef ENABLE_SP
    if (startsWith(buf, "sinf ")) {
      uint32_t u;
      sscanf(buf, "sinf %x", &u);
      u = f2u(xxsinf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "cosf ")) {
      uint32_t u;
      sscanf(buf, "cosf %x", &u);
      u = f2u(xxcosf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "tanf ")) {
      uint32_t u;
      sscanf(buf, "tanf %x", &u);
      u = f2u(xxtanf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "asinf ")) {
      uint32_t u;
      sscanf(buf, "asinf %x", &u);
      u = f2u(xxasinf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "acosf ")) {
      uint32_t u;
      sscanf(buf, "acosf %x", &u);
      u = f2u(xxacosf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atanf ")) {
      uint32_t u;
      sscanf(buf, "atanf %x", &u);
      u = f2u(xxatanf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "logf ")) {
      uint32_t u;
      sscanf(buf, "logf %x", &u);
      u = f2u(xxlogf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "expf ")) {
      uint32_t u;
      sscanf(buf, "expf %x", &u);
      u = f2u(xxexpf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atan2f ")) {
      uint32_t u, v;
      sscanf(buf, "atan2f %x %x", &u, &v);
      u = f2u(xxatan2f(u2f(u), u2f(v)));
      printf("%x\n", u);
    } else if (startsWith(buf, "cbrtf ")) {
      uint32_t u;
      sscanf(buf, "cbrtf %x", &u);
      u = f2u(xxcbrtf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sincosf ")) {
      uint32_t u;
      sscanf(buf, "sincosf %x", &u);
      Sleef_float2 x = xxsincosf(u2f(u));
      printf("%x %x\n", f2u(x.x), f2u(x.y));
    } else if (startsWith(buf, "sincospif_u05 ")) {
      uint32_t u;
      sscanf(buf, "sincospif_u05 %x", &u);
      Sleef_float2 x = xxsincospif_u05(u2f(u));
      printf("%x %x\n", f2u(x.x), f2u(x.y));
    } else if (startsWith(buf, "sincospif_u35 ")) {
      uint32_t u;
      sscanf(buf, "sincospif_u35 %x", &u);
      Sleef_float2 x = xxsincospif_u35(u2f(u));
      printf("%x %x\n", f2u(x.x), f2u(x.y));
    } else if (startsWith(buf, "powf ")) {
      uint32_t u, v;
      sscanf(buf, "powf %x %x", &u, &v);
      u = f2u(xxpowf(u2f(u), u2f(v)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sinhf ")) {
      uint32_t u;
      sscanf(buf, "sinhf %x", &u);
      u = f2u(xxsinhf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "coshf ")) {
      uint32_t u;
      sscanf(buf, "coshf %x", &u);
      u = f2u(xxcoshf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "tanhf ")) {
      uint32_t u;
      sscanf(buf, "tanhf %x", &u);
      u = f2u(xxtanhf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "asinhf ")) {
      uint32_t u;
      sscanf(buf, "asinhf %x", &u);
      u = f2u(xxasinhf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "acoshf ")) {
      uint32_t u;
      sscanf(buf, "acoshf %x", &u);
      u = f2u(xxacoshf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atanhf ")) {
      uint32_t u;
      sscanf(buf, "atanhf %x", &u);
      u = f2u(xxatanhf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sqrtf ")) {
      uint32_t u;
      sscanf(buf, "sqrtf %x", &u);
      u = f2u(xxsqrtf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "exp2f ")) {
      uint32_t u;
      sscanf(buf, "exp2f %x", &u);
      u = f2u(xxexp2f(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "exp10f ")) {
      uint32_t u;
      sscanf(buf, "exp10f %x", &u);
      u = f2u(xxexp10f(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "expm1f ")) {
      uint32_t u;
      sscanf(buf, "expm1f %x", &u);
      u = f2u(xxexpm1f(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "log10f ")) {
      uint32_t u;
      sscanf(buf, "log10f %x", &u);
      u = f2u(xxlog10f(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "log1pf ")) {
      uint32_t u;
      sscanf(buf, "log1pf %x", &u);
      u = f2u(xxlog1pf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sinf_u1 ")) {
      uint32_t u;
      sscanf(buf, "sinf_u1 %x", &u);
      u = f2u(xxsinf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "cosf_u1 ")) {
      uint32_t u;
      sscanf(buf, "cosf_u1 %x", &u);
      u = f2u(xxcosf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "tanf_u1 ")) {
      uint32_t u;
      sscanf(buf, "tanf_u1 %x", &u);
      u = f2u(xxtanf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "asinf_u1 ")) {
      uint32_t u;
      sscanf(buf, "asinf_u1 %x", &u);
      u = f2u(xxasinf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "acosf_u1 ")) {
      uint32_t u;
      sscanf(buf, "acosf_u1 %x", &u);
      u = f2u(xxacosf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atanf_u1 ")) {
      uint32_t u;
      sscanf(buf, "atanf_u1 %x", &u);
      u = f2u(xxatanf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "logf_u1 ")) {
      uint32_t u;
      sscanf(buf, "logf_u1 %x", &u);
      u = f2u(xxlogf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atan2f_u1 ")) {
      uint32_t u, v;
      sscanf(buf, "atan2f_u1 %x %x", &u, &v);
      u = f2u(xxatan2f_u1(u2f(u), u2f(v)));
      printf("%x\n", u);
    } else if (startsWith(buf, "cbrtf_u1 ")) {
      uint32_t u;
      sscanf(buf, "cbrtf_u1 %x", &u);
      u = f2u(xxcbrtf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sincosf_u1 ")) {
      uint32_t u;
      sscanf(buf, "sincosf_u1 %x", &u);
      Sleef_float2 x = xxsincosf_u1(u2f(u));
      printf("%x %x\n", f2u(x.x), f2u(x.y));
    }
    /*
    else if (startsWith(buf, "ldexpf ")) {
      uint32_t u, v;
      sscanf(buf, "ldexpf %x %x", &u, &v);
      u = f2u(xxldexpf(u2f(u), (int)u2f(v)));
      printf("%x\n", u);
      } else if (startsWith(buf, "ilogbf ")) {
      uint32_t u;
      int i;
      sscanf(buf, "ilogbf %x", &u);
      i = xxilogbf(u2f(u));
      printf("%d\n", i);
      }*/
#endif

    else {
      break;
    }

    fflush(stdout);
  }

  return 0;
}
