#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <math.h>
#include <inttypes.h>

#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <signal.h>

#include "sleef.h"

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

#define BUFSIZE 1024

int main(int argc, char **argv) {
  char buf[BUFSIZE];

  //fprintf(stderr, "IUT start\n");

  for(;;) {
    if (readln(STDIN_FILENO, buf, BUFSIZE-1) < 1) break;

    //fprintf(stderr, "iut: got %s\n", buf);

    if (startsWith(buf, "sin ")) {
      uint64_t u;
      sscanf(buf, "sin %" PRIx64, &u);
      u = d2u(xsin(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sin_u1 ")) {
      uint64_t u;
      sscanf(buf, "sin_u1 %" PRIx64, &u);
      u = d2u(xsin_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cos ")) {
      uint64_t u;
      sscanf(buf, "cos %" PRIx64, &u);
      u = d2u(xcos(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cos_u1 ")) {
      uint64_t u;
      sscanf(buf, "cos_u1 %" PRIx64, &u);
      u = d2u(xcos_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sincos ")) {
      uint64_t u;
      sscanf(buf, "sincos %" PRIx64, &u);
      double2 x = xsincos(u2d(u));
      printf("%" PRIx64 " %" PRIx64 "\n", d2u(x.x), d2u(x.y));
    } else if (startsWith(buf, "sincos_u1 ")) {
      uint64_t u;
      sscanf(buf, "sincos_u1 %" PRIx64, &u);
      double2 x = xsincos_u1(u2d(u));
      printf("%" PRIx64 " %" PRIx64 "\n", d2u(x.x), d2u(x.y));
    } else if (startsWith(buf, "tan ")) {
      uint64_t u;
      sscanf(buf, "tan %" PRIx64, &u);
      u = d2u(xtan(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "tan_u1 ")) {
      uint64_t u;
      sscanf(buf, "tan_u1 %" PRIx64, &u);
      u = d2u(xtan_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "asin ")) {
      uint64_t u;
      sscanf(buf, "asin %" PRIx64, &u);
      u = d2u(xasin(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "acos ")) {
      uint64_t u;
      sscanf(buf, "acos %" PRIx64, &u);
      u = d2u(xacos(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atan ")) {
      uint64_t u;
      sscanf(buf, "atan %" PRIx64, &u);
      u = d2u(xatan(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "log ")) {
      uint64_t u;
      sscanf(buf, "log %" PRIx64, &u);
      u = d2u(xlog(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "exp ")) {
      uint64_t u;
      sscanf(buf, "exp %" PRIx64, &u);
      u = d2u(xexp(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atan2 ")) {
      uint64_t u, v;
      sscanf(buf, "atan2 %" PRIx64 " %" PRIx64, &u, &v);
      u = d2u(xatan2(u2d(u), u2d(v)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "asin_u1 ")) {
      uint64_t u;
      sscanf(buf, "asin_u1 %" PRIx64, &u);
      u = d2u(xasin_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "acos_u1 ")) {
      uint64_t u;
      sscanf(buf, "acos_u1 %" PRIx64, &u);
      u = d2u(xacos_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atan_u1 ")) {
      uint64_t u;
      sscanf(buf, "atan_u1 %" PRIx64, &u);
      u = d2u(xatan_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atan2_u1 ")) {
      uint64_t u, v;
      sscanf(buf, "atan2_u1 %" PRIx64 " %" PRIx64, &u, &v);
      u = d2u(xatan2_u1(u2d(u), u2d(v)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "log_u1 ")) {
      uint64_t u;
      sscanf(buf, "log_u1 %" PRIx64, &u);
      u = d2u(xlog_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "pow ")) {
      uint64_t u, v;
      sscanf(buf, "pow %" PRIx64 " %" PRIx64, &u, &v);
      u = d2u(xpow(u2d(u), u2d(v)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sinh ")) {
      uint64_t u;
      sscanf(buf, "sinh %" PRIx64, &u);
      u = d2u(xsinh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cosh ")) {
      uint64_t u;
      sscanf(buf, "cosh %" PRIx64, &u);
      u = d2u(xcosh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "tanh ")) {
      uint64_t u;
      sscanf(buf, "tanh %" PRIx64, &u);
      u = d2u(xtanh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "asinh ")) {
      uint64_t u;
      sscanf(buf, "asinh %" PRIx64, &u);
      u = d2u(xasinh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "acosh ")) {
      uint64_t u;
      sscanf(buf, "acosh %" PRIx64, &u);
      u = d2u(xacosh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "atanh ")) {
      uint64_t u;
      sscanf(buf, "atanh %" PRIx64, &u);
      u = d2u(xatanh(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "fma ")) {
      uint64_t u, v, w;
      sscanf(buf, "fma %" PRIx64 " %" PRIx64 " %" PRIx64, &u, &v, &w);
      u = d2u(xfma(u2d(u), u2d(v), u2d(w)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sqrt ")) {
      uint64_t u;
      sscanf(buf, "sqrt %" PRIx64, &u);
      u = d2u(xsqrt(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cbrt ")) {
      uint64_t u;
      sscanf(buf, "cbrt %" PRIx64, &u);
      u = d2u(xcbrt(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "cbrt_u1 ")) {
      uint64_t u;
      sscanf(buf, "cbrt_u1 %" PRIx64, &u);
      u = d2u(xcbrt_u1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "exp2 ")) {
      uint64_t u;
      sscanf(buf, "exp2 %" PRIx64, &u);
      u = d2u(xexp2(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "exp10 ")) {
      uint64_t u;
      sscanf(buf, "exp10 %" PRIx64, &u);
      u = d2u(xexp10(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "expm1 ")) {
      uint64_t u;
      sscanf(buf, "expm1 %" PRIx64, &u);
      u = d2u(xexpm1(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "log10 ")) {
      uint64_t u;
      sscanf(buf, "log10 %" PRIx64, &u);
      u = d2u(xlog10(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "log1p ")) {
      uint64_t u;
      sscanf(buf, "log1p %" PRIx64, &u);
      u = d2u(xlog1p(u2d(u)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "ldexp ")) {
      uint64_t u, v;
      sscanf(buf, "ldexp %" PRIx64 " %" PRIx64, &u, &v);
      u = d2u(xldexp(u2d(u), (int)u2d(v)));
      printf("%" PRIx64 "\n", u);
    } else if (startsWith(buf, "sinf ")) {
      uint32_t u;
      sscanf(buf, "sinf %x", &u);
      u = f2u(xsinf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "cosf ")) {
      uint32_t u;
      sscanf(buf, "cosf %x", &u);
      u = f2u(xcosf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sincosf ")) {
      uint32_t u;
      sscanf(buf, "sincosf %x", &u);
      float2 x = xsincosf(u2f(u));
      printf("%x %x\n", f2u(x.x), f2u(x.y));
    } else if (startsWith(buf, "tanf ")) {
      uint32_t u;
      sscanf(buf, "tanf %x", &u);
      u = f2u(xtanf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "asinf ")) {
      uint32_t u;
      sscanf(buf, "asinf %x", &u);
      u = f2u(xasinf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "acosf ")) {
      uint32_t u;
      sscanf(buf, "acosf %x", &u);
      u = f2u(xacosf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atanf ")) {
      uint32_t u;
      sscanf(buf, "atanf %x", &u);
      u = f2u(xatanf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atan2f ")) {
      uint32_t u, v;
      sscanf(buf, "atan2f %x %x", &u, &v);
      u = f2u(xatan2f(u2f(u), u2f(v)));
      printf("%x\n", u);
    } else if (startsWith(buf, "logf ")) {
      uint32_t u;
      sscanf(buf, "logf %x", &u);
      u = f2u(xlogf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "expf ")) {
      uint32_t u;
      sscanf(buf, "expf %x", &u);
      u = f2u(xexpf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "cbrtf ")) {
      uint32_t u;
      sscanf(buf, "cbrtf %x", &u);
      u = f2u(xcbrtf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sqrtf ")) {
      uint32_t u;
      sscanf(buf, "sqrtf %x", &u);
      u = f2u(sqrt(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "ldexpf ")) {
      uint32_t u, v;
      sscanf(buf, "ldexpf %x %x", &u, &v);
      u = f2u(xldexpf(u2f(u), (int)u2f(v)));
      printf("%x\n", u);
    } else if (startsWith(buf, "powf ")) {
      uint32_t u, v;
      sscanf(buf, "powf %x %x", &u, &v);
      u = f2u(xpowf(u2f(u), u2f(v)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sinhf ")) {
      uint32_t u;
      sscanf(buf, "sinhf %x", &u);
      u = f2u(xsinhf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "coshf ")) {
      uint32_t u;
      sscanf(buf, "coshf %x", &u);
      u = f2u(xcoshf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "tanhf ")) {
      uint32_t u;
      sscanf(buf, "tanhf %x", &u);
      u = f2u(xtanhf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "asinhf ")) {
      uint32_t u;
      sscanf(buf, "asinhf %x", &u);
      u = f2u(xasinhf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "acoshf ")) {
      uint32_t u;
      sscanf(buf, "acoshf %x", &u);
      u = f2u(xacoshf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atanhf ")) {
      uint32_t u;
      sscanf(buf, "atanhf %x", &u);
      u = f2u(xatanhf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "exp2f ")) {
      uint32_t u;
      sscanf(buf, "exp2f %x", &u);
      u = f2u(xexp2f(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "exp10f ")) {
      uint32_t u;
      sscanf(buf, "exp10f %x", &u);
      u = f2u(xexp10f(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "expm1f ")) {
      uint32_t u;
      sscanf(buf, "expm1f %x", &u);
      u = f2u(xexpm1f(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "log10f ")) {
      uint32_t u;
      sscanf(buf, "log10f %x", &u);
      u = f2u(xlog10f(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "log1pf ")) {
      uint32_t u;
      sscanf(buf, "log1pf %x", &u);
      u = f2u(xlog1pf(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sinf_u1 ")) {
      uint32_t u;
      sscanf(buf, "sinf_u1 %x", &u);
      u = f2u(xsinf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "cosf_u1 ")) {
      uint32_t u;
      sscanf(buf, "cosf_u1 %x", &u);
      u = f2u(xcosf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "sincosf_u1 ")) {
      uint32_t u;
      sscanf(buf, "sincosf_u1 %x", &u);
      float2 x = xsincosf_u1(u2f(u));
      printf("%x %x\n", f2u(x.x), f2u(x.y));
    } else if (startsWith(buf, "tanf_u1 ")) {
      uint32_t u;
      sscanf(buf, "tanf_u1 %x", &u);
      u = f2u(xtanf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "asinf_u1 ")) {
      uint32_t u;
      sscanf(buf, "asinf_u1 %x", &u);
      u = f2u(xasinf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "acosf_u1 ")) {
      uint32_t u;
      sscanf(buf, "acosf_u1 %x", &u);
      u = f2u(xacosf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atanf_u1 ")) {
      uint32_t u;
      sscanf(buf, "atanf_u1 %x", &u);
      u = f2u(xatanf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "atan2f_u1 ")) {
      uint32_t u, v;
      sscanf(buf, "atan2f_u1 %x %x", &u, &v);
      u = f2u(xatan2f_u1(u2f(u), u2f(v)));
      printf("%x\n", u);
    } else if (startsWith(buf, "logf_u1 ")) {
      uint32_t u;
      sscanf(buf, "logf_u1 %x", &u);
      u = f2u(xlogf_u1(u2f(u)));
      printf("%x\n", u);
    } else if (startsWith(buf, "cbrtf_u1 ")) {
      uint32_t u;
      sscanf(buf, "cbrtf_u1 %x", &u);
      u = f2u(xcbrtf_u1(u2f(u)));
      printf("%x\n", u);
    } else {
      break;
    }

    fflush(stdout);
  }

  return 0;
}
