#include <setjmp.h>
#include <signal.h>
#include <stdio.h>

#if defined(ENABLE_SSE4) || defined(ENABLE_SSE2)
#define ISA_TOKEN b
#define VLEN_SP 4
#define VLEN_DP 2
#endif /* defined(ENABLE_SSE4) || defined(ENABLE_SSE2) */

#ifdef ENABLE_AVX
#define ISA_TOKEN c
#define VLEN_SP 8
#define VLEN_DP 4
#endif /* ENABLE_AVX */

#ifdef ENABLE_AVX2
#define ISA_TOKEN d
#define VLEN_SP 8
#define VLEN_DP 4
#endif /* ENABLE_AVX2 */

#ifdef ENABLE_AVX512F
#define ISA_TOKEN e
#define VLEN_SP 16
#define VLEN_DP 8
#endif /* ENABLE_AVX512F */

#ifdef ENABLE_ADVSIMD
#define ISA_TOKEN n
#define VLEN_SP 4
#define VLEN_DP 2
#endif /* ENABLE_ADVSIMDF */

#define __MAKE_FN_NAME(name, t, vl, p) _ZGV##t##N##vl##p##_##name

#define __DECLARE(name, t, vl, p) void __MAKE_FN_NAME(name, t, vl, p)(int *)
#define __CALL(name, t, vl, p) __MAKE_FN_NAME(name, t, vl, p)(a)

#ifndef ISA_TOKEN
#error "Missing ISA token"
#endif

#ifndef VLEN_DP
#error "Missing VLEN_DP"
#endif

#ifndef VLEN_DP
#error "Missing VLEN_SP"
#endif

#define DECLARE_DP(name, p) __DECLARE(name, ISA_TOKEN, VLEN_DP, p)
#define CALL_DP(name, p) __CALL(name, ISA_TOKEN, VLEN_DP, p)

DECLARE_DP(__acos_finite, v);
DECLARE_DP(__asin_finite, v);

#define DECLARE_SP(name, p) __DECLARE(name, ISA_TOKEN, VLEN_SP, p)
#define CALL_SP(name, p) __CALL(name, ISA_TOKEN, VLEN_SP, p)

DECLARE_SP(__acosf_finite, v);
DECLARE_SP(__asinf_finite, v);

int b;
int *a = &b;

static jmp_buf sigjmp;

static void sighandler(int signum) { longjmp(sigjmp, 1); }

int detectFeature() {
  signal(SIGILL, sighandler);

  if (setjmp(sigjmp) == 0) {
    CALL_DP(__acos_finite, v);
    signal(SIGILL, SIG_DFL);
    return 1;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}

int main(void) {

  if (!detectFeature()) {
    return 0;
  }

  CALL_DP(__acos_finite, v);
  CALL_DP(__asin_finite, v);

  CALL_SP(__acosf_finite, v);
  CALL_SP(__asinf_finite, v);

  printf("%d\n", b);
  return 0;
}
