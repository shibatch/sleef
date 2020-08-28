//   Copyright Naoki Shibata and contributors 2010 - 2020.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "funcproto.h"
#include <sleef-config.h>

static const size_t vecSizes[] = {
  0, 16, 32, 64,
};

static const struct variantInfo {
  size_t vecSize;
  const char* ppTest;
  const char* suffix;
} variantList[] = {

  /* Arm */

  #if defined(COMPILER_SUPPORTS_SVE)
    { 0, "defined(__ARM_SVE)", "sve" },
  #endif
  #if defined(COMPILER_SUPPORTS_SVENOFMA)
    { 0, "defined(__ARM_SVE)", "svenofma" },
  #endif

  #if defined(COMPILER_SUPPORTS_ADVSIMD)
    /* Not sure what the feature check for FMA should look like */
    { 16, "defined(__ARM_NEON) && defined(__aarch64__)", "advsimd" },
  #endif
  #if defined(COMPILER_SUPPORTS_ADVSIMDNOFMA)
    { 16, "defined(__ARM_NEON) && defined(__aarch64__)", "advsimdnofma" },
  #endif

  #if defined(COMPILER_SUPPORTS_NEON32VFPV4)
    { 16, "defined(__ARM_NEON) && (__ARM_NEON_FP >= 4)", "neonvfpv4" },
  #endif
  #if defined(COMPILER_SUPPORTS_NEON32VFPV4)
    { 16, "defined(__ARM_NEON)", "neon" },
  #endif

  /* POWER */

  #if defined(COMPILER_SUPPORTS_VSX)
    /* Not sure what the feature check for FMA should look like */
    { 16, "defined(__VSX__) && defined(__ARCH_PWR8) && defined(__LITTLE_ENDIAN__)", "vsx" },
  #endif
  #if defined(COMPILER_SUPPORTS_VSXNOFMA)
    { 16, "defined(__VSX__) && defined(__ARCH_PWR8) && defined(__LITTLE_ENDIAN__)", "vsxnofma" },
  #endif

  /* x86 */

  #if defined(COMPILER_SUPPORTS_AVX512F)
    { 64, "defined(__AVX512F__) && defined(__FMA4__)", "avx512f" },
  #endif
  #if defined(COMPILER_SUPPORTS_AVX512FNOFMA)
    { 64, "defined(__AVX512F__)", "avx512fnofma" },
  #endif

  #if defined(COMPILER_SUPPORTS_FMA4)
    { 32, "defined(__FMA4__)", "fma4" },
  #endif
  #if defined(COMPILER_SUPPORTS_AVX2)
    { 32, "defined(__AVX2__)", "avx2" },
  #endif
  #if defined(COMPILER_SUPPORTS_AVX)
    { 32, "defined(__AVX__)", "avx" },
  #endif

  #if defined(COMPILER_SUPPORTS_AVX2128)
    { 16, "defined(__AVX2__)", "avx2128" },
  #endif
  #if defined(COMPILER_SUPPORTS_SSE4)
    { 16, "defined(__SSE4_1__)", "sse4" },
  #endif
  #if defined(COMPILER_SUPPORTS_SSE2)
    { 16, "defined(__SSE2__)", "sse2" },
  #endif

  { 0, NULL, NULL }
};

void
write_static_dispatch(FILE* fp, size_t vecSize) {
  int first = 1;
  for (const struct variantInfo* vi = variantList ; vi->suffix != NULL ; vi++) {
    if (vi->vecSize != vecSize)
      continue;

    if (first) {
      first = 0;
      fprintf(fp, "#if %s\n", vi->ppTest);
    } else {
      fprintf(fp, "#elif %s\n", vi->ppTest);
    }

    for (funcSpec* func = funcList ; func->name != NULL ; func++) {
      if (func->funcType >= 7) {
        continue;
      }

      if (vecSize != 0) {
        if (func->ulp > 0) {
          fprintf(fp, "#  define Sleef_%sf%zu_u%02dstatic Sleef_%sf%zu_u%02d%s\n",
                  func->name, vecSize / sizeof(float), func->ulp,
                  func->name, vecSize / sizeof(float), func->ulp, vi->suffix);
        } else {
          fprintf(fp, "#  define Sleef_%sf%zu_static Sleef_%sf%zu_%s\n",
                  func->name, vecSize / sizeof(float),
                  func->name, vecSize / sizeof(float), vi->suffix);
        }
      } else {
        if (func->ulp > 0) {
          fprintf(fp, "#  define Sleef_%sfx_u%02dstatic Sleef_%sfx_u%02d%s\n",
                  func->name, func->ulp,
                  func->name, func->ulp, vi->suffix);
        } else {
          fprintf(fp, "#  define Sleef_%sfx_static Sleef_%sfx_%s\n",
                  func->name,
                  func->name, vi->suffix);
        }
      }
    }
  }

  if (!first) {
    fputs("#endif\n\n", fp);
  }
}

int
main(void) {
  fprintf(stdout, "\n/* Begin static dispatch macros */\n\n");

  for (size_t i = 0 ; i < (sizeof(vecSizes) / sizeof(vecSizes[0])) ; i++) {
    write_static_dispatch(stdout, vecSizes[i]);
  }

  fprintf(stdout, "/* End static distpatch macros */\n\n");

  return 0;
}
