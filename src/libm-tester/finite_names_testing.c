#include <stdio.h>

#if defined(ENABLE_SSE2) || defined(ENABLE_SSE4)

#ifdef ENABLE_SSE2
#define NAME "SSE2"
#define CONFIG 2
#endif

#ifdef ENABLE_SSE4
#define NAME "SSE4"
#define CONFIG 4
#endif

#include "helpersse2.h"
#define vacos_finite _ZGVbN2v___acos_finite
#define vasin_finite _ZGVbN2v___asin_finite

#endif /* defined(ENABLE_SSE2) || defined(ENABLE_SSE4) */

#ifdef NAME

extern vdouble vacos_finite(vdouble);
extern vdouble vasin_finite(vdouble);

int main(void) {
  double in[VECTLENDP], out[VECTLENDP];
  vdouble vin, vout;
  int i;

  for (i = 0; i < VECTLENDP; ++i)
    in[i] = i;

  vin = vload_vd_p(in);
  vout = vacos_finite(vin);
  vout = vasin_finite(vout);
  vstore_v_p_vd(out, vout);

  printf("%s = ", NAME);
  for (i = 0; i < VECTLENDP; ++i)
    printf("%g ", out[i]);
  printf("\n");

  return 0;
}

#else

int main(void) {
  printf("Nothing here.\n");
  return 0;
}

#endif
