#include <stdio.h>

#ifdef ENABLE_SSE2
#define CONFIG 2
#include "helpersse2.h"
#define vcos_finite _ZGVbN2v___cos_finite
#define vsin_finite _ZGVbN2v___sin_finite
// #endif

extern vdouble vcos_finite(vdouble);
extern vdouble vsin_finite(vdouble);

int main(void) {
  double in[VECTLENDP], out[VECTLENDP];
  vdouble vin, vout;
  int i;
  vin = vload_vd_p(in);
  vout = vcos_finite(vin);
  vout = vsin_finite(vout);
  vstore_v_p_vd(out, vout);

  for (i = 0; i < VECTLENDP; ++i)
    printf("%g\n", out[i]);

  return 0;
}

#else

int main(void) { printf("Nothing here.\n"); }

#endif
