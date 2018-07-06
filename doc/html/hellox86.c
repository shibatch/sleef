#include <stdio.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <sleef.h>

int main(int argc, char **argv) {
  double a[] = {2, 10};
  double b[] = {3, 20};

  __m128d va, vb, vc;
  
  va = _mm_loadu_pd(a);
  vb = _mm_loadu_pd(b);

  vc = Sleef_powd2_u10(va, vb);

  double c[2];

  _mm_storeu_pd(c, vc);

  printf("pow(%g, %g) = %g\n", a[0], b[0], c[0]);
  printf("pow(%g, %g) = %g\n", a[1], b[1], c[1]);
}
