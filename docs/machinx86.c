#include <sleefquad.h>

int main(int argc, char **argv) {
  __float128 a0[] = { 5, 239 };
  Sleef_quadx2 q0 = Sleef_loadq2_sse2(a0);

  __float128 a1[] = { 1, 1 };
  Sleef_quadx2 q1 = Sleef_loadq2_sse2(a1);

  __float128 a2[] = { 16, 4 };
  Sleef_quadx2 q2 = Sleef_loadq2_sse2(a2);

  Sleef_quadx2 q3;
  q3 = Sleef_divq2_u05sse2(q1, q0);
  q3 = Sleef_atanq2_u10sse2(q3);
  q3 = Sleef_mulq2_u05sse2(q3, q2);

  __float128 pi = Sleef_getq2_sse2(q3, 0) - Sleef_getq2_sse2(q3, 1);

  Sleef_printf("%.40Pg\n", &pi);
}
