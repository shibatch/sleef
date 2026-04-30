#include <stdio.h>
#include <riscv_vector.h>
#include <sleef.h>

int main(int argc, char **argv) {
  double a[] = {2, 10};
  double b[] = {3, 20};

  size_t vl = __riscv_vsetvl_e64m1(2);

  vfloat64m1_t va, vb, vc;

  va = __riscv_vle64_v_f64m1(a, vl);
  vb = __riscv_vle64_v_f64m1(b, vl);

  vc = Sleef_powdx_u10rvvm1(va, vb);

  double c[2];

  __riscv_vse64_v_f64m1(c, vc, vl);

  printf("pow(%g, %g) = %g\n", a[0], b[0], c[0]);
  printf("pow(%g, %g) = %g\n", a[1], b[1], c[1]);
}
