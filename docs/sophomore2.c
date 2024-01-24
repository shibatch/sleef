#include <stdio.h>
#include <math.h>

#define N 65536
#define M (N + 3)

static double func(double x) { return pow(x, -x); }

double int_simpson(double a, double b) {
   double h = (b - a) / M;
   double sum_odd = 0.0, sum_even = 0.0;
   for(int i = 1;i <= M-3;i += 2) {
     sum_odd  += func(a + h * i);
     sum_even += func(a + h * (i + 1));
   }
   return h / 3 * (func(a) + 4 * sum_odd + 2 * sum_even + 4 * func(b - h) + func(b));
}

int main() {
  double sum = 0;
  for(int i=1;i<N;i++) sum += pow(i, -i);
  printf("%g %g\n", int_simpson(0, 1), sum);
}
