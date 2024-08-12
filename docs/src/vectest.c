#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 256
__attribute__ ((__aligned__(16))) double a[N];

int main(int argc, char **argv) {
  srand(time(NULL));
  for(int i = 0;i < N;i++) a[i] = rand();
  for(int i = 0;i < N;i++) a[i] = sin(a[i]);
  for(int i = 0;i < N;i++) printf("%g\n", a[i]);
}
