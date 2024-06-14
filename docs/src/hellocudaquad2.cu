// nvcc -O3 hellocudaquad2.cu -I./include --fmad=false -Xcompiler -ffp-contract=off

#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <ctype.h>
#include <assert.h>
#include <emmintrin.h>

#include "sleefquadinline_sse2.h"
#include "sleefquadinline_purec_scalar.h"
#include "sleefquadinline_cuda.h"
#include "sleefinline_sse2.h"

// Based on the tutorial code at https://developer.nvidia.com/blog/even-easier-introduction-cuda/

__global__ void pow_gpu(int n, Sleef_quadx1 *r, Sleef_quadx1 *x, Sleef_quadx1 *y) {
  int index = threadIdx.x, stride = blockDim.x;

  for (int i = index; i < n; i += stride)
    r[i] = Sleef_powq1_u10cuda(x[i], y[i]);
}

int main(void) {
  int N = 1 << 20;

  Sleef_quadx1 *rd, *xd, *yd;
  cudaMallocManaged(&rd, N*sizeof(Sleef_quadx1));
  cudaMallocManaged(&xd, N*sizeof(Sleef_quadx1));
  cudaMallocManaged(&yd, N*sizeof(Sleef_quadx1));

  Sleef_quad *r = (Sleef_quad *)rd, *x = (Sleef_quad *)xd, *y = (Sleef_quad *)yd;

  //

  for (int i = 0; i < N; i++) {
    r[i] = Sleef_cast_from_doubleq1_purec(0);
    x[i] = Sleef_cast_from_doubleq1_purec(1.00001);
    y[i] = Sleef_cast_from_doubleq1_purec(i);
  }

  pow_gpu<<<1, 256>>>(N, rd, xd, yd);

  cudaDeviceSynchronize();

  Sleef_quadx2 maxError = Sleef_splatq2_sse2(Sleef_strtoq("0.0", NULL));

  for (int i = 0; i < N; i += 2) {
    Sleef_quadx2 r2 = Sleef_loadq2_sse2(&r[i]);
    Sleef_quadx2 x2 = Sleef_loadq2_sse2(&x[i]);
    Sleef_quadx2 y2 = Sleef_loadq2_sse2(&y[i]);

    Sleef_quadx2 q = Sleef_fabsq2_sse2(Sleef_subq2_u05sse2(r2, Sleef_powq2_u10sse2(x2, y2)));
    maxError = Sleef_fmaxq2_sse2(maxError, q);
  }
  
  Sleef_printf("Max error: %Qg\n",
	       Sleef_fmaxq1_purec(Sleef_getq2_sse2(maxError, 0), Sleef_getq2_sse2(maxError, 1)));

  //

  cudaFree(yd);
  cudaFree(xd);
  cudaFree(rd);
  
  return 0;
}
