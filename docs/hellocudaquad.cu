#include <iostream>
#include <quadmath.h>

#include "sleefquadinline_cuda.h"

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

  __float128 *r = (__float128 *)rd, *x = (__float128 *)xd, *y = (__float128 *)yd;

  for (int i = 0; i < N; i++) {
    r[i] = 0.0;
    x[i] = 1.00001Q;
    y[i] = i;
  }

  pow_gpu<<<1, 256>>>(N, rd, xd, yd);

  cudaDeviceSynchronize();

  double maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabsq(r[i]-powq(x[i], y[i])));
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(yd);
  cudaFree(xd);
  cudaFree(rd);
  
  return 0;
}
