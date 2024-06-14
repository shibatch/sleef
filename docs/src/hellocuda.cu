#include <iostream>
#include <math.h>

#include "sleefinline_cuda.h"

// Based on the tutorial code at https://developer.nvidia.com/blog/even-easier-introduction-cuda/

__global__ void pow_gpu(int n, double *r, double *x, double *y)
{
  int index = threadIdx.x, stride = blockDim.x;

  for (int i = index; i < n; i += stride)
    r[i] = Sleef_powd1_u10cuda(x[i], y[i]);
}

int main(void)
{
  int N = 1 << 20;

  double *r, *x, *y;
  cudaMallocManaged(&r, N*sizeof(double));
  cudaMallocManaged(&x, N*sizeof(double));
  cudaMallocManaged(&y, N*sizeof(double));

  for (int i = 0; i < N; i++) {
    r[i] = 0.0;
    x[i] = 1.00001;
    y[i] = i;
  }

  pow_gpu<<<1, 256>>>(N, r, x, y);

  cudaDeviceSynchronize();

  double maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(r[i]-pow(x[i], y[i])));
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(y);
  cudaFree(x);
  cudaFree(r);
  
  return 0;
}
