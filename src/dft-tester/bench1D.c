//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define _BSD_SOURCE
#define _XOPEN_SOURCE 700

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#ifdef USEFFTW
#include <fftw3.h>
#else
#include "sleef.h"
#include "sleefdft.h"
#endif

typedef double real;

static uint64_t gettime() {
  struct timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);
  return (uint64_t)tp.tv_sec * 1000000000 + ((uint64_t)tp.tv_nsec);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    fprintf(stderr, "%s <log2n>\n", argv[0]);
    exit(-1);
  }

  int backward = 0;

  int log2n = atoi(argv[1]);
  if (log2n < 0) {
    backward = 1;
    log2n = -log2n;
  }

  const int n = 1 << log2n;
  const int niter = (int)(30000000000.0 / n / log2n);
  //const int niter = (int)(30000000.0 / n / log2n);

  printf("Number of iterations = %d\n", niter);

#ifdef USEFFTW
  fftw_complex *in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

#if 1
  int fftw_init_threads(void);
  fftw_plan_with_nthreads(8);
#endif
  
  fftw_plan w = fftw_plan_dft_1d(n, in, out, backward ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_MEASURE);
  //fftw_plan w = fftw_plan_dft_1d(n, in, out, backward ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_PATIENT);

  for(int i=0;i<n;i++) {
    in[i] = (2.0 * (rand() / (double)RAND_MAX) - 1) + (2.0 * (rand() / (double)RAND_MAX) - 1) * _Complex_I;
  }

  for(int i=0;i<niter/2;i++) fftw_execute(w);
#else
  real *in  = (real *)Sleef_malloc(n*2 * sizeof(real));
  real *out = (real *)Sleef_malloc(n*2 * sizeof(real));

  int mode = SLEEF_MODE_MEASURE;
  if (argc >= 3) mode = SLEEF_MODE_VERBOSE | SLEEF_MODE_ESTIMATE;

  if (backward) mode |= SLEEF_MODE_BACKWARD;
  struct SleefDFT *p = SleefDFT_double_init1d(n, NULL, NULL, mode);

  if (argc >= 3) SleefDFT_setPath(p, argv[2]);

  for(int i=0;i<n*2;i++) {
    in[i] = (2.0 * (rand() / (double)RAND_MAX) - 1);
  }

  for(int i=0;i<niter/2;i++) SleefDFT_double_execute(p, in, out);
#endif

  uint64_t tm0 = gettime();
  for(int i=0;i<niter;i++) {
#ifdef USEFFTW
    fftw_execute(w);
#else
    SleefDFT_double_execute(p, in, out);
#endif
  }
  uint64_t tm1 = gettime();

  printf("Actual    time = %g ns\n", (double)(tm1 - tm0) / niter);
  double timeus = (tm1 - tm0) / ((double)niter * 1000);

  double mflops = 5 * n * log2n / timeus;

  printf("%g Mflops\n", mflops);

  //

  exit(0);
}
