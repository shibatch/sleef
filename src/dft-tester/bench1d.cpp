//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <complex.h>
#include <ctime>
#include <chrono>

#ifdef USEFFTW
#include <fftw3.h>
#include <omp.h>
#else
#include "sleef.h"
#include "sleefdft.h"
#endif

typedef double real;

static uint64_t gettime() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>
    (std::chrono::system_clock::now() - std::chrono::system_clock::from_time_t(0)).count();
}

#define REPEAT 8

int main(int argc, char **argv) {
  if (argc == 1) {
#ifdef USEFFTW
    fprintf(stderr, "%s <log2n>\n", argv[0]);
#else
    fprintf(stderr, "%s <log2n> <plan file name> <plan string>\n", argv[0]);
#endif
    exit(-1);
  }

  int backward = 0;

  int log2n = atoi(argv[1]);
  if (log2n < 0) {
    backward = 1;
    log2n = -log2n;
  }

  const int n = 1 << log2n;
  const int64_t niter = (int64_t)(100000000000.0 / n / log2n);

  printf("Number of iterations = %lld\n", (long long int)niter);

#ifdef USEFFTW
  fftw_complex *in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

#ifdef MULTITHREAD
  int fftw_init_threads(void);
  fftw_plan_with_nthreads(omp_get_max_threads());
#endif
  
  fftw_plan w = fftw_plan_dft_1d(n, in, out, backward ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_MEASURE);
  //fftw_plan w = fftw_plan_dft_1d(n, in, out, backward ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_PATIENT);

  for(int i=0;i<n;i++) {
    auto a = (2.0 * (rand() / (double)RAND_MAX) - 1) + (2.0 * (rand() / (double)RAND_MAX) - 1) * _Complex_I;
    in[i][0] = creal(a);
    in[i][1] = cimag(a);
  }

  for(int64_t i=0;i<niter/2;i++) fftw_execute(w);
#else
  const char *planfn = (argc < 3 || strcmp(argv[2], "-") == 0) ? NULL : argv[2];

  SleefDFT_setPlanFilePath(planfn, NULL, SLEEF_PLAN_AUTOMATIC);

  real *in  = (real *)Sleef_malloc(n*2 * sizeof(real));
  real *out = (real *)Sleef_malloc(n*2 * sizeof(real));

#ifdef MULTITHREAD
  int mode = SLEEF_MODE_MEASURE | SLEEF_MODE_VERBOSE;
#else
  int mode = SLEEF_MODE_MEASURE | SLEEF_MODE_VERBOSE | SLEEF_MODE_NO_MT;
#endif
  if (argc >= 4) mode = SLEEF_MODE_VERBOSE | SLEEF_MODE_ESTIMATE;

  if (backward) mode |= SLEEF_MODE_BACKWARD;
  struct SleefDFT *p = SleefDFT_double_init1d(n, in, out, mode);

  if (argc >= 4) SleefDFT_setPath(p, argv[3]);

  for(int i=0;i<n*2;i++) {
    in[i] = (2.0 * (rand() / (double)RAND_MAX) - 1);
  }

  for(int64_t i=0;i<niter/2;i++) SleefDFT_double_execute(p, in, out);
#endif

  for(int rep=0;rep<REPEAT;rep++) {
    uint64_t tm0 = gettime();
    for(int64_t i=0;i<niter;i++) {
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
  }

  //

  exit(0);
}
