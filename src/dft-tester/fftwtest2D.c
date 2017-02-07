//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define _BSD_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <complex.h>

#include <unistd.h>
#include <sys/time.h>

#include <fftw3.h>

#include "sleef.h"
#include "sleefdft.h"

#define THRES 1e-2

typedef double real;

// complex forward
int check_cf(int n, int m) {
  int i;

  //

  fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * m);
  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * m);

  fftw_plan w = fftw_plan_dft_2d(n, m, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  //
  
  real *sx = (real *)Sleef_malloc(n*m*2*sizeof(real));
  real *sy = (real *)Sleef_malloc(n*m*2*sizeof(real));

  struct SleefDFT *p = SleefDFT_double_init2d(n, m, sx, sy, SLEEF_MODE_ESTIMATE);

  //
  
  for(i=0;i<n*m;i++) {
    double re = (2.0 * random() - 1) / (double)RAND_MAX;
    double im = (2.0 * random() - 1) / (double)RAND_MAX;
    sx[(i*2+0)] = re;
    sx[(i*2+1)] = im;
    in[i] = re + im * _Complex_I;
  }

  //

  SleefDFT_double_execute(p, NULL, NULL);

  fftw_execute(w);

  //

  int success = 1;

  for(i=0;i<n*m;i++) {
    if (fabs(sy[(i*2+0)] - creal(out[i])) > THRES) success = 0;
    if (fabs(sy[(i*2+1)] - cimag(out[i])) > THRES) success = 0;
  }

  //

  fftw_destroy_plan(w);
  fftw_free(in);
  fftw_free(out);

  Sleef_free(sx);
  Sleef_free(sy);

  SleefDFT_dispose(p);

  //

  return success;
}

// complex backward
int check_cb(int n, int m) {
  int i;

  //

  fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * m);
  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * m);

  fftw_plan w = fftw_plan_dft_2d(n, m, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

  //
  
  real *sx = (real *)Sleef_malloc(n*m*2*sizeof(real));
  real *sy = (real *)Sleef_malloc(n*m*2*sizeof(real));

  struct SleefDFT *p = SleefDFT_double_init2d(n, m, sx, sy, SLEEF_MODE_BACKWARD | SLEEF_MODE_ESTIMATE);

  //
  
  for(i=0;i<n*m;i++) {
    double re = (2.0 * random() - 1) / (double)RAND_MAX;
    double im = (2.0 * random() - 1) / (double)RAND_MAX;
    sx[(i*2+0)] = re;
    sx[(i*2+1)] = im;
    in[i] = re + im * _Complex_I;
  }

  //

  SleefDFT_double_execute(p, NULL, NULL);

  fftw_execute(w);

  //

  int success = 1;

  for(i=0;i<n*m;i++) {
    if (fabs(sy[(i*2+0)] - creal(out[i])) > THRES) success = 0;
    if (fabs(sy[(i*2+1)] - cimag(out[i])) > THRES) success = 0;
  }

  //

  fftw_destroy_plan(w);
  fftw_free(in);
  fftw_free(out);

  Sleef_free(sx);
  Sleef_free(sy);

  SleefDFT_dispose(p);

  //

  return success;
}


int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "%s <log2n> <log2m>\n", argv[0]);
    exit(-1);
  }

  const int n = 1 << atoi(argv[1]);
  const int m = 1 << atoi(argv[2]);

  srand(time(NULL));

  //

  int cf_ok = check_cf(n, m);
  printf("complex forward   : %s\n", cf_ok ? "OK" : "NG");
  int cb_ok = check_cb(n, m);
  printf("complex backward  : %s\n", cb_ok ? "OK" : "NG");

  if (!cf_ok || !cb_ok) exit(-1);

  exit(0);
}
