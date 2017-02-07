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

#define THRES 1e-4

// complex forward
int check_cf(int n) {
  int i;

  //

  fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

  fftw_plan w = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);


  //
  
  double *sx = (double *)Sleef_malloc(n*2*sizeof(double));
  double *sy = (double *)Sleef_malloc(n*2*sizeof(double));

  struct SleefDFT *p = SleefDFT_double_init1d(n, sx, sy, SLEEF_MODE_ESTIMATE);

  //
  
  for(i=0;i<n;i++) {
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

  for(i=0;i<n;i++) {
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
int check_cb(int n) {
  int i;

  //

  fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

  fftw_plan w = fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

  //
  
  double *sx = (double *)Sleef_malloc(n*2*sizeof(double));
  double *sy = (double *)Sleef_malloc(n*2*sizeof(double));

  struct SleefDFT *p = SleefDFT_double_init1d(n, sx, sy, SLEEF_MODE_BACKWARD | SLEEF_MODE_ESTIMATE);

  //
  
  for(i=0;i<n;i++) {
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

  for(i=0;i<n;i++) {
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

// real forward
int check_rf(int n) {
  int i;

  //

  double       *in  = (double *)      fftw_malloc(sizeof(double) * n);
  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (n/2+1));

  fftw_plan w = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);

  //
  
  double *sx = (double *)Sleef_malloc(n*sizeof(double));
  double *sy = (double *)Sleef_malloc((n/2+1)*sizeof(double)*2);

  struct SleefDFT *p = SleefDFT_double_init1d(n, sx, sy, SLEEF_MODE_REAL | SLEEF_MODE_ESTIMATE);

  //
  
  for(i=0;i<n;i++) {
    double re = (2.0 * random() - 1) / (double)RAND_MAX;
    sx[i] = re;
    in[i] = re;
  }

  //

  SleefDFT_double_execute(p, NULL, NULL);

  fftw_execute(w);

  //

  int success = 1;

  for(i=0;i<n/2+1;i++) {
    if (fabs(sy[(2*i+0)] - creal(out[i])) > THRES) success = 0;
    if (fabs(sy[(2*i+1)] - cimag(out[i])) > THRES) success = 0;
    //printf("%d : (%g %g) (%g %g)\n", i, sy[(2*i+0)], sy[(2*i+1)], creal(out[i]), cimag(out[i]));
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

// real backward
int check_rb(int n) {
  int i;

  //

  fftw_complex *in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (n/2+1));
  double       *out = (double *)      fftw_malloc(sizeof(double) * n);

  fftw_plan w = fftw_plan_dft_c2r_1d(n, in, out, FFTW_ESTIMATE);

  //
  
  double *sx = (double *)Sleef_malloc((n/2+1) * sizeof(double)*2);
  double *sy = (double *)Sleef_malloc(sizeof(double)*n);

  struct SleefDFT *p = SleefDFT_double_init1d(n, sx, sy, SLEEF_MODE_REAL | SLEEF_MODE_BACKWARD | SLEEF_MODE_ESTIMATE);

  //
  
  for(i=0;i<n/2;i++) {
    if (i == 0) {
      in[0  ] = (2.0 * (rand() / (double)RAND_MAX) - 1);
      in[n/2] = (2.0 * (rand() / (double)RAND_MAX) - 1);
    } else {
      in[i  ] = (2.0 * (rand() / (double)RAND_MAX) - 1) + (2.0 * (rand() / (double)RAND_MAX) - 1) * _Complex_I;
    }
  }


  for(i=0;i<n/2+1;i++) {
    sx[2*i+0] = creal(in[i]);
    sx[2*i+1] = cimag(in[i]);
  }
  
  //

  SleefDFT_double_execute(p, NULL, NULL);

  fftw_execute(w);

  //

  int success = 1;

  for(i=0;i<n;i++) {
    if ((fabs(sy[i] - out[i]) > THRES)) success = 0;
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
  if (argc != 2) {
    fprintf(stderr, "%s <log2n>\n", argv[0]);
    exit(-1);
  }

  const int n = 1 << atoi(argv[1]);

  srand(time(NULL));

  //

  printf("complex forward   : %s\n", check_cf(n) ? "OK" : "NG");
  printf("complex backward  : %s\n", check_cb(n) ? "OK" : "NG");
  printf("real    forward   : %s\n", check_rf(n) ? "OK" : "NG");
  printf("real    backward  : %s\n", check_rb(n) ? "OK" : "NG");

  exit(0);
}
