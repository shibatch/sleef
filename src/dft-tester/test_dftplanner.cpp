//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "sleef.h"
#include "sleefdft.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc == 1) {
    fprintf(stderr, "%s <plan file name>\n", argv[0]);
    exit(-1);
  }

  SleefDFT_setPlanFilePath(argv[1], NULL, SLEEF_PLAN_AUTOMATIC);

  double *din  = (double *)Sleef_malloc(2048*64*2 * sizeof(double));
  double *dout = (double *)Sleef_malloc(2048*64*2 * sizeof(double));

  float *fin  = (float *)Sleef_malloc(2048*64*2 * sizeof(double));
  float *fout = (float *)Sleef_malloc(2048*64*2 * sizeof(double));

#ifdef MEASURE
#ifdef MULTITHREAD
  int mode = SLEEF_MODE_MEASURE | SLEEF_MODE_VERBOSE;
#else
  int mode = SLEEF_MODE_MEASURE | SLEEF_MODE_VERBOSE | SLEEF_MODE_NO_MT;
#endif
#else
#ifdef MULTITHREAD
  int mode = SLEEF_MODE_ESTIMATE | SLEEF_MODE_VERBOSE;
#else
  int mode = SLEEF_MODE_ESTIMATE | SLEEF_MODE_VERBOSE | SLEEF_MODE_NO_MT;
#endif
#endif

  SleefDFT *p;

  //

  p = SleefDFT_double_init1d(1024, din, dout, mode);

  vector<char> pathd1024(1024);
  SleefDFT_getPath(p, pathd1024.data(), pathd1024.size());

  cout << "Path (D1024) : " << pathd1024.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_double_init1d(512, din, dout, mode);

  vector<char> pathd512(1024);
  SleefDFT_getPath(p, pathd512.data(), pathd512.size());

  cout << "Path (D512) : " << pathd512.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_float_init1d(1024, fin, fout, mode);

  vector<char> pathf1024(1024);
  SleefDFT_getPath(p, pathf1024.data(), pathf1024.size());

  cout << "Path (F1024) : " << pathf1024.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_float_init1d(512, fin, fout, mode);

  vector<char> pathf512(1024);
  SleefDFT_getPath(p, pathf512.data(), pathf512.size());

  cout << "Path (F512) : " << pathf512.data() << endl;

  SleefDFT_dispose(p);


  //


  p = SleefDFT_double_init2d(2048, 64, din, dout, mode);

  vector<char> pathd128x128(1024);
  SleefDFT_getPath(p, pathd128x128.data(), pathd128x128.size());

  cout << "Path (D128x128) : " << pathd128x128.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_double_init2d(64, 64, din, dout, mode);

  vector<char> pathd64x64(1024);
  SleefDFT_getPath(p, pathd64x64.data(), pathd64x64.size());

  cout << "Path (D64x64) : " << pathd64x64.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_float_init2d(2048, 64, fin, fout, mode);

  vector<char> pathf128x128(1024);
  SleefDFT_getPath(p, pathf128x128.data(), pathf128x128.size());

  cout << "Path (F128x128) : " << pathf128x128.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_float_init2d(64, 64, fin, fout, mode);

  vector<char> pathf64x64(1024);
  SleefDFT_getPath(p, pathf64x64.data(), pathf64x64.size());

  cout << "Path (F64x64) : " << pathf64x64.data() << endl;

  SleefDFT_dispose(p);


  //

  cout << endl;

  p = SleefDFT_double_init1d(1024, din, dout, mode);

  vector<char> pathd1024_2(1024);
  SleefDFT_getPath(p, pathd1024_2.data(), pathd1024_2.size());

  cout << "Path2 (D1024) : " << pathd1024_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathd1024.size() != pathd1024_2.size() || memcmp(pathd1024.data(), pathd1024_2.data(), pathd1024.size()) != 0) {
    cerr << "Paths do not match" << endl;
    cerr << pathd1024.data() << endl;
    cerr << pathd1024_2.data() << endl;
    exit(-1);
  }

  //

  p = SleefDFT_double_init1d(512, din, dout, mode);

  vector<char> pathd512_2(1024);
  SleefDFT_getPath(p, pathd512_2.data(), pathd512_2.size());

  cout << "Path2 (D512) : " << pathd512_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathd512.size() != pathd512_2.size() || memcmp(pathd512.data(), pathd512_2.data(), pathd512.size()) != 0) {
    cerr << "Paths do not match" << endl;
    cerr << pathd512.data() << endl;
    cerr << pathd512_2.data() << endl;
    exit(-1);
  }

  //

  p = SleefDFT_float_init1d(1024, fin, fout, mode);

  vector<char> pathf1024_2(1024);
  SleefDFT_getPath(p, pathf1024_2.data(), pathf1024_2.size());

  cout << "Path2 (F1024) : " << pathf1024_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathf1024.size() != pathf1024_2.size() || memcmp(pathf1024.data(), pathf1024_2.data(), pathf1024.size()) != 0) {
    cerr << "Paths do not match" << endl;
    cerr << pathf1024.data() << endl;
    cerr << pathf1024_2.data() << endl;
    exit(-1);
  }

  //

  p = SleefDFT_float_init1d(512, fin, fout, mode);

  vector<char> pathf512_2(1024);
  SleefDFT_getPath(p, pathf512_2.data(), pathf512_2.size());

  cout << "Path2 (F512) : " << pathf512_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathf512.size() != pathf512_2.size() || memcmp(pathf512.data(), pathf512_2.data(), pathf512.size()) != 0) {
    cerr << "Paths do not match" << endl;
    cerr << pathf512.data() << endl;
    cerr << pathf512_2.data() << endl;
    exit(-1);
  }

  //


  p = SleefDFT_double_init2d(2048, 64, din, dout, mode);

  vector<char> pathd128x128_2(1024);
  SleefDFT_getPath(p, pathd128x128_2.data(), pathd128x128_2.size());

  cout << "Path2 (D128x128) : " << pathd128x128_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathd128x128.size() != pathd128x128_2.size() || memcmp(pathd128x128.data(), pathd128x128_2.data(), pathd128x128.size()) != 0) {
    cerr << "Paths do not match" << endl;
    cerr << pathd128x128.data() << endl;
    cerr << pathd128x128_2.data() << endl;
    exit(-1);
  }

  //

  p = SleefDFT_double_init2d(64, 64, din, dout, mode);

  vector<char> pathd64x64_2(1024);
  SleefDFT_getPath(p, pathd64x64_2.data(), pathd64x64_2.size());

  cout << "Path2 (D64x64) : " << pathd64x64_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathd64x64.size() != pathd64x64_2.size() || memcmp(pathd64x64.data(), pathd64x64_2.data(), pathd64x64.size()) != 0) {
    cerr << "Paths do not match" << endl;
    cerr << pathd64x64.data() << endl;
    cerr << pathd64x64_2.data() << endl;
    exit(-1);
  }

  //

  p = SleefDFT_float_init2d(2048, 64, fin, fout, mode);

  vector<char> pathf128x128_2(1024);
  SleefDFT_getPath(p, pathf128x128_2.data(), pathf128x128_2.size());

  cout << "Path2 (F128x128) : " << pathf128x128_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathf128x128.size() != pathf128x128_2.size() || memcmp(pathf128x128.data(), pathf128x128_2.data(), pathf128x128.size()) != 0) {
    cerr << "Paths do not match" << endl;
    cerr << pathf128x128.data() << endl;
    cerr << pathf128x128_2.data() << endl;
    exit(-1);
  }

  //

  p = SleefDFT_float_init2d(64, 64, fin, fout, mode);

  vector<char> pathf64x64_2(1024);
  SleefDFT_getPath(p, pathf64x64_2.data(), pathf64x64_2.size());

  cout << "Path2 (F64x64) : " << pathf64x64_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathf64x64.size() != pathf64x64_2.size() || memcmp(pathf64x64.data(), pathf64x64_2.data(), pathf64x64.size()) != 0) {
    cerr << "Paths do not match" << endl;
    cerr << pathf64x64.data() << endl;
    cerr << pathf64x64_2.data() << endl;
    exit(-1);
  }

  //

  Sleef_free(din);
  Sleef_free(dout);
  Sleef_free(fin);
  Sleef_free(fout);

  cerr << "OK" << endl;

  exit(0);
}
