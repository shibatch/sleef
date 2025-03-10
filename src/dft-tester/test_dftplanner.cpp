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

  double *din  = (double *)Sleef_malloc(1024*2 * sizeof(double));
  double *dout = (double *)Sleef_malloc(1024*2 * sizeof(double));

  float *fin  = (float *)Sleef_malloc(1024*2 * sizeof(double));
  float *fout = (float *)Sleef_malloc(1024*2 * sizeof(double));

#ifdef MULTITHREAD
  int mode = SLEEF_MODE_MEASURE | SLEEF_MODE_VERBOSE;
#else
  int mode = SLEEF_MODE_MEASURE | SLEEF_MODE_VERBOSE | SLEEF_MODE_NO_MT;
#endif

  SleefDFT *p;

  //

  p = SleefDFT_double_init1d(1024, din, dout, mode);

  vector<char> pathd1024(1024);
  SleefDFT_getPath(p, pathd1024.data(), pathd1024.size());

  cout << "Path (1024) : " << pathd1024.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_double_init1d(512, din, dout, mode);

  vector<char> pathd512(512);
  SleefDFT_getPath(p, pathd512.data(), pathd512.size());

  cout << "Path (512) : " << pathd512.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_float_init1d(1024, fin, fout, mode);

  vector<char> pathf1024(1024);
  SleefDFT_getPath(p, pathf1024.data(), pathf1024.size());

  cout << "Path (1024) : " << pathf1024.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_float_init1d(512, fin, fout, mode);

  vector<char> pathf512(512);
  SleefDFT_getPath(p, pathf512.data(), pathf512.size());

  cout << "Path (512) : " << pathf512.data() << endl;

  SleefDFT_dispose(p);

  //

  p = SleefDFT_double_init1d(1024, din, dout, mode);

  vector<char> pathd1024_2(1024);
  SleefDFT_getPath(p, pathd1024_2.data(), pathd1024_2.size());

  cout << "Path2 (1024) : " << pathd1024_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathd1024.size() != pathd1024_2.size() || memcmp(pathd1024.data(), pathd1024_2.data(), pathd1024.size()) != 0) {
    cerr << "Paths do not match" << endl;
    exit(-1);
  }

  //

  p = SleefDFT_double_init1d(512, din, dout, mode);

  vector<char> pathd512_2(512);
  SleefDFT_getPath(p, pathd512_2.data(), pathd512_2.size());

  cout << "Path2 (512) : " << pathd512_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathd512.size() != pathd512_2.size() || memcmp(pathd512.data(), pathd512_2.data(), pathd512.size()) != 0) {
    cerr << "Paths do not match" << endl;
    exit(-1);
  }

  //

  p = SleefDFT_float_init1d(1024, fin, fout, mode);

  vector<char> pathf1024_2(1024);
  SleefDFT_getPath(p, pathf1024_2.data(), pathf1024_2.size());

  cout << "Path2 (1024) : " << pathf1024_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathf1024.size() != pathf1024_2.size() || memcmp(pathf1024.data(), pathf1024_2.data(), pathf1024.size()) != 0) {
    cerr << "Paths do not match" << endl;
    exit(-1);
  }

  //

  p = SleefDFT_float_init1d(512, fin, fout, mode);

  vector<char> pathf512_2(512);
  SleefDFT_getPath(p, pathf512_2.data(), pathf512_2.size());

  cout << "Path2 (512) : " << pathf512_2.data() << endl;

  SleefDFT_dispose(p);

  if (pathf512.size() != pathf512_2.size() || memcmp(pathf512.data(), pathf512_2.data(), pathf512.size()) != 0) {
    cerr << "Paths do not match" << endl;
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
