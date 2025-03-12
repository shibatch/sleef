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
#include <iostream>
#include <complex>
#include <ctime>
#include <chrono>
#include <thread>
#include <memory>
#include <vector>

#include <fftw3.h>
#include <omp.h>

#include "sleef.h"
#include "sleefdft.h"

using namespace std;

#if BASETYPEID == 1
typedef double xreal;
#define FFTW_COMPLEX fftw_complex
#define FFTW_PLAN_WITH_NTHREADS fftw_plan_with_nthreads
#define FFTW_PLAN fftw_plan
#define FFTW_MALLOC fftw_malloc
#define FFTW_FREE fftw_free
#define FFTW_PLAN_DFT_1D fftw_plan_dft_1d
#define FFTW_EXECUTE fftw_execute
#define FFTW_DESTROY_PLAN fftw_destroy_plan
#define SLEEFDFT_INIT1D SleefDFT_double_init1d
#elif BASETYPEID == 2
typedef float xreal;
#define FFTW_COMPLEX fftwf_complex
#define FFTW_PLAN_WITH_NTHREADS fftwf_plan_with_nthreads
#define FFTW_PLAN fftwf_plan
#define FFTW_MALLOC fftwf_malloc
#define FFTW_FREE fftwf_free
#define FFTW_PLAN_DFT_1D fftwf_plan_dft_1d
#define FFTW_EXECUTE fftwf_execute
#define FFTW_DESTROY_PLAN fftwf_destroy_plan
#define SLEEFDFT_INIT1D SleefDFT_float_init1d
#else
#error BASETYPEID not set
#endif

static uint64_t timens() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>
    (std::chrono::system_clock::now() - std::chrono::system_clock::from_time_t(0)).count();
}

template<typename cplx>
class FFTFramework {
public:
  virtual void execute() = 0;
  virtual cplx* getInPtr() = 0;
  virtual cplx* getOutPtr() = 0;
  virtual ~FFTFramework() {};

  int64_t niter(int64_t ns) {
    int64_t niter = 10, t0, t1;

    for(;;) {
      t0 = timens();
      for(int64_t i=0;i<niter;i++) execute();
      t1 = timens();
      if (t1 - t0 > 1000LL * 1000 * 10) break;
      niter *= 2;
    }

    return (double)niter * ns / (t1 - t0);
  }
};

template<typename cplx>
class FWSleefDFT : public FFTFramework<cplx> {
  const int n;
  cplx* in;
  cplx* out;
  SleefDFT *plan;

public:
  FWSleefDFT(int n_, bool forward, bool mt=false) : n(n_) {
    SleefDFT_setDefaultVerboseFP(stderr);
    SleefDFT_setPlanFilePath(NULL, NULL, SLEEF_PLAN_RESET);
    in  = (cplx*)Sleef_malloc(sizeof(cplx) * n);
    out = (cplx*)Sleef_malloc(sizeof(cplx) * n);
    plan = SLEEFDFT_INIT1D(n, (xreal*)in, (xreal*)out,
      (forward ? SLEEF_MODE_FORWARD : SLEEF_MODE_BACKWARD) | SLEEF_MODE_MEASURE /* | SLEEF_MODE_VERBOSE*/ | (mt ? 0 : SLEEF_MODE_NO_MT));
    vector<char> pathstr(1024);
    SleefDFT_getPath(plan, pathstr.data(), pathstr.size());
    cerr << pathstr.data() << endl;
  }

  ~FWSleefDFT() { SleefDFT_dispose(plan); }

  cplx* getInPtr () { return in ; }
  cplx* getOutPtr() { return out; }

  void execute() { SleefDFT_execute(plan, NULL, NULL); }
};

template<typename cplx>
class FWFFTW3 : public FFTFramework<cplx> {
  const int n;
  cplx* in;
  cplx* out;
  FFTW_PLAN plan;

public:
  FWFFTW3(int n_, bool forward, bool mt=false) : n(n_) {
    FFTW_PLAN_WITH_NTHREADS(mt ? omp_get_max_threads() : 1);
    in  = (cplx*)FFTW_MALLOC(sizeof(FFTW_COMPLEX) * n);
    out = (cplx*)FFTW_MALLOC(sizeof(FFTW_COMPLEX) * n);
    plan = FFTW_PLAN_DFT_1D(n, (FFTW_COMPLEX*)in, (FFTW_COMPLEX*)out,
			    forward ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_MEASURE);
  }

  ~FWFFTW3() {
    FFTW_DESTROY_PLAN(plan);
    FFTW_FREE(out);
    FFTW_FREE(in);
  }

  cplx* getInPtr() { return in; }
  cplx* getOutPtr() { return out; }

  void execute() { FFTW_EXECUTE(plan); }
};

int main(int argc, char **argv) {
  if (argc == 1) {
    fprintf(stderr, "%s <log2n> <plan file name> <plan string>\n", argv[0]);
    exit(-1);
  }

  fftw_init_threads();

  double measureTimeMillis = 3000;

  int backward = 0;

  int log2n = atoi(argv[1]);
  if (log2n < 0) {
    backward = 1;
    log2n = -log2n;
  }

  const int n = 1 << log2n;
  const int nrepeat = 8;

  vector<double> mflops_sleefdftst, mflops_fftwst, mflops_sleefdftmt, mflops_fftwmt;

#ifdef CHECK
  {
    // Test if we are really computing the same values

    auto sleefdft = make_shared<FWSleefDFT<complex<xreal>>>(n, true);
    auto fftw     = make_shared<FWFFTW3   <complex<xreal>>>(n, true);

    complex<xreal> *in0  = sleefdft->getInPtr();
    complex<xreal> *out0 = sleefdft->getOutPtr();
    complex<xreal> *in1  = fftw->getInPtr();
    complex<xreal> *out1 = fftw->getOutPtr();

    for(int i=0;i<n;i++) {
      in0[i] = in1[i] = (2.0 * (rand() / (double)RAND_MAX) - 1) + (2.0 * (rand() / (double)RAND_MAX) - 1) * 1i;
    }

    sleefdft->execute();
    fftw    ->execute();

    for(int i=0;i<n;i++) {
      if (std::real(abs((out0[i] - out1[i]) * (out0[i] - out1[i]))) > 0.1) {
	cerr << "NG " << i << " : " << out0[i] << ", " << out1[i] << endl;
	exit(-1);
      }
    }
  }
#endif

  for(int nr = 0;nr < nrepeat;nr++) {
    cerr << endl << "n = " << n << "(" << log2n << "), nr = " << nr << endl;

    cerr << "Planning SleefDFT ST" << endl;
    auto sleefdftst = make_shared<FWSleefDFT<complex<xreal>>>(n, true, false);
    cerr << "Planning FFTW ST" << endl;
    auto fftwst     = make_shared<FWFFTW3   <complex<xreal>>>(n, true, false);
    cerr << "Planning SleefDFT MT" << endl;
    auto sleefdftmt = make_shared<FWSleefDFT<complex<xreal>>>(n, true, true);
    cerr << "Planning FFTW MT" << endl;
    auto fftwmt     = make_shared<FWFFTW3   <complex<xreal>>>(n, true, true);
    cerr << "Planning done" << endl;

    complex<xreal> *in0  = sleefdftst->getInPtr();
    complex<xreal> *in1  = fftwst->getInPtr();

    for(int i=0;i<n;i++) {
      in0[i] = in1[i] = (2.0 * (rand() / (double)RAND_MAX) - 1) + (2.0 * (rand() / (double)RAND_MAX) - 1) * 1i;
    }

    //

    {
      auto niter = sleefdftst->niter(1000LL * 1000 * measureTimeMillis);

      cerr << "SleefDFT ST niter = " << niter << endl;

      for(int64_t i=0;i<niter/10;i++) sleefdftst->execute(); // warm up

      int64_t tm0 = timens();
      for(int64_t i=0;i<niter;i++) sleefdftst->execute();
      int64_t tm1 = timens();

      double mflops = 5 * n * log2n / ((tm1 - tm0) / (double(niter)*1000));

      fprintf(stderr, "%g Mflops\n", mflops);

      mflops_sleefdftst.push_back(mflops);
    }

    //

    {
      auto niter  = fftwst->niter(1000LL * 1000 * measureTimeMillis);

      cerr << "FFTW ST niter = " << niter << endl;

      for(int64_t i=0;i<niter/10;i++) fftwst->execute(); // warm up

      int64_t tm0 = timens();
      for(int64_t i=0;i<niter;i++) fftwst->execute();
      int64_t tm1 = timens();

      double mflops = 5 * n * log2n / ((tm1 - tm0) / (double(niter)*1000));

      fprintf(stderr, "%g Mflops\n", mflops);

      mflops_fftwst.push_back(mflops);
    }

    //

    {
      auto niter = sleefdftmt->niter(1000LL * 1000 * measureTimeMillis);

      cerr << "SleefDFT MT niter = " << niter << endl;

      for(int64_t i=0;i<niter/10;i++) sleefdftmt->execute(); // warm up

      int64_t tm0 = timens();
      for(int64_t i=0;i<niter;i++) sleefdftmt->execute();
      int64_t tm1 = timens();

      double mflops = 5 * n * log2n / ((tm1 - tm0) / (double(niter)*1000));

      fprintf(stderr, "%g Mflops\n", mflops);

      mflops_sleefdftmt.push_back(mflops);
    }

    //

    {
      auto niter  = fftwmt->niter(1000LL * 1000 * measureTimeMillis);

      cerr << "FFTW MT niter = " << niter << endl;

      for(int64_t i=0;i<niter/10;i++) fftwmt->execute(); // warm up

      int64_t tm0 = timens();
      for(int64_t i=0;i<niter;i++) fftwmt->execute();
      int64_t tm1 = timens();

      double mflops = 5 * n * log2n / ((tm1 - tm0) / (double(niter)*1000));

      fprintf(stderr, "%g Mflops\n", mflops);

      mflops_fftwmt.push_back(mflops);
    }
  }

  cout << log2n << ", ";

  {
    double f = 0;
    for(auto a : mflops_sleefdftst) {
      if (a > f) f = a;
    }
    cout << f << ", ";
  }

  {
    double f = 0;
    for(auto a : mflops_sleefdftmt) {
      if (a > f) f = a;
    }
    cout << f << ", ";
  }

  {
    double f = 0;
    for(auto a : mflops_fftwst) {
      if (a > f) f = a;
    }
    cout << f << ", ";
  }

  {
    double f = 0;
    for(auto a : mflops_fftwmt) {
      if (a > f) f = a;
    }
    cout << f << endl;
  }

  //

  exit(0);
}
