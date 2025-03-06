//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "dispatchparam.h"

#define CONFIG_STREAM 1
#define CONFIG_MT 2

#define MAXLOG2LEN 32

template<typename real>
struct SleefDFTXX {
  uint32_t magic;
  uint64_t mode, mode2, mode3;
  int baseTypeID;
  const real *in;
  real *out;
  
  //

  uint32_t log2len;

  real **tbl[MAXBUTWIDTH+1];
  real *rtCoef0, *rtCoef1;
  uint32_t **perm;

  real **x0, **x1;

  int isa;
  int planMode;

  int vecwidth, log2vecwidth;
  int nThread;
  
  uint64_t tm[CONFIGMAX][(MAXBUTWIDTH+1)*32];
  uint64_t bestTime;
  int16_t bestPath[32], bestPathConfig[32], pathLen;
};

template<typename real>
struct SleefDFT2DXX {
  uint32_t magic;
  uint64_t mode, mode2, mode3;
  int baseTypeID;
  const real *in;
  real *out;
  
  //

  int32_t hlen, vlen;
  int32_t log2hlen, log2vlen;
  uint64_t tmNoMT, tmMT;
  real *tBuf;

  struct SleefDFTXX<real> *instH, *instV;
};

struct SleefDFT {
  uint32_t magic;
  union {
    SleefDFTXX<double> *double_;
    SleefDFTXX<float> *float_;
    SleefDFT2DXX<double> *double2d_;
    SleefDFT2DXX<float> *float2d_;
  };
};

#define SLEEF_MODE2_MT1D       (1 << 0)
#define SLEEF_MODE3_MT2D       (1 << 0)

#define PLANFILEID "SLEEFDFT0\n"
#define ENVVAR "SLEEFDFTPLAN"

#define SLEEF_MODE_MEASUREBITS (3 << 20)

int omp_thread_count();
void startAllThreads(const int nth);

template<typename real> void freeTables(SleefDFTXX<real> *p);
uint32_t ilog2(uint32_t q);

template<typename real> int PlanManager_loadMeasurementResultsT(SleefDFT2DXX<real> *p);
template<typename real> void PlanManager_saveMeasurementResultsT(SleefDFT2DXX<real> *p);
template<typename real> int PlanManager_loadMeasurementResultsP(SleefDFTXX<real> *p, int pathCat);
template<typename real> void PlanManager_saveMeasurementResultsP(SleefDFTXX<real> *p, int pathCat);
template<typename real> void freeTables(SleefDFTXX<real> *p);

#define GETINT_VECWIDTH 100
#define GETINT_DFTPRIORITY 101
