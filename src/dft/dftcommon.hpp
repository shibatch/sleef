//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "dispatchparam.h"

#define CONFIG_STREAM 1
#define CONFIG_MT 2

#define MAXLOG2LEN 32

template<typename real, typename real2>
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

  SleefDFTXX(uint32_t n, const real *in, real *out, uint64_t mode, const char *baseTypeString, int BASETYPEID_, int MAGIC_,
	     int (*GETINT_[16])(int), const void *(*GETPTR_[16])(int), real2 (*SINCOSPI_)(real),
	     void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	     void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	     void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	     void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	     void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
	     void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int));

  ~SleefDFTXX();

  void execute(const real *s0, real *d0, int MAGIC_, int MAGIC2D_,
	       void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	       void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	       void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	       void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	       void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
	       void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
	       void (*REALSUB0_[ISAMAX])(real *, const real *, const int, const real *, const real *),
	       void (*REALSUB1_[ISAMAX])(real *, const real *, const int, const real *, const real *, const int));
};

template<typename real, typename real2>
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

  SleefDFTXX<real, real2> *instH, *instV;

  SleefDFT2DXX(uint32_t vlen, uint32_t hlen, const real *in, real *out, uint64_t mode, const char *baseTypeString,
	       int BASETYPEID_, int MAGIC_, int MAGIC2D_,
	       int (*GETINT_[16])(int), const void *(*GETPTR_[16])(int), real2 (*SINCOSPI_)(real),
	       void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	       void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	       void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	       void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	       void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
	       void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int));

  ~SleefDFT2DXX();

  void execute(const real *s0, real *d0, int MAGIC_, int MAGIC2D_,
	       void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	       void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	       void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	       void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	       void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
	       void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
	       void (*REALSUB0_[ISAMAX])(real *, const real *, const int, const real *, const real *),
	       void (*REALSUB1_[ISAMAX])(real *, const real *, const int, const real *, const real *, const int));
};

struct SleefDFT {
  uint32_t magic;
  union {
    SleefDFTXX<double, Sleef_double2> *double_;
    SleefDFTXX<float, Sleef_float2> *float_;
    SleefDFT2DXX<double, Sleef_double2> *double2d_;
    SleefDFT2DXX<float, Sleef_float2> *float2d_;
  };
};

#define SLEEF_MODE2_MT1D       (1 << 0)
#define SLEEF_MODE3_MT2D       (1 << 0)

#define PLANFILEID "SLEEFDFT0\n"
#define ENVVAR "SLEEFDFTPLAN"

#define SLEEF_MODE_MEASUREBITS (3 << 20)

int omp_thread_count();
void startAllThreads(const int nth);

template<typename real, typename real2> void freeTables(SleefDFTXX<real, real2> *p);
uint32_t ilog2(uint32_t q);

template<typename real, typename real2> int PlanManager_loadMeasurementResultsT(SleefDFT2DXX<real, real2> *p);
template<typename real, typename real2> void PlanManager_saveMeasurementResultsT(SleefDFT2DXX<real, real2> *p);
template<typename real, typename real2> int PlanManager_loadMeasurementResultsP(SleefDFTXX<real, real2> *p, int pathCat);
template<typename real, typename real2> void PlanManager_saveMeasurementResultsP(SleefDFTXX<real, real2> *p, int pathCat);
template<typename real, typename real2> void freeTables(SleefDFTXX<real, real2> *p);

#define GETINT_VECWIDTH 100
#define GETINT_DFTPRIORITY 101
