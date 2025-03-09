//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include "dispatchparam.h"

#define CONFIG_STREAM 1
#define CONFIG_MT 2

#define MAXLOG2LEN 32

template<typename real, typename real2, int MAXBUTWIDTH>
struct SleefDFTXX {
  uint32_t magic;
  const int baseTypeID;
  const real * const in;
  real * const out;
  const int nThread;
  const uint32_t log2len;
  const uint64_t mode;

  uint64_t mode2 = 0, mode3 = 0;
  
  //

  real **tbl[MAXBUTWIDTH+1];
  real *rtCoef0, *rtCoef1;
  uint32_t **perm;

  real **x0, **x1;

  int isa = 0;
  int planMode = 0;

  int vecwidth, log2vecwidth;
  
  uint64_t tm[CONFIGMAX][(MAXBUTWIDTH+1)*32];
  uint64_t bestTime = 0;
  int16_t bestPath[32], bestPathConfig[32], pathLen = 0;

  void (*(* const DFTF)[ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int);
  void (*(* const DFTB)[ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int);
  void (*(* const TBUTF)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int);
  void (*(* const TBUTB)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int);
  void (*(* const BUTF)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int);
  void (*(* const BUTB)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int);
  void (** const REALSUB0)(real *, const real *, const int, const real *, const real *);
  void (** const REALSUB1)(real *, const real *, const int, const real *, const real *, const int);

  SleefDFTXX(uint32_t n, const real *in, real *out, uint64_t mode, const char *baseTypeString, int BASETYPEID_, int MAGIC_,
    int (*GETINT_[16])(int), const void *(*GETPTR_[16])(int), real2 (*SINCOSPI_)(real),
    void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
    void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
    void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
    void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
    void (*REALSUB0_[ISAMAX])(real *, const real *, const int, const real *, const real *),
    void (*REALSUB1_[ISAMAX])(real *, const real *, const int, const real *, const real *, const int));

  ~SleefDFTXX();

  void dispatch(const int N, real *d, const real *s, const int level, const int config);
  void execute(const real *s0, real *d0, int MAGIC_, int MAGIC2D_);
  void freeTables();
  int searchForRandomPathRecurse(int level, int *path, int *pathConfig, uint64_t tm, int nTrial);
  void searchForBestPath();
  void measureBut();
  void estimateBut();
  bool measure(bool randomize);
  int loadMeasurementResults(int pathCat);
  void saveMeasurementResults(int pathCat);
  void setPath(const char *pathStr);
};

template<typename real, typename real2, int MAXBUTWIDTH>
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

  SleefDFTXX<real, real2, MAXBUTWIDTH> *instH, *instV;

  SleefDFT2DXX(uint32_t vlen, uint32_t hlen, const real *in, real *out, uint64_t mode, const char *baseTypeString,
    int BASETYPEID_, int MAGIC_, int MAGIC2D_,
    int (*GETINT_[16])(int), const void *(*GETPTR_[16])(int), real2 (*SINCOSPI_)(real),
    void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
    void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
    void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
    void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
    void (*REALSUB0_[ISAMAX])(real *, const real *, const int, const real *, const real *),
    void (*REALSUB1_[ISAMAX])(real *, const real *, const int, const real *, const real *, const int));

  ~SleefDFT2DXX();

  void execute(const real *s0, real *d0, int MAGIC_, int MAGIC2D_);
  void measureTranspose();
  int loadMeasurementResults();
  void saveMeasurementResults();
};

struct SleefDFT {
  uint32_t magic;
  union {
    SleefDFTXX<double, Sleef_double2, MAXBUTWIDTHDP> *double_;
    SleefDFTXX<float, Sleef_float2, MAXBUTWIDTHSP> *float_;
    SleefDFT2DXX<double, Sleef_double2, MAXBUTWIDTHDP> *double2d_;
    SleefDFT2DXX<float, Sleef_float2, MAXBUTWIDTHSP> *float2d_;
  };
};

#define SLEEF_MODE2_MT1D       (1 << 0)
#define SLEEF_MODE3_MT2D       (1 << 0)

#define PLANFILEID "SLEEFDFT0\n"
#define ENVVAR "SLEEFDFTPLAN"

#define SLEEF_MODE_MEASUREBITS (3 << 20)

int omp_thread_count();
void startAllThreads(const int nth);

uint32_t ilog2(uint32_t q);

#define GETINT_VECWIDTH 100
#define GETINT_DFTPRIORITY 101
