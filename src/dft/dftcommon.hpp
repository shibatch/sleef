//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <unordered_map>
#include <mutex>

#include "dispatchparam.h"

#define CONFIG_STREAM 1
#define CONFIG_MT 2

#define MAXLOG2LEN 32

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
struct SleefDFTXX {
  int magic;
  const int baseTypeID;
  const real * const in;
  real * const out;
  const int nThread;
  const uint32_t log2len;
  const uint64_t mode;
  const int minshift;

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

  FILE *verboseFP = NULL;

  void (*(* const DFTF)[ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int);
  void (*(* const DFTB)[ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int);
  void (*(* const TBUTF)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int);
  void (*(* const TBUTB)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int);
  void (*(* const BUTF)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int);
  void (*(* const BUTB)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int);
  void (** const REALSUB0)(real *, const real *, const int, const real *, const real *);
  void (** const REALSUB1)(real *, const real *, const int, const real *, const real *, const int);
  void (*(* const DFTFS)[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *);
  void (*(* const DFTBS)[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *);
  void (*(* const TBUTFS)[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int);
  void (*(* const TBUTBS)[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int);
  void (*(* const BUTFS)[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int);
  void (*(* const BUTBS)[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int);

  SleefDFTXX(uint32_t n, const real *in, real *out, uint64_t mode, const char *baseTypeString, int BASETYPEID_, int MAGIC_, int minshift_,
    int (*GETINT_[16])(int), const void *(*GETPTR_[16])(int), real2 (*SINCOSPI_)(real),
    void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
    void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
    void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
    void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
    void (*REALSUB0_[ISAMAX])(real *, const real *, const int, const real *, const real *),
    void (*REALSUB1_[ISAMAX])(real *, const real *, const int, const real *, const real *, const int), 
    void (*DFTFS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *),
    void (*DFTBS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *),
    void (*TBUTFS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int),
    void (*TBUTBS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int),
    void (*BUTFS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*BUTBS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int)
  );

  ~SleefDFTXX();

  void dispatch(const int N, real *d, const real *s, const int level, const int config);
  void execute(const real *s0, real *d0, int MAGIC_, int MAGIC2D_);
  void freeTables();
  int searchForRandomPathRecurse(int level, int *path, int *pathConfig, uint64_t tm, int nTrial);
  void searchForBestPath(int nPaths);
  void measureBut();
  void estimateBut();
  bool measure(bool randomize);
  int loadMeasurementResults(int pathCat);
  void saveMeasurementResults(int pathCat);
  void setPath(const char *pathStr);
  size_t getPath(char *pathStr, size_t pathStrSize);
};

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
struct SleefDFT2DXX {
  int magic;
  uint64_t mode, mode2, mode3;
  int baseTypeID;
  const real *in;
  real *out;
  
  //

  int32_t hlen, vlen;
  int32_t log2hlen, log2vlen;
  uint64_t tmNoMT, tmMT;
  real *tBuf;

  SleefDFTXX<real, real2, MAXSHIFT, MAXBUTWIDTH> *instH, *instV;

  FILE *verboseFP = NULL;

  SleefDFT2DXX(uint32_t vlen, uint32_t hlen, const real *in, real *out, uint64_t mode, const char *baseTypeString,
    int BASETYPEID_, int MAGIC_, int MAGIC2D_, int minshift_,
    int (*GETINT_[16])(int), const void *(*GETPTR_[16])(int), real2 (*SINCOSPI_)(real),
    void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
    void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
    void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
    void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
    void (*REALSUB0_[ISAMAX])(real *, const real *, const int, const real *, const real *),
    void (*REALSUB1_[ISAMAX])(real *, const real *, const int, const real *, const real *, const int),
    void (*DFTFS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *),
    void (*DFTBS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *),
    void (*TBUTFS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int),
    void (*TBUTBS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int),
    void (*BUTFS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
    void (*BUTBS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int)
  );

  ~SleefDFT2DXX();

  void execute(const real *s0, real *d0, int MAGIC_, int MAGIC2D_);
  void measureTranspose();
  int loadMeasurementResults();
  void saveMeasurementResults();
};

struct SleefDFT {
  uint32_t magic;
  union {
    SleefDFTXX<double, Sleef_double2, MAXSHIFTDP, MAXBUTWIDTHDP> *double_;
    SleefDFTXX<float, Sleef_float2, MAXSHIFTSP, MAXBUTWIDTHSP> *float_;
    SleefDFT2DXX<double, Sleef_double2, MAXSHIFTDP, MAXBUTWIDTHDP> *double2d_;
    SleefDFT2DXX<float, Sleef_float2, MAXSHIFTSP, MAXBUTWIDTHSP> *float2d_;
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

class PlanManager {
  static const int CATBIT = 8;
  static const int BASETYPEIDBIT = 2;
  static const int LOG2LENBIT = 8;
  static const int DIRBIT = 1;
  static const int BUTSTATBIT = 16;
  static const int LEVELBIT = LOG2LENBIT;
  static const int BUTCONFIGBIT = 8;
  static const int TRANSCONFIGBIT = 8;

  char *dftPlanFilePath = nullptr;
  char *archID = nullptr;
  std::unordered_map<uint64_t, uint64_t> planMap;
  uint64_t planMode_ = SLEEF_PLAN_REFERTOENVVAR;
  int planFilePathSet_ = 0, planFileLoaded_ = 0;

public:
  std::mutex mtx;

  static uint64_t keyButStat(int baseTypeID, int log2len, int dir, int butStat);
  static uint64_t keyTrans(int baseTypeID, int hlen, int vlen, int transConfig);
  static uint64_t keyPath(int baseTypeID, int log2len, int dir, int level, int config);
  static uint64_t keyPathConfig(int baseTypeID, int log2len, int dir, int level, int config);

  uint64_t planMode() { return planMode_; }
  int planFilePathSet() { return planFilePathSet_; }
  int planFileLoaded() { return planFileLoaded_; }

  void setPlanFilePath(const char *path, const char *arch, uint64_t mode);
  void loadPlanFromFile();
  void savePlanToFile();
  uint64_t getU64(uint64_t key);
  void putU64(uint64_t key, uint64_t value);
};

extern PlanManager planManager;
extern FILE *defaultVerboseFP;
