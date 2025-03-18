//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>
#include <vector>
#include <climits>
#include <unordered_map>
#include <tuple>
#include <mutex>

using namespace std;

#include "dispatchparam.h"

#define CONFIG_STREAM 1
#define CONFIG_MT 2

#define MAXLOG2LEN 32

#define INFINITY_ (1e+300 * 1e+300)

class Action {
public:
  int config, level, N;

  Action(const Action& a) = default;

  Action(int config_, int level_, int N_) : config(config_), level(level_), N(N_) {}

  bool operator==(const Action& rhs) const {
    return config == rhs.config && level == rhs.level && N == rhs.N;
  }
  bool operator!=(const Action& rhs) const { return !(*this == rhs); }

  friend ostream& operator<<(ostream &os, const Action &ac) {
    return os << "[" << ac.config << ", " << ac.level << ", " << ac.N << "]";
  }
};

template <>
struct std::hash<Action> {
  size_t operator()(const Action &a) const {
    size_t u = 0;
    u ^= a.config;
    u = (u << 7) | (u >> ((sizeof(u)*8)-7));
    u ^= a.level;
    u = (u << 7) | (u >> ((sizeof(u)*8)-7));
    u ^= a.N;
    return u;
  }
};

template <>
struct std::hash<vector<Action>> {
  size_t operator()(const vector<Action> &v) const {
    size_t u = 0;
    for(auto a : v) {
      hash<Action> hash;
      u ^= hash(a);
      u = (u << 19) | (u >> ((sizeof(u)*8)-19));
    }
    return u;
  }
};

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
  
  bool executable[CONFIGMAX][MAXLOG2LEN][MAXLOG2LEN];
  vector<Action> bestPath;

  FILE *verboseFP = NULL;

  void (*(* const DFTF)[ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int);
  void (*(* const DFTB)[ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int);
  void (*(* const TBUTF)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int);
  void (*(* const TBUTB)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int);
  void (*(* const BUTF)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int);
  void (*(* const BUTB)[ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int);
  void (** const REALSUB0)(real *, const real *, const int, const real *, const real *);
  void (** const REALSUB1)(real *, const real *, const int, const real *, const real *, const int);
  void (*(* const TBUTFS)[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int);
  void (*(* const TBUTBS)[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int);

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
    void (*TBUTFS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int),
    void (*TBUTBS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int)
  );

  ~SleefDFTXX();

  void dispatch(const int N, real *d, const real *s, const int level, const int config);
  void execute(const real *s0, real *d0, int MAGIC_, int MAGIC2D_);
  void freeTables();
  void generatePerm(const vector<Action> &);

  uint64_t measurePath(const vector<Action> &path, uint64_t niter);
  void searchForBestPath(int nPaths);
  void searchForRandomPath();
  bool measure(bool randomize);

  vector<Action> parsePathStr(const char *);

  string planKeyString(string = "");
  bool loadMeasurementResults();
  void saveMeasurementResults();
  void setPath(const char *pathStr);
  string getPath();
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
  unsigned long long tmNoMT, tmMT;
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
    void (*TBUTFS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int),
    void (*TBUTBS_[MAXSHIFT][CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const real *, const int)
  );

  ~SleefDFT2DXX();

  void execute(const real *s0, real *d0, int MAGIC_, int MAGIC2D_);
  void measureTranspose();

  string planKeyString(string = "");
  bool loadMeasurementResults();
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

#define PLANFILEID "SLEEFDFT1"
#define ENVVAR "SLEEFDFTPLAN"

#define SLEEF_MODE_MEASUREBITS (3 << 20)

int omp_thread_count();
void startAllThreads(const int nth);

uint32_t ilog2(uint32_t q);

#define GETINT_VECWIDTH 100
#define GETINT_DFTPRIORITY 101

class PlanManager {
  string dftPlanFilePath;
  bool planFilePathSet_ = 0, planFileLoaded_ = 0;
  uint64_t planMode_ = SLEEF_PLAN_REFERTOENVVAR;

  string planID;
  tuple<unordered_map<string, unordered_map<string, string>>, string> thePlan;

public:
  PlanManager();

  recursive_mutex mtx;

  uint64_t planMode() { return planMode_; }
  int planFilePathSet() { return planFilePathSet_; }
  int planFileLoaded() { return planFileLoaded_; }

  void setPlanFilePath(const char *path, const char *arch, uint64_t mode);
  void loadPlanFromFile();
  void savePlanToFile();

  string get(const string& key);
  void put(const string& key, const string& value);
};

extern PlanManager planManager;
extern FILE *defaultVerboseFP;
