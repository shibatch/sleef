//   Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <inttypes.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <vector>

#include "compat.h"
#include "misc.h"
#include "sleef.h"

#define IMPORT_IS_EXPORT
#include "sleefdft.h"
#include "dftcommon.hpp"
#include "common.h"
#include "serializer.hpp"

#define MAGIC_FLOAT 0x31415926
#define MAGIC_DOUBLE 0x27182818

#define MAGIC2D_FLOAT 0x22360679
#define MAGIC2D_DOUBLE 0x17320508

const char *configStr[] = { "ST", "ST stream", "MT", "MT stream" };

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
vector<Action> SleefDFTXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::parsePathStr(const char *p) {
  vector<Action> v;

  int level = log2len;
  for(;;) {
    while(isspace((int)*p)) p++;
    if (*p == '\0') break;
    if (!isdigit((int)*p)) throw(runtime_error("Unexpected character"));

    int N = 0;
    while(isdigit((int)*p)) N = N * 10 + *p++ - '0';

    if (N > MAXBUTWIDTHALL) throw(runtime_error("N too large"));
    if (N > level) throw(runtime_error("N larger than level"));

    int config = 0;
    if (*p == '(') {
      p++;

      for(config=4;config>=0;config--) {
	if (strncmp(p, configStr[config], strlen(configStr[config])) == 0) break;
      }
      if (config == -1) throw(runtime_error("Unknown config"));
      p += strlen(configStr[config]);
      if (*p++ != ')') throw(runtime_error("No ')' after config"));
    }

    v.push_back(Action(config, level, N));
    level -= N;
  }

  if (level != 0) throw(runtime_error("Sum of N less than level"));

  return v;
}

static string to_string(vector<Action> v) {
  string s = "";
  for(auto e : v) {
    string c = "? " + to_string(e.config);
    if (0 <= e.config && e.config < 4) c = configStr[e.config];
    s += to_string(e.N) + "(" + c + ") ";
  }
  return s;
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
void SleefDFTXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::setPath(const char *pathStr) {
  assert(magic == MAGIC_FLOAT || magic == MAGIC_DOUBLE);

  try {
    bestPath = parsePathStr(pathStr);

    if ((mode & SLEEF_MODE_VERBOSE) != 0) fprintf(verboseFP, "Set path : %s\n", to_string(bestPath).c_str());
  } catch(exception &ex) {
    if ((mode & SLEEF_MODE_VERBOSE) != 0) fprintf(verboseFP, "Parse error : %s\n", ex.what());
  }
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
string SleefDFTXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::getPath() {
  assert(magic == MAGIC_FLOAT || magic == MAGIC_DOUBLE);
  return to_string(bestPath);
}

EXPORT void SleefDFT_setPath(SleefDFT *p, char *pathStr) {
  assert(p != NULL);
  switch(p->magic) {
  case MAGIC_DOUBLE:
    p->double_->setPath(pathStr);
    break;
  case MAGIC_FLOAT:
    p->float_->setPath(pathStr);
    break;
  default: abort();
  }
}

EXPORT int SleefDFT_getPath(SleefDFT *p, char *pathStr, int pathStrSize) {
  assert(p != NULL);

  string str;
  switch(p->magic) {
  case MAGIC_DOUBLE:
    str = p->double_->getPath();
    break;
  case MAGIC_FLOAT:
    str = p->float_->getPath();
    break;
  case MAGIC2D_DOUBLE:
    str = to_string(p->double2d_->tmNoMT) + "," + to_string(p->double2d_->tmMT) + "," +
      p->double2d_->instH->getPath() + "," + p->double2d_->instV->getPath();
    break;
  case MAGIC2D_FLOAT:
    str = to_string(p->float2d_->tmNoMT) + "," + to_string(p->float2d_->tmMT) + "," +
      p->float2d_->instH->getPath() + "," + p->float2d_->instV->getPath();
    break;
  default: abort();
  }

  strncpy(pathStr, str.c_str(), pathStrSize);

  return pathStrSize == 0 ? 0 : strlen(pathStr);
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
void SleefDFTXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::freeTables() {
  for(int N=1;N<=MAXBUTWIDTH;N++) {
    for(uint32_t level=N;level<=log2len;level++) {
      Sleef_free(tbl[N][level]);
      tbl[N][level] = nullptr;
    }
    free(tbl[N]);
    tbl[N] = NULL;
  }

  for(int i=0;i<nThread;i++) {
    Sleef_free(x1[i]);
    x1[i] = nullptr;
    Sleef_free(x0[i]);
    x0[i] = nullptr;
  }

  free(x1);
  x1 = nullptr;
  free(x0);
  x0 = nullptr;
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
SleefDFTXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::~SleefDFTXX() {
  assert(magic == MAGIC_FLOAT || magic == MAGIC_DOUBLE);

  if (log2len <= 1) {
    magic = 0;
    return;
  }
  
  if ((mode & SLEEF_MODE_REAL) != 0) {
    Sleef_free(rtCoef1);
    rtCoef1 = nullptr;
    Sleef_free(rtCoef0);
    rtCoef0 = nullptr;
  }
  
  for(int level = log2len;level >= 1;level--) {
    Sleef_free(perm[level]);
    perm[level] = nullptr;
  }
  free(perm);
  perm = NULL;

  freeTables();

  magic = 0;
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
SleefDFT2DXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::~SleefDFT2DXX() {
  assert(magic == MAGIC2D_FLOAT || magic == MAGIC2D_DOUBLE);

  Sleef_free(tBuf);
  tBuf = nullptr;
  delete instH;
  instH = nullptr;
  if (hlen != vlen) {
    delete instV;
    instV = nullptr;
  }
  
  magic = 0;
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
string SleefDFTXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::planKeyString(string suffix) {
  string s;
  s += baseTypeID == 1 ? "D" : "S";
  s += (mode & SLEEF_MODE_REAL) ? "r" : "c";
  s += (mode & SLEEF_MODE_BACKWARD) ? "b" : "f";
  s += (mode & SLEEF_MODE_ALT) ? "o" : "w";
  s += (mode & SLEEF_MODE_NO_MT) ? "s" : "m";
  s += to_string(log2len) + "," + "0";
  if (suffix != "") s += ":" + suffix;
  return s;
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
string SleefDFT2DXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::planKeyString(string suffix) {
  string s;
  s += baseTypeID == 1 ? "D" : "S";
  s += (mode & SLEEF_MODE_REAL) ? "r" : "c";
  s += (mode & SLEEF_MODE_BACKWARD) ? "b" : "f";
  s += (mode & SLEEF_MODE_ALT) ? "o" : "w";
  s += (mode & SLEEF_MODE_NO_MT) ? "s" : "m";
  s += to_string(log2hlen) + "," + to_string(log2vlen);
  if (suffix != "") s += ":" + suffix;
  return s;
}

EXPORT void SleefDFT_dispose(SleefDFT *p) {
  assert(p != NULL);
  switch(p->magic) {
  case MAGIC_DOUBLE:
    delete p->double_;
    p->magic = 0;
    p->double_ = nullptr;
    free(p);
    break;
  case MAGIC2D_DOUBLE:
    delete p->double2d_;
    p->magic = 0;
    p->double_ = nullptr;
    free(p);
    break;
  case MAGIC_FLOAT:
    delete p->float_;
    p->magic = 0;
    p->float_ = nullptr;
    free(p);
    break;
  case MAGIC2D_FLOAT:
    delete p->float2d_;
    p->magic = 0;
    p->float_ = nullptr;
    free(p);
    break;
  default: abort();
  }
}

uint32_t ilog2(uint32_t q) {
  static const uint32_t tab[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
  uint32_t r = 0,qq;

  if (q & 0xffff0000) r = 16;

  q >>= r;
  qq = q | (q >> 1);
  qq |= (qq >> 2);
  qq = ((qq & 0x10) >> 4) | ((qq & 0x100) >> 7) | ((qq & 0x1000) >> 10);

  return r + tab[qq] * 4 + tab[q >> (tab[qq] * 4)] - 1;
}

// Utility functions

int omp_thread_count() {
  int n = 0;
#pragma omp parallel reduction(+:n)
  n += 1;
  return n;
}

void startAllThreads(const int nth) {
  volatile int8_t *state = (int8_t *)calloc(nth, 1);
  int th=0;
#pragma omp parallel for
  for(th=0;th<nth;th++) {
    state[th] = 1;
    for(;;) {
      int i;
      for(i=0;i<nth;i++) if (state[i] == 0) break;
      if (i == nth) break;
    }
  }
  free((void *)state);
}

// PlanManager

static string getPlanIdPrefix() {
  string s;

#ifdef ENABLE_STREAM
  s += "s";
#else
  s += "n";
#endif
  s += to_string(CONFIGMAX) + ",";
  s += to_string(ISAMAX) + ",";
  s += to_string(MAXBUTWIDTHDP) + ",";
  s += to_string(MAXBUTWIDTHSP) + ",";
  s += to_string(MINSHIFTDP) + ",";
  s += to_string(MAXSHIFTDP) + ",";
  s += to_string(MINSHIFTSP) + ",";
  s += to_string(MAXSHIFTSP) + ":";

  return s;
}

PlanManager::PlanManager() {
  planID = getPlanIdPrefix() + Sleef_getCpuIdString();
}

void PlanManager::setPlanFilePath(const char *path, const char *arch, uint64_t mode) {
  if ((mode & SLEEF_PLAN_RESET) != 0) {
    std::get<0>(thePlan)[planID].clear();
    planFileLoaded_ = false;
    planFilePathSet_ = false;
  }

  dftPlanFilePath = "";
  if (path != NULL) dftPlanFilePath = path;

  planID = Sleef_getCpuIdString();
  if (arch != NULL) planID = arch;
  planID = getPlanIdPrefix() + planID;

  planMode_ = mode;
  planFilePathSet_ = true;
}

void PlanManager::loadPlanFromFile() {
  if (!planFilePathSet_ && (planMode_ & SLEEF_PLAN_REFERTOENVVAR) != 0) {
    char *s = std::getenv(ENVVAR);
    if (s != NULL) SleefDFT_setPlanFilePath(s, NULL, planMode_);
  }
  
  if (dftPlanFilePath != "" && (planMode_ & SLEEF_PLAN_RESET) == 0) {
    FILE *fp = fopen(dftPlanFilePath.c_str(), "rb");
    if (fp) {
      if (!(planMode_ & SLEEF_PLAN_NOLOCK)) FLOCK(fp);
      FileDeserializer d(fp);
      tuple<unordered_map<string, unordered_map<string, string>>, string> plan;
      d >> plan;
      if (!(planMode_ & SLEEF_PLAN_NOLOCK)) FUNLOCK(fp);
      fclose(fp);
      if (std::get<1>(plan) == PLANFILEID) thePlan = plan;
    }
  }

  planFileLoaded_ = true;
}

void PlanManager::savePlanToFile() {
  assert(planFileLoaded_);
  if ((planMode_ & SLEEF_PLAN_READONLY) == 0 && dftPlanFilePath != "") {
    FILE *fp = fopen(dftPlanFilePath.c_str(), "wb");
    if (fp) {
      FLOCK(fp);
      FileSerializer s(fp);
      std::get<1>(thePlan) = PLANFILEID;
      s << thePlan;
      FUNLOCK(fp);
      fclose(fp);
    }
  }
}

EXPORT void SleefDFT_setPlanFilePath(const char *path, const char *arch, uint64_t mode) {
  planManager.setPlanFilePath(path, arch, mode);
}

string PlanManager::get(const string& key) {
  if (std::get<0>(thePlan)[planID].count(key) == 0) return "";

  return std::get<0>(thePlan)[planID].at(key);
}

void PlanManager::put(const string& key, const string& value) {
  std::get<0>(thePlan)[planID][key] = value;
}

//

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
bool SleefDFTXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::loadMeasurementResults() {
  assert(magic == MAGIC_FLOAT || magic == MAGIC_DOUBLE);

  std::unique_lock<recursive_mutex> lock(planManager.mtx);

  if (!planManager.planFileLoaded()) planManager.loadPlanFromFile();

  string path = planManager.get(planKeyString());
  if (path == "") return false;

  try {
    bestPath = parsePathStr(path.c_str());
  } catch(exception &ex) {
    if ((mode & SLEEF_MODE_VERBOSE) != 0)
      fprintf(verboseFP, "SleefDFTXX::loadMeasurementResults : %s\n", ex.what());
    return false;
  }

  return true;
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
void SleefDFTXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::saveMeasurementResults() {
  assert(magic == MAGIC_FLOAT || magic == MAGIC_DOUBLE);

  unique_lock<recursive_mutex> lock(planManager.mtx);

  if (!planManager.planFileLoaded()) planManager.loadPlanFromFile();

  planManager.put(planKeyString(), to_string(bestPath));
  
  if ((planManager.planMode() & SLEEF_PLAN_READONLY) == 0) planManager.savePlanToFile();
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
bool SleefDFT2DXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::loadMeasurementResults() {
  assert(magic == MAGIC2D_FLOAT || magic == MAGIC2D_DOUBLE);

  std::unique_lock<recursive_mutex> lock(planManager.mtx);

  if (!planManager.planFileLoaded()) planManager.loadPlanFromFile();

  string mtstring = planManager.get(planKeyString());
  if (mtstring == "") return false;

  if (sscanf(mtstring.c_str(), "%llu,%llu", &tmNoMT, &tmMT) != 2) return false;

  if (!instH->loadMeasurementResults()) return false;
  if (instH != instV && !instH->loadMeasurementResults()) return false;

  return true;
}

template<typename real, typename real2, int MAXSHIFT, int MAXBUTWIDTH>
void SleefDFT2DXX<real, real2, MAXSHIFT, MAXBUTWIDTH>::saveMeasurementResults() {
  assert(magic == MAGIC2D_FLOAT || magic == MAGIC2D_DOUBLE);

  std::unique_lock<recursive_mutex> lock(planManager.mtx);

  if (!planManager.planFileLoaded()) planManager.loadPlanFromFile();

  vector<char> str(1000);
  snprintf(str.data(), str.size(), "%llu,%llu", tmNoMT, tmMT);
  planManager.put(planKeyString(), str.data());

  instH->saveMeasurementResults();
  if (instH != instV) instV->saveMeasurementResults();
}

// Instantiation

template void SleefDFTXX<double, Sleef_double2, MAXSHIFTDP, MAXBUTWIDTHDP>::freeTables();
template void SleefDFTXX<float, Sleef_float2, MAXSHIFTSP, MAXBUTWIDTHSP>::freeTables();
template SleefDFTXX<double, Sleef_double2, MAXSHIFTDP, MAXBUTWIDTHDP>::~SleefDFTXX();
template SleefDFTXX<float, Sleef_float2, MAXSHIFTSP, MAXBUTWIDTHSP>::~SleefDFTXX();
template SleefDFT2DXX<double, Sleef_double2, MAXSHIFTDP, MAXBUTWIDTHDP>::~SleefDFT2DXX();
template SleefDFT2DXX<float, Sleef_float2, MAXSHIFTSP, MAXBUTWIDTHSP>::~SleefDFT2DXX();

template bool SleefDFTXX<double, Sleef_double2, MAXSHIFTDP, MAXBUTWIDTHDP>::loadMeasurementResults();
template bool SleefDFTXX<float, Sleef_float2, MAXSHIFTSP, MAXBUTWIDTHSP>::loadMeasurementResults();
template void SleefDFTXX<double, Sleef_double2, MAXSHIFTDP, MAXBUTWIDTHDP>::saveMeasurementResults();
template void SleefDFTXX<float, Sleef_float2, MAXSHIFTSP, MAXBUTWIDTHSP>::saveMeasurementResults();
template bool SleefDFT2DXX<double, Sleef_double2, MAXSHIFTDP, MAXBUTWIDTHDP>::loadMeasurementResults();
template bool SleefDFT2DXX<float, Sleef_float2, MAXSHIFTSP, MAXBUTWIDTHSP>::loadMeasurementResults();
template void SleefDFT2DXX<double, Sleef_double2, MAXSHIFTDP, MAXBUTWIDTHDP>::saveMeasurementResults();
template void SleefDFT2DXX<float, Sleef_float2, MAXSHIFTSP, MAXBUTWIDTHSP>::saveMeasurementResults();

PlanManager planManager;

FILE *defaultVerboseFP = stdout;

EXPORT void SleefDFT_setDefaultVerboseFP(FILE *fp) {
  defaultVerboseFP = fp;
}
