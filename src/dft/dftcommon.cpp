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

static int parsePathStr(const char *p, int *path, int *config, int pathLenMax, int log2len) {
  int pathLen = 0, l2l = 0;

  for(;;) {
    while(*p == ' ') p++;
    if (*p == '\0') break;
    if (!isdigit((int)*p)) return -1;

    pathLen++;
    if (pathLen >= pathLenMax) return -2;

    int n = 0;
    while(isdigit((int)*p)) n = n * 10 + *p++ - '0';

    if (n > MAXBUTWIDTHALL) return -6;
    path[pathLen-1] = n;
    l2l += n;
    config[pathLen-1] = 0;

    if (*p != '(') continue;

    int c;
    for(c=3;c>=0;c--) if (strncmp(p+1, configStr[c], strlen(configStr[c])) == 0) break;
    if (c == -1) return -3;
    p += strlen(configStr[c]) + 1;
    if (*p != ')') return -4;
    p++;

    config[pathLen-1] = c;
  }

  if (l2l != log2len) return -5;

  return pathLen;
}

template<typename real, typename real2, int MAXBUTWIDTH>
void SleefDFTXX<real, real2, MAXBUTWIDTH>::setPath(const char *pathStr) {
  assert(magic == MAGIC_FLOAT || magic == MAGIC_DOUBLE);

  int path[32], config[32];
  int pathLen = parsePathStr(pathStr, path, config, 31, log2len);

  if (pathLen < 0) {
    if ((mode & SLEEF_MODE_VERBOSE) != 0) fprintf(verboseFP, "Error %d in parsing path string : %s\n", pathLen, pathStr);
    return;
  }

  for(uint32_t j = 0;j <= log2len;j++) bestPath[j] = 0;

  for(int level = log2len, j=0;level > 0 && j < pathLen;) {
    bestPath[level] = path[j];
    bestPathConfig[level] = config[j];
    level -= path[j];
    j++;
  }

  pathLen = 0;
  for(int j = log2len;j >= 0;j--) if (bestPath[j] != 0) pathLen++;

  if ((mode & SLEEF_MODE_VERBOSE) != 0) {
    fprintf(verboseFP, "Set path : ");
    for(int j = log2len;j >= 0;j--) if (bestPath[j] != 0) fprintf(verboseFP, "%d(%s) ", bestPath[j], configStr[bestPathConfig[j]]);
    fprintf(verboseFP, "\n");
  }
}

template<typename real, typename real2, int MAXBUTWIDTH>
size_t SleefDFTXX<real, real2, MAXBUTWIDTH>::getPath(char *pathStr, size_t pathStrSize) {
  assert(magic == MAGIC_FLOAT || magic == MAGIC_DOUBLE);
  string s = "";
  for(int j = log2len;j >= 0;j--) {
    vector<char> buf(1024);
    if (bestPath[j] != 0) snprintf(buf.data(), buf.size(), "%d(%s) ", bestPath[j], configStr[bestPathConfig[j]]);
    s += buf.data();
  }
  strncpy(pathStr, s.c_str(), pathStrSize);
  return MIN(pathStrSize-1, s.size());
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
  switch(p->magic) {
  case MAGIC_DOUBLE:
    return (int)p->double_->getPath(pathStr, pathStrSize);
  case MAGIC_FLOAT:
    return (int)p->float_->getPath(pathStr, pathStrSize);
  default: abort();
  }
}

template<typename real, typename real2, int MAXBUTWIDTH>
void SleefDFTXX<real, real2, MAXBUTWIDTH>::freeTables() {
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

template<typename real, typename real2, int MAXBUTWIDTH>
SleefDFTXX<real, real2, MAXBUTWIDTH>::~SleefDFTXX() {
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

template<typename real, typename real2, int MAXBUTWIDTH>
SleefDFT2DXX<real, real2, MAXBUTWIDTH>::~SleefDFT2DXX() {
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

uint64_t PlanManager::keyButStat(int baseTypeID, int log2len, int dir, int butStat) {
  dir = (dir & SLEEF_MODE_BACKWARD) == 0;
  int cat = 0;
  uint64_t k = 0;
  k = (k << BUTSTATBIT) | (butStat & ~(~(uint64_t)0 << BUTSTATBIT));
  k = (k << LOG2LENBIT) | (log2len & ~(~(uint64_t)0 << LOG2LENBIT));
  k = (k << DIRBIT) | (dir & ~(~(uint64_t)0 << LOG2LENBIT));
  k = (k << BASETYPEIDBIT) | (baseTypeID & ~(~(uint64_t)0 << BASETYPEIDBIT));
  k = (k << CATBIT) | (cat & ~(~(uint64_t)0 << CATBIT));
  return k;
}

uint64_t PlanManager::keyTrans(int baseTypeID, int hlen, int vlen, int transConfig) {
  int max = MAX(hlen, vlen), min = MIN(hlen, vlen);
  int cat = 2;
  uint64_t k = 0;
  k = (k << TRANSCONFIGBIT) | (transConfig & ~(~(uint64_t)0 << TRANSCONFIGBIT));
  k = (k << LOG2LENBIT) | (max & ~(~(uint64_t)0 << LOG2LENBIT));
  k = (k << LOG2LENBIT) | (min & ~(~(uint64_t)0 << LOG2LENBIT));
  k = (k << BASETYPEIDBIT) | (baseTypeID & ~(~(uint64_t)0 << BASETYPEIDBIT));
  k = (k << CATBIT) | (cat & ~(~(uint64_t)0 << CATBIT));
  return k;
}

uint64_t PlanManager::keyPath(int baseTypeID, int log2len, int dir, int level, int config) {
  dir = (dir & SLEEF_MODE_BACKWARD) == 0;
  int cat = 3;
  uint64_t k = 0;
  k = (k << BUTCONFIGBIT) | (config & ~(~(uint64_t)0 << BUTCONFIGBIT));
  k = (k << LEVELBIT) | (level & ~(~(uint64_t)0 << LEVELBIT));
  k = (k << LOG2LENBIT) | (log2len & ~(~(uint64_t)0 << LOG2LENBIT));
  k = (k << DIRBIT) | (dir & ~(~(uint64_t)0 << LOG2LENBIT));
  k = (k << BASETYPEIDBIT) | (baseTypeID & ~(~(uint64_t)0 << BASETYPEIDBIT));
  k = (k << CATBIT) | (cat & ~(~(uint64_t)0 << CATBIT));
  return k;
}

uint64_t PlanManager::keyPathConfig(int baseTypeID, int log2len, int dir, int level, int config) {
  dir = (dir & SLEEF_MODE_BACKWARD) == 0;
  int cat = 4;
  uint64_t k = 0;
  k = (k << BUTCONFIGBIT) | (config & ~(~(uint64_t)0 << BUTCONFIGBIT));
  k = (k << LEVELBIT) | (level & ~(~(uint64_t)0 << LEVELBIT));
  k = (k << LOG2LENBIT) | (log2len & ~(~(uint64_t)0 << LOG2LENBIT));
  k = (k << DIRBIT) | (dir & ~(~(uint64_t)0 << LOG2LENBIT));
  k = (k << BASETYPEIDBIT) | (baseTypeID & ~(~(uint64_t)0 << BASETYPEIDBIT));
  k = (k << CATBIT) | (cat & ~(~(uint64_t)0 << CATBIT));
  return k;
}

void PlanManager::setPlanFilePath(const char *path, const char *arch, uint64_t mode) {
  if ((mode & SLEEF_PLAN_RESET) != 0) {
    planMap.clear();
    planFileLoaded_ = 0;
    planFilePathSet_ = 0;
  }

  if (dftPlanFilePath != NULL) free(dftPlanFilePath);
  if (path != NULL) {
    dftPlanFilePath = (char *)malloc(strlen(path)+10);
    strncpy(dftPlanFilePath, path, strlen(path)+1);
  } else {
    dftPlanFilePath = NULL;
  }

  if (archID != NULL) free(archID);
  if (arch == NULL) arch = Sleef_getCpuIdString();
  archID = (char *)malloc(strlen(arch)+10);
  strncpy(archID, arch, strlen(arch)+1);

  planMode_ = mode;
  planFilePathSet_ = 1;
}

void PlanManager::loadPlanFromFile() {
  if (planFilePathSet_ == 0 && (planMode_ & SLEEF_PLAN_REFERTOENVVAR) != 0) {
    char *s = std::getenv(ENVVAR);
    if (s != NULL) SleefDFT_setPlanFilePath(s, NULL, planMode_);
  }

  planMap.clear();
  
  if (dftPlanFilePath != NULL && (planMode_ & SLEEF_PLAN_RESET) == 0) {
    //planMap = ArrayMap_load(dftPlanFilePath, archID, PLANFILEID, (planMode_ & SLEEF_PLAN_NOLOCK) == 0);
    FILE *fp = fopen(dftPlanFilePath, "rb");
    if (fp) {
      if (!(planMode_ & SLEEF_PLAN_NOLOCK)) FLOCK(fp);
      FileDeserializer d(fp);
      d >> planMap;
      if (!(planMode_ & SLEEF_PLAN_NOLOCK)) FUNLOCK(fp);
      fclose(fp);
    }
  }

  planFileLoaded_ = 1;
}

void PlanManager::savePlanToFile() {
  assert(planFileLoaded_);
  if ((planMode_ & SLEEF_PLAN_READONLY) == 0 && dftPlanFilePath != NULL) {
    //ArrayMap_save(planMap, dftPlanFilePath, archID, PLANFILEID);
    FILE *fp = fopen(dftPlanFilePath, "wb");
    if (fp) {
      FLOCK(fp);
      FileSerializer s(fp);
      s << planMap;
      FUNLOCK(fp);
      fclose(fp);
    }
  }
}

uint64_t PlanManager::getU64(uint64_t key) {
  if (!planMap.count(key)) return 0;
  return planMap.at(key);
}

void PlanManager::putU64(uint64_t key, uint64_t value) {
  planMap[key] = value;
}

EXPORT void SleefDFT_setPlanFilePath(const char *path, const char *arch, uint64_t mode) {
  planManager.setPlanFilePath(path, arch, mode);
}

template<typename real, typename real2, int MAXBUTWIDTH>
int SleefDFTXX<real, real2, MAXBUTWIDTH>::loadMeasurementResults(int pathCat) {
  assert(magic == MAGIC_FLOAT || magic == MAGIC_DOUBLE);

  std::unique_lock<mutex> lock(planManager.mtx);

  if (!planManager.planFileLoaded()) planManager.loadPlanFromFile();

  int stat = planManager.getU64(PlanManager::keyButStat(baseTypeID, log2len, mode, pathCat+10));
  if (stat == 0) return 0;

  int ret = 1;
  
  for(int j = log2len;j >= 0;j--) {
    bestPath[j] = planManager.getU64(PlanManager::keyPath(baseTypeID, log2len, mode, j, pathCat));
    bestPathConfig[j] = planManager.getU64(PlanManager::keyPathConfig(baseTypeID, log2len, mode, j, pathCat));
    if (bestPath[j] > MAXBUTWIDTH) ret = 0;
  }

  pathLen = 0;
  for(int j = log2len;j >= 0;j--) if (bestPath[j] != 0) pathLen++;
  
  return ret;
}

template<typename real, typename real2, int MAXBUTWIDTH>
void SleefDFTXX<real, real2, MAXBUTWIDTH>::saveMeasurementResults(int pathCat) {
  assert(magic == MAGIC_FLOAT || magic == MAGIC_DOUBLE);

  std::unique_lock<mutex> lock(planManager.mtx);

  if (!planManager.planFileLoaded()) planManager.loadPlanFromFile();

  if (planManager.getU64(PlanManager::keyButStat(baseTypeID, log2len, mode, pathCat+10)) != 0) {
    return;
  }
  
  for(int j = log2len;j >= 0;j--) {
    planManager.putU64(PlanManager::keyPath(baseTypeID, log2len, mode, j, pathCat), bestPath[j]);
    planManager.putU64(PlanManager::keyPathConfig(baseTypeID, log2len, mode, j, pathCat), bestPathConfig[j]);
  }

  planManager.putU64(PlanManager::keyButStat(baseTypeID, log2len, mode, pathCat+10), 1);

  if ((planManager.planMode() & SLEEF_PLAN_READONLY) == 0) planManager.savePlanToFile();
}

template<typename real, typename real2, int MAXBUTWIDTH>
int SleefDFT2DXX<real, real2, MAXBUTWIDTH>::loadMeasurementResults() {
  assert(magic == MAGIC2D_FLOAT || magic == MAGIC2D_DOUBLE);

  std::unique_lock<mutex> lock(planManager.mtx);

  if (!planManager.planFileLoaded()) planManager.loadPlanFromFile();

  tmNoMT = planManager.getU64(PlanManager::keyTrans(baseTypeID, log2hlen, log2vlen, 0));
  tmMT   = planManager.getU64(PlanManager::keyTrans(baseTypeID, log2hlen, log2vlen, 1));
  
  return tmNoMT != 0;
}

template<typename real, typename real2, int MAXBUTWIDTH>
void SleefDFT2DXX<real, real2, MAXBUTWIDTH>::saveMeasurementResults() {
  assert(magic == MAGIC2D_FLOAT || magic == MAGIC2D_DOUBLE);

  std::unique_lock<mutex> lock(planManager.mtx);

  if (!planManager.planFileLoaded()) planManager.loadPlanFromFile();

  planManager.putU64(PlanManager::keyTrans(baseTypeID, log2hlen, log2vlen, 0), tmNoMT);
  planManager.putU64(PlanManager::keyTrans(baseTypeID, log2hlen, log2vlen, 1), tmMT  );
  
  if ((planManager.planMode() & SLEEF_PLAN_READONLY) == 0) planManager.savePlanToFile();
}

// Instantiation

template void SleefDFTXX<double, Sleef_double2, MAXBUTWIDTHDP>::freeTables();
template void SleefDFTXX<float, Sleef_float2, MAXBUTWIDTHSP>::freeTables();
template SleefDFTXX<double, Sleef_double2, MAXBUTWIDTHDP>::~SleefDFTXX();
template SleefDFTXX<float, Sleef_float2, MAXBUTWIDTHSP>::~SleefDFTXX();
template SleefDFT2DXX<double, Sleef_double2, MAXBUTWIDTHDP>::~SleefDFT2DXX();
template SleefDFT2DXX<float, Sleef_float2, MAXBUTWIDTHSP>::~SleefDFT2DXX();

template int SleefDFTXX<double, Sleef_double2, MAXBUTWIDTHDP>::loadMeasurementResults(int pathCat);
template int SleefDFTXX<float, Sleef_float2, MAXBUTWIDTHSP>::loadMeasurementResults(int pathCat);
template void SleefDFTXX<double, Sleef_double2, MAXBUTWIDTHDP>::saveMeasurementResults(int pathCat);
template void SleefDFTXX<float, Sleef_float2, MAXBUTWIDTHSP>::saveMeasurementResults(int pathCat);
template int SleefDFT2DXX<double, Sleef_double2, MAXBUTWIDTHDP>::loadMeasurementResults();
template int SleefDFT2DXX<float, Sleef_float2, MAXBUTWIDTHSP>::loadMeasurementResults();
template void SleefDFT2DXX<double, Sleef_double2, MAXBUTWIDTHDP>::saveMeasurementResults();
template void SleefDFT2DXX<float, Sleef_float2, MAXBUTWIDTHSP>::saveMeasurementResults();

PlanManager planManager;

FILE *defaultVerboseFP = stdout;

EXPORT void SleefDFT_setDefaultVerboseFP(FILE *fp) {
  defaultVerboseFP = fp;
}
