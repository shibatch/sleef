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

#include "misc.h"
#include "sleef.h"

#define IMPORT_IS_EXPORT
#include "sleefdft.h"
#include "dispatchparam.h"
#include "dftcommon.hpp"
#include "common.h"
#include "serializer.hpp"

#define MAGIC_FLOAT 0x31415926
#define MAGIC_DOUBLE 0x27182818

#define MAGIC2D_FLOAT 0x22360679
#define MAGIC2D_DOUBLE 0x17320508

const char *configStr[] = { "ST", "ST stream", "MT", "MT stream" };

static int parsePathStr(char *p, int *path, int *config, int pathLenMax, int log2len) {
  int pathLen = 0, l2l = 0;

  for(;;) {
    while(*p == ' ') p++;
    if (*p == '\0') break;
    if (!isdigit((int)*p)) return -1;

    pathLen++;
    if (pathLen >= pathLenMax) return -2;

    int n = 0;
    while(isdigit((int)*p)) n = n * 10 + *p++ - '0';

    if (n > MAXBUTWIDTH) return -6;
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

template<typename real>
static void SleefDFTXX_setPath(SleefDFTXX<real> *p, char *pathStr) {
  assert(p != NULL && (p->magic == MAGIC_FLOAT || p->magic == MAGIC_DOUBLE));

  int path[32], config[32];
  int pathLen = parsePathStr(pathStr, path, config, 31, p->log2len);

  if (pathLen < 0) {
    if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("Error %d in parsing path string : %s\n", pathLen, pathStr);
    return;
  }

  for(uint32_t j = 0;j <= p->log2len;j++) p->bestPath[j] = 0;

  for(int level = p->log2len, j=0;level > 0 && j < pathLen;) {
    p->bestPath[level] = path[j];
    p->bestPathConfig[level] = config[j];
    level -= path[j];
    j++;
  }

  p->pathLen = 0;
  for(int j = p->log2len;j >= 0;j--) if (p->bestPath[j] != 0) p->pathLen++;

  if ((p->mode & SLEEF_MODE_VERBOSE) != 0) {
    printf("Set path : ");
    for(int j = p->log2len;j >= 0;j--) if (p->bestPath[j] != 0) printf("%d(%s) ", p->bestPath[j], configStr[p->bestPathConfig[j]]);
    printf("\n");
  }
}

EXPORT void SleefDFT_setPath(SleefDFT *p, char *pathStr) {
  assert(p != NULL);
  switch(p->magic) {
  case MAGIC_DOUBLE:
    SleefDFTXX_setPath<double>(p->double_, pathStr);
    break;
  case MAGIC_FLOAT:
    SleefDFTXX_setPath<float>(p->float_, pathStr);
    break;
  default: abort();
  }
}

template<typename real>
void freeTables(SleefDFTXX<real> *p) {
  for(int N=1;N<=MAXBUTWIDTH;N++) {
    for(uint32_t level=N;level<=p->log2len;level++) {
      Sleef_free(p->tbl[N][level]);
      p->tbl[N][level] = nullptr;
    }
    free(p->tbl[N]);
    p->tbl[N] = NULL;
  }

  for(int i=0;i<p->nThread;i++) {
    Sleef_free(p->x1[i]);
    p->x1[i] = nullptr;
    Sleef_free(p->x0[i]);
    p->x0[i] = nullptr;
  }

  free(p->x1);
  p->x1 = nullptr;
  free(p->x0);
  p->x0 = nullptr;
}

template<typename real>
static void SleefDFTXX_dispose(SleefDFTXX<real> *p) {
  assert(p != NULL && (p->magic == MAGIC_FLOAT || p->magic == MAGIC_DOUBLE));

  if (p->log2len <= 1) {
    p->magic = 0;
    free(p);
    return;
  }
  
  if ((p->mode & SLEEF_MODE_REAL) != 0) {
    Sleef_free(p->rtCoef1);
    p->rtCoef1 = nullptr;
    Sleef_free(p->rtCoef0);
    p->rtCoef0 = nullptr;
  }
  
  for(int level = p->log2len;level >= 1;level--) {
    Sleef_free(p->perm[level]);
    p->perm[level] = nullptr;
  }
  free(p->perm);
  p->perm = NULL;

  freeTables<real>(p);

  p->magic = 0;
  free(p);
}

template<typename real>
static void SleefDFT2DXX_dispose(SleefDFT2DXX<real> *p) {
  assert(p != NULL && (p->magic == MAGIC2D_FLOAT || p->magic == MAGIC2D_DOUBLE));

  Sleef_free(p->tBuf);
  p->tBuf = nullptr;
  SleefDFTXX_dispose(p->instH);
  p->instH = nullptr;
  if (p->hlen != p->vlen) {
    SleefDFTXX_dispose(p->instV);
    p->instV = nullptr;
  }
  
  p->magic = 0;
  free(p);
}

EXPORT void SleefDFT_dispose(SleefDFT *p) {
  assert(p != NULL);
  switch(p->magic) {
  case MAGIC_DOUBLE:
    SleefDFTXX_dispose<double>(p->double_);
    p->magic = 0;
    p->double_ = nullptr;
    free(p);
    break;
  case MAGIC2D_DOUBLE:
    SleefDFT2DXX_dispose<double>(p->double2d_);
    p->magic = 0;
    p->double_ = nullptr;
    free(p);
    break;
  case MAGIC_FLOAT:
    SleefDFTXX_dispose<float>(p->float_);
    p->magic = 0;
    p->float_ = nullptr;
    free(p);
    break;
  case MAGIC2D_FLOAT:
    SleefDFT2DXX_dispose<float>(p->float2d_);
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

//

char *dftPlanFilePath = NULL;
char *archID = NULL;
uint64_t planMode = SLEEF_PLAN_REFERTOENVVAR;
unordered_map<uint64_t, uint64_t> planMap;
int planFilePathSet = 0, planFileLoaded = 0;
omp_lock_t planMapLock;
int planMapLockInitialized = 0;

static void initPlanMapLock() {
#pragma omp critical
  {
    if (!planMapLockInitialized) {
      planMapLockInitialized = 1;
      omp_init_lock(&planMapLock);
    }
  }
}

EXPORT void SleefDFT_setPlanFilePath(const char *path, const char *arch, uint64_t mode) {
  initPlanMapLock();

  if ((mode & SLEEF_PLAN_RESET) != 0) {
    planMap.clear();
    planFileLoaded = 0;
    planFilePathSet = 0;
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

  planMode = mode;
  planFilePathSet = 1;
}

static void loadPlanFromFile() {
  if (planFilePathSet == 0 && (planMode & SLEEF_PLAN_REFERTOENVVAR) != 0) {
    char *s = getenv(ENVVAR);
    if (s != NULL) SleefDFT_setPlanFilePath(s, NULL, planMode);
  }

  planMap.clear();
  
  if (dftPlanFilePath != NULL && (planMode & SLEEF_PLAN_RESET) == 0) {
    //planMap = ArrayMap_load(dftPlanFilePath, archID, PLANFILEID, (planMode & SLEEF_PLAN_NOLOCK) == 0);
    FILE *fp = fopen(dftPlanFilePath, "rb");
    if (fp) {
      FileDeserializer d(fp);
      d >> planMap;
      fclose(fp);
    }
  }

  planFileLoaded = 1;
}

static void savePlanToFile() {
  assert(planFileLoaded);
  if ((planMode & SLEEF_PLAN_READONLY) == 0 && dftPlanFilePath != NULL) {
    //ArrayMap_save(planMap, dftPlanFilePath, archID, PLANFILEID);
    FILE *fp = fopen(dftPlanFilePath, "wb");
    if (fp) {
      FileSerializer s(fp);
      s << planMap;
      fclose(fp);
    }
  }
}

#define CATBIT 8
#define BASETYPEIDBIT 2
#define LOG2LENBIT 8
#define DIRBIT 1

#define BUTSTATBIT 16

static uint64_t keyButStat(int baseTypeID, int log2len, int dir, int butStat) {
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

#define LEVELBIT LOG2LENBIT
#define BUTCONFIGBIT 8
#define TRANSCONFIGBIT 8

static uint64_t keyTrans(int baseTypeID, int hlen, int vlen, int transConfig) {
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

static uint64_t keyPath(int baseTypeID, int log2len, int dir, int level, int config) {
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

static uint64_t keyPathConfig(int baseTypeID, int log2len, int dir, int level, int config) {
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

static uint64_t planMap_getU64(uint64_t key) {
  if (!planMap.count(key)) return 0;
  return planMap.at(key);
}

static void planMap_putU64(uint64_t key, uint64_t value) {
  planMap[key] = value;
}

template<typename real>
int PlanManager_loadMeasurementResultsP(SleefDFTXX<real> *p, int pathCat) {
  assert(p != NULL && (p->magic == MAGIC_FLOAT || p->magic == MAGIC_DOUBLE));

  initPlanMapLock();

  omp_set_lock(&planMapLock);

  if (!planFileLoaded) loadPlanFromFile();

  int stat = planMap_getU64(keyButStat(p->baseTypeID, p->log2len, p->mode, pathCat+10));
  if (stat == 0) {
    omp_unset_lock(&planMapLock);
    return 0;
  }

  int ret = 1;
  
  for(int j = p->log2len;j >= 0;j--) {
    p->bestPath[j] = planMap_getU64(keyPath(p->baseTypeID, p->log2len, p->mode, j, pathCat));
    p->bestPathConfig[j] = planMap_getU64(keyPathConfig(p->baseTypeID, p->log2len, p->mode, j, pathCat));
    if (p->bestPath[j] > MAXBUTWIDTH) ret = 0;
  }

  p->pathLen = 0;
  for(int j = p->log2len;j >= 0;j--) if (p->bestPath[j] != 0) p->pathLen++;
  
  omp_unset_lock(&planMapLock);
  return ret;
}

template<typename real>
void PlanManager_saveMeasurementResultsP(SleefDFTXX<real> *p, int pathCat) {
  assert(p != NULL && (p->magic == MAGIC_FLOAT || p->magic == MAGIC_DOUBLE));

  initPlanMapLock();

  omp_set_lock(&planMapLock);
  if (!planFileLoaded) loadPlanFromFile();

  if (planMap_getU64(keyButStat(p->baseTypeID, p->log2len, p->mode, pathCat+10)) != 0) {
    omp_unset_lock(&planMapLock);
    return;
  }
  
  for(int j = p->log2len;j >= 0;j--) {
    planMap_putU64(keyPath(p->baseTypeID, p->log2len, p->mode, j, pathCat), p->bestPath[j]);
    planMap_putU64(keyPathConfig(p->baseTypeID, p->log2len, p->mode, j, pathCat), p->bestPathConfig[j]);
  }

  planMap_putU64(keyButStat(p->baseTypeID, p->log2len, p->mode, pathCat+10), 1);

  if ((planMode & SLEEF_PLAN_READONLY) == 0) savePlanToFile();

  omp_unset_lock(&planMapLock);
}

template<typename real>
int PlanManager_loadMeasurementResultsT(SleefDFT2DXX<real> *p) {
  assert(p != NULL && (p->magic == MAGIC2D_FLOAT || p->magic == MAGIC2D_DOUBLE));

  initPlanMapLock();

  omp_set_lock(&planMapLock);
  if (!planFileLoaded) loadPlanFromFile();

  p->tmNoMT = planMap_getU64(keyTrans(p->baseTypeID, p->log2hlen, p->log2vlen, 0));
  p->tmMT   = planMap_getU64(keyTrans(p->baseTypeID, p->log2hlen, p->log2vlen, 1));
  
  omp_unset_lock(&planMapLock);
  return p->tmNoMT != 0;
}

template<typename real>
void PlanManager_saveMeasurementResultsT(SleefDFT2DXX<real> *p) {
  assert(p != NULL && (p->magic == MAGIC2D_FLOAT || p->magic == MAGIC2D_DOUBLE));

  initPlanMapLock();

  omp_set_lock(&planMapLock);
  if (!planFileLoaded) loadPlanFromFile();

  planMap_putU64(keyTrans(p->baseTypeID, p->log2hlen, p->log2vlen, 0), p->tmNoMT);
  planMap_putU64(keyTrans(p->baseTypeID, p->log2hlen, p->log2vlen, 1), p->tmMT  );
  
  if ((planMode & SLEEF_PLAN_READONLY) == 0) savePlanToFile();

  omp_unset_lock(&planMapLock);
}

template void freeTables(SleefDFTXX<double> *p);
template void freeTables(SleefDFTXX<float> *p);
template int PlanManager_loadMeasurementResultsP<double>(SleefDFTXX<double> *p, int pathCat);
template int PlanManager_loadMeasurementResultsP<float>(SleefDFTXX<float> *p, int pathCat);
template int PlanManager_loadMeasurementResultsT<double>(SleefDFT2DXX<double> *p);
template int PlanManager_loadMeasurementResultsT<float>(SleefDFT2DXX<float> *p);
template void PlanManager_saveMeasurementResultsP<double>(SleefDFTXX<double> *p, int pathCat);
template void PlanManager_saveMeasurementResultsP<float>(SleefDFTXX<float> *p, int pathCat);
template void PlanManager_saveMeasurementResultsT<double>(SleefDFT2DXX<double> *p);
template void PlanManager_saveMeasurementResultsT<float>(SleefDFT2DXX<float> *p);
