//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc.h"
#include "sleef.h"

#define IMPORT_IS_EXPORT
#include "sleefdft.h"
#include "dispatchparam.h"
#include "dftcommon.h"
#include "common.h"
#include "arraymap.h"

#define MAGIC_FLOAT 0x31415926
#define MAGIC_DOUBLE 0x27182818
#define MAGIC_LONGDOUBLE 0x14142135
#define MAGIC_QUAD 0x33166247

#define MAGIC2D_FLOAT 0x22360679
#define MAGIC2D_DOUBLE 0x17320508
#define MAGIC2D_LONGDOUBLE 0x26457513
#define MAGIC2D_QUAD 0x36055512

EXPORT void SleefDFT_setPath(SleefDFT *p, char *pathStr) {
  assert(p != NULL && (p->magic == MAGIC_FLOAT || p->magic == MAGIC_DOUBLE || p->magic == MAGIC_LONGDOUBLE || p->magic == MAGIC_QUAD));

  int path[32];

  int pathLen = sscanf(pathStr, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", 
		       &path[ 0], &path[ 1], &path[ 2], &path[ 3], &path[ 4], &path[ 5], &path[ 6], &path[ 7], 
		       &path[ 8], &path[ 9], &path[10], &path[11], &path[12], &path[13], &path[14], &path[15], 
		       &path[16], &path[17], &path[18], &path[19], &path[20], &path[21], &path[22], &path[23], 
		       &path[24], &path[25], &path[26], &path[27], &path[28], &path[29], &path[30], &path[31]);

  for(uint32_t j = 0;j <= p->log2len;j++) p->bestPath[j] = 0;

  for(int level = p->log2len, j=0;level > 0 && j < pathLen;) {
    p->bestPath[level] = path[j] % 10;
    p->bestPathConfig[level] = path[j] / 10;
    level -= ABS(path[j] % 10);
    j++;
  }

  p->pathLen = 0;
  for(int j = p->log2len;j >= 0;j--) if (p->bestPath[j] != 0) p->pathLen++;

  if ((p->mode & SLEEF_MODE_VERBOSE) != 0) {
    printf("set path : ");
    for(int j = p->log2len;j >= 0;j--) if (p->bestPath[j] != 0) printf("%d(%d) ", p->bestPath[j], p->bestPathConfig[j]);
    printf("\n");
  }
}

void freeTables(SleefDFT *p) {
  for(int N=1;N<=MAXBUTWIDTH;N++) {
    for(uint32_t level=N;level<=p->log2len;level++) {
      Sleef_free(p->tbl[N][level]);
    }
    free(p->tbl[N]);
    p->tbl[N] = NULL;
  }
}

EXPORT void SleefDFT_dispose(SleefDFT *p) {
  if (p != NULL && (p->magic == MAGIC2D_FLOAT || p->magic == MAGIC2D_DOUBLE || p->magic == MAGIC2D_LONGDOUBLE || p->magic == MAGIC2D_QUAD)) {
    Sleef_free(p->tBuf);
    SleefDFT_dispose(p->instH);
    if (p->hlen != p->vlen) SleefDFT_dispose(p->instV);
  
    p->magic = 0;
    free(p);
    return;
  }

  assert(p != NULL && (p->magic == MAGIC_FLOAT || p->magic == MAGIC_DOUBLE || p->magic == MAGIC_LONGDOUBLE || p->magic == MAGIC_QUAD));

  if (p->log2len <= 1) {
    p->magic = 0;
    free(p);
    return;
  }
  
  if ((p->mode & SLEEF_MODE_REAL) != 0) {
    Sleef_free(p->rtCoef1);
    Sleef_free(p->rtCoef0);
    p->rtCoef0 = p->rtCoef1 = NULL;
  }

  for(int i=0;i<p->nThread;i++) {
    Sleef_free(p->x1[i]);
    Sleef_free(p->x0[i]);
    p->x0[i] = p->x1[i] = NULL;
  }

  free(p->x1);
  free(p->x0);
  p->x0 = p->x1 = NULL;
  
  for(int level = p->log2len;level >= 1;level--) {
    Sleef_free(p->perm[level]);
  }
  free(p->perm);
  p->perm = NULL;

  freeTables(p);

  p->magic = 0;
  free(p);
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

//

char *dftPlanFilePath = NULL;
char *archID = NULL;
uint64_t planMode = SLEEF_PLAN_REFERTOENVVAR;
ArrayMap *planMap = NULL;
int planFilePathSet = 0, planFileLoaded = 0;
#ifdef _OPENMP
omp_lock_t planMapLock;
int planMapLockInitialized = 0;
#endif

static void initPlanMapLock() {
#ifdef _OPENMP
#pragma omp critical
  {
    if (!planMapLockInitialized) {
      planMapLockInitialized = 1;
      omp_init_lock(&planMapLock);
    }
  }
#endif
}

EXPORT void SleefDFT_setPlanFilePath(const char *path, const char *arch, uint64_t mode) {
  initPlanMapLock();

  if (dftPlanFilePath != NULL) free(dftPlanFilePath);
  if (path != NULL) {
    dftPlanFilePath = malloc(strlen(path)+10);
    strcpy(dftPlanFilePath, path);
  } else {
    dftPlanFilePath = NULL;
  }

  if (archID != NULL) free(archID);
  if (arch == NULL) arch = Sleef_getCpuIdString();
  archID = malloc(strlen(arch)+10);
  strcpy(archID, arch);

  planMode = mode;
  planFilePathSet = 1;
}

static void loadPlanFromFile() {
  if (planFilePathSet == 0 && (planMode & SLEEF_PLAN_REFERTOENVVAR) != 0) {
    char *s = getenv(ENVVAR);
    if (s != NULL) SleefDFT_setPlanFilePath(s, NULL, planMode);
  }

  if (planMap != NULL) ArrayMap_dispose(planMap);
  
  if (dftPlanFilePath != NULL) {
    planMap = ArrayMap_load(dftPlanFilePath, archID, PLANFILEID, (planMode & SLEEF_PLAN_NOLOCK) == 0);
  }

  if (planMap == NULL) planMap = initArrayMap();

  planFileLoaded = 1;
}

static void savePlanToFile() {
  assert(planFileLoaded);
  if ((planMode & SLEEF_PLAN_READONLY) == 0 && dftPlanFilePath != NULL) {
    ArrayMap_save(planMap, dftPlanFilePath, archID, PLANFILEID);
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
  k = (k << BUTSTATBIT) | (butStat & ~(-1 << BUTSTATBIT));
  k = (k << LOG2LENBIT) | (log2len & ~(-1 << LOG2LENBIT));
  k = (k << DIRBIT) | (dir & ~(-1 << LOG2LENBIT));
  k = (k << BASETYPEIDBIT) | (baseTypeID & ~(-1 << BASETYPEIDBIT));
  k = (k << CATBIT) | (cat & ~(-1 << CATBIT));
  return k;
}

#define LEVELBIT LOG2LENBIT
#define BUTCONFIGBIT 8
#define TRANSCONFIGBIT 8

static uint64_t keyTrans(int baseTypeID, int hlen, int vlen, int transConfig) {
  int max = MAX(hlen, vlen), min = MIN(hlen, vlen);
  int cat = 2;
  uint64_t k = 0;
  k = (k << TRANSCONFIGBIT) | (transConfig & ~(-1 << TRANSCONFIGBIT));
  k = (k << LOG2LENBIT) | (max & ~(-1 << LOG2LENBIT));
  k = (k << LOG2LENBIT) | (min & ~(-1 << LOG2LENBIT));
  k = (k << BASETYPEIDBIT) | (baseTypeID & ~(-1 << BASETYPEIDBIT));
  k = (k << CATBIT) | (cat & ~(-1 << CATBIT));
  return k;
}

static uint64_t keyPath(int baseTypeID, int log2len, int dir, int level, int config) {
  dir = (dir & SLEEF_MODE_BACKWARD) == 0;
  int cat = 3;
  uint64_t k = 0;
  k = (k << BUTCONFIGBIT) | (config & ~(-1 << BUTCONFIGBIT));
  k = (k << LEVELBIT) | (level & ~(-1 << LEVELBIT));
  k = (k << LOG2LENBIT) | (log2len & ~(-1 << LOG2LENBIT));
  k = (k << DIRBIT) | (dir & ~(-1 << LOG2LENBIT));
  k = (k << BASETYPEIDBIT) | (baseTypeID & ~(-1 << BASETYPEIDBIT));
  k = (k << CATBIT) | (cat & ~(-1 << CATBIT));
  return k;
}

static uint64_t keyPathConfig(int baseTypeID, int log2len, int dir, int level, int config) {
  dir = (dir & SLEEF_MODE_BACKWARD) == 0;
  int cat = 4;
  uint64_t k = 0;
  k = (k << BUTCONFIGBIT) | (config & ~(-1 << BUTCONFIGBIT));
  k = (k << LEVELBIT) | (level & ~(-1 << LEVELBIT));
  k = (k << LOG2LENBIT) | (log2len & ~(-1 << LOG2LENBIT));
  k = (k << DIRBIT) | (dir & ~(-1 << LOG2LENBIT));
  k = (k << BASETYPEIDBIT) | (baseTypeID & ~(-1 << BASETYPEIDBIT));
  k = (k << CATBIT) | (cat & ~(-1 << CATBIT));
  return k;
}

static uint64_t planMap_getU64(uint64_t key) {
  char *s = ArrayMap_get(planMap, key);
  if (s == NULL) return 0;
  uint64_t ret;
  if (sscanf(s, "%" SCNx64, &ret) != 1) return 0;
  return ret;
}

static void planMap_putU64(uint64_t key, uint64_t value) {
  char *s = malloc(100);
  sprintf(s, "%" PRIx64, value);
  s = ArrayMap_put(planMap, key, s);
  if (s != NULL) free(s);
}

int PlanManager_loadMeasurementResultsP(SleefDFT *p, int pathCat) {
  assert(p != NULL && (p->magic == MAGIC_FLOAT || p->magic == MAGIC_DOUBLE || p->magic == MAGIC_LONGDOUBLE || p->magic == MAGIC_QUAD));

  initPlanMapLock();

#ifdef _OPENMP
  omp_set_lock(&planMapLock);
#endif
  if (!planFileLoaded) loadPlanFromFile();

  int stat = planMap_getU64(keyButStat(p->baseTypeID, p->log2len, p->mode, pathCat+10));
  if (stat == 0) {
#ifdef _OPENMP
    omp_unset_lock(&planMapLock);
#endif
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
  
#ifdef _OPENMP
  omp_unset_lock(&planMapLock);
#endif
  return ret;
}

void PlanManager_saveMeasurementResultsP(SleefDFT *p, int pathCat) {
  assert(p != NULL && (p->magic == MAGIC_FLOAT || p->magic == MAGIC_DOUBLE || p->magic == MAGIC_LONGDOUBLE || p->magic == MAGIC_QUAD));

  initPlanMapLock();

#ifdef _OPENMP
  omp_set_lock(&planMapLock);
#endif
  if (!planFileLoaded) loadPlanFromFile();

  if (planMap_getU64(keyButStat(p->baseTypeID, p->log2len, p->mode, pathCat+10)) != 0) {
#ifdef _OPENMP
    omp_unset_lock(&planMapLock);
#endif
    return;
  }
  
  for(int j = p->log2len;j >= 0;j--) {
    planMap_putU64(keyPath(p->baseTypeID, p->log2len, p->mode, j, pathCat), p->bestPath[j]);
    planMap_putU64(keyPathConfig(p->baseTypeID, p->log2len, p->mode, j, pathCat), p->bestPathConfig[j]);
  }

  planMap_putU64(keyButStat(p->baseTypeID, p->log2len, p->mode, pathCat+10), 1);

  if ((planMode & SLEEF_PLAN_READONLY) == 0) savePlanToFile();

#ifdef _OPENMP
  omp_unset_lock(&planMapLock);
#endif
}

int PlanManager_loadMeasurementResultsT(SleefDFT *p) {
  assert(p != NULL && (p->magic == MAGIC2D_FLOAT || p->magic == MAGIC2D_DOUBLE || p->magic == MAGIC2D_LONGDOUBLE || p->magic == MAGIC2D_QUAD));

  initPlanMapLock();

  int ret = 0;
  
#ifdef _OPENMP
  omp_set_lock(&planMapLock);
#endif
  if (!planFileLoaded) loadPlanFromFile();

  p->tmNoMT = planMap_getU64(keyTrans(p->baseTypeID, p->log2hlen, p->log2vlen, 0));
  p->tmMT   = planMap_getU64(keyTrans(p->baseTypeID, p->log2hlen, p->log2vlen, 1));
  
#ifdef _OPENMP
  omp_unset_lock(&planMapLock);
#endif
  return p->tmNoMT != 0;
}

void PlanManager_saveMeasurementResultsT(SleefDFT *p) {
  assert(p != NULL && (p->magic == MAGIC2D_FLOAT || p->magic == MAGIC2D_DOUBLE || p->magic == MAGIC2D_LONGDOUBLE || p->magic == MAGIC2D_QUAD));

  initPlanMapLock();

  int ret = 0;
  
#ifdef _OPENMP
  omp_set_lock(&planMapLock);
#endif
  if (!planFileLoaded) loadPlanFromFile();

  planMap_putU64(keyTrans(p->baseTypeID, p->log2hlen, p->log2vlen, 0), p->tmNoMT);
  planMap_putU64(keyTrans(p->baseTypeID, p->log2hlen, p->log2vlen, 1), p->tmMT  );
  
  if ((planMode & SLEEF_PLAN_READONLY) == 0) savePlanToFile();

#ifdef _OPENMP
  omp_unset_lock(&planMapLock);
#endif
}
