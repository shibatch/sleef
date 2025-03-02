//   Copyright Naoki Shibata and contributors 2010 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <setjmp.h>
#include <math.h>

#include "sleef.h"

#include "misc.h"
#include "common.h"
#include "arraymap.h"
#include "dftcommon.hpp"

#include <omp.h>

#if BASETYPEID == 1
typedef double real;
#include "dispatchdp.hpp"
#elif BASETYPEID == 2
typedef float real;
#include "dispatchsp.hpp"
#else
#error No BASETYPEID specified
#endif


#define IMPORT_IS_EXPORT
#include "sleefdft.h"

//

#ifndef ENABLE_STREAM
#error ENABLE_STREAM not defined
#endif

static const int constK[] = { 0, 2, 6, 14, 38, 94, 230, 542, 1254 };

extern const char *configStr[];

extern int planFilePathSet;

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
static jmp_buf sigjmp;
#define SETJMP(x) setjmp(x)
#define LONGJMP longjmp
#else
static sigjmp_buf sigjmp;
#define SETJMP(x) sigsetjmp(x, 1)
#define LONGJMP siglongjmp
#endif

static void sighandler(int signum) { LONGJMP(sigjmp, 1); }

template<int (*GETINT_[16])(int)>
static int checkISAAvailability(int isa) {
  signal(SIGILL, sighandler);

  if (SETJMP(sigjmp) == 0) {
    int ret = GETINT_[isa] != NULL && (*GETINT_[isa])(BASETYPEID);
    signal(SIGILL, SIG_DFL);
    return ret;
  }

  signal(SIGILL, SIG_DFL);
  return 0;
}

// Dispatcher

template<typename real,
	 void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	 void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	 void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	 void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	 void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
	 void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int)>
static void dispatch(SleefDFT *p, const int N, real *d, const real *s, const int level, const int config) {
  const int K = constK[N], log2len = p->log2len;
  if (level == N) {
    if ((p->mode & SLEEF_MODE_BACKWARD) == 0) {
      void (*func)(real *, const real *, const int) = DFTF_[config][p->isa][N];
      (*func)(d, s, log2len-N);
    } else {
      void (*func)(real *, const real *, const int) = DFTB_[config][p->isa][N];
      (*func)(d, s, log2len-N);
    }
  } else if (level == log2len) {
    assert(p->vecwidth <= (1 << N));
    if ((p->mode & SLEEF_MODE_BACKWARD) == 0) {
      void (*func)(real *, uint32_t *, const real *, const int, const real *, const int) = TBUTF_[config][p->isa][N];
      (*func)(d, p->perm[level], s, log2len-N, (const real *)p->tbl[N][level], K);
    } else {
      void (*func)(real *, uint32_t *, const real *, const int, const real *, const int) = TBUTB_[config][p->isa][N];
      (*func)(d, p->perm[level], s, log2len-N, (const real *)p->tbl[N][level], K);
    }
  } else {
    if ((p->mode & SLEEF_MODE_BACKWARD) == 0) {
      void (*func)(real *, uint32_t *, const int, const real *, const int, const real *, const int) = BUTF_[config][p->isa][N];
      (*func)(d, p->perm[level], log2len-level, s, log2len-N, (const real *)p->tbl[N][level], K);
    } else {
      void (*func)(real *, uint32_t *, const int, const real *, const int, const real *, const int) = BUTB_[config][p->isa][N];
      (*func)(d, p->perm[level], log2len-level, s, log2len-N, (const real *)p->tbl[N][level], K);
    }
  }
}

// Transposer

#define LOG2BS 4

#define BS (1 << LOG2BS)
#define TRANSPOSE_BLOCK(y2) do {					\
    for(int x2=y2+1;x2<BS;x2++) {					\
      element_t r = *(element_t *)&row[y2].r[x2*2+0];			\
      *(element_t *)&row[y2].r[x2*2+0] = *(element_t *)&row[x2].r[y2*2+0]; \
      *(element_t *)&row[x2].r[y2*2+0] = r;				\
    }} while(0)

template<typename real>
static void transpose(real *RESTRICT ALIGNED(256) d, real *RESTRICT ALIGNED(256) s, const int log2n, const int log2m) {
  if (log2n < LOG2BS || log2m < LOG2BS) {
    for(int y=0;y<(1 << log2n);y++) {
      for(int x=0;x<(1 << log2m);x++) {
	real r0 = s[((y << log2m)+x)*2+0];
	real r1 = s[((y << log2m)+x)*2+1];
	d[((x << log2n)+y)*2+0] = r0;
	d[((x << log2n)+y)*2+1] = r1;
      }
    }
  } else {
#if defined(__GNUC__) && !defined(__clang__)
    typedef struct { real __attribute__((vector_size(sizeof(real)*BS*2))) r; } row_t;
    typedef struct { real __attribute__((vector_size(sizeof(real)*2))) r; } element_t;
#else
    typedef struct { real r[BS*2]; } row_t;
    typedef struct { real r0, r1; } element_t;
#endif
    for(int y=0;y<(1 << log2n);y+=BS) {
      for(int x=0;x<(1 << log2m);x+=BS) {
	row_t row[BS];
	for(int y2=0;y2<BS;y2++) {
	  row[y2] = *(row_t *)&s[(((y+y2) << log2m)+x)*2];
	}

	TRANSPOSE_BLOCK( 0); TRANSPOSE_BLOCK( 1);
	TRANSPOSE_BLOCK( 2); TRANSPOSE_BLOCK( 3);
	TRANSPOSE_BLOCK( 4); TRANSPOSE_BLOCK( 5);
	TRANSPOSE_BLOCK( 6); TRANSPOSE_BLOCK( 7);
	TRANSPOSE_BLOCK( 8); TRANSPOSE_BLOCK( 9);
	TRANSPOSE_BLOCK(10); TRANSPOSE_BLOCK(11);
	TRANSPOSE_BLOCK(12); TRANSPOSE_BLOCK(13);
	TRANSPOSE_BLOCK(14); TRANSPOSE_BLOCK(15);

	for(int y2=0;y2<BS;y2++) {
	  *(row_t *)&d[(((x+y2) << log2n)+y)*2] = row[y2];
	}
      }
    }
  }
}

template<typename real>
static void transposeMT(real *RESTRICT ALIGNED(256) d, real *RESTRICT ALIGNED(256) s, int log2n, int log2m) {
  if (log2n < LOG2BS || log2m < LOG2BS) {
    for(int y=0;y<(1 << log2n);y++) {
      for(int x=0;x<(1 << log2m);x++) {
	real r0 = s[((y << log2m)+x)*2+0];
	real r1 = s[((y << log2m)+x)*2+1];
	d[((x << log2n)+y)*2+0] = r0;
	d[((x << log2n)+y)*2+1] = r1;
      }
    }
  } else {
#if defined(__GNUC__) && !defined(__clang__)
    typedef struct { real __attribute__((vector_size(sizeof(real)*BS*2))) r; } row_t;
    typedef struct { real __attribute__((vector_size(sizeof(real)*2))) r; } element_t;
#else
    typedef struct { real r[BS*2]; } row_t;
    typedef struct { real r0, r1; } element_t;
#endif
    int y=0;
#pragma omp parallel for
    for(y=0;y<(1 << log2n);y+=BS) {
      for(int x=0;x<(1 << log2m);x+=BS) {
	row_t row[BS];
	for(int y2=0;y2<BS;y2++) {
	  row[y2] = *(row_t *)&s[(((y+y2) << log2m)+x)*2];
	}

	TRANSPOSE_BLOCK( 0); TRANSPOSE_BLOCK( 1);
	TRANSPOSE_BLOCK( 2); TRANSPOSE_BLOCK( 3);
	TRANSPOSE_BLOCK( 4); TRANSPOSE_BLOCK( 5);
	TRANSPOSE_BLOCK( 6); TRANSPOSE_BLOCK( 7);
	TRANSPOSE_BLOCK( 8); TRANSPOSE_BLOCK( 9);
	TRANSPOSE_BLOCK(10); TRANSPOSE_BLOCK(11);
	TRANSPOSE_BLOCK(12); TRANSPOSE_BLOCK(13);
	TRANSPOSE_BLOCK(14); TRANSPOSE_BLOCK(15);

	for(int y2=0;y2<BS;y2++) {
	  *(row_t *)&d[(((x+y2) << log2n)+y)*2] = row[y2];
	}
      }
    }
  }
}

// Table generator

template<typename real, typename real2, real2 (*SINCOSPI_)(real)>
static real2 r2coefsc(int i, int log2len, int level) {
  return (*SINCOSPI_)((i & ((-1 << (log2len - level)) & ~(-1 << log2len))) * ((real)1.0/(1 << (log2len-1))));
}

template<typename real, typename real2, real2 (*SINCOSPI_)(real)>
static real2 srcoefsc(int i, int log2len, int level) {
  return (*SINCOSPI_)(((3*(i & (-1 << (log2len - level)))) & ~(-1 << log2len)) * ((real)1.0/(1 << (log2len-1))));
}

template<typename real, typename real2, real2 (*SINCOSPI_)(real)>
static int makeTableRecurse(real *x, int *p, const int log2len, const int levelorg, const int levelinc, const int sign, const int top, const int bot, const int N, int cnt) {
  if (levelinc >= N-1) return cnt;
  const int level = levelorg - levelinc;
  if (bot - top > 4) {
    const int bl = 1 << (N - levelinc);
    const int w = bl/4;
    for(int j=0;j<(bot-top)/bl;j++) {
      for(int i=0;i<w;i++) {
	int a = sign*(p[(levelinc << N) + top+bl*j+i] & (-1 << (log2len - level)));
	real2 sc;
	sc = r2coefsc<real, real2, SINCOSPI_>(a, log2len, level);
	x[cnt++] = -sc.x; x[cnt++] = -sc.y; 
	sc = srcoefsc<real, real2, SINCOSPI_>(a, log2len, level);
	x[cnt++] = -sc.x; x[cnt++] = -sc.y; 
      }
      cnt = makeTableRecurse<real, real2, SINCOSPI_>(x, p, log2len, levelorg, levelinc+1, sign, top+bl*j       , top+bl*j + bl/2, N, cnt);
      cnt = makeTableRecurse<real, real2, SINCOSPI_>(x, p, log2len, levelorg, levelinc+2, sign, top+bl*j + bl/2, top+bl*j + bl  , N, cnt);
    }
  } else if (bot - top == 4) {
    int a = sign*(p[(levelinc << N) + top] & (-1 << (log2len - level)));
    real2 sc;
    sc = r2coefsc<real, real2, SINCOSPI_>(a, log2len, level);
    x[cnt++] = -sc.x; x[cnt++] = -sc.y; 
    sc = srcoefsc<real, real2, SINCOSPI_>(a, log2len, level);
    x[cnt++] = -sc.x; x[cnt++] = -sc.y; 
  }

  return cnt;
}

static uint32_t perm(int nbits, uint32_t k, int s, int d) {
  s = MIN(MAX(s, 0), nbits);
  d = MIN(MAX(d, 0), nbits);
  uint32_t r;
  r = (((k & 0xaaaaaaaa) >> 1) | ((k & 0x55555555) << 1));
  r = (((r & 0xcccccccc) >> 2) | ((r & 0x33333333) << 2));
  r = (((r & 0xf0f0f0f0) >> 4) | ((r & 0x0f0f0f0f) << 4));
  r = (((r & 0xff00ff00) >> 8) | ((r & 0x00ff00ff) << 8));
  r = ((r >> 16) | (r << 16)) >> (32-nbits);

  return (((r << s) | (k & ~(-1 << s))) & ~(-1 << d)) |
    ((((k >> s) | (r & (-1 << (nbits-s)))) << d) & ~(-1 << nbits));
}

template<typename real, typename real2, real2 (*SINCOSPI_)(real)>
static real **makeTable(int sign, int vecwidth, int log2len, const int N, const int K) {
  if (log2len < N) return NULL;

  int *p = (int *)malloc(sizeof(int)*((N+1)<<N));
  
  real **tbl = (real **)calloc(sizeof(real *), (log2len+1));

  for(int level=N;level<=log2len;level++) {
    if (level == log2len && (1 << (log2len-N)) < vecwidth) { tbl[level] = NULL; continue; }

    int tblOffset = 0;
    tbl[level] = (real *)Sleef_malloc(sizeof(real) * (K << (level-N)));

    for(int i0=0;i0 < (1 << (log2len-N));i0+=(1 << (log2len - level))) {
      for(int j=0;j<N+1;j++) {
	for(int i=0;i<(1 << N);i++) {
	  p[(j << N) + i] = perm(log2len, i0 + (i << (log2len-N)), log2len-level, log2len-(level-j));
	}
      }

      int a = -sign*(p[((N-1) << N) + 0] & (-1 << (log2len - level)));
      real2 sc = r2coefsc<real, real2, SINCOSPI_>(a, log2len, level-N+1);
      tbl[level][tblOffset++] = sc.y; tbl[level][tblOffset++] = sc.x;
      
      tblOffset = makeTableRecurse<real, real2, SINCOSPI_>(tbl[level], p, log2len, level, 0, sign, 0, 1 << N, N, tblOffset);
    }

    if (level == log2len) {
      real *atbl = (real *)Sleef_malloc(sizeof(real)*(K << (log2len-N))*2);
      tblOffset = 0;
      while(tblOffset < (K << (log2len-N))) {
	for(int k=0;k < K;k++) {
	  for(int v = 0;v < vecwidth;v++) {
	    assert((tblOffset + k * vecwidth + v)*2 + 1 < (K << (log2len-N))*2);
	    atbl[(tblOffset + k * vecwidth + v)*2 + 0] = tbl[log2len][tblOffset + v * K + k];
	    atbl[(tblOffset + k * vecwidth + v)*2 + 1] = tbl[log2len][tblOffset + v * K + k];
	  }
	}
	tblOffset += K * vecwidth;
      }
      Sleef_free(tbl[log2len]);
      tbl[log2len] = atbl;
    }
  }

  free(p);
  
  return tbl;
}

// Random planner (for debugging)

static int searchForRandomPathRecurse(SleefDFT *p, int level, int *path, int *pathConfig, uint64_t tm, int nTrial) {
  if (level == 0) {
    p->bestTime = tm;
    for(uint32_t j = 0;j < p->log2len+1;j++) {
      p->bestPathConfig[j] = pathConfig[j];
      p->bestPath[j] = path[j];
    }
    return nTrial;
  }

  if (level < 1) return nTrial-1;
  
  for(int i=0;i<10;i++) {
    int N;

    do {
      N = 1 + rand() % MAXBUTWIDTH;
    } while(p->tm[0][level*(MAXBUTWIDTH+1)+N] >= 1ULL << 60);

    if (p->vecwidth > (1 << N) || N == p->log2len) continue;

    path[level] = N;
    for(;;) {
      pathConfig[level] = rand() % CONFIGMAX;
#if ENABLE_STREAM == 0
      pathConfig[level] &= ~1;
#endif
      if ((p->mode2 & SLEEF_MODE2_MT1D) == 0 && (pathConfig[level] & CONFIG_MT) != 0) continue;
      break;
    }
    for(int j = level-1;j >= 0;j--) path[j] = 0;
    nTrial = searchForRandomPathRecurse(p, level - N, path, pathConfig, 0, nTrial);
    if (nTrial <= 0) break;
    if (p->bestTime < 1ULL << 60) break;
  }

  return nTrial - 1;
}

// Planner

#define NSHORTESTPATHS 15
#define MAXPATHLEN (MAXLOG2LEN+1)
#define POSMAX (CONFIGMAX * MAXLOG2LEN * (MAXBUTWIDTH+1))

static int cln2pos(int config, int level, int N) { return (config * MAXLOG2LEN + level) * MAXBUTWIDTH + N; }
static int pos2config(int pos) { return pos == -1 ? -1 : ((pos - 1) / (MAXBUTWIDTH * MAXLOG2LEN)); }
static int pos2level(int pos) { return pos == -1 ? -1 : (((pos - 1) / MAXBUTWIDTH) % MAXLOG2LEN); }
static int pos2N(int pos) { return pos == -1 ? -1 : ((pos - 1) % MAXBUTWIDTH + 1); }

struct ks_t {
  SleefDFT *p;

  int countu[POSMAX];
  int path[NSHORTESTPATHS][MAXPATHLEN];
  int pathLen[NSHORTESTPATHS];
  uint64_t cost[NSHORTESTPATHS];
  int nPaths;

  int heapSize, nPathsInHeap;
  int *heap;
  uint64_t *heapCost;
  int *heapLen;

  ks_t(SleefDFT *p_) :
    p(p_), nPaths(0), heapSize(10), nPathsInHeap(0),
    heap((int *)calloc(heapSize, sizeof(int)*MAXPATHLEN)),
    heapCost((uint64_t *)calloc(heapSize, sizeof(uint64_t))), 
    heapLen((int *)calloc(heapSize, sizeof(int))) {
    memset(countu, 0, sizeof(countu));
    memset(path, 0, sizeof(path));
    memset(pathLen, 0, sizeof(pathLen));
    memset(cost, 0, sizeof(cost));
  }

  ~ks_t() {
    free(heapCost);
    free(heapLen);
    free(heap);
  }

  /** returns the number of paths in the heap */
  int ksSize() { return nPathsInHeap; }

  /** returns the cost of n-th paths in the heap */
  uint64_t ksCost(int n) {
    assert(0 <= n && n < nPathsInHeap);
    return heapCost[n];
  }

  /** adds a path to the heap */
  void ksAddPath(int *path, int pathLen, uint64_t cost) {
    assert(pathLen <= MAXPATHLEN);

    if (nPathsInHeap == heapSize) {
      heapSize *= 2;
      heap = (int *)realloc(heap, heapSize * sizeof(int)*MAXPATHLEN);
      heapCost = (uint64_t *)realloc(heapCost, heapSize * sizeof(uint64_t));
      heapLen = (int *)realloc(heapLen, heapSize * sizeof(int));
    }

    for(int i=0;i<pathLen;i++) heap[nPathsInHeap * MAXPATHLEN + i] = path[i];
    heapLen[nPathsInHeap] = pathLen;
    heapCost[nPathsInHeap] = cost;
    nPathsInHeap++;
  }

  /** copies the n-th paths in the heap to path, returns its length */
  int ksGetPath(int *path, int n) {
    assert(0 <= n && n < nPathsInHeap);
    int len = heapLen[n];
    for(int i=0;i<len;i++) path[i] = heap[n * MAXPATHLEN + i];
    return len;
  }

  /** removes the n-th paths in the heap */
  void ksRemove(int n) {
    assert(0 <= n && n < nPathsInHeap);

    for(int i=n;i<nPathsInHeap-1;i++) {
      int len = heapLen[i+1];
      assert(len < MAXPATHLEN);
      for(int j=0;j<len;j++) heap[i * MAXPATHLEN + j] = heap[(i+1) * MAXPATHLEN + j];
      heapLen[i] = heapLen[i+1];
      heapCost[i] = heapCost[i+1];
    }
    nPathsInHeap--;
  }

  /** returns the countu value at pos */
  int ksCountu(int pos) {
    assert(0 <= pos && pos < POSMAX);
    return countu[pos];
  }

  /** set the countu value at pos to n */
  void ksSetCountu(int pos, int n) {
    assert(0 <= pos && pos < POSMAX);
    countu[pos] = n;
  }

  /** adds a path as one of the best k paths, returns the number best paths */
  int ksAddBestPath(int *path_, int pathLen_, uint64_t cost_) {
    assert(pathLen_ <= MAXPATHLEN);
    assert(nPaths < NSHORTESTPATHS);
    for(int i=0;i<pathLen_;i++) path[nPaths][i] = path_[i];
    pathLen[nPaths] = pathLen_;
    cost[nPaths] = cost_;
    nPaths++;
    return nPaths;
  }

  /** returns if pos is a destination */
  int ksIsDest(int pos) { return pos2level(pos) == 0; }

  /** returns n-th adjacent nodes at pos */
  int ksAdjacent(int pos, int n) {
    if (pos != -1 && pos2level(pos) == 0) return -1;

    int NMAX = MIN(MIN(p->log2len, MAXBUTWIDTH+1), p->log2len - p->log2vecwidth + 1);

    if (pos == -1) {
      int N = n / 2 + MAX(p->log2vecwidth, 1);
      if (N >= NMAX) return -1;
      return cln2pos((n & 1) * CONFIG_MT, p->log2len, N);
    }

    int config = (pos2config(pos) & CONFIG_MT);
    int N = n + 1;
    int level = pos2level(pos) - pos2N(pos);

    if (level < 0 || N >= NMAX) return -1;
    if (level == 0) return n == 0 ? cln2pos(0, 0, 0) : -1;

    return cln2pos(config, level, N);
  }

  uint64_t ksAdjacentCost(int pos, int n) {
    int nxpos = ksAdjacent(pos, n);
    if (nxpos == -1) return 0;
    int config = pos2config(nxpos), level = pos2level(nxpos), N = pos2N(nxpos);
    uint64_t ret0 = p->tm[config | 0][level*(MAXBUTWIDTH+1) + N];
    uint64_t ret1 = p->tm[config | 1][level*(MAXBUTWIDTH+1) + N];
    return MIN(ret0, ret1);
  }
};

//

template<typename real,
  void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
  void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
  void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
  void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
  void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
  void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int)>
static void searchForBestPath(SleefDFT *p) {
  ks_t *q = new ks_t(p);

  for(int i=0;;i++) {
    int v = q->ksAdjacent(-1, i);
    if (v == -1) break;
    uint64_t c = q->ksAdjacentCost(-1, i);
    int path[1] = { v };
    q->ksAddPath(path, 1, c);
  }

  while(q->ksSize() != 0) {
    uint64_t bestCost = 1ULL << 60;
    int bestPathNum = -1;

    for(int i=0;i<q->ksSize();i++) {
      if (q->ksCost(i) < bestCost) {
	bestCost = q->ksCost(i);
	bestPathNum = i;
      }
    }
    if (bestPathNum == -1) break;

    int path[MAXPATHLEN];
    int pathLen = q->ksGetPath(path, bestPathNum);
    uint64_t cost = q->ksCost(bestPathNum);
    q->ksRemove(bestPathNum);

    int lastPos = path[pathLen-1];
    if (q->ksCountu(lastPos) >= NSHORTESTPATHS) continue;
    q->ksSetCountu(lastPos, q->ksCountu(lastPos)+1);

    if (q->ksIsDest(lastPos)) {
      if (q->ksAddBestPath(path, pathLen, cost) >= NSHORTESTPATHS) break;
      continue;
    }

    for(int i=0;;i++) {
      int v = q->ksAdjacent(lastPos, i);
      if (v == -1) break;
      assert(0 <= pos2N(v) && pos2N(v) <= q->p->log2len);
      uint64_t c = q->ksAdjacentCost(lastPos, i);
      path[pathLen] = v;
      q->ksAddPath(path, pathLen+1, cost + c);
    }
  }

  for(int j = p->log2len;j >= 0;j--) p->bestPath[j] = 0;

  if (((p->mode & SLEEF_MODE_MEASURE) != 0 || (planFilePathSet && (p->mode & SLEEF_MODE_MEASUREBITS) == 0))) {
    uint64_t besttm = 1ULL << 62;
    int bestPath = -1;
    const int niter =  1 + 5000000 / ((1 << p->log2len) + 1);

    real *s2 = NULL, *d2 = NULL;
    const real *s = p->in  == NULL ? (s2 = (real *)memset(Sleef_malloc((2 << p->log2len) * sizeof(real)), 0, sizeof(real) * (2 << p->log2len))) : (const real *)p->in;
    real       *d = p->out == NULL ? (d2 = (real *)memset(Sleef_malloc((2 << p->log2len) * sizeof(real)), 0, sizeof(real) * (2 << p->log2len))) : (real *)p->out;

    const int tn = omp_get_thread_num();

    real *t[] = { (real *)p->x1[tn], (real *)p->x0[tn], d };

    for(int mt=0;mt<2;mt++) {
      for(int i=q->nPaths-1;i>=0;i--) {
	if (((pos2config(q->path[i][0]) & CONFIG_MT) != 0) != mt) continue;

	if ((p->mode & SLEEF_MODE_VERBOSE) != 0) {
	  for(int j=0;j<q->pathLen[i];j++) {
	    int N = pos2N(q->path[i][j]);
	    int level = pos2level(q->path[i][j]);
	    int config = pos2config(q->path[i][j]) & ~1;
	    uint64_t t0 = q->p->tm[config | 0][level*(MAXBUTWIDTH+1) + N];
	    uint64_t t1 = q->p->tm[config | 1][level*(MAXBUTWIDTH+1) + N];
	    config = t0 < t1 ? config : (config | 1);

	    if (N != 0) printf("%d(%s) ", N, configStr[config]);
	  }
	}

	if (mt) startAllThreads(p->nThread);

	uint64_t tm0 = Sleef_currentTimeMicros();
	for(int k=0;k<niter;k++) {
	  int nb = 0;
	  const real *lb = s;
	  if ((p->pathLen & 1) == 1) nb = -1;
	  for(int level = p->log2len, j=0;level >= 1;j++) {
	    assert(pos2level(q->path[i][j]) == level);
	    int N = pos2N(q->path[i][j]);
	    int config = pos2config(q->path[i][j]) & ~1;
	    uint64_t t0 = q->p->tm[config | 0][level*(MAXBUTWIDTH+1) + N];
	    uint64_t t1 = q->p->tm[config | 1][level*(MAXBUTWIDTH+1) + N];
	    config = t0 < t1 ? config : (config | 1);
	    dispatch<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, N, t[nb+1], lb, level, config);
	    level -= N;
	    lb = t[nb+1];
	    nb = (nb + 1) & 1;
	  }
	}
	uint64_t tm1 = Sleef_currentTimeMicros();
	for(int k=0;k<niter;k++) {
	  int nb = 0;
	  const real *lb = s;
	  if ((p->pathLen & 1) == 1) nb = -1;
	  for(int level = p->log2len, j=0;level >= 1;j++) {
	    assert(pos2level(q->path[i][j]) == level);
	    int N = pos2N(q->path[i][j]);
	    int config = pos2config(q->path[i][j]) & ~1;
	    uint64_t t0 = q->p->tm[config | 0][level*(MAXBUTWIDTH+1) + N];
	    uint64_t t1 = q->p->tm[config | 1][level*(MAXBUTWIDTH+1) + N];
	    config = t0 < t1 ? config : (config | 1);
	    dispatch<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, N, t[nb+1], lb, level, config);
	    level -= N;
	    lb = t[nb+1];
	    nb = (nb + 1) & 1;
	  }
	}
	uint64_t tm2 = Sleef_currentTimeMicros();

	if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf(" : %lld %lld\n", (long long int)(tm1 - tm0), (long long int)(tm2 - tm1));
	if ((tm1 - tm0) < besttm) {
	  bestPath = i;
	  besttm = tm1 - tm0;
	}
	if ((tm2 - tm1) < besttm) {
	  bestPath = i;
	  besttm = tm2 - tm1;
	}
      }
    }

    for(int level = p->log2len, j=0;level >= 1;j++) {
      assert(pos2level(q->path[bestPath][j]) == level);
      int N = pos2N(q->path[bestPath][j]);

      int config = pos2config(q->path[bestPath][j]) & ~1;
      uint64_t t0 = q->p->tm[config | 0][level*(MAXBUTWIDTH+1) + N];
      uint64_t t1 = q->p->tm[config | 1][level*(MAXBUTWIDTH+1) + N];
      config = t0 < t1 ? config : (config | 1);

      p->bestPath[level] = N;
      p->bestPathConfig[level] = config;
      level -= N;
    }

    if (d2 != NULL) Sleef_free(d2);
    if (s2 != NULL) Sleef_free(s2);
  } else {
    for(int level = p->log2len, j=0;level >= 1;j++) {
      int bestPath = 0;
      assert(pos2level(q->path[bestPath][j]) == level);
      int N = pos2N(q->path[bestPath][j]);
      int config = pos2config(q->path[bestPath][j]);
      p->bestPath[level] = N;
      p->bestPathConfig[level] = config;
      level -= N;
    }
  }

  delete q;
}

//

static uint64_t estimate(int log2len, int level, int N, int config) {
  uint64_t ret = N * 1000 + ABS(N-3) * 1000;
  if (log2len >= 14 && (config & CONFIG_MT) != 0) ret /= 2;
  return ret;
}

template<typename real,
  void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
  void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
  void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
  void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
  void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
  void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int)>
static void measureBut(SleefDFT *p) {
  if (p->x0 == NULL) return;

  //

  const int tn = omp_get_thread_num();

  real *s = (real *)memset(p->x0[tn], 0, sizeof(real) * (2 << p->log2len));
  real *d = (real *)memset(p->x1[tn], 0, sizeof(real) * (2 << p->log2len));

  const int niter =  1 + 100000 / ((1 << p->log2len) + 1);

#define MEASURE_REPEAT 4

  for(int rep=1;rep<=MEASURE_REPEAT;rep++) {
    for(int config=0;config<CONFIGMAX;config++) {
#if ENABLE_STREAM == 0
      if ((config & 1) != 0) continue;
#endif
      if ((p->mode2 & SLEEF_MODE2_MT1D) == 0 && (config & CONFIG_MT) != 0) continue;
      for(uint32_t level = p->log2len;level >= 1;level--) {
	for(uint32_t N=1;N<=MAXBUTWIDTH;N++) {
	  if (level < N || p->log2len <= N) continue;
	  if (level == N) {
	    if ((int)p->log2len - (int)level < p->log2vecwidth) continue;

	    uint64_t tm = Sleef_currentTimeMicros();
	    for(int i=0;i<niter*2;i++) {
	      dispatch<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, N, d, s, level, config);
	    }
	    tm = Sleef_currentTimeMicros() - tm + 1;
	    p->tm[config][level*(MAXBUTWIDTH+1)+N] = MIN(p->tm[config][level*(MAXBUTWIDTH+1)+N], tm);
	  } else if (level == p->log2len) {
	    if (p->tbl[N] == NULL || p->tbl[N][level] == NULL) continue;
	    if (p->vecwidth > (1 << N)) continue;
	    if ((config & CONFIG_MT) != 0) {
	      int i1=0;

#pragma omp parallel for
	      for(i1=0;i1 < (1 << (p->log2len-N-p->log2vecwidth));i1++) {
		int i0 = i1 << p->log2vecwidth;
		p->perm[level][i1] = 2*perm(p->log2len, i0, p->log2len-level, p->log2len-(level-N));
	      }
	    } else {
	      for(int i0=0, i1=0;i0 < (1 << (p->log2len-N));i0+=p->vecwidth, i1++) {
		p->perm[level][i1] = 2*perm(p->log2len, i0, p->log2len-level, p->log2len-(level-N));
	      }
	    }

	    uint64_t tm = Sleef_currentTimeMicros();
	    for(int i=0;i<niter;i++) {
	      dispatch<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, N, d, s, level, config);
	      dispatch<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, N, s, d, level, config);
	    }
	    tm = Sleef_currentTimeMicros() - tm + 1;
	    p->tm[config][level*(MAXBUTWIDTH+1)+N] = MIN(p->tm[config][level*(MAXBUTWIDTH+1)+N], tm);
	  } else {
	    if (p->tbl[N] == NULL || p->tbl[N][level] == NULL) continue;
	    if (p->vecwidth > 2 && p->log2len <= N+2) continue;
	    if ((int)p->log2len - (int)level < p->log2vecwidth) continue;
	    if ((config & CONFIG_MT) != 0) {
	      int i1=0;

#pragma omp parallel for
	      for(i1=0;i1 < (1 << (p->log2len-N-p->log2vecwidth));i1++) {
		int i0 = i1 << p->log2vecwidth;
		p->perm[level][i1] = 2*perm(p->log2len, i0, p->log2len-level, p->log2len-(level-N));
	      }
	    } else {
	      for(int i0=0, i1=0;i0 < (1 << (p->log2len-N));i0+=p->vecwidth, i1++) {
		p->perm[level][i1] = 2*perm(p->log2len, i0, p->log2len-level, p->log2len-(level-N));
	      }
	    }

	    uint64_t tm = Sleef_currentTimeMicros();
	    for(int i=0;i<niter;i++) {
	      dispatch<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, N, d, s, level, config);
	      dispatch<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, N, s, d, level, config);
	    }
	    tm = Sleef_currentTimeMicros() - tm + 1;
	    p->tm[config][level*(MAXBUTWIDTH+1)+N] = MIN(p->tm[config][level*(MAXBUTWIDTH+1)+N], tm);
	  }
	}
      }
    }
  }

  if ((p->mode & SLEEF_MODE_VERBOSE) != 0) {
    for(uint32_t level = p->log2len;level >= 1;level--) {
      for(uint32_t N=1;N<=MAXBUTWIDTH;N++) {
	if (level < N || p->log2len <= N) continue;
	if (level == N) {
	  if ((int)p->log2len - (int)level < p->log2vecwidth) continue;
	  printf("bot %d, %d, %d, ", p->log2len, level, N);
	  for(int config=0;config<CONFIGMAX;config++) {
	    if (p->tm[config][level*(MAXBUTWIDTH+1)+N] == 1ULL << 60) {
	      printf("N/A, ");
	    } else {
	      printf("%lld, ", (long long int)p->tm[config][level*(MAXBUTWIDTH+1)+N]);
	    }
	  }
	  printf("\n");
	} else if (level == p->log2len) {
	  if (p->tbl[N] == NULL || p->tbl[N][level] == NULL) continue;
	  if (p->vecwidth > (1 << N)) continue;
	  printf("top %d, %d, %d, ", p->log2len, level, N);
	  for(int config=0;config<CONFIGMAX;config++) {
	    if (p->tm[config][level*(MAXBUTWIDTH+1)+N] == 1ULL << 60) {
	      printf("N/A, ");
	    } else {
	      printf("%lld, ", (long long int)p->tm[config][level*(MAXBUTWIDTH+1)+N]);
	    }
	  }
	  printf("\n");
	} else {
	  if (p->tbl[N] == NULL || p->tbl[N][level] == NULL) continue;
	  if (p->vecwidth > 2 && p->log2len <= N+2) continue;
	  if ((int)p->log2len - (int)level < p->log2vecwidth) continue;
	  printf("mid %d, %d, %d, ", p->log2len, level, N);
	  for(int config=0;config<CONFIGMAX;config++) {
	    if (p->tm[config][level*(MAXBUTWIDTH+1)+N] == 1ULL << 60) {
	      printf("N/A, ");
	    } else {
	      printf("%lld, ", (long long int)p->tm[config][level*(MAXBUTWIDTH+1)+N]);
	    }
	  }
	  printf("\n");
	}
      }
    }
  }
}

static void estimateBut(SleefDFT *p) {
  for(uint32_t level = p->log2len;level >= 1;level--) {
    for(uint32_t N=1;N<=MAXBUTWIDTH;N++) {
      if (level < N || p->log2len <= N) continue;
      if (level == N) {
	if ((int)p->log2len - (int)level < p->log2vecwidth) continue;
	for(int config=0;config<CONFIGMAX;config++) {
#if ENABLE_STREAM == 0
	  if ((config & 1) != 0) continue;
#endif
	  p->tm[config][level*(MAXBUTWIDTH+1)+N] = estimate(p->log2len, level, N, config);
	}
      } else if (level == p->log2len) {
	if (p->tbl[N] == NULL || p->tbl[N][level] == NULL) continue;
	if (p->vecwidth > (1 << N)) continue;
	for(int config=0;config<CONFIGMAX;config++) {
#if ENABLE_STREAM == 0
	  if ((config & 1) != 0) continue;
#endif
	  p->tm[config][level*(MAXBUTWIDTH+1)+N] = estimate(p->log2len, level, N, config);
	}
      } else {
	if (p->tbl[N] == NULL || p->tbl[N][level] == NULL) continue;
	if (p->vecwidth > 2 && p->log2len <= N+2) continue;
	if ((int)p->log2len - (int)level < p->log2vecwidth) continue;
	for(int config=0;config<CONFIGMAX;config++) {
#if ENABLE_STREAM == 0
	  if ((config & 1) != 0) continue;
#endif
	  p->tm[config][level*(MAXBUTWIDTH+1)+N] = estimate(p->log2len, level, N, config);
	}
      }
    }
  }
}

template<typename real,
  void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
  void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
  void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
  void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
  void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
  void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int)>
static int measure(SleefDFT *p, int randomize) {
  if (p->log2len == 1) {
    p->bestTime = 1ULL << 60;

    p->pathLen = 1;
    p->bestPath[1] = 1;

    return 1;
  }

  if (PlanManager_loadMeasurementResultsP(p, (p->mode & SLEEF_MODE_NO_MT) != 0 ? 1 : 0)) {
    if ((p->mode & SLEEF_MODE_VERBOSE) != 0) {
      printf("Path(loaded) : ");
      for(int j = p->log2len;j >= 0;j--) if (p->bestPath[j] != 0) printf("%d(%s) ", p->bestPath[j], configStr[p->bestPathConfig[j]]);
      printf("\n");
    }
    
    return 1;
  }
  
  int toBeSaved = 0;

  for(uint32_t level = p->log2len;level >= 1;level--) {
    for(uint32_t N=1;N<=MAXBUTWIDTH;N++) {
      for(int config=0;config<CONFIGMAX;config++) {
	p->tm[config][level*(MAXBUTWIDTH+1)+N] = 1ULL << 60;
      }
    }
  }
  
  if (((p->mode & SLEEF_MODE_MEASURE) != 0 || (planFilePathSet && (p->mode & SLEEF_MODE_MEASUREBITS) == 0)) && !randomize) {
    measureBut<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p);
    toBeSaved = 1;
  } else {
    estimateBut(p);
  }

  int executable = 0;
  for(int i=1;i<=MAXBUTWIDTH && !executable;i++) {
    if (p->tm[0][p->log2len*(MAXBUTWIDTH+1)+i] < (1ULL << 60)) executable = 1;
  }

  if (!executable) return 0;

  p->bestTime = 1ULL << 60;

  p->bestPath[p->log2len] = 0;
  
  if (!randomize) {
    searchForBestPath<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p);
  } else {
    int path[MAXLOG2LEN+1];
    int pathConfig[MAXLOG2LEN+1];
    for(int j = p->log2len;j >= 0;j--) path[j] = pathConfig[j] = 0;

    int nTrial = 100000;
    do {
      nTrial = searchForRandomPathRecurse(p, p->log2len, path, pathConfig, 0, nTrial);
    } while(p->bestTime == 1ULL << 60 && nTrial >= 0);
  }

  if (p->bestPath[p->log2len] == 0) return 0;
  
  p->pathLen = 0;
  for(int j = p->log2len;j >= 0;j--) if (p->bestPath[j] != 0) p->pathLen++;

  if ((p->mode & SLEEF_MODE_VERBOSE) != 0) {
    printf("Path");
    if (randomize) printf("(random) :");
    else if (toBeSaved) printf("(measured) :");
    else printf("(estimated) :");

    for(int j = p->log2len;j >= 0;j--) if (p->bestPath[j] != 0) printf("%d(%s) ", p->bestPath[j], configStr[p->bestPathConfig[j]]);
    printf("\n");
  }

  if (toBeSaved) {
    PlanManager_saveMeasurementResultsP(p, (p->mode & SLEEF_MODE_NO_MT) != 0 ? 1 : 0);
  }
  
  return 1;
}

template<typename real>
static void measureTranspose(SleefDFT *p) {
  if (PlanManager_loadMeasurementResultsT(p)) {
    if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("transpose NoMT(loaded): %lld\n", (long long int)p->tmNoMT);
    if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("transpose   MT(loaded): %lld\n", (long long int)p->tmMT);
    return;
  }

  if ((p->mode & SLEEF_MODE_MEASURE) == 0 && (!planFilePathSet || (p->mode & SLEEF_MODE_MEASUREBITS) != 0)) {
    if (p->log2hlen + p->log2vlen >= 14) {
      p->tmNoMT = 20;
      p->tmMT = 10;
      if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("transpose : selected MT(estimated)\n");
    } else {
      p->tmNoMT = 10;
      p->tmMT = 20;
      if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("transpose : selected NoMT(estimated)\n");
    }
    return;
  }
  
  real *tBuf2 = (real *)Sleef_malloc(sizeof(real)*2*p->hlen*p->vlen);

  const int niter =  1 + 5000000 / (p->hlen * p->vlen + 1);
  uint64_t tm;

  tm = Sleef_currentTimeMicros();
  for(int i=0;i<niter;i++) {
    transpose<real>(tBuf2, (real *)p->tBuf, p->log2hlen, p->log2vlen);
    transpose<real>(tBuf2, (real *)p->tBuf, p->log2vlen, p->log2hlen);
  }
  p->tmNoMT = Sleef_currentTimeMicros() - tm + 1;

  if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("transpose NoMT(measured): %lld\n", (long long int)p->tmNoMT);

  tm = Sleef_currentTimeMicros();
  for(int i=0;i<niter;i++) {
    transposeMT<real>(tBuf2, (real *)p->tBuf, p->log2hlen, p->log2vlen);
    transposeMT<real>(tBuf2, (real *)p->tBuf, p->log2vlen, p->log2hlen);
  }
  p->tmMT = Sleef_currentTimeMicros() - tm + 1;

  if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("transpose   MT(measured): %lld\n", (long long int)p->tmMT);
  
  Sleef_free(tBuf2);

  PlanManager_saveMeasurementResultsT(p);
}

// Implementation of SleefDFT_*_init1d

template<typename real, typename real2, int BASETYPEID_, int MAGIC_,
	 int (*GETINT_[16])(int), const void *(*GETPTR_[16])(int), real2 (*SINCOSPI_)(real),
	 void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	 void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	 void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	 void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	 void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
	 void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int)>
static SleefDFT *SleefDFT_init1d(uint32_t n, const real *in, real *out, uint64_t mode, const char *baseTypeString) {
  SleefDFT *p = (SleefDFT *)calloc(1, sizeof(SleefDFT));
  p->magic = MAGIC_;
  p->baseTypeID = BASETYPEID_;
  p->in = (const void *)in;
  p->out = (void *)out;
  
  // Mode

  p->mode = mode;

  if ((p->mode & SLEEF_MODE_NO_MT) == 0) {
    p->mode2 |= SLEEF_MODE2_MT1D;
  }
  
  if ((mode & SLEEF_MODE_REAL) != 0) n /= 2;
  p->log2len = ilog2(n);

  if (p->log2len <= 1) return p;

  if ((mode & SLEEF_MODE_ALT) != 0) p->mode = mode = mode ^ SLEEF_MODE_BACKWARD;

  p->nThread = omp_thread_count();

  // ISA availability

  int bestPriority = -1;
  p->isa = -1;

  for(int i=0;i<ISAMAX;i++) {
    if (checkISAAvailability<GETINT_>(i) && bestPriority < (*GETINT_[i])(GETINT_DFTPRIORITY) && n >= (uint32_t)((*GETINT_[i])(GETINT_VECWIDTH) * (*GETINT_[i])(GETINT_VECWIDTH))) {
      bestPriority = (*GETINT_[i])(GETINT_DFTPRIORITY);
      p->isa = i;
    }
  }

  if (p->isa == -1) {
    if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("ISA not available\n");
    p->magic = 0;
    free(p);
    return NULL;
  }

  // Tables
  
  p->perm = (uint32_t **)calloc(sizeof(uint32_t *), p->log2len+1);
  for(int level = p->log2len;level >= 1;level--) {
    p->perm[level] = (uint32_t *)Sleef_malloc(sizeof(uint32_t) * ((1 << p->log2len) + 8));
  }

  p->x0 = (void **)malloc(sizeof(real *) * p->nThread);
  p->x1 = (void **)malloc(sizeof(real *) * p->nThread);

  for(int i=0;i<p->nThread;i++) {
    p->x0[i] = (real *)Sleef_malloc(sizeof(real) * 2 * n);
    p->x1[i] = (real *)Sleef_malloc(sizeof(real) * 2 * n);
  }
  
  if ((mode & SLEEF_MODE_REAL) != 0) {
    p->rtCoef0 = (real *)Sleef_malloc(sizeof(real) * n);
    p->rtCoef1 = (real *)Sleef_malloc(sizeof(real) * n);

    if ((mode & SLEEF_MODE_BACKWARD) == 0) {
      for(uint32_t i=0;i<n/2;i++) {
	real2 sc = SINCOSPI_(i*((real)-1.0/n));
	((real *)p->rtCoef0)[i*2+0] = ((real *)p->rtCoef0)[i*2+1] = (real)0.5 - (real)0.5 * sc.x;
	((real *)p->rtCoef1)[i*2+0] = ((real *)p->rtCoef1)[i*2+1] = (real)0.5*sc.y;
      }
    } else {
      for(uint32_t i=0;i<n/2;i++) {
	real2 sc = SINCOSPI_(i*((real)-1.0/n));
	((real *)p->rtCoef0)[i*2+0] = ((real *)p->rtCoef0)[i*2+1] = (real)0.5 + (real)0.5 * sc.x;
	((real *)p->rtCoef1)[i*2+0] = ((real *)p->rtCoef1)[i*2+1] = (real)0.5*sc.y;
      }
    }
  }

  // Measure
  
  int sign = (mode & SLEEF_MODE_BACKWARD) != 0 ? -1 : 1;
  
  p->vecwidth = (*GETINT_[p->isa])(GETINT_VECWIDTH);
  p->log2vecwidth = ilog2(p->vecwidth);

  for(int i=1;i<=MAXBUTWIDTH;i++) {
    ((real ***)p->tbl)[i] = makeTable<real, real2, SINCOSPI_>(sign, p->vecwidth, p->log2len, i, constK[i]);
  }

  if (!measure<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, (mode & SLEEF_MODE_DEBUG))) {
    // Fall back to the first ISA
    freeTables(p);
    p->isa = 0;

    p->vecwidth = (*GETINT_[p->isa])(GETINT_VECWIDTH);
    p->log2vecwidth = ilog2(p->vecwidth);

    for(int i=1;i<=MAXBUTWIDTH;i++) {
      ((real ***)p->tbl)[i] = makeTable<real, real2, SINCOSPI_>(sign, p->vecwidth, p->log2len, i, constK[i]);
    }

    for(int level = p->log2len;level >= 1;) {
      int N = ABS(p->bestPath[level]);
      if (level == N) { level -= N; continue; }

      int i1 = 0;
      for(int i0=0;i0 < (1 << (p->log2len-N));i0+=p->vecwidth, i1++) {
	p->perm[level][i1] = 2*perm(p->log2len, i0, p->log2len-level, p->log2len-(level-N));
      }
      for(;i1 < (1 << p->log2len) + 8;i1++) p->perm[level][i1] = 0;

      level -= N;
    }  

    if (!measure<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, (mode & SLEEF_MODE_DEBUG))) {
      if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("Suitable ISA not found. This should not happen.\n");
      return NULL;
    }
  }
  
  for(int level = p->log2len;level >= 1;) {
    int N = ABS(p->bestPath[level]);
    if (level == N) { level -= N; continue; }

    int i1 = 0;
    for(int i0=0;i0 < (1 << (p->log2len-N));i0+=p->vecwidth, i1++) {
      p->perm[level][i1] = 2*perm(p->log2len, i0, p->log2len-level, p->log2len-(level-N));
    }
    for(;i1 < (1 << p->log2len) + 8;i1++) p->perm[level][i1] = 0;

    level -= N;
  }  
  
  if ((p->mode & SLEEF_MODE_VERBOSE) != 0) printf("ISA : %s %d bit %s\n", (char *)(*GETPTR_[p->isa])(0), (int)(GETINT_[p->isa](GETINT_VECWIDTH) * sizeof(real) * 16), baseTypeString);

  return p;
}

// Implementation of SleefDFT_*_init2d

template<typename real, typename real2, int BASETYPEID_, int MAGIC_, int MAGIC2D_,
	 int (*GETINT_[16])(int), const void *(*GETPTR_[16])(int), real2 (*SINCOSPI_)(real),
	 void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	 void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	 void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	 void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	 void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
	 void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int)>
static SleefDFT *SleefDFT_init2d(uint32_t vlen, uint32_t hlen, const real *in, real *out, uint64_t mode, const char *baseTypeString) {
  SleefDFT *p = (SleefDFT *)calloc(1, sizeof(SleefDFT));
  p->magic = MAGIC2D_;
  p->mode = mode;
  p->baseTypeID = BASETYPEID_;
  p->in = in;
  p->out = out;
  p->hlen = hlen;
  p->log2hlen = ilog2(hlen);
  p->vlen = vlen;
  p->log2vlen = ilog2(vlen);
  
  uint64_t mode1D = mode;
  mode1D |= SLEEF_MODE_NO_MT;

  if ((mode & SLEEF_MODE_NO_MT) == 0) p->mode3 |= SLEEF_MODE3_MT2D;
  
  p->instH = p->instV = SleefDFT_init1d<real, real2, BASETYPEID_, MAGIC_, GETINT_, GETPTR_, SINCOSPI_,
					DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(hlen, NULL, NULL, mode1D, baseTypeString);
  if (hlen != vlen) p->instV = SleefDFT_init1d<real, real2, BASETYPEID_, MAGIC_, GETINT_, GETPTR_, SINCOSPI_,
					       DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(vlen, NULL, NULL, mode1D, baseTypeString);

  p->tBuf = (void *)Sleef_malloc(sizeof(real)*2*hlen*vlen);

  measureTranspose<real>(p);
  
  return p;
}

// Implementation of SleefDFT_*_execute

template<typename real, int MAGIC_, int MAGIC2D_,
	 void (*DFTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	 void (*DFTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, const real *, const int),
	 void (*TBUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	 void (*TBUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const real *, const int, const real *, const int),
	 void (*BUTF_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
         void (*BUTB_[CONFIGMAX][ISAMAX][MAXBUTWIDTH+1])(real *, uint32_t *, const int, const real *, const int, const real *, const int),
         void (*REALSUB0_[ISAMAX])(real *, const real *, const int, const real *, const real *),
         void (*REALSUB1_[ISAMAX])(real *, const real *, const int, const real *, const real *, const int)>
static void SleefDFT_execute(SleefDFT *p, const real *s0, real *d0) {
  assert(p != NULL && (p->magic == MAGIC_ || p->magic == MAGIC2D_));

  const real *s = s0 == NULL ? (const real *)p->in : s0;
  real *d = d0 == NULL ? (real *)p->out : d0;

  if (p->magic == MAGIC2D_) {
  // S -> T -> D -> T -> D

    real *tBuf = (real *)(p->tBuf);

    if ((p->mode3 & SLEEF_MODE3_MT2D) != 0 &&
	(((p->mode & SLEEF_MODE_DEBUG) == 0 && p->tmMT < p->tmNoMT) ||
	 ((p->mode & SLEEF_MODE_DEBUG) != 0 && (rand() & 1))))
      {
	int y=0;
#pragma omp parallel for
	for(y=0;y<p->vlen;y++) {
	  SleefDFT_execute<real, MAGIC_, MAGIC2D_, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_, REALSUB0_, REALSUB1_>(p->instH, &s[p->hlen*2*y], &tBuf[p->hlen*2*y]);
	}

	transposeMT<real>(d, tBuf, p->log2vlen, p->log2hlen);

#pragma omp parallel for
	for(y=0;y<p->hlen;y++) {
	  SleefDFT_execute<real, MAGIC_, MAGIC2D_, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_, REALSUB0_, REALSUB1_>(p->instV, &d[p->vlen*2*y], &tBuf[p->vlen*2*y]);
	}

	transposeMT<real>(d, tBuf, p->log2hlen, p->log2vlen);
      } else {
	for(int y=0;y<p->vlen;y++) {
	  SleefDFT_execute<real, MAGIC_, MAGIC2D_, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_, REALSUB0_, REALSUB1_>(p->instH, &s[p->hlen*2*y], &tBuf[p->hlen*2*y]);
	}

	transpose<real>(d, tBuf, p->log2vlen, p->log2hlen);

	for(int y=0;y<p->hlen;y++) {
	  SleefDFT_execute<real, MAGIC_, MAGIC2D_, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_, REALSUB0_, REALSUB1_>(p->instV, &d[p->vlen*2*y], &tBuf[p->vlen*2*y]);
	}

	transpose<real>(d, tBuf, p->log2hlen, p->log2vlen);
      }

    return;
  }
  
  if (p->log2len <= 1) {
    if ((p->mode & SLEEF_MODE_REAL) == 0) {
      real r0 = s[0] + s[2];
      real r1 = s[1] + s[3];
      real r2 = s[0] - s[2];
      real r3 = s[1] - s[3];
      d[0] = r0; d[1] = r1; d[2] = r2; d[3] = r3;
    } else {
      if ((p->mode & SLEEF_MODE_ALT) == 0) {
	if (p->log2len == 1) {
	  if ((p->mode & SLEEF_MODE_BACKWARD) == 0) {
	    real r0 = s[0] + s[2] + (s[1] + s[3]);
	    real r1 = s[0] + s[2] - (s[1] + s[3]);
	    real r2 = s[0] - s[2];
	    real r3 = s[3] - s[1];
	    d[0] = r0; d[1] = 0; d[2] = r2; d[3] = r3; d[4] = r1; d[5] = 0;
	  } else {
	    real r0 = (s[0] + s[4])*(real)0.5 + s[2];
	    real r1 = (s[0] - s[4])*(real)0.5 - s[3];
	    real r2 = (s[0] + s[4])*(real)0.5 - s[2];
	    real r3 = (s[0] - s[4])*(real)0.5 + s[3];
	    d[0] = r0*2; d[1] = r1*2; d[2] = r2*2; d[3] = r3*2;
	  }
	} else {
	  if ((p->mode & SLEEF_MODE_BACKWARD) == 0) {
	    real r0 = s[0] + s[1];
	    real r1 = s[0] - s[1];
	    d[0] = r0; d[1] = 0; d[2] = r1; d[3] = 0;
	  } else {
	    real r0 = s[0] + s[2];
	    real r1 = s[0] - s[2];
	    d[0] = r0; d[1] = r1;
	  }
	}
      } else {
	if (p->log2len == 1) {
	  if ((p->mode & SLEEF_MODE_BACKWARD) == 0) {
	    real r0 = s[0] + s[2] + (s[1] + s[3]);
	    real r1 = s[0] + s[2] - (s[1] + s[3]);
	    real r2 = s[0] - s[2];
	    real r3 = s[1] - s[3];
	    d[0] = r0; d[1] = r1; d[2] = r2; d[3] = r3;
	  } else {
	    real r0 = (s[0] + s[1])*(real)0.5 + s[2];
	    real r1 = (s[0] - s[1])*(real)0.5 + s[3];
	    real r2 = (s[0] + s[1])*(real)0.5 - s[2];
	    real r3 = (s[0] - s[1])*(real)0.5 - s[3];
	    d[0] = r0; d[1] = r1; d[2] = r2; d[3] = r3;
	  }
	} else {
	  real c = ((p->mode & SLEEF_MODE_BACKWARD) != 0) ? (real)0.5 : (real)1.0;
	  real r0 = s[0] + s[1];
	  real r1 = s[0] - s[1];
	  d[0] = r0 * c; d[1] = r1 * c;
	}
      }
    }
    return;
  }

  //

  const int tn = omp_get_thread_num();
  real *t[] = { (real *)p->x1[tn], (real *)p->x0[tn], d };
  
  const real *lb = s;
  int nb = 0;

  if ((p->mode & SLEEF_MODE_REAL) != 0 && (p->pathLen & 1) == 0 &&
      ((p->mode & SLEEF_MODE_BACKWARD) != 0) != ((p->mode & SLEEF_MODE_ALT) != 0)) nb = -1;
  if ((p->mode & SLEEF_MODE_REAL) == 0 && (p->pathLen & 1) == 1) nb = -1;
  
  if ((p->mode & SLEEF_MODE_REAL) != 0 &&
      ((p->mode & SLEEF_MODE_BACKWARD) != 0) != ((p->mode & SLEEF_MODE_ALT) != 0)) {
    (*REALSUB1_[p->isa])(t[nb+1], s, p->log2len, (const real *)p->rtCoef0, (const real *)p->rtCoef1, (p->mode & SLEEF_MODE_ALT) == 0);
    if ((p-> mode & SLEEF_MODE_ALT) == 0) t[nb+1][(1 << p->log2len)+1] = -s[(1 << p->log2len)+1] * 2;
    lb = t[nb+1];
    nb = (nb + 1) & 1;
  }

  for(int level = p->log2len;level >= 1;) {
    int N = ABS(p->bestPath[level]), config = p->bestPathConfig[level];
    dispatch<real, DFTF_, DFTB_, TBUTF_, TBUTB_, BUTF_, BUTB_>(p, N, t[nb+1], lb, level, config);
    level -= N;
    lb = t[nb+1];
    nb = (nb + 1) & 1;
  }

  if ((p->mode & SLEEF_MODE_REAL) != 0 && 
      ((p->mode & SLEEF_MODE_BACKWARD) == 0) != ((p->mode & SLEEF_MODE_ALT) != 0)) {
    (*REALSUB0_[p->isa])(d, lb, p->log2len, (const real *)p->rtCoef0, (const real *)p->rtCoef1);
    if ((p->mode & SLEEF_MODE_ALT) == 0) {
      d[(1 << p->log2len)+1] = -d[(1 << p->log2len)+1];
      d[(2 << p->log2len)+0] =  d[1];
      d[(2 << p->log2len)+1] =  0;
      d[1] = 0;
    }
  }
}

//

#if BASETYPEID == 1
typedef Sleef_double2 sc_t;
#define BASETYPESTRING "double"
#define MAGIC 0x27182818
#define MAGIC2D 0x17320508
#define INIT SleefDFT_double_init1d
#define EXECUTE SleefDFT_double_execute
#define INIT2D SleefDFT_double_init2d
#define REALSUB0 realSub0_double
#define REALSUB1 realSub1_double
#define GETINT getInt_double
#define GETPTR getPtr_double
#define DFTF dftf_double
#define DFTB dftb_double
#define TBUTF tbutf_double
#define TBUTB tbutb_double
#define BUTF butf_double
#define BUTB butb_double
#define SINCOSPI Sleef_sincospi_u05
#elif BASETYPEID == 2
typedef Sleef_float2 sc_t;
#define BASETYPESTRING "float"
#define MAGIC 0x31415926
#define MAGIC2D 0x22360679
#define INIT SleefDFT_float_init1d
#define EXECUTE SleefDFT_float_execute
#define INIT2D SleefDFT_float_init2d
#define REALSUB0 realSub0_float
#define REALSUB1 realSub1_float
#define GETINT getInt_float
#define GETPTR getPtr_float
#define DFTF dftf_float
#define DFTB dftb_float
#define TBUTF tbutf_float
#define TBUTB tbutb_float
#define BUTF butf_float
#define BUTB butb_float
#define SINCOSPI Sleef_sincospif_u05
#else
#error No BASETYPEID specified
#endif

EXPORT SleefDFT *INIT(uint32_t n, const real *in, real *out, uint64_t mode) {
  return SleefDFT_init1d<real, sc_t, BASETYPEID, MAGIC, GETINT, GETPTR, SINCOSPI,
			 DFTF, DFTB, TBUTF, TBUTB, BUTF, BUTB>(n, in, out, mode, BASETYPESTRING);
}

EXPORT SleefDFT *INIT2D(uint32_t vlen, uint32_t hlen, const real *in, real *out, uint64_t mode) {
  return SleefDFT_init2d<real, sc_t, BASETYPEID, MAGIC, MAGIC2D, GETINT, GETPTR, SINCOSPI,
			 DFTF, DFTB, TBUTF, TBUTB, BUTF, BUTB>(vlen, hlen, in, out, mode, BASETYPESTRING);
}

EXPORT void EXECUTE(SleefDFT *p, const real *s0, real *d0) {
  SleefDFT_execute<real, MAGIC, MAGIC2D, DFTF, DFTB, TBUTF, TBUTB, BUTF, BUTB, REALSUB0, REALSUB1>(p, s0, d0);
}
