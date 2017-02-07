// Put pi_fft.c in Prof. Takuya Ooura's FFT package.
// http://www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz

#include <malloc.h>
#include "sleef.h"
#include "sleefdft.h"

int ctz(unsigned int v) {
  int c = 32; // c will be the number of zero bits on the right
  if (v) c--;
  if (v & 0x0000FFFF) c -= 16;
  if (v & 0x00FF00FF) c -= 8;
  if (v & 0x0F0F0F0F) c -= 4;
  if (v & 0x33333333) c -= 2;
  if (v & 0x55555555) c -= 1;
  return c;
}

struct SleefDFT *dftf[32], *dftb[32];

void rdft(int n, int isgn, double *a, int *ip, double *w) {
  if (n <= 1) return;

  int c = ctz(n);
  if (dftf[c] == NULL) {
    uint64_t mode = SLEEF_MODE_REAL | SLEEF_MODE_ALT;
    dftf[c] = SleefDFT_double_init1d(n, NULL, NULL, SLEEF_MODE_FORWARD  | mode);
    dftb[c] = SleefDFT_double_init1d(n, NULL, NULL, SLEEF_MODE_BACKWARD | mode);
  }
  
  if (isgn == 1) {
    SleefDFT_double_execute(dftf[c], a, a);
  } else {
    SleefDFT_double_execute(dftb[c], a, a);
  }
}

#define malloc(s) (memalign(256, 256+(s)) + 256 - sizeof(double))
#define free(s) (free(((void *)s) + sizeof(double) - 256))
