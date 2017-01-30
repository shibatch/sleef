// This is part of SLEEF, written by Naoki Shibata. http://shibatch.sourceforge.net

// Since the original code for simplex algorithm is developed by Haruhiko Okumura and
// the code is distributed under the Creative Commons Attribution 4.0 International License,
// the contents under this directory are also distributed under the same license.
// https://creativecommons.org/licenses/by/4.0/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <mpfr.h>

#define PREC 4096
#define EPS 1e-50

void regressMinRelError_fr(int n, int m, mpfr_t **x, mpfr_t *w,mpfr_t *result);

static mpfr_t zero, one;
static mpfr_t fra, frb, frc, frd;
char *mpfrToStr(mpfr_t m);
typedef long double REAL;

//

#if 1
#define N 9           // Degree of equation
#define S 40          // Number of samples for phase 1
#define L 2           // Number of high precision coefficients
#define MIN 0.0       // Min argument
#define MAX (M_PI/2)  // Max argument
#define PMUL 2        // The form of polynomial is y = x^(PADD+PMUL*0) + x^(PADD+PMUL*1) + ...
#define PADD 1

void FRFUNC(mpfr_t ret, mpfr_t a) { mpfr_sin(ret, a, GMP_RNDN); } // The function to approximate
#define FIXCOEF0 1.0  // Fix coef 0 to 1.0
#endif

#if 0
#define N 9
#define S 40
#define L 2
#define MIN 0.0
#define MAX (M_PI/2)
void FRFUNC(mpfr_t ret, mpfr_t a) { // cos(x) - 1
  mpfr_t x;
  mpfr_init(x);
  mpfr_cos(ret, a, GMP_RNDN);
  mpfr_set_ld(x, 1, GMP_RNDN);
  mpfr_sub(ret, ret, x, GMP_RNDN);
  mpfr_clear(x);
}

#define PMUL 2
#define PADD 2
#define FIXCOEF0 (-0.5)
#endif

#if 0
#define N 17
#define S 40
#define L 0
#define MIN 0.0
#define MAX (M_PI/4)
#define PMUL 2
#define PADD 1

void FRFUNC(mpfr_t ret, mpfr_t a) { mpfr_tan(ret, a, GMP_RNDN); }
#define FIXCOEF0 1.0
#endif

#if 0
#define N 9
#define S 40
#define L 2
#define MIN 1 //0.75
#define MAX 1.5
#define PMUL 2
#define PADD 1

#define LOGMODE
void FRFUNC(mpfr_t ret, mpfr_t a) { mpfr_log(ret, a, GMP_RNDN); }

#define FIXCOEF0 2.0
#endif

#if 0
#define N 12
#define S 50
#define L 0
#define MIN -0.347
#define MAX 0.347 // 0.5 log 2
#define PMUL 1
#define PADD 0

void FRFUNC(mpfr_t ret, mpfr_t a) { mpfr_exp(ret, a, GMP_RNDN); }
#define FIXCOEF0 1.0
#define FIXCOEF1 1.0
#define FIXCOEF2 0.5
#endif

#if 0
#define N 22
#define S 100
#define L 2
#define MIN 0.0
#define MAX 1.0
#define PMUL 2
#define PADD 1

void FRFUNC(mpfr_t ret, mpfr_t a) { mpfr_atan(ret, a, GMP_RNDN); }
#define FIXCOEF0 1.0
#endif

static long double xmlal(long double x, long double y, long double z) { return x * y + z; }
static double xmla(double x, double y, double z) { return x * y + z; }

static inline long double func(long double x, long double *coef, int n) {
  long double s = coef[n-1];
  switch(n-1) {
  case 32: s = xmla(s, x, coef[31]);
  case 31: s = xmla(s, x, coef[30]);
  case 30: s = xmla(s, x, coef[29]);
  case 29: s = xmla(s, x, coef[28]);
  case 28: s = xmla(s, x, coef[27]);
  case 27: s = xmla(s, x, coef[26]);
  case 26: s = xmla(s, x, coef[25]);
  case 25: s = xmla(s, x, coef[24]);
  case 24: s = xmla(s, x, coef[23]);
  case 23: s = xmla(s, x, coef[22]);
  case 22: s = xmla(s, x, coef[21]);
  case 21: s = xmla(s, x, coef[20]);
  case 20: s = xmla(s, x, coef[19]);
  case 19: s = xmla(s, x, coef[18]);
  case 18: s = xmla(s, x, coef[17]);
  case 17: s = xmla(s, x, coef[16]);
  case 16: s = xmla(s, x, coef[15]);
  case 15: s = xmla(s, x, coef[14]);
  case 14: s = xmla(s, x, coef[13]);
  case 13: s = xmla(s, x, coef[12]);
  case 12: s = xmla(s, x, coef[11]);
  case 11: s = xmla(s, x, coef[10]);
  case 10: s = xmla(s, x, coef[ 9]);
  case  9: s = xmla(s, x, coef[ 8]);
  case  8: s = xmla(s, x, coef[ 7]);
  case  7: s = xmla(s, x, coef[ 6]);
  case  6: s = xmla(s, x, coef[ 5]);
  case  5: s = xmla(s, x, coef[ 4]);
  case  4: s = L >= 4 ? xmlal(s, x, coef[ 3]) : xmla(s, x, coef[ 3]);
  case  3: s = L >= 3 ? xmlal(s, x, coef[ 2]) : xmla(s, x, coef[ 2]);
  case  2: s = L >= 2 ? xmlal(s, x, coef[ 1]) : xmla(s, x, coef[ 1]);
  case  1: s = L >= 1 ? xmlal(s, x, coef[ 0]) : xmla(s, x, coef[ 0]);
  }
  return s;
}

static inline long double ulpadd(long double d, int n) {
  union {
    long double dbl;
    uint64_t u64;
  } tmp;

  tmp.dbl = d;
  tmp.u64 += n;
  return isfinite(tmp.dbl) ? tmp.dbl : d;
}

#define Q 100000

void refine(long double *initcoef) {
  static long double a[Q];
  static long double v[Q], w[Q], am[Q], aa[Q];
  static long double bestcoef[Q], curcoef[Q];
  long double best = 1e+100, bestworstx = 0, bestsum = 0;
  int i, j, k;

  srandom(time(NULL));

  for(i=0;i<Q;i++) {
    long double lda = ((long double)MAX - (long double)MIN) * i / (long double)(Q-1) + (long double)MIN;
    mpfr_set_ld(fra, lda, GMP_RNDN);

    FRFUNC(frb, fra);
    v[i] = w[i] = mpfr_get_ld(frb, GMP_RNDN);

#ifdef LOGMODE
    mpfr_add(frb, fra, one, GMP_RNDN);
    mpfr_sub(frc, fra, one, GMP_RNDN);
    mpfr_div(fra, frc, frb, GMP_RNDN);
#endif

    a[i] = mpfr_get_ld(fra, GMP_RNDN);

    am[i] = powl(a[i], PMUL);
    aa[i] = powl(a[i], PADD);

    bestcoef[i] = 0;
  }

  for(i=0;i<N;i++) {
    bestcoef[i] = curcoef[i] = initcoef[N-1-i];
    if (i >= L) bestcoef[i] = curcoef[i] = (double)initcoef[N-1-i];
  }

  for(k=0;k<1000;k++) {
    long double emax = 0, esum = 0, worstx = 0;

#ifdef FIXCOEF0
    curcoef[0] = FIXCOEF0;
#endif

#ifdef FIXCOEF1
    curcoef[1] = FIXCOEF1;
#endif

#ifdef FIXCOEF2
    curcoef[2] = FIXCOEF2;
#endif
    
    for(i=0;i<Q;i++) {
      long double ld = w[i];
      ld = fabsl(ld) * powl(2, -52);
      if (ld == 0) ld = DBL_MAX;
      long double f = func(am[i], curcoef, N)*aa[i];
      long double e = fabsl(f - v[i])/ld;
      //printf("%d : correct = %.20Lg, test = %.20Lg, error = %Lg\n", i, v[i], f, e);
      if (!isfinite(e)) continue;
      if (e > emax) { emax = e; worstx = a[i]; }
      esum += e;
    }

    if (emax < best || (emax == best && esum < bestsum)) {
      for(i=0;i<N;i++) {
	bestcoef[i] = curcoef[i];
      }
      best = emax;
      bestsum = esum;
      bestworstx = worstx;
      if (k > 10) printf("Max error = %.20Lg ULP, Sum error = %.20Lg (Max error at %.20Lg)\n", best, bestsum, bestworstx);
      k = 0;
    }

    for(i=0;i<N;i++) {
      curcoef[i] = bestcoef[i];
    }
    for(i=0;i<N;i++) {
      static int tab[] = {0, 0, -4, 4, 2, -2, 1, -1};
      if (i >= L) {
	curcoef[i] = ulpadd(bestcoef[i], tab[random() & 7] * (1 << (64-53)));
	curcoef[i] = (double)curcoef[i];
      } else {
	curcoef[i] = ulpadd(bestcoef[i], tab[random() & 7]);
	curcoef[i] = (double)curcoef[i];
      }
    }
  }

  printf("\nMax error = %.20Lg ULP at %.20Lg\n", best, bestworstx);

  for(i=N-1;i>=0;i--) {
    if (i >= L) {
      printf("%.24Lg,\n", bestcoef[i]);
    } else {
      printf("%.24g + %.24g\n", (double)bestcoef[i], (double)(bestcoef[i] - (double)bestcoef[i]));
    }
  }
  printf("\n");
}

int main(int argc, char **argv)
{
  mpfr_t *x[1000], w[1000], result[1000];
  int i,j;
  int n,m;

  mpfr_set_default_prec(PREC);
  mpfr_init(one); mpfr_set_d(one, 1, GMP_RNDN);

  for(i=0;i<1000;i++) {
    x[i] = calloc(sizeof(mpfr_t),1000);
    for(j=0;j<1000;j++) mpfr_zinit(x[i][j]);
    mpfr_zinit(result[i]);
    mpfr_zinit(w[i]);
  }

  m = N+1;
  n = argc >= 2 ? atoi(argv[1]) : S;

  mpfr_zinit(fra);
  mpfr_zinit(frb);
  mpfr_zinit(frc);
  mpfr_zinit(frd);

  for(i=0;i<n;i++) {
    REAL b = (REAL)i / (n-1);
    REAL a = ((REAL)MAX - MIN) * b + MIN;

    mpfr_set_ld(fra, a, GMP_RNDN);
    mpfr_set(frd, fra, GMP_RNDN);

#ifdef LOGMODE
    mpfr_add(frb, fra, one, GMP_RNDN);
    mpfr_sub(frd, fra, one, GMP_RNDN);
    mpfr_div(frd, frd, frb, GMP_RNDN);
#endif

    for(j=0;j<m-1;j++) {
      mpfr_set_ld(frb, (long double)j*PMUL+PADD, GMP_RNDN);
      mpfr_pow(x[j][i], frd, frb, GMP_RNDN);
    }

    FRFUNC(x[m-1][i], fra);
    FRFUNC(w[i], fra);
  }
  
  for(i=0;i<m-1;i++) mpfr_set_d(result[i], 0, GMP_RNDN);
  
  regressMinRelError_fr(n,m-1,x,w,result);

  mpfr_init(one); mpfr_set_d(one, 1, GMP_RNDN);

  for(i=m-2;i>=0;i--)
    printf("%.20Lg, \n", mpfr_get_ld(result[i], GMP_RNDN));
  printf("\n");

  double emaxd = 0;
  mpfr_t emax;
  mpfr_zinit(emax);

  for(i=0;i<n*10;i++) {
    REAL a = (MAX - MIN) / (n*10) * i + MIN;
    mpfr_set_ld(fra, a, GMP_RNDN);

#ifdef LOGMODE
    mpfr_add(frb, fra, one, GMP_RNDN);
    mpfr_sub(frd, fra, one, GMP_RNDN);
    mpfr_div(frd, frd, frb, GMP_RNDN);
#else
    mpfr_set(frd, fra, GMP_RNDN);
#endif
    mpfr_set_ld(frb, 0, GMP_RNDN);

    for(j=m-1;j>=0;j--) {
      mpfr_set_ld(frc, j*PMUL+PADD, GMP_RNDN);
      mpfr_pow(frc, frd, frc, GMP_RNDN);
      mpfr_fma(frb, frc, result[j], frb, GMP_RNDN);
    }

    long double b = mpfr_get_ld(frb, GMP_RNDN);
    FRFUNC(frc, fra);
    long double c = mpfr_get_ld(frc, GMP_RNDN);
#if 0
#ifdef LOGMODE
    printf("arg = %g : c = %.20g x = %.20g ", (double)((a-1)/(a+1)), (double)c, (double)b);
#else
    printf("arg = %g : c = %.20g x = %.20g ", (double)a, (double)c, (double)b);
#endif
#endif
    c = fabsl((b-c)/c);
#if 0
    printf("e = %g\n", (double)c * pow(2, 52));
#endif
    if (c > emaxd) emaxd = c;
  }

  printf("error max = %g ULP\n", emaxd * pow(2, 52));

  long double initcoef[N];

  for(i=N-1;i>=0;i--)
    initcoef[N-1-i] = mpfr_get_ld(result[i], GMP_RNDN);

  refine(initcoef);
  
  exit(0);
}
