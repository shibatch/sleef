//          Copyright Naoki Shibata 2010 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

void xsrand(uint64_t s);
uint64_t xrand();
void memrand(void *p, int size);
Sleef_quad rndf128(Sleef_quad min, Sleef_quad max);
Sleef_quad rndf128x();

int readln(int fd, char *buf, int cnt);
int startsWith(char *str, char *prefix);

int iszerof128(Sleef_quad a);
int isnegf128(Sleef_quad a);
int isinff128(Sleef_quad a);
int isnonnumberf128(Sleef_quad a);
int isnanf128(Sleef_quad a);

#ifdef USEMPFR
void mpfr_set_f128(mpfr_t frx, Sleef_quad a, mpfr_rnd_t rnd);
Sleef_quad mpfr_get_f128(mpfr_t m, mpfr_rnd_t rnd);

double countULPf128(Sleef_quad d, mpfr_t c, int checkNegZero);
char *sprintfr(mpfr_t fr);
char *sprintf128(Sleef_quad x);

double cast_d_q(Sleef_quad q);
Sleef_quad cast_q_str(const char *s);
Sleef_quad add_q_d(Sleef_quad q, double d);
#endif
