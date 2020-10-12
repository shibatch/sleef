#include <stdio.h>
#include <stdlib.h>
#include <sleefquad.h>

int main(int argc, char **argv) {
  if (argc == 1) {
    printf("Usage : %s <FP number>\n", argv[0]);
    exit(-1);
  }
  union {
    struct {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
      unsigned sign : 1;
      unsigned e : 15;
      unsigned long long h : 48;
      unsigned long long l : 64;
#else
      unsigned long long l : 64;
      unsigned long long h : 48;
      unsigned e : 15;
      unsigned sign : 1;
#endif
    };
    Sleef_quad q;
  } cnv = { .q = Sleef_strtoq(argv[1], NULL) };

  Sleef_printf("%+Pa\nSLEEF_Q(%c0x1%012lxLL, 0x%016lxULL, %d)\n",
	       &cnv.q, cnv.sign ? '-' : '+', (long unsigned int)cnv.h,
	       (long unsigned int)cnv.l, cnv.e - 16383);
}
