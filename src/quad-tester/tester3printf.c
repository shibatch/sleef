//   Copyright Naoki Shibata and contributors 2010 - 2020.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <openssl/md5.h>

#include "sleefquad.h"

static void convertEndianness(void *ptr, int len) {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
  for(int k=0;k<len/2;k++) {
    unsigned char t = ((unsigned char *)ptr)[k];
    ((unsigned char *)ptr)[k] = ((unsigned char *)ptr)[len-1-k];
    ((unsigned char *)ptr)[len-1-k] = t;
  }
#else
#endif
}

static void testem(MD5_CTX *ctx, Sleef_quad val, char *fmt) {
  for(int alt=0;alt<2;alt++) {
    for(int zero=0;zero<2;zero++) {
      for(int left=0;left<2;left++) {
	for(int blank=0;blank<2;blank++) {
	  for(int sign=0;sign<2;sign++) {
	    static char buf[100];
	    Sleef_quad q;
	    int r;
	    snprintf(buf, 99, "%%%s%s%s%s%s%s",
		     alt ? "#" : "", 
		     zero ? "0" : "", 
		     left ? "-" : "", 
		     blank ? " " : "", 
		     sign ? "+" : "",
		     fmt);

	    r = Sleef_snprintf(buf, 99, buf, val);
	    assert(r < 100);
	    MD5_Update(ctx, buf, r < 0 ? 0 : r);
	    q = Sleef_strtoq(buf, NULL);
	    convertEndianness(&q, sizeof(q));
	    MD5_Update(ctx, &q, sizeof(Sleef_quad));

	    for(int width=0;width<=40;width += 2) {
	      snprintf(buf, 99, "%%%s%s%s%s%s%d.%s",
		       alt ? "#" : "", 
		       zero ? "0" : "", 
		       left ? "-" : "", 
		       blank ? " " : "", 
		       sign ? "+" : "",
		       width, fmt);

	      r = Sleef_snprintf(buf, 99, buf, val);
	      assert(r < 100);
	      MD5_Update(ctx, buf, r < 0 ? 0 : r);
	      q = Sleef_strtoq(buf, NULL);
	      convertEndianness(&q, sizeof(q));
	      MD5_Update(ctx, &q, sizeof(Sleef_quad));
	    }

	    for(int prec=0;prec<=40;prec += 3) {
	      for(int width=0;width<=40;width += 3) {
		snprintf(buf, 99, "%%%s%s%s%s%s%d.%d%s",
			 alt ? "#" : "", 
			 zero ? "0" : "", 
			 left ? "-" : "", 
			 blank ? " " : "", 
			 sign ? "+" : "",
			 width, prec, fmt);

		r = Sleef_snprintf(buf, 99, buf, val);
		assert(r < 100);
		MD5_Update(ctx, buf, r < 0 ? 0 : r);
		q = Sleef_strtoq(buf, NULL);
		convertEndianness(&q, sizeof(q));
		MD5_Update(ctx, &q, sizeof(Sleef_quad));
	      }

	      snprintf(buf, 99, "%%%s%s%s%s%s.%d%s",
		       alt ? "#" : "", 
		       zero ? "0" : "", 
		       left ? "-" : "", 
		       blank ? " " : "", 
		       sign ? "+" : "",
		       prec, fmt);

	      r = Sleef_snprintf(buf, 99, buf, val);
	      assert(r < 100);
	      MD5_Update(ctx, buf, r < 0 ? 0 : r);
	      q = Sleef_strtoq(buf, NULL);
	      convertEndianness(&q, sizeof(q));
	      MD5_Update(ctx, &q, sizeof(Sleef_quad));
	    }
	  }
	}
      }
    }
  }
}

int main(int argc, char **argv) {
#if defined(__GLIBC__)
  Sleef_registerPrintfHook();
  static char buf[110];
  Sleef_quad q = Sleef_strtoq("3.1415926535897932384626433832795028842", NULL);

  snprintf(buf, 100, "%50.40Pe", &q);
  if (strcmp(buf, "    3.1415926535897932384626433832795027974791e+00") != 0) exit(-1);
  snprintf(buf, 100, "%50.40Pf", &q);
  if (strcmp(buf, "        3.1415926535897932384626433832795027974791") != 0) exit(-1);
  snprintf(buf, 100, "%50.40Pg", &q);
  if (strcmp(buf, "         3.141592653589793238462643383279502797479") != 0) exit(-1);
  snprintf(buf, 100, "%Pa", &q);
  if (strcmp(buf, "0x1.921fb54442d18469898cc51701b8p+1") != 0) exit(-1);
#endif

  //

  FILE *fp = NULL;

  if (argc != 1) {
    fp = fopen(argv[1], "r");
    if (fp == NULL) {
      fprintf(stderr, "Could not open %s\n", argv[1]);
      exit(-1);
    }
  }

  //

  static char *types[] = { "Qe", "Qf", "Qg", "Qa" };

  static const char *strvals[] = {
    "1.2345678912345678912345e+0Q",
    "1.2345678912345678912345e+1Q",
    "1.2345678912345678912345e-1Q",
    "1.2345678912345678912345e+2Q",
    "1.2345678912345678912345e-2Q",
    "1.2345678912345678912345e+3Q",
    "1.2345678912345678912345e-3Q",
    "1.2345678912345678912345e+4Q",
    "1.2345678912345678912345e-4Q",
    "1.2345678912345678912345e+5Q",
    "1.2345678912345678912345e-5Q",
    "1.2345678912345678912345e+10Q",
    "1.2345678912345678912345e-10Q",
    "1.2345678912345678912345e+15Q",
    "1.2345678912345678912345e-15Q",
    "1.2345678912345678912345e+30Q",
    "1.2345678912345678912345e-30Q",
    "1.2345678912345678912345e+1000Q",
    "1.2345678912345678912345e-1000Q",
    "1.2345678912345678912345e-4950Q",
    "1.2345678912345678912345e+4920Q",
    "3.36210314311209350626267781732175260e-4932",
    "1.18973149535723176508575932662800702e+4932",
    "6.475175119438025110924438958227646552e-4966",
    "0.0Q", "1.0Q",
    "1e+1Q", "1e+2Q", "1e+3Q", "1e+4Q", "1e+5Q", "1e+6Q", 
    "1e-1Q", "1e-2Q", "1e-3Q", "1e-4Q", "1e-5Q", "1e-6Q", 
    "inf", "nan",
  };
  Sleef_quad vals[sizeof(strvals) / sizeof(char *)];
  for(int i=0;i<sizeof(strvals) / sizeof(char *);i++) {
    vals[i] = Sleef_strtoq(strvals[i], NULL);
  }

  int success = 1;

  for(int j=0;j<4;j++) {
    MD5_CTX ctx;
    memset(&ctx, 0, sizeof(MD5_CTX));
    MD5_Init(&ctx);

    for(int i=0;i<sizeof(vals)/sizeof(Sleef_quad);i++) {
      testem(&ctx, vals[i], types[j]);
      testem(&ctx, Sleef_negq1_purec(vals[i]), types[j]);
    }

    unsigned char d[16], mes[64], buf[64];
    MD5_Final(d, &ctx);

    snprintf((char *)mes, 60, "%s %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
	     types[j], d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
	     d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]);

    if (fp != NULL) {
      fgets((char *)buf, 60, fp);
      if (strncmp((char *)mes, (char *)buf, strlen((char *)mes)) != 0) {
	puts((char *)mes);
	puts((char *)buf);
	success = 0;
      }
    } else puts((char *)mes);
  }

  if (fp != NULL) fclose(fp);

  //

  exit(success ? 0 : -1);
}
