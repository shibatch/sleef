//          Copyright Naoki Shibata 2010 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>

#ifndef _MSC_VER
#include <unistd.h>
#include <sys/types.h>
int xgetpid() { return (int)getpid(); }
#define _CRT_SECURE_NO_WARNINGS
#else
#include <process.h>
int xgetpid() { return _getpid(); }
#endif

char buf[1024], fnout[256];

void sub(FILE *fpin, FILE *fpout) {
  while(!feof(fpin)) {
    size_t z = fread(buf, sizeof(unsigned char), sizeof(buf), fpin);
    fwrite(buf, sizeof(unsigned char), z, fpout);
  }
}

int main(int argc, char **argv) {
  if (argc == 1) {
    fprintf(stderr, "Usage : %s [<source> ...] <destination>\n\n", argv[0]);
    fprintf(stderr, "This program concatenates the source files and write it in\n");
    fprintf(stderr, "the destination file. If no source file is given, it copies\n");
    fprintf(stderr, "STDIN to the destination. The destination file is once made\n");
    fprintf(stderr, "with a name with PID suffix, and it is renamed to the\n");
    fprintf(stderr, "specified name after it is closed. In this way, the other\n");
    fprintf(stderr, "processes do not see incomplete contents of the destination\n");
    fprintf(stderr, "file.\n");
    exit(-1);
  }

  snprintf(fnout, sizeof(fnout), "%s.%d", argv[argc-1], xgetpid());
  FILE *fpout = fopen(fnout, "w");

  if (fpout == NULL) { perror("copycat"); exit(-1); }
  
  if (argc == 2) {
    sub(stdin, fpout);
  } else {
    for(int i=1;i<argc-1;i++) {
      FILE *fpin = fopen(argv[i], "r");
      if (fpin == NULL) {
	fclose(fpout);
	remove(fnout);
	perror("copycat");
	exit(-1);
      }
      sub(fpin, fpout);
      fclose(fpin);
    }
  }

  fclose(fpout);
  if (rename(fnout, argv[argc-1]) != 0) remove(fnout);
  
  exit(0);
}
