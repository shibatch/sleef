//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "funcproto.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage : %s <isa> <DP width> <SP width>\n", argv[0]);
    exit(-1);
  }

  char *isaname = argv[1];
  int wdp = atoi(argv[2]);
  int wsp = atoi(argv[3]);
  
  static char *ulpSuffixStr[] = { "", "_u1", "_u05", "_u35", "_u15" };
  
  for(int i=0;funcList[i].name != NULL;i++) {
    if (funcList[i].ulp < 0) {
      printf("#define x%s __llvm_%s_v%df64_%s\n", funcList[i].name, funcList[i].name, wdp, isaname);
      printf("#define x%sf __llvm_%s_v%df32_%s\n", funcList[i].name, funcList[i].name, wsp, isaname);
    } else {
      printf("#define x%s%s __llvm_%s%s_v%df64_%s\n", funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	     funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],wdp, isaname);
      printf("#define x%sf%s __llvm_%s%s_v%df32_%s\n", funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	     funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],wsp, isaname);
    }
  }

  printf("\n");
  
  exit(0);
}
