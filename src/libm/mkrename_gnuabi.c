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
  if (argc < 5) {
    fprintf(stderr, "Usage : %s <isa> <Mangled ISA> <DP width> <SP width>\n", argv[0]);
    exit(-1);
  }

  char *isaname = argv[1];
  char *mangledisa = argv[2];
  int wdp = atoi(argv[3]);
  int wsp = atoi(argv[4]);

  static char *ulpSuffixStr[] = { "", "_u1", "_u05", "_u35", "_u15" };
  static char *vparameterStrDP[] = { "v", "vv", "vl8l8", "vv", "v", "vvv", "vl8" };
  static char *vparameterStrSP[] = { "v", "vv", "vl4l4", "vv", "v", "vvv", "vl4" };
  
  for(int i=0;funcList[i].name != NULL;i++) {
    if (funcList[i].ulp < 0) {
      printf("#define x%s _ZGV%sN%d%s_%s\n", funcList[i].name,
	     mangledisa, wdp, vparameterStrDP[funcList[i].funcType], funcList[i].name);
    } else if (funcList[i].ulp < 20) {
      printf("#define x%s%s _ZGV%sN%d%s_%s\n", 
	     funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	     mangledisa, wdp, vparameterStrDP[funcList[i].funcType], funcList[i].name);
    } else {
      printf("#define x%s%s _ZGV%sN%d%s_%s_u%d\n", 
	     funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	     mangledisa, wdp, vparameterStrDP[funcList[i].funcType], funcList[i].name, funcList[i].ulp);
    }
  }

  printf("\n");

  for(int i=0;funcList[i].name != NULL;i++) {
    if (funcList[i].ulp < 0) {
      printf("#define x%sf _ZGV%sN%d%s_%sf\n", funcList[i].name,
	     mangledisa, wsp, vparameterStrSP[funcList[i].funcType], funcList[i].name);
    } else if (funcList[i].ulp < 20) {
      printf("#define x%sf%s _ZGV%sN%d%s_%sf\n", 
	     funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	     mangledisa, wsp, vparameterStrSP[funcList[i].funcType], funcList[i].name);
    } else {
      printf("#define x%sf%s _ZGV%sN%d%s_%sf_u%d\n", 
	     funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	     mangledisa, wsp, vparameterStrSP[funcList[i].funcType], funcList[i].name, funcList[i].ulp);
    }
  }
  
  exit(0);
}
  
