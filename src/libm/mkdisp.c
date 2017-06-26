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
  if (argc < 7) {
    fprintf(stderr, "Usage : %s <DP width> <SP width> <vdouble type> <vfloat type> <vint type> <isa0> [<isa1> ...]\n", argv[0]);
    fprintf(stderr, "\n");
    exit(-1);
  }

  const int wdp = atoi(argv[1]), wsp = atoi(argv[2]);
  const char *vdoublename = argv[3], *vfloatname = argv[4], *vintname = argv[5];
  const int isastart = 6, nisa = argc - isastart;
  
  for(int i=0;funcList[i].name != NULL;i++) {
    char ulpSuffix0[100] = "", ulpSuffix1[100] = "_";
    if (funcList[i].ulp >= 0) {
      sprintf(ulpSuffix0, "_u%02d", funcList[i].ulp);
      sprintf(ulpSuffix1, "_u%02d", funcList[i].ulp);
    }

    switch(funcList[i].funcType) {
    case 0:
      printf("DISPATCH_d_d(%s, Sleef_%sd%d%s, pnt_%sd%d%s, disp_%sd%d%s",
	     vdoublename,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sd%d%s%s", funcList[i].name, wdp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");

      printf("DISPATCH_d_d(%s, Sleef_%sf%d%s, pnt_%sf%d%s, disp_%sf%d%s",
	     vfloatname,
	     funcList[i].name, wsp, ulpSuffix0,
	     funcList[i].name, wsp, ulpSuffix0,
	     funcList[i].name, wsp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sf%d%s%s", funcList[i].name, wsp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");

      break;
    case 1:
      printf("DISPATCH_d_d_d(%s, Sleef_%sd%d%s, pnt_%sd%d%s, disp_%sd%d%s",
	     vdoublename,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sd%d%s%s", funcList[i].name, wdp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");

      printf("DISPATCH_d_d_d(%s, Sleef_%sf%d%s, pnt_%sf%d%s, disp_%sf%d%s",
	     vfloatname,
	     funcList[i].name, wsp, ulpSuffix0,
	     funcList[i].name, wsp, ulpSuffix0,
	     funcList[i].name, wsp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sf%d%s%s", funcList[i].name, wsp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");

      break;
    case 2:
    case 6:
      printf("DISPATCH_d2_d(%s, Sleef_%s_2, Sleef_%sd%d%s, pnt_%sd%d%s, disp_%sd%d%s",
	     vdoublename, vdoublename,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sd%d%s%s", funcList[i].name, wdp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");

      printf("DISPATCH_d2_d(%s, Sleef_%s_2, Sleef_%sf%d%s, pnt_%sf%d%s, disp_%sf%d%s",
	     vfloatname, vfloatname, 
	     funcList[i].name, wsp, ulpSuffix0,
	     funcList[i].name, wsp, ulpSuffix0,
	     funcList[i].name, wsp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sf%d%s%s", funcList[i].name, wsp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");

      break;
    case 3:
      printf("DISPATCH_d_d_i(%s, %s, Sleef_%sd%d%s, pnt_%sd%d%s, disp_%sd%d%s",
	     vdoublename, vintname,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sd%d%s%s", funcList[i].name, wdp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");
      break;
    case 4:
      printf("DISPATCH_i_d(%s, %s, Sleef_%sd%d%s, pnt_%sd%d%s, disp_%sd%d%s",
	     vdoublename, vintname,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sd%d%s%s", funcList[i].name, wdp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");
      break;
    case 5:
      printf("DISPATCH_d_d_d_d(%s, Sleef_%sd%d%s, pnt_%sd%d%s, disp_%sd%d%s",
	     vdoublename,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0,
	     funcList[i].name, wdp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sd%d%s%s", funcList[i].name, wdp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");

      printf("DISPATCH_d_d_d_d(%s, Sleef_%sf%d%s, pnt_%sf%d%s, disp_%sf%d%s",
	     vfloatname,
	     funcList[i].name, wsp, ulpSuffix0,
	     funcList[i].name, wsp, ulpSuffix0,
	     funcList[i].name, wsp, ulpSuffix0);
      for(int j=0;j<nisa;j++) printf(", Sleef_%sf%d%s%s", funcList[i].name, wsp, ulpSuffix1, argv[isastart + j]);
      printf(")\n");

      break;
    }
  }
  
  exit(0);
}
