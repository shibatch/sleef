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
    fprintf(stderr, "Usage : %s <isa> <DP width> <SP width> [<vdouble type> <vfloat type> <vint type> <vint2 type> <Macro to enable>]\n", argv[0]);
    exit(-1);
  }

  char *isaname = argv[1];
  int wdp = atoi(argv[2]);
  int wsp = atoi(argv[3]);

  static char *ulpSuffixStr[] = { "", "_u1", "_u05", "_u35", "_u15" };
  
  if (argc == 4) {
    for(int i=0;funcList[i].name != NULL;i++) {
      if (funcList[i].ulp >= 0) {
	printf("#define x%s%s Sleef_%sd%d_u%02d%s\n",
	       funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	       funcList[i].name, wdp,
	       funcList[i].ulp, isaname);
      } else {
	printf("#define x%s Sleef_%sd%d_%s\n", funcList[i].name, funcList[i].name, wdp, isaname);
      }
    }

    printf("\n");
  
    for(int i=0;funcList[i].name != NULL;i++) {
      if (funcList[i].ulp >= 0) {
	printf("#define x%sf%s Sleef_%sf%d_u%02d%s\n",
	       funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	       funcList[i].name, wsp,
	       funcList[i].ulp, isaname);
      } else {
	printf("#define x%sf Sleef_%sf%d_%s\n", funcList[i].name, funcList[i].name, wsp, isaname);
      }
    }
  } else {
    char *vdoublename = argv[4];
    char *vfloatname = argv[5];
    char *vintname = argv[6];
    char *vint2name = argv[7];
    printf("#ifdef %s\n", argv[8]);

    if (strcmp(vdoublename, "-") != 0) {
      printf("\n");
      printf("#ifndef Sleef_%s_2_DEFINED\n", vdoublename);
      printf("typedef struct {\n");
      printf("  %s x, y;\n", vdoublename);
      printf("} Sleef_%s_2;\n", vdoublename);
      printf("#define Sleef_%s_2_DEFINED\n", vdoublename);
      printf("#endif\n");
      printf("\n");

      for(int i=0;funcList[i].name != NULL;i++) {
	switch(funcList[i].funcType) {
	case 0:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT %s Sleef_%sd%d_u%02d%s(%s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename);
	  } else {
	    printf("IMPORT %s Sleef_%sd%d_%s(%s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaname,
		   vdoublename);
	  }
	  break;
	case 1:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT %s Sleef_%sd%d_u%02d%s(%s, %s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename, vdoublename);
	  } else {
	    printf("IMPORT %s Sleef_%sd%d_%s(%s, %s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaname,
		   vdoublename, vdoublename);
	  }
	  break;
	case 2:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT Sleef_%s_2 Sleef_%sd%d_u%02d%s(%s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename);
	  } else {
	    printf("IMPORT Sleef_%s_2 Sleef_%sd%d_%s(%s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaname,
		   vdoublename);
	  }
	  break;
	case 3:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT %s Sleef_%sd%d_u%02d%s(%s, %s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename, vintname);
	  } else {
	    printf("IMPORT %s Sleef_%sd%d_%s(%s, %s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaname,
		   vdoublename, vintname);
	  }
	  break;
	case 4:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT %s Sleef_%sd%d_u%02d%s(%s);\n",
		   vintname,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename);
	  } else {
	    printf("IMPORT %s Sleef_%sd%d_%s(%s);\n",
		   vintname,
		   funcList[i].name, wdp,
		   isaname,
		   vdoublename);
	  }
	  break;
	case 5:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT %s Sleef_%sd%d_u%02d%s(%s, %s, %s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename, vdoublename, vdoublename);
	  } else {
	    printf("IMPORT %s Sleef_%sd%d_%s(%s, %s, %s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaname,
		   vdoublename, vdoublename, vdoublename);
	  }
	  break;
	}
      }
    }

    printf("\n");
    printf("#ifndef Sleef_%s_2_DEFINED\n", vfloatname);
    printf("typedef struct {\n");
    printf("  %s x, y;\n", vfloatname);
    printf("} Sleef_%s_2;\n", vfloatname);
    printf("#define Sleef_%s_2_DEFINED\n", vfloatname);
    printf("#endif\n");
    printf("\n");

    //printf("typedef %s vint2_%s;\n", vint2name, isaname);
    //printf("\n");
    
    for(int i=0;funcList[i].name != NULL;i++) {
      switch(funcList[i].funcType) {
      case 0:
	if (funcList[i].ulp >= 0) {
	  printf("IMPORT %s Sleef_%sf%d_u%02d%s(%s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 funcList[i].ulp, isaname,
		 vfloatname);
	} else {
	  printf("IMPORT %s Sleef_%sf%d_%s(%s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaname,
		 vfloatname);
	}
	break;
      case 1:
	if (funcList[i].ulp >= 0) {
	  printf("IMPORT %s Sleef_%sf%d_u%02d%s(%s, %s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 funcList[i].ulp, isaname,
		 vfloatname, vfloatname);
	} else {
	  printf("IMPORT %s Sleef_%sf%d_%s(%s, %s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaname,
		 vfloatname, vfloatname);
	}
	break;
      case 2:
	if (funcList[i].ulp >= 0) {
	  printf("IMPORT Sleef_%s_2 Sleef_%sf%d_u%02d%s(%s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 funcList[i].ulp, isaname,
		 vfloatname);
	} else {
	  printf("IMPORT Sleef_%s_2 Sleef_%sf%d_%s(%s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaname,
		 vfloatname);
	}
	break;
	/*
	  case 3:
	  printf("IMPORT %s Sleef_%sf%d_%s(%s, vint2_%s);\n",
	  vfloatname,
	  funcList[i].name, wsp,
	  isaname,
	  vfloatname, isaname);
	  break;
	  case 4:
	  printf("IMPORT vint2_%s Sleef_%sf%d_%s(%s);\n",
	  isaname,
	  funcList[i].name, wsp,
	  isaname,
	  vfloatname);
	  break;
	*/
      case 5:
	if (funcList[i].ulp >= 0) {
	  printf("IMPORT %s Sleef_%sf%d_u%02d%s(%s, %s, %s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 funcList[i].ulp, isaname,
		 vfloatname, vfloatname, vfloatname);
	} else {
	  printf("IMPORT %s Sleef_%sf%d_%s(%s, %s, %s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaname,
		 vfloatname, vfloatname, vfloatname);
	}
	break;
      }
    }

    printf("#endif\n");
  }

  exit(0);
}
  
