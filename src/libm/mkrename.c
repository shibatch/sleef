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
  if (argc < 3) {
    fprintf(stderr, "Generate a header for renaming functions\n");
    fprintf(stderr, "Usage : %s <DP width> <SP width> [<isa>]\n", argv[0]);
    fprintf(stderr, "\n");

    fprintf(stderr, "Generate a part of header for library functions\n");
    fprintf(stderr, "Usage : %s <DP width> <SP width> <vdouble type> <vfloat type> <vint type> <vint2 type> <Macro to enable> [<isa>]\n", argv[0]);
    fprintf(stderr, "\n");

    exit(-1);
  }

  static char *ulpSuffixStr[] = { "", "_u1", "_u05", "_u35", "_u15" };
  
  if (argc == 3 || argc == 4) {
    int wdp = atoi(argv[1]);
    int wsp = atoi(argv[2]);
    char *isaname = argc == 3 ? "" : argv[3];
    char *isaub = argc == 4 ? "_" : "";

    for(int i=0;funcList[i].name != NULL;i++) {
      if (funcList[i].ulp >= 0) {
	printf("#define x%s%s Sleef_%sd%d_u%02d%s\n",
	       funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	       funcList[i].name, wdp,
	       funcList[i].ulp, isaname);
      } else {
	printf("#define x%s Sleef_%sd%d%s%s\n", funcList[i].name, funcList[i].name, wdp, isaub, isaname);
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
	printf("#define x%sf Sleef_%sf%d%s%s\n", funcList[i].name, funcList[i].name, wsp, isaub, isaname);
      }
    }
  } else {
    int wdp = atoi(argv[1]);
    int wsp = atoi(argv[2]);
    char *vdoublename = argv[3];
    char *vfloatname = argv[4];
    char *vintname = argv[5];
    char *vint2name = argv[6];
    char *isaname = argc == 9 ? argv[8] : "";
    char *isaub = argc == 9 ? "_" : "";
    printf("#ifdef %s\n", argv[7]);

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
	    printf("IMPORT %s Sleef_%sd%d%s%s(%s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaub, isaname,
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
	    printf("IMPORT %s Sleef_%sd%d%s%s(%s, %s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaub, isaname,
		   vdoublename, vdoublename);
	  }
	  break;
	case 2:
	case 6:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT Sleef_%s_2 Sleef_%sd%d_u%02d%s(%s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename);
	  } else {
	    printf("IMPORT Sleef_%s_2 Sleef_%sd%d%s%s(%s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaub, isaname,
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
	    printf("IMPORT %s Sleef_%sd%d%s%s(%s, %s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaub, isaname,
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
	    printf("IMPORT %s Sleef_%sd%d%s%s(%s);\n",
		   vintname,
		   funcList[i].name, wdp,
		   isaub, isaname,
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
	    printf("IMPORT %s Sleef_%sd%d%s%s(%s, %s, %s);\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaub, isaname,
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
	  printf("IMPORT %s Sleef_%sf%d%s%s(%s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaub, isaname,
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
	  printf("IMPORT %s Sleef_%sf%d%s%s(%s, %s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaub, isaname,
		 vfloatname, vfloatname);
	}
	break;
      case 2:
      case 6:
	if (funcList[i].ulp >= 0) {
	  printf("IMPORT Sleef_%s_2 Sleef_%sf%d_u%02d%s(%s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 funcList[i].ulp, isaname,
		 vfloatname);
	} else {
	  printf("IMPORT Sleef_%s_2 Sleef_%sf%d%s%s(%s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaub, isaname,
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
	  printf("IMPORT %s Sleef_%sf%d%s%s(%s, %s, %s);\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaub, isaname,
		 vfloatname, vfloatname, vfloatname);
	}
	break;
      }
    }

    printf("#endif\n");
  }

  exit(0);
}
  
