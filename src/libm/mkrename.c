//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "funcproto.h"

// In VSX intrinsics, vector data types are like "vector float".
// This function replaces space characters with '_'.
char *escapeSpace(char *str) {
  char *ret = malloc(strlen(str) + 10);
  strcpy(ret, str);
  for(char *p = ret;*p != '\0';p++) if (*p == ' ') *p = '_';
  return ret;
}

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
    char *wdp = argv[1];
    char *wsp = argv[2];
    char *isaname = argc == 3 ? "" : argv[3];
    char *isaub = argc == 4 ? "_" : "";

    if (strcmp(isaname, "sve") == 0)
      wdp = wsp = "x";

    for(int i=0;funcList[i].name != NULL;i++) {
      if (funcList[i].ulp >= 0) {
	printf("#define x%s%s Sleef_%sd%s_u%02d%s\n",
	       funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	       funcList[i].name, wdp,
	       funcList[i].ulp, isaname);
      } else {
	printf("#define x%s Sleef_%sd%s%s%s\n", funcList[i].name, funcList[i].name, wdp, isaub, isaname);
      }
    }

    printf("\n");
  
    for(int i=0;funcList[i].name != NULL;i++) {
      if (funcList[i].ulp >= 0) {
	printf("#define x%sf%s Sleef_%sf%s_u%02d%s\n",
	       funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	       funcList[i].name, wsp,
	       funcList[i].ulp, isaname);
      } else {
	printf("#define x%sf Sleef_%sf%s%s%s\n", funcList[i].name, funcList[i].name, wsp, isaub, isaname);
      }
    }
  }
  else {
    char *wdp = argv[1];
    char *wsp = argv[2];
    char *vdoublename = argv[3], *vdoublename_escspace = escapeSpace(vdoublename);
    char *vfloatname = argv[4], *vfloatname_escspace = escapeSpace(vfloatname);
    char *vintname = argv[5], *vintname_escspace = escapeSpace(vintname);
    char *vint2name = argv[6], *vint2name_escspace = escapeSpace(vint2name);
    char *architecture = argv[7];
    char *isaname = argc == 9 ? argv[8] : "";
    char *isaub = argc == 9 ? "_" : "";

    if (strcmp(isaname, "sve") == 0)
      wdp = wsp = "x";

    char * vectorcc = "";
    #ifdef ENABLE_AAVPCS
    if (strcmp(isaname, "advsimd") == 0)
      vectorcc =" __attribute__((aarch64_vector_pcs))";
    #endif

    printf("#ifdef %s\n", architecture);

    if (strcmp(architecture, "__ARM_FEATURE_SVE") == 0)
      printf("#define STRUCT_KEYWORD_%s __sizeless_struct\n", architecture);
    else printf("#define STRUCT_KEYWORD_%s struct\n", architecture);

    if (strcmp(vdoublename, "-") != 0) {
      printf("\n");
      printf("#ifndef Sleef_%s_2_DEFINED\n", vdoublename_escspace);
      printf("typedef STRUCT_KEYWORD_%s {\n", architecture);
      printf("  %s x, y;\n", vdoublename);
      printf("} Sleef_%s_2;\n", vdoublename_escspace);
      printf("#define Sleef_%s_2_DEFINED\n", vdoublename_escspace);
      printf("#endif\n");
      printf("\n");

      for(int i=0;funcList[i].name != NULL;i++) {
	switch(funcList[i].funcType) {
	case 0:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sd%s_u%02d%s(%s)%s;\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename,
                   vectorcc);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sd%s%s%s(%s)%s;\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaub, isaname,
		   vdoublename,
                   vectorcc);
	  }
	  break;
	case 1:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sd%s_u%02d%s(%s, %s)%s;\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename, vdoublename,
                   vectorcc);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sd%s%s%s(%s, %s)%s;\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaub, isaname,
		   vdoublename, vdoublename,
                   vectorcc);
	  }
	  break;
	case 2:
	case 6:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST Sleef_%s_2 Sleef_%sd%s_u%02d%s(%s)%s;\n",
		   vdoublename_escspace,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename,
                   vectorcc);
	  } else {
	    printf("IMPORT CONST Sleef_%s_2 Sleef_%sd%s%s%s(%s)%s;\n",
		   vdoublename_escspace,
		   funcList[i].name, wdp,
		   isaub, isaname,
		   vdoublename,
                   vectorcc);
	  }
	  break;
	case 3:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sd%s_u%02d%s(%s, %s)%s;\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename, vintname,
                   vectorcc);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sd%s%s%s(%s, %s)%s;\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaub, isaname,
		   vdoublename, vintname,
                   vectorcc);
	  }
	  break;
	case 4:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sd%s_u%02d%s(%s)%s;\n",
		   vintname,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename,
                   vectorcc);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sd%s%s%s(%s)%s;\n",
		   vintname,
		   funcList[i].name, wdp,
		   isaub, isaname,
		   vdoublename,
                   vectorcc);
	  }
	  break;
	case 5:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sd%s_u%02d%s(%s, %s, %s)%s;\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   funcList[i].ulp, isaname,
		   vdoublename, vdoublename, vdoublename,
                   vectorcc);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sd%s%s%s(%s, %s, %s)%s;\n",
		   vdoublename,
		   funcList[i].name, wdp,
		   isaub, isaname,
		   vdoublename, vdoublename, vdoublename,
                   vectorcc);
	  }
	  break;
	case 7:
	  printf("IMPORT CONST int Sleef_%sd%s%s%s(int)%s;\n", funcList[i].name, wdp, isaub, isaname, vectorcc);
	  break;
	case 8:
	  printf("IMPORT CONST void *Sleef_%sd%s%s%s(int)%s;\n", funcList[i].name, wdp, isaub, isaname, vectorcc);
	  break;
	}
      }
    }

    printf("\n");
    printf("#ifndef Sleef_%s_2_DEFINED\n", vfloatname_escspace);
    printf("typedef STRUCT_KEYWORD_%s {\n", architecture);
    printf("  %s x, y;\n", vfloatname);
    printf("} Sleef_%s_2;\n", vfloatname_escspace);
    printf("#define Sleef_%s_2_DEFINED\n", vfloatname_escspace);
    printf("#endif\n");
    printf("\n");

    //printf("typedef %s vint2_%s;\n", vint2name, isaname);
    //printf("\n");
    
    for(int i=0;funcList[i].name != NULL;i++) {
      switch(funcList[i].funcType) {
      case 0:
	if (funcList[i].ulp >= 0) {
	  printf("IMPORT CONST %s Sleef_%sf%s_u%02d%s(%s)%s;\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 funcList[i].ulp, isaname,
		 vfloatname,
                 vectorcc);
	} else {
	  printf("IMPORT CONST %s Sleef_%sf%s%s%s(%s)%s;\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaub, isaname,
		 vfloatname,
                 vectorcc);
	}
	break;
      case 1:
	if (funcList[i].ulp >= 0) {
	  printf("IMPORT CONST %s Sleef_%sf%s_u%02d%s(%s, %s)%s;\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 funcList[i].ulp, isaname,
		 vfloatname, vfloatname,
                 vectorcc);
	} else {
	  printf("IMPORT CONST %s Sleef_%sf%s%s%s(%s, %s)%s;\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaub, isaname,
		 vfloatname, vfloatname,
                 vectorcc);
	}
	break;
      case 2:
      case 6:
	if (funcList[i].ulp >= 0) {
	  printf("IMPORT CONST Sleef_%s_2 Sleef_%sf%s_u%02d%s(%s)%s;\n",
		 vfloatname_escspace,
		 funcList[i].name, wsp,
		 funcList[i].ulp, isaname,
		 vfloatname,
                 vectorcc);
	} else {
	  printf("IMPORT CONST Sleef_%s_2 Sleef_%sf%s%s%s(%s)%s;\n",
		 vfloatname_escspace,
		 funcList[i].name, wsp,
		 isaub, isaname,
		 vfloatname,
                 vectorcc);
	}
	break;
	/*
	  case 3:
	  printf("IMPORT CONST %s Sleef_%sf%d_%s(%s, vint2_%s);\n",
	  vfloatname,
	  funcList[i].name, wsp,
	  isaname,
	  vfloatname, isaname);
	  break;
	  case 4:
	  printf("IMPORT CONST vint2_%s Sleef_%sf%d_%s(%s);\n",
	  isaname,
	  funcList[i].name, wsp,
	  isaname,
	  vfloatname);
	  break;
	*/
      case 5:
	if (funcList[i].ulp >= 0) {
	  printf("IMPORT CONST %s Sleef_%sf%s_u%02d%s(%s, %s, %s)%s;\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 funcList[i].ulp, isaname,
		 vfloatname, vfloatname, vfloatname,
                 vectorcc);
	} else {
	  printf("IMPORT CONST %s Sleef_%sf%s%s%s(%s, %s, %s)%s;\n",
		 vfloatname,
		 funcList[i].name, wsp,
		 isaub, isaname,
		 vfloatname, vfloatname, vfloatname,
                 vectorcc);
	}
	break;
      case 7:
	printf("IMPORT CONST int Sleef_%sf%s%s%s(int)%s;\n", funcList[i].name, wsp, isaub, isaname, vectorcc);
	break;
      case 8:
	printf("IMPORT CONST void *Sleef_%sf%s%s%s(int)%s;\n", funcList[i].name, wsp, isaub, isaname, vectorcc);
	break;
      }
    }

    printf("#endif\n");

    free(vdoublename_escspace);
    free(vfloatname_escspace);
    free(vintname_escspace);
    free(vint2name_escspace);
  }

  exit(0);
}
  
