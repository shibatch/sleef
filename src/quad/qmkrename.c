//          Copyright Naoki Shibata 2010 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "qfuncproto.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Generate a header for renaming functions\n");
    fprintf(stderr, "Usage : %s <width> [<isa>]\n", argv[0]);
    fprintf(stderr, "\n");

    fprintf(stderr, "Generate a part of header for library functions\n");
    fprintf(stderr, "Usage : %s <width> <vargquad type> <vargquad2 type> <vdouble type> <vfloat type> <vmask type> <vint type> <Macro to enable> [<isa>]\n", argv[0]);
    fprintf(stderr, "\n");

    exit(-1);
  }

  static char *ulpSuffixStr[] = { "", "_u10", "_u05" };
  
  if (argc == 2 || argc == 3) {
    char *wqp = argv[1];
    char *isaname = argc == 2 ? "" : argv[2];
    char *isaub = argc == 3 ? "_" : "";

    if (strcmp(isaname, "sve") == 0) wqp = "x";

    for(int i=0;funcList[i].name != NULL;i++) {
      if (funcList[i].ulp >= 0) {
	printf("#define x%sq%s Sleef_%sq%s_u%02d%s\n",
	       funcList[i].name, ulpSuffixStr[funcList[i].ulpSuffix],
	       funcList[i].name, wqp,
	       funcList[i].ulp, isaname);
      } else {
	printf("#define x%sq Sleef_%sq%s%s%s\n", funcList[i].name, funcList[i].name, wqp, isaub, isaname);
      }
    }
  } else {
    char *wqp = argv[1];
    char *vargquadname = argv[2];
    char *vargquad2name = argv[3];
    char *vdoublename = argv[4];
    char *vfloatname = argv[5];
    char *vmaskname = argv[6];
    char *vintname = argv[7];
    char *architecture = argv[8];
    char *isaname = argc == 10 ? argv[9] : "";
    char *isaub = argc == 10 ? "_" : "";

    if (strcmp(isaname, "sve") == 0) wqp = "x";

    printf("#ifdef %s\n", architecture);

    if (strcmp(vargquadname, "-") != 0) {
      for(int i=0;funcList[i].name != NULL;i++) {
	switch(funcList[i].funcType) {
	case 0:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vargquadname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vargquadname);
	  }
	  break;
	case 1:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s, %s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vargquadname, vargquadname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s, %s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vargquadname, vargquadname);
	  }
	  break;
	case 2:
	case 6:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s);\n",
		   vargquad2name,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vargquadname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s);\n",
		   vargquad2name,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vargquadname);
	  }
	  break;
	case 3:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s, %s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vargquadname, vintname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s, %s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vargquadname, vintname);
	  }
	  break;
	case 4:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s);\n",
		   vintname,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vargquadname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s);\n",
		   vintname,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vargquadname);
	  }
	  break;
	case 5:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s, %s, %s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vargquadname, vargquadname, vargquadname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s, %s, %s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vargquadname, vargquadname, vargquadname);
	  }
	  break;
	case 7:
	  printf("IMPORT CONST int Sleef_%sq%s%s%s(int);\n", funcList[i].name, wqp, isaub, isaname);
	  break;
	case 8:
	  printf("IMPORT CONST void *Sleef_%sq%s%s%s(int);\n", funcList[i].name, wqp, isaub, isaname);
	  break;
	case 9:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s, %s);\n",
		   vintname,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vargquadname, vargquadname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s, %s);\n",
		   vintname,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vargquadname, vargquadname);
	  }
	  break;
	case 10:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s);\n",
		   vdoublename,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vargquadname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s);\n",
		   vdoublename,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vargquadname);
	  }
	  break;
	case 11:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vdoublename);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vdoublename);
	  }
	  break;
	case 12:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s);\n",
		   vmaskname,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vargquadname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s);\n",
		   vmaskname,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vargquadname);
	  }
	  break;
	case 13:
	  if (funcList[i].ulp >= 0) {
	    printf("IMPORT CONST %s Sleef_%sq%s_u%02d%s(%s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   funcList[i].ulp, isaname,
		   vmaskname);
	  } else {
	    printf("IMPORT CONST %s Sleef_%sq%s%s%s(%s);\n",
		   vargquadname,
		   funcList[i].name, wqp,
		   isaub, isaname,
		   vmaskname);
	  }
	  break;
	}
      }
    }

    printf("#endif\n");
  }

  exit(0);
}
  
