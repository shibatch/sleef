//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  char *name;
  int ulp;
  int ulpSuffix;
  int funcType;
} funcSpec;

/*
  ulp : (error bound in ulp) * 10

  ulpSuffix:
  0 : ""
  1 : "_u1"
  2 : "_u05"
  3 : "_u35"
  4 : "_u15"

  funcType:
  0 : double func(double);
  1 : double func(double, double);
  2 : double2 func(double);
  3 : double func(double, int);
  4 : int func(double);
  5 : double func(double, double, double);
 */

funcSpec funcList[] = {
  { "sin", 35, 0, 0 },
  { "cos", 35, 0, 0 },
  { "sincos", 35, 0, 2 },
  { "tan", 35, 0, 0 },
  { "asin", 35, 0, 0 },
  { "acos", 35, 0, 0 },
  { "atan", 35, 0, 0 },
  { "atan2", 35, 0, 1 },
  { "log", 35, 0, 0 },
  { "cbrt", 35, 0, 0 },
  { "sin", 10, 1, 0 },
  { "cos", 10, 1, 0 },
  { "sincos", 10, 1, 2 },
  { "tan", 10, 1, 0 },
  { "asin", 10, 1, 0 },
  { "acos", 10, 1, 0 },
  { "atan", 10, 1, 0 },
  { "atan2", 10, 1, 1 },
  { "log", 10, 1, 0 },
  { "cbrt", 10, 1, 0 },
  { "exp", 10, 0, 0 },
  { "pow", 10, 0, 1 },
  { "sinh", 10, 0, 0 },
  { "cosh", 10, 0, 0 },
  { "tanh", 10, 0, 0 },
  { "asinh", 10, 0, 0 },
  { "acosh", 10, 0, 0 },
  { "atanh", 10, 0, 0 },
  { "exp2", 10, 0, 0 },
  { "exp10", 10, 0, 0 },
  { "expm1", 10, 0, 0 },
  { "log10", 10, 0, 0 },
  { "log1p", 10, 0, 0 },
  { "sincospi", 5, 2, 2 },
  { "sincospi", 35, 3, 2 },
  { "sinpi", 5, 2, 0 },
  { "ldexp", -1, 0, 3 },
  { "ilogb", -1, 0, 4 },

  { "fma", -1, 0, 5 },
  { "sqrt", 5, 2, 0 },
  { "sqrt", 35, 3, 0 },
  { "hypot", 5, 2, 1 },
  { "hypot", 35, 3, 1 },
  { "fabs", -1, 0, 0 },
  { "copysign", -1, 0, 1 },
  { "fmax", -1, 0, 1 },
  { "fmin", -1, 0, 1 },
  { "fdim", -1, 0, 1 },
  { "trunc", -1, 0, 0 },
  { "floor", -1, 0, 0 },
  { "ceil", -1, 0, 0 },
  { "round", -1, 0, 0 },
  { "rint", -1, 0, 0 },
  { "nextafter", -1, 0, 1 },
  { "frfrexp", -1, 0, 0 },
  { "expfrexp", -1, 0, 4 },
  { "fmod", -1, 0, 1 },
  { "modf", -1, 0, 2 },

  { "lgamma", 10, 1, 0 },
  { "tgamma", 10, 1, 0 },
  { "erf", 10, 1, 0 },
  { "erfc", 15, 4, 0 },
  
  { NULL, -1, 0, 0 },
};

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
  
