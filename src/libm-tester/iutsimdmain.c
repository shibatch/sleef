//   Copyright Naoki Shibata and contributors 2010 - 2020.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>

static jmp_buf sigjmp;

int do_test(int argc, char **argv);
int check_featureDP(double d);
int check_featureSP(float d);

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#define SETJMP(x) setjmp(x)
#define LONGJMP longjmp
#else
#define SETJMP(x) sigsetjmp(x, 1)
#define LONGJMP siglongjmp
#endif

static void sighandler(int signum) {
  LONGJMP(sigjmp, 1);
}

int detectFeatureDP() {
  signal(SIGILL, sighandler);

  if (SETJMP(sigjmp) == 0) {
    int r = check_featureDP(1.0);
    signal(SIGILL, SIG_DFL);
    return r;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}

int detectFeatureSP() {
  signal(SIGILL, sighandler);

  if (SETJMP(sigjmp) == 0) {
    int r = check_featureSP(1.0);
    signal(SIGILL, SIG_DFL);
    return r;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}

int main(int argc, char **argv) {
  if (!detectFeatureDP() && !detectFeatureSP()) {
    fprintf(stderr, "\n\n***** This host does not support the necessary CPU features to execute this program *****\n\n\n");
    printf("0\n");
    fclose(stdout);
    exit(-1);
  }

  return do_test(argc, argv);
}
