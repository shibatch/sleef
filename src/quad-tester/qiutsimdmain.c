//   Copyright Naoki Shibata and contributors 2010 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
static jmp_buf sigjmp;
#define SETJMP(x) setjmp(x)
#define LONGJMP longjmp
#else
static sigjmp_buf sigjmp;
#define SETJMP(x) sigsetjmp(x, 1)
#define LONGJMP siglongjmp
#endif

int do_test(int argc, char **argv);
int check_featureQP();

static void sighandler(int signum) {
  LONGJMP(sigjmp, 1);
}

int detectFeatureQP() {
  signal(SIGILL, sighandler);

  if (SETJMP(sigjmp) == 0) {
    int r = check_featureQP();
    signal(SIGILL, SIG_DFL);
    return r;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}

int main(int argc, char **argv) {
  if (!detectFeatureQP()) {
    fprintf(stderr, "\n\n***** This host does not support the necessary CPU features to execute this program *****\n\n\n");
    printf("0\n");
    fclose(stdout);
    exit(-1);
  }

  return do_test(argc, argv);
}
