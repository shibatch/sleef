//          Copyright Naoki Shibata 2010 - 2017.
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
void check_featureDP();
void check_featureSP();

static void sighandler(int signum) {
  longjmp(sigjmp, 1);
}

int detectFeatureDP() {
  signal(SIGILL, sighandler);

  if (setjmp(sigjmp) == 0) {
    check_featureDP();
    signal(SIGILL, SIG_DFL);
    return 1;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}

int detectFeatureSP() {
  signal(SIGILL, sighandler);

  if (setjmp(sigjmp) == 0) {
    check_featureSP();
    signal(SIGILL, SIG_DFL);
    return 1;
  } else {
    signal(SIGILL, SIG_DFL);
    return 0;
  }
}

int main(int argc, char **argv) {
  printf("SLEEF_IUT\n");

  if (!(detectFeatureDP() && detectFeatureSP())) {
    printf("0\n");
    fclose(stdout);
    exit(-1);
  }

  return do_test(argc, argv);
}
