//   Copyright Naoki Shibata and contributors 2010 - 2020.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>

int do_test(int argc, char **argv);

static jmp_buf sigjmp;

static void sighandler(int signum) {
  longjmp(sigjmp, 1);
}

int main(int argc, char **argv) {
  signal(SIGILL, sighandler);
  if (setjmp(sigjmp) == 0) do_test(argc, argv);
  signal(SIGILL, SIG_DFL);
  exit(0);
}
