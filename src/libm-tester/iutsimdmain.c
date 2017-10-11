//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "check_feature.h"

int do_test(int argc, char **argv);

int main(int argc, char **argv) {
  if (!(detectFeatureDP() && detectFeatureSP())) {
    fprintf(stderr, "\n\n***** This host does not support the necessary CPU features to execute this program *****\n\n\n");
    printf("0\n");
    fclose(stdout);
    exit(-1);
  }

  return do_test(argc, argv);
}
