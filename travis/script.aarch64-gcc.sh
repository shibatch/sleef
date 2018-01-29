#!/bin/sh
cd sleef.build
VERBOSE=1 make -j 1 all
export CTEST_OUTPUT_ON_FAILURE=TRUE
ctest --verbose -j 2
make install
