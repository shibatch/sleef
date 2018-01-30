#!/bin/bash
set -ev
cd /build/build-cross
make -j 2 all
export CTEST_OUTPUT_ON_FAILURE=TRUE
ctest --verbose -j 2
make install
