#!/bin/bash
set -ev
cd /build
cd sleef.build
make -j 2 all
export CTEST_OUTPUT_ON_FAILURE=TRUE
ctest --verbose -j 2
make install
