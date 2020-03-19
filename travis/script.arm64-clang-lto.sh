#!/bin/bash
set -ev
cd sleef.build
make -j `nproc` all
export OMP_WAIT_POLICY=passive
export CTEST_OUTPUT_ON_FAILURE=TRUE
test -f lib/sleefdp.ll
ctest -j `nproc`
make install
