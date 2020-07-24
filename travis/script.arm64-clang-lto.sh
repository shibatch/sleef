#!/bin/bash
set -ev
cd sleef.build
ninja all
export OMP_WAIT_POLICY=passive
export CTEST_OUTPUT_ON_FAILURE=TRUE
test -f lib/sleefdp.ll
ctest -j `nproc`
ninja install
