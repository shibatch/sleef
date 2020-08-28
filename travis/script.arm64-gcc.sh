#!/bin/bash
set -ev
cd sleef.build
ninja all
export OMP_WAIT_POLICY=passive
export CTEST_OUTPUT_ON_FAILURE=TRUE
ctest -j `nproc`
ninja install
