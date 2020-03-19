#!/bin/bash
set -ev
cd sleef.build
make -j `nproc` all
export OMP_WAIT_POLICY=passive
export CTEST_OUTPUT_ON_FAILURE=TRUE
ctest -j `nproc`
make install
