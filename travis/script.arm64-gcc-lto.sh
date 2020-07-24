#!/bin/bash
set -ev
cd sleef.build
ninja all
export OMP_WAIT_POLICY=passive
export CTEST_OUTPUT_ON_FAILURE=TRUE
readelf -a lib/libsleef.a|grep -cq __gnu_lto
ctest -j `nproc`
ninja install
