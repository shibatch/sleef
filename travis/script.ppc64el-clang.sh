#!/bin/bash
set -ev
export QEMU_CPU=POWER8
cd /build/build-cross
ninja all
export OMP_WAIT_POLICY=passive
export CTEST_OUTPUT_ON_FAILURE=TRUE
ctest -j `nproc`
ninja install
