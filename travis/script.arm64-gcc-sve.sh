#!/bin/bash
set -ev
cd sleef.build
ninja all
export PATH=/opt/local/bin:$PATH
export LD_LIBRARY_PATH=/opt/local/lib:$LD_LIBRARY_PATH
export QEMU_CPU=max,sve-max-vq=1
export OMP_WAIT_POLICY=passive
export CTEST_OUTPUT_ON_FAILURE=TRUE
ctest -j `nproc`
ninja install
