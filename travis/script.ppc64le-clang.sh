#!/bin/bash
set -ev
cd build
ninja all
ctest -j `nproc`
ninja install
