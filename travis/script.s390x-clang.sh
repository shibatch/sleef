#!/bin/bash
set -ev
cd build
ninja all
ctest -V -j `nproc`
ninja install
