#!/bin/bash
set -ev
cd build
make -j `nproc` all
ctest -j `nproc`
make install
