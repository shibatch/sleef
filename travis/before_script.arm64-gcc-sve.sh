#!/bin/bash
set -ev
mkdir sleef.build
cd sleef.build
export PATH=/opt/local/bin:$PATH
export LD_LIBRARY_PATH=/opt/local/lib:$LD_LIBRARY_PATH
export CC=gcc-10
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DEMULATOR=qemu-aarch64 -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_QUAD=FALSE -DBUILD_DFT=TRUE -DENFORCE_SVE=TRUE ..
