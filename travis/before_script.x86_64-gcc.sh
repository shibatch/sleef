#!/bin/bash
set -ev
mkdir sleef.build
cd sleef.build
export CC=gcc-8
export CXX=g++-8
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DCMAKE_C_FLAGS_RELEASE='-mtune=knl -march=native -O3 -DNDEBUG' ..
