#!/bin/bash
set -ev
mkdir sleef.build
cd sleef.build
export CC=clang-8
cmake -G Ninja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_INLINE_HEADERS=TRUE  -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
