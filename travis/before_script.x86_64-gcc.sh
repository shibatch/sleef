#!/bin/bash
set -ev
mkdir sleef.build
cd sleef.build
export CC=gcc-7
export CXX=g++-7
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE ..
