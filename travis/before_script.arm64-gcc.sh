#!/bin/bash
set -ev
mkdir sleef.build
cd sleef.build
export CC=gcc-7
export CXX=g++-7
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE ..
