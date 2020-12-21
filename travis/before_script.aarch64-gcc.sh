#!/bin/bash
set -ev
cd /build
mkdir build-native
cd build-native
cmake -G Ninja -DBUILD_QUAD=TRUE ..
ninja all
cd /build
mkdir build-cross
cd build-cross
cmake -G Ninja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_TOOLCHAIN_FILE=../travis/toolchain-aarch64.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DEMULATOR=qemu-aarch64-static -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE ..

