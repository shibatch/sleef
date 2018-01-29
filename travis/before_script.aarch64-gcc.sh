#!/bin/bash
set -ev
cd /build
mkdir sleef.build
cd sleef.build
cmake -DCMAKE_TOOLCHAIN_FILE=../travis/toolchain-aarch64.cmake -DEMULATOR=qemu-aarch64-static -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
