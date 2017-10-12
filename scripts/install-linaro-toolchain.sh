#!/bin/bash
set -ex

NAME=gcc-linaro-7.1.1-2017.08-x86_64_aarch64-linux-gnu
TOOLCHAIN=/opt/linaro-toolchain

wget https://releases.linaro.org/components/toolchain/binaries/latest/aarch64-linux-gnu/$NAME.tar.xz
tar -xvf $NAME.tar.xz

mkdir $TOOLCHAIN
mv $NAME/* $TOOLCHAIN

$TOOLCHAIN/bin/aarch64-linux-gnu-gcc --version
