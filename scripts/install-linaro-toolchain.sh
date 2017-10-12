#!/bin/bash
TXZ=https://releases.linaro.org/components/toolchain/binaries/latest/aarch64-linux-gnu/gcc-linaro-7.1.1-2017.08-x86_64_aarch64-linux-gnu.tar.xz
wget $TGZ
tar -xvf $TGZ

# gcc-linaro-7.1.1-2017.08-x86_64_aarch64-linux-gnu
./gcc-linaro-7.1.1-2017.08-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc --version

mkdir /opt/linaro-toolchain
mv gcc-linaro-7.1.1-2017.08-x86_64_aarch64-linux-gnu/* /opt/linaro-toolchain
