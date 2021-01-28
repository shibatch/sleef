#!/bin/bash
set -ev
sudo apt-get -qq update
sudo apt-get install -y libmpfr-dev libfftw3-dev libssl-dev ninja-build
wget -nv https://shibata.naist.jp/~n-sibata/travis/binutils-2.34-aarch64.tar.xz
tar xf binutils-2.34-aarch64.tar.xz -C /
