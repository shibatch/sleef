#!/bin/bash
set -ev
sudo apt-get -qq update
sudo apt-get install -y cmake libmpfr-dev libssl-dev libfftw3-dev ninja-build
wget -q https://shibata.naist.jp/~n-sibata/travis/binutils-2.34-aarch64.tar.xz
tar xv binutils-2.34-aarch64.tar.xz -C /
