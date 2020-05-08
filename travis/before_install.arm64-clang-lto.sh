#!/bin/bash
set -ev
sudo apt-get -qq update
sudo apt-get install -y cmake ninja-build libmpfr-dev libssl-dev libfftw3-dev clang-8 lld-8
