#!/bin/bash
set -ev
uname -a
sudo apt-get -qq update
sudo apt-get install -y libmpfr-dev libssl-dev libfftw3-dev ninja-build
