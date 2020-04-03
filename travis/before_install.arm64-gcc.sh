#!/bin/bash
set -ev
sudo apt-get -qq update
sudo apt-get install -y cmake libmpfr-dev libssl-dev libfftw3-dev ninja-build
