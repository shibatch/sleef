#!/bin/bash
set -ev
sudo apt-get -qq update
sudo apt-get install -y libmpfr-dev libfftw3-dev libssl-dev ninja-build
