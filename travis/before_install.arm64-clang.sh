#!/bin/bash
set -ev
sudo apt-get -qq update
sudo apt-get install -y cmake libmpfr-dev libfftw3-dev libssl-dev
