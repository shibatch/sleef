#!/bin/bash
set -ev
sudo add-apt-repository -y ppa:adrozdoff/cmake
sudo apt-get -qq update
sudo apt-get install -y cmake libmpfr-dev libssl-dev
