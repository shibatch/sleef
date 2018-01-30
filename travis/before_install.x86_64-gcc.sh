#!/bin/bash
set -ev
tar xf sde-external-8.12.0-2017-10-23-lin.tar.bz2
sudo add-apt-repository -y ppa:adrozdoff/cmake
sudo apt-get -qq update
sudo apt-get install -y cmake libmpfr-dev
