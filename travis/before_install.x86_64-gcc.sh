#!/bin/bash
set -ev
export PATH=$PATH:/usr/bin:`pwd`/sde-external-8.12.0-2017-10-23-lin
tar xf sde-external-8.12.0-2017-10-23-lin.tar.bz2
add-apt-repository -y ppa:adrozdoff/cmake
apt-get -qq update
apt-get install -y cmake libmpfr-dev
export CC=gcc-7
export CXX=g++-7
