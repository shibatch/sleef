#!/bin/bash
set -ev
brew update
brew install gcc@6
export CC=gcc-6
