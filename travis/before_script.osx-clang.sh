#!/bin/bash
set -ev
mkdir sleef.build
cd sleef.build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
