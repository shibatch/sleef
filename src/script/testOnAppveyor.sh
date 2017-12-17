#!/bin/sh
HOME=/cygdrive/c/projects/sleef
for i in $HOME/build/bin/Release/iut*.exe; do
    $HOME/src/libm-tester/test.exe $i || exit 1
done
