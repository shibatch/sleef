#!/bin/sh
if [ `uname` = Darwin ]; then
    find `cd $1; pwd` -name $2 -print
elif [ `uname -o` = Cygwin ]; then
    find `cd $1; pwd` -name $2 -exec cygpath -w -p -a {} \;
else
    find `cd $1; pwd` -name $2 -print
fi
