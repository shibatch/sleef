#!/bin/sh
if [ -f $1 -o -f $1.exe ]
then
    ./runtestersub1.sh $1 $2 $3
    rc=$?
    if [ $rc != 0 ]; then exit 1; fi
fi

exit 0
