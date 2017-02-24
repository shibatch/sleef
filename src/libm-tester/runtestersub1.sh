#!/bin/sh
if [ -f ./tester -o -f ./tester.exe ]
then
    ./runtestersub0.sh ./tester $1 $2 $3
    rc=$?
    if [ $rc != 0 ]; then exit 1; fi
fi

if [ -f ./testeru1 -o -f ./testeru1.exe ]
then
    ./runtestersub0.sh ./testeru1 $1 $2 $3
    rc=$?
    if [ $rc != 0 ]; then exit 1; fi
fi

if [ -f ./testersp -o -f ./testersp.exe ]
then
    ./runtestersub0.sh ./testersp $1 $2 $3
    rc=$?
    if [ $rc != 0 ]; then exit 1; fi
fi

if [ -f ./testerspu1 -o -f ./testerspu1.exe ]
then
    ./runtestersub0.sh ./testerspu1 $1 $2 $3
    rc=$?
    if [ $rc != 0 ]; then exit 1; fi
fi

exit 0
