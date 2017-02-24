#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib
./runtestersub2.sh ./iutneon32 --flushtozero
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./runtestersub2.sh ./iutadvsimd
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi


./runtestersub2.sh ./iutavx512f sde64 --
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./runtestersub2.sh ./iutfma4
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./runtestersub2.sh ./iutpurec
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./runtestersub2.sh ./iutavx2
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./runtestersub2.sh ./iutavx
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./runtestersub2.sh ./iut
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./runtestersub2.sh ./iutvecext
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./runtestersub2.sh ./iutsse2
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

echo "All tests passed."

exit 0
