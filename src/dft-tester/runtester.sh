#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib
./naivetestdp 1
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestdp 2
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestdp 3
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestdp 4
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestdp 6
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestdp 8
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestdp 10
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestsp 1
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestsp 2
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestsp 3
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestsp 4
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestsp 6
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestsp 8
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi

./naivetestsp 10
rc=$?
if [ $rc != 0 ]
then
    echo "Not all tests were passed."
    exit 1
fi


echo "All tests passed."

exit 0
