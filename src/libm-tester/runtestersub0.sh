#!/bin/sh
if [ -f $2 -o -f $2.exe ]
then
    echo $1 $3 $4 $2
    $1 $3 $4 $2
    rc=$?
    if [ $rc != 0 ]
    then
	exit 1
    fi
fi
exit 0
