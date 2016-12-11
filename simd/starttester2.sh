#!/bin/sh
nohup nice ./tester2sse2 >> ../tester2.out &
nohup nice ./tester2fsse2 >> ../tester2.out &
nohup nice ./tester2avx >> ../tester2.out &
nohup nice ./tester2favx >> ../tester2.out &
nohup nice ./tester2avx2 >> ../tester2.out &
nohup nice ./tester2favx2 >> ../tester2.out &
nohup nice sde64 -- ./tester2avx512f >> ../tester2.out &
nohup nice sde64 -- ./tester2favx512f >> ../tester2.out &
