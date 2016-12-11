CC=gcc
OPT=-O -Wall -Wno-unused -Wno-attributes -ffp-contract=off -fmax-errors=3
SDE=sde64

all : testsse2 testavx testavx2

iutsse2 : sleefsimddp.c sleefsimdsp.c helpersse2.h iut.c
	$(CC) $(OPT) -DENABLE_SSE2 -msse2 iut.c sleefsimddp.c sleefsimdsp.c -o iutsse2 -lm

iutavx : sleefsimddp.c sleefsimdsp.c helperavx.h iut.c
	$(CC) $(OPT) -DENABLE_AVX -mavx iut.c sleefsimddp.c sleefsimdsp.c -o iutavx -lm

iutavx2 : sleefsimddp.c sleefsimdsp.c helperavx2.h iut.c
	$(CC) $(OPT) -DENABLE_AVX2 -mavx2 -mfma iut.c sleefsimddp.c sleefsimdsp.c -o iutavx2 -lm

iutavx512f : sleefsimddp.c sleefsimdsp.c helperavx512f.h iut.c
	$(CC) $(OPT) -DENABLE_AVX512F -mavx512f iut.c sleefsimddp.c sleefsimdsp.c -o iutavx512f -lm

iutfma4 : sleefsimddp.c sleefsimdsp.c helperfma4.h iut.c
	$(CC) $(OPT) -DENABLE_FMA4 -mavx -mfma4 iut.c sleefsimddp.c sleefsimdsp.c -o iutfma4 -lm

#

tester2 : tester2sse2 tester2avx tester2avx2 tester2avx512f tester2fma4 tester2fsse2 tester2favx tester2favx2 tester2favx512f tester2ffma4

tester2sse2 : sleefsimddp.c helpersse2.h tester2simd.c
	$(CC) $(OPT) -DENABLE_SSE2 -msse2 tester2simd.c sleefsimddp.c -o tester2sse2 -lm -lmpfr

tester2avx : sleefsimddp.c helperavx.h tester2simd.c
	$(CC) $(OPT) -DENABLE_AVX -mavx tester2simd.c sleefsimddp.c -o tester2avx -lm -lmpfr

tester2avx2 : sleefsimddp.c helperavx2.h tester2simd.c
	$(CC) $(OPT) -DENABLE_AVX2 -mavx2 -mfma tester2simd.c sleefsimddp.c -o tester2avx2 -lm -lmpfr

tester2avx512f : sleefsimddp.c helperavx512f.h tester2simd.c
	$(CC) $(OPT) -DENABLE_AVX512F -mavx512f tester2simd.c sleefsimddp.c -o tester2avx512f -lm -lmpfr

tester2fma4 : sleefsimddp.c helperfma4.h tester2simd.c
	$(CC) $(OPT) -DENABLE_FMA4 -mavx -mfma4 tester2simd.c sleefsimddp.c -o tester2fma4 -lm -lmpfr

tester2fsse2 : sleefsimdsp.c helpersse2.h tester2fsimd.c
	$(CC) $(OPT) -DENABLE_SSE2 -msse2 tester2fsimd.c sleefsimdsp.c -o tester2fsse2 -lm -lmpfr

tester2favx : sleefsimdsp.c helperavx.h tester2fsimd.c
	$(CC) $(OPT) -DENABLE_AVX -mavx tester2fsimd.c sleefsimdsp.c -o tester2favx -lm -lmpfr

tester2favx2 : sleefsimdsp.c helperavx2.h tester2fsimd.c
	$(CC) $(OPT) -DENABLE_AVX2 -mavx2 -mfma tester2fsimd.c sleefsimdsp.c -o tester2favx2 -lm -lmpfr

tester2favx512f : sleefsimdsp.c helperavx512f.h tester2fsimd.c
	$(CC) $(OPT) -DENABLE_AVX512F -mavx512f tester2fsimd.c sleefsimdsp.c -o tester2favx512f -lm -lmpfr

tester2ffma4 : sleefsimdsp.c helperfma4.h tester2fsimd.c
	$(CC) $(OPT) -DENABLE_FMA4 -mavx -mfma4 tester2fsimd.c sleefsimdsp.c -o tester2ffma4 -lm -lmpfr

#

../tester/tester :
	cd ../tester; make tester

../tester/testeru1 :
	cd ../tester; make testeru1

../tester/testersp :
	cd ../tester; make testersp

../tester/testerspu1 :
	cd ../tester; make testerspu1

testsse2 : iutsse2 ../tester/tester ../tester/testeru1 ../tester/testersp ../tester/testerspu1
	../tester/tester ./iutsse2
	../tester/testersp ./iutsse2
	../tester/testeru1 ./iutsse2
	../tester/testerspu1 ./iutsse2

testavx : iutavx ../tester/tester ../tester/testeru1 ../tester/testersp ../tester/testerspu1
	../tester/tester ./iutavx
	../tester/testersp ./iutavx
	../tester/testeru1 ./iutavx
	../tester/testerspu1 ./iutavx

testavx2 : iutavx2 ../tester/tester ../tester/testeru1 ../tester/testersp ../tester/testerspu1
	../tester/tester ./iutavx2
	../tester/testersp ./iutavx2
	../tester/testeru1 ./iutavx2
	../tester/testerspu1 ./iutavx2

testavx512f : iutavx512f ../tester/tester ../tester/testeru1 ../tester/testersp ../tester/testerspu1
	../tester/tester $(SDE) -- ./iutavx512f
	../tester/testersp $(SDE) -- ./iutavx512f
	../tester/testeru1 $(SDE) -- ./iutavx512f
	../tester/testerspu1 $(SDE) -- ./iutavx512f

testfma4 : iutfma4 ../tester/tester ../tester/testeru1 ../tester/testersp ../tester/testerspu1
	../tester/tester ./iutfma4
	../tester/testersp ./iutfma4
	../tester/testeru1 ./iutfma4
	../tester/testerspu1 ./iutfma4

clean :
	rm -f *~ *.o *.s *.out iutsse2 iutavx iutavx2 iutavx512f iutfma4 iutneon iutclangvec tester2sse2 tester2avx tester2avx2 tester2avx512f tester2fma4 tester2clangvec tester2fsse2 tester2favx tester2favx2 tester2favx512f tester2ffma4 tester2fclangvec
