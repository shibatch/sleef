include ../../config.mk

CFLAGS+=$(STRICTMATHFLAG) -O -g -I../common -I../arch -I../../include -I../libm -L../../lib

ifeq ($(OS), Darwin)
CFLAGS += -I/opt/local/include -L/opt/local/lib
endif

#

TARGET=tester iut
TARGET2=tester2dp tester2sp

ifeq ($(ENABLEVECEXT), 1)
TARGET+=iutvecext
TARGET2+=tester2vecextdp tester2vecextsp
endif

ifeq ($(COMPILER), clang)
TARGET+=iutvecext
TARGET2+=tester2vecextdp tester2vecextsp
endif

ifeq ($(COMPILER), icc)
TARGET+=iutpurec
TARGET2+=tester2purecdp tester2purecsp
endif

ifdef ENABLEFLOAT80
TARGET2+=tester2ld
endif

ifdef ENABLEFLOAT128
TARGET2+=tester2qp
endif

ifeq ($(ARCH), x86_64)

TARGET += iutsse2 iutavx
TARGET2 += tester2sse2dp tester2sse2sp tester2avxdp tester2avxsp

ifeq ($(ENABLEAVX2), 1)
TARGET+=iutavx2
TARGET2+=tester2avx2dp tester2avx2sp
endif

ifdef ENABLEFMA4
TARGET += iutfma4
TARGET2 += tester2fma4dp tester2fma4sp
endif

ifeq ($(ENABLEAVX512F),1)
TARGET += iutavx512f
TARGET2 += tester2avx512fdp tester2avx512fsp
endif

ifeq ($(shell $(CC) -v 2>&1 | grep -c "mingw"), 0)
ifeq ($(shell $(CC) -v 2>&1 | grep -c "cygwin"), 0)
ifneq ($(OS), Darwin)
CFLAGS+=-DENABLE_SYS_getrandom
endif
endif
endif

else ifeq ($(ARCH), arm)
TARGET += iutneon32
else ifeq ($(ARCH), aarch64)
TARGET += iutadvsimd
TARGET2 += tester2advsimddp tester2advsimdsp
endif

.PHONY: all test
all : $(TARGET) $(TARGET2)

test: $(TARGET) tester
	$(foreach var,$(TARGET), LD_LIBRARY_PATH=../../lib:$LD_LIBRARY_PATH ./tester ./$(var);)

#

iut : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iut.c testerutil.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) iut.c testerutil.c -o iut -lsleef -lm -lrt

tester2dp : testerutil.c ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2dp.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DUSEMPFR tester2dp.c testerutil.c -o tester2dp -lsleef -lm -lmpfr

tester2sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2sp.c testerutil.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DUSEMPFR tester2sp.c testerutil.c -o tester2sp -lsleef -lm -lmpfr

tester2ld : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2ld.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) tester2ld.c -o tester2ld -lsleef -lm -lmpfr

tester2qp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2qp.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) tester2qp.c -o tester2qp -lsleef -lmpfr -lquadmath -lm

#

iutsse2 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c testerutil.c ../libm/renamesse2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE2 -msse2 iutsimd.c testerutil.c -o iutsse2 -lsleef -lm

tester2sse2dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renamesse2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE2 -msse2 -DUSEMPFR tester2simddp.c testerutil.c -o tester2sse2dp -lsleef -lm -lmpfr

tester2sse2sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renamesse2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE2 -msse2 -DUSEMPFR tester2simdsp.c testerutil.c -o tester2sse2sp -lsleef -lm -lmpfr

../libm/renamesse2.h :
	+"$(MAKE)" --directory=../libm renamesse2.h

#

iutavx : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c testerutil.c ../libm/renameavx.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX -mavx iutsimd.c testerutil.c -o iutavx -lsleef -lm

tester2avxdp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renameavx.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX -mavx -DUSEMPFR tester2simddp.c testerutil.c -o tester2avxdp -lsleef -lm -lmpfr

tester2avxsp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renameavx.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX -mavx -DUSEMPFR tester2simdsp.c testerutil.c -o tester2avxsp -lsleef -lm -lmpfr

../libm/renameavx.h :
	+"$(MAKE)" --directory=../libm renameavx.h

#

iutfma4 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c testerutil.c ../libm/renamefma4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_FMA4 -mavx -mfma4 iutsimd.c testerutil.c -o iutfma4 -lsleef -lm

tester2fma4dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renamefma4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_FMA4 -mfma4 -DUSEMPFR tester2simddp.c testerutil.c -o tester2fma4dp -lsleef -lm -lmpfr

tester2fma4sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renamefma4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_FMA4 -mfma4 -DUSEMPFR tester2simdsp.c testerutil.c -o tester2fma4sp -lsleef -lm -lmpfr

../libm/renamefma4.h :
	+"$(MAKE)" --directory=../libm renamefma4.h

#

iutavx2 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c testerutil.c ../libm/renameavx2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2 $(AVX2FLAG) iutsimd.c testerutil.c -o iutavx2 -lsleef -lm

tester2avx2dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renameavx2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2 $(AVX2FLAG) -DUSEMPFR tester2simddp.c testerutil.c -o tester2avx2dp -lsleef -lm -lmpfr

tester2avx2sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renameavx2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2 $(AVX2FLAG) -DUSEMPFR tester2simdsp.c testerutil.c -o tester2avx2sp -lsleef -lm -lmpfr

../libm/renameavx2.h :
	+"$(MAKE)" --directory=../libm renameavx2.h

#

iutavx512f : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c testerutil.c ../libm/renameavx512f.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX512F $(AVX512FLAG) iutsimd.c testerutil.c -o iutavx512f -lsleef -lm

tester2avx512fdp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renameavx512f.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX512F $(AVX512FLAG) -DUSEMPFR tester2simddp.c testerutil.c -o tester2avx512fdp -lsleef -lm -lmpfr

tester2avx512fsp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renameavx512f.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX512F $(AVX512FLAG) -DUSEMPFR tester2simdsp.c testerutil.c -o tester2avx512fsp -lsleef -lm -lmpfr

../libm/renameavx512f.h :
	+"$(MAKE)" --directory=../libm renameavx512f.h

#

iutneon32 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c testerutil.c ../libm/renameneon32.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_NEON32 -mfpu=neon iutsimd.c testerutil.c -o iutneon32 -lsleef -lm


#

iutadvsimd : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c ../arch/helperadvsimd.h
	$(CC) $(CFLAGS) -DENABLE_ADVSIMD $(ADVSIMDFLAG) ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c testerutil.c -o iutadvsimd -lm

tester2advsimddp :  ../libm/sleefsimddp.c ../libm/sleefsimdsp.c ../arch/helperadvsimd.h tester2simddp.c testerutil.c
	$(CC) $(CFLAGS) -DENABLE_ADVSIMD $(ADVSIMDFLAG) -DUSEMPFR ../libm/sleefsimddp.c ../libm/sleefsimdsp.c tester2simddp.c testerutil.c -o tester2advsimddp  -lm -lmpfr

tester2advsimdsp : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c ../arch/helperadvsimd.h tester2simdsp.c testerutil.c
	$(CC) $(CFLAGS) -DENABLE_ADVSIMD $(ADVSIMDFLAG) -DUSEMPFR ../libm/sleefsimddp.c ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c -o tester2advsimdsp  -lm -lmpfr

#

tester2vecextdp : ../libm/sleefsimddp.c tester2simddp.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_VECEXT -DUSEMPFR ../libm/sleefsimddp.c tester2simddp.c testerutil.c -lm -lmpfr -o tester2vecextdp

tester2vecextsp : ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_VECEXT -DUSEMPFR ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c -lm -lmpfr -o tester2vecextsp

iutvecext : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_VECEXT ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c testerutil.c -lm -o iutvecext

#

tester2purecdp : ../libm/sleefsimddp.c tester2simddp.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_PUREC -DUSEMPFR ../libm/sleefsimddp.c tester2simddp.c testerutil.c -lm -lmpfr -o tester2purecdp

tester2purecsp : ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_PUREC -DUSEMPFR ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c -lm -lmpfr -o tester2purecsp

iutpurec : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_PUREC ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c testerutil.c -lm -o iutpurec

#

../../lib/libsleef.$(DLLSUFFIX) :
	+"$(MAKE)" --directory=../../lib libsleef.$(DLLSUFFIX)

../../include/sleef.h :
	+"$(MAKE)" --directory=../../include sleef.h

#

tester : tester.c testerutil.c
	$(CC) $(CFLAGS) -Wno-unused-result -DUSEMPFR tester.c testerutil.c -lm -lmpfr -o tester

#

.PHONY: bench
bench : bench_s0_10 bench_s0_50 bench_s1_10 bench_s1_50 bench_s2_10 bench_s2_50 bench_a0_10 bench_a0_50

bench_s0_10 : bench_s0.c
	icc bench_s0.c -Wall -fimf-max-error=1.0 -fimf-domain-exclusion=0 -O0 -march=native -I../common -I../arch -I../../include -I../libm -L../../lib -Wno-attributes -I../libm -lsleef -lm -o bench_s0_10

bench_s0_50 : bench_s0.c
	icc bench_s0.c -Wall -fimf-max-error=5.0 -fimf-domain-exclusion=0 -O0 -march=native -I../common -I../arch -I../../include -I../libm -L../../lib -Wno-attributes -I../libm -lsleef -lm -o bench_s0_50

bench_s1_10 : bench_s1.c
	icc -DSVMLULP='"(1ulp)"' bench_s1.c -Wall -fimf-max-error=1.0 -fimf-domain-exclusion=0 -O0 -march=native -I../common -I../arch -I../../include -I../libm -L../../lib -Wno-attributes -I../libm -lsleef -lm -o bench_s1_10

bench_s1_50 : bench_s1.c
	icc -DSVMLULP='"(5ulp)"' -DSVMLONLY bench_s1.c -Wall -fimf-max-error=5.0 -fimf-domain-exclusion=0 -O0 -march=native -I../common -I../arch -I../../include -I../libm -L../../lib -Wno-attributes -I../libm -lsleef -lm -o bench_s1_50

bench_s2_10 : bench_s2.c doNothing.o
	icc -DSVMLULP='"(1ulp)"' bench_s2.c doNothing.o -Wall -fimf-max-error=1.0 -fimf-domain-exclusion=0 -O0 -march=native -I../common -I../arch -I../../include -I../libm -L../../lib -Wno-attributes -I../libm -lsleef -lm -o bench_s2_10

bench_s2_50 : bench_s2.c doNothing.o
	icc -DSVMLULP='"(5ulp)"' -DSVMLONLY bench_s2.c doNothing.o -Wall -fimf-max-error=5.0 -fimf-domain-exclusion=0 -O0 -march=native -I../common -I../arch -I../../include -I../libm -L../../lib -Wno-attributes -I../libm -lsleef -lm -o bench_s2_50

doNothing.o : doNothing.c
	icc doNothing.c -Wall -march=native -I../common -I../arch -I../../include -I../libm -L../../lib -Wno-attributes -I../libm -lsleef -lm -c

bench_a0_10 : bench_a0.c
	icc -DSVMLULP='"(1ulp)"' bench_a0.c -march=native -Wall -fimf-max-error=1.0 -fimf-domain-exclusion=0 -I../common -I../arch -I../../include -I../libm -L../../lib -Wno-attributes -I../libm -lm -lmpfr -lsleef -o bench_a0_10

bench_a0_50 : bench_a0.c
	icc -DSVMLULP='"(5ulp)"' -DSVMLONLY bench_a0.c -march=native -Wall -fimf-max-error=5.0 -fimf-domain-exclusion=0 -I../common -I../arch -I../../include -I../libm -L../../lib -Wno-attributes -I../libm -lm -lmpfr -lsleef -o bench_a0_50

#

.PHONY: clean
clean :
	rm -f *~ a.out *.obj *.lib *.dll *.exp *.exe *.stackdump
	rm -rf *.dSYM *.dylib
	rm -f *.so *.so.* *.a *.s *.o
	rm -f iut iutsse2 iutavx iutfma4 iutavx2 iutavx512f iutneon32 iutadvsimd
	rm -f tester tester2vecextdp tester2vecextsp iutvecext tester2purecdp tester2purecsp iutpurec
	rm -f tester2dp tester2sp tester2qp tester2ld tester2sse2dp tester2sse2sp tester2fma4dp tester2fma4sp tester2avxdp tester2avxsp tester2avx2dp tester2avx2sp tester2avx512fdp tester2avx512fsp
	rm -f tester2advsimddp tester2advsimdsp
	rm -f bench_s0_10 bench_s0_50 bench_s1_10 bench_s1_50 bench_s2_10 bench_s2_50 bench_a0_10 bench_a0_50
	rm -f *.csv *.xlsx

.PHONY: distclean
distclean : clean
