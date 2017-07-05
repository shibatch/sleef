include ../../config.mk

CFLAGS+=$(STRICTMATHFLAG) -O -g -I../common -I../arch -I../../include -I../libm -L../../lib

ifeq ($(OS), Darwin)
CFLAGS += -I/opt/local/include -L/opt/local/lib
ENVNAME_LD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH)
LIB=
else
CFLAGS += -I/opt/local/include -L/opt/local/lib
ENVNAME_LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)
LIB=-lrt
endif

#

TARGET_IUT=iut
TARGET_TESTER=tester tester2dp tester2sp

ifeq ($(ENABLEVECEXT), 1)
TARGET_IUT+=iutvecext
TARGET_TESTER+=tester2vecextdp tester2vecextsp
endif

ifeq ($(COMPILER), clang)
TARGET_IUT+=iutvecext
TARGET_TESTER+=tester2vecextdp tester2vecextsp
endif

ifeq ($(COMPILER), icc)
TARGET_IUT+=iutpurec
TARGET_TESTER+=tester2purecdp tester2purecsp
endif

ifdef ENABLEFLOAT80
TARGET_TESTER+=tester2ld
endif

ifdef ENABLEFLOAT128
TARGET_TESTER+=tester2qp
endif

ifeq ($(X86ARCH),1)

TARGET_IUT += iutsse2 iutsse4 iutdsp128
TARGET_TESTER += tester2sse2dp tester2sse2sp tester2sse4dp tester2sse4sp

ifeq ($(ENABLEAVX), 1)
TARGET_IUT += iutavx iutdsp256
TARGET_TESTER += tester2avxdp tester2avxsp

ifeq ($(ENABLEAVX2), 1)
TARGET_IUT+=iutavx2
TARGET_TESTER+=tester2avx2dp tester2avx2sp
endif

ifdef ENABLEFMA4
TARGET_IUT += iutfma4
TARGET_TESTER += tester2fma4dp tester2fma4sp
endif
endif #ifeq ($(ENABLEAVX), 1)

ifeq ($(ENABLEAVX2), 1)
TARGET_IUT+=iutavx2128
endif

ifeq ($(ENABLEAVX512F),1)
TARGET_IUT += iutavx512f
TARGET_TESTER += tester2avx512fdp tester2avx512fsp
endif

ifeq ($(shell $(CC) -v 2>&1 | grep -c "mingw"), 0)
ifeq ($(shell $(CC) -v 2>&1 | grep -c "cygwin"), 0)
ifneq ($(OS), Darwin)
CFLAGS+=-DENABLE_SYS_getrandom
endif
endif
endif

else ifeq ($(ARCH), arm)
TARGET_IUT += iutneon32
else ifeq ($(ARCH), aarch64)
TARGET_IUT += iutadvsimd
TARGET_TESTER += tester2advsimddp tester2advsimdsp
endif

.PHONY: all test
all : $(TARGET_IUT) $(TARGET_TESTER)

ifeq ($(OS),MinGW)
test : $(TARGET_IUT) tester
	cp ../../lib/*.dll .
	$(foreach var,$(TARGET_IUT), ./tester ./$(var) || exit;)
else
test : $(TARGET_IUT) tester
	$(foreach var,$(TARGET_IUT), $(ENVNAME_LD_LIBRARY_PATH)=../../lib:$$$(ENVNAME_LD_LIBRARY_PATH) ./tester ./$(var) || exit;)
endif

#

iut : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iut.c testerutil.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) iut.c testerutil.c -o iut -lsleef -lm $(LIB)

tester2dp : testerutil.c ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2dp.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DUSEMPFR tester2dp.c testerutil.c -o tester2dp -lsleef -lm -lmpfr $(LIB)

tester2sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2sp.c testerutil.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DUSEMPFR tester2sp.c testerutil.c -o tester2sp -lsleef -lm -lmpfr $(LIB)

tester2ld : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2ld.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) tester2ld.c -o tester2ld -lsleef -lm -lmpfr $(LIB)

tester2qp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2qp.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) tester2qp.c -o tester2qp -lsleef -lmpfr -lquadmath -lm $(LIB)

#

iutsimdmain.o : iutsimdmain.c
	$(CC) $(CFLAGS) iutsimdmain.c -c -o iutsimdmain.o $(LIB)

#

iutsse2 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renamesse2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE2 -msse2 iutsimdmain.o iutsimd.c testerutil.c -o iutsse2 -lsleef -lm $(LIB)

tester2sse2dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renamesse2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE2 -msse2 -DUSEMPFR tester2simddp.c testerutil.c -o tester2sse2dp -lsleef -lm -lmpfr $(LIB)

tester2sse2sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renamesse2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE2 -msse2 -DUSEMPFR tester2simdsp.c testerutil.c -o tester2sse2sp -lsleef -lm -lmpfr $(LIB)

../libm/renamesse2.h :
	+"$(MAKE)" --directory=../libm renamesse2.h

#

iutsse4 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renamesse4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE4 -msse4.1 iutsimdmain.o iutsimd.c testerutil.c -o iutsse4 -lsleef -lm $(LIB)

tester2sse4dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renamesse4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE4 -msse4.1 -DUSEMPFR tester2simddp.c testerutil.c -o tester2sse4dp -lsleef -lm -lmpfr $(LIB)

tester2sse4sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renamesse4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE4 -msse4.1 -DUSEMPFR tester2simdsp.c testerutil.c -o tester2sse4sp -lsleef -lm -lmpfr $(LIB)

../libm/renamesse4.h :
	+"$(MAKE)" --directory=../libm renamesse4.h

#

iutavx : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renameavx.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX -mavx iutsimdmain.o iutsimd.c testerutil.c -o iutavx -lsleef -lm $(LIB)

tester2avxdp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renameavx.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX -mavx -DUSEMPFR tester2simddp.c testerutil.c -o tester2avxdp -lsleef -lm -lmpfr $(LIB)

tester2avxsp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renameavx.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX -mavx -DUSEMPFR tester2simdsp.c testerutil.c -o tester2avxsp -lsleef -lm -lmpfr $(LIB)

../libm/renameavx.h :
	+"$(MAKE)" --directory=../libm renameavx.h

#

iutfma4 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renamefma4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_FMA4 -mavx -mfma4 iutsimdmain.o iutsimd.c testerutil.c -o iutfma4 -lsleef -lm $(LIB)

tester2fma4dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renamefma4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_FMA4 -mfma4 -DUSEMPFR tester2simddp.c testerutil.c -o tester2fma4dp -lsleef -lm -lmpfr $(LIB)

tester2fma4sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renamefma4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_FMA4 -mfma4 -DUSEMPFR tester2simdsp.c testerutil.c -o tester2fma4sp -lsleef -lm -lmpfr $(LIB)

../libm/renamefma4.h :
	+"$(MAKE)" --directory=../libm renamefma4.h

#

iutavx2 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renameavx2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2 $(AVX2FLAG) iutsimdmain.o iutsimd.c testerutil.c -o iutavx2 -lsleef -lm $(LIB)

tester2avx2dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renameavx2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2 $(AVX2FLAG) -DUSEMPFR tester2simddp.c testerutil.c -o tester2avx2dp -lsleef -lm -lmpfr $(LIB)

tester2avx2sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renameavx2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2 $(AVX2FLAG) -DUSEMPFR tester2simdsp.c testerutil.c -o tester2avx2sp -lsleef -lm -lmpfr $(LIB)

../libm/renameavx2.h :
	+"$(MAKE)" --directory=../libm renameavx2.h

#

iutavx2128 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renameavx2128.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2128 $(AVX2FLAG) iutsimdmain.o iutsimd.c testerutil.c -o iutavx2128 -lsleef -lm $(LIB)

#

iutavx512f : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renameavx512f.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX512F $(AVX512FLAG) iutsimdmain.o iutsimd.c testerutil.c -o iutavx512f -lsleef -lm $(LIB)

tester2avx512fdp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c testerutil.c ../libm/renameavx512f.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX512F $(AVX512FLAG) -DUSEMPFR tester2simddp.c testerutil.c -o tester2avx512fdp -lsleef -lm -lmpfr $(LIB)

tester2avx512fsp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c testerutil.c ../libm/renameavx512f.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX512F $(AVX512FLAG) -DUSEMPFR tester2simdsp.c testerutil.c -o tester2avx512fsp -lsleef -lm -lmpfr $(LIB)

../libm/renameavx512f.h :
	+"$(MAKE)" --directory=../libm renameavx512f.h

#

iutdsp128 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renamedsp128.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_DSP128 iutsimdmain.o iutsimd.c testerutil.c -o iutdsp128 -lsleef -lm $(LIB)

../libm/renamedsp128.h :
	+"$(MAKE)" --directory=../libm renamedsp128.h

iutdsp256 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renamedsp256.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_DSP256 -mavx iutsimdmain.o iutsimd.c testerutil.c -o iutdsp256 -lsleef -lm $(LIB)

../libm/renamedsp256.h :
	+"$(MAKE)" --directory=../libm renamedsp256.h

#

iutneon32 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimdmain.o iutsimd.c testerutil.c ../libm/renameneon32.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_NEON32 -mfpu=neon iutsimdmain.o iutsimd.c testerutil.c -o iutneon32 -lsleef -lm $(LIB)


#

iutadvsimd : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c ../arch/helperadvsimd.h iutsimdmain.o iutsimd.c 
	$(CC) $(CFLAGS) -DENABLE_ADVSIMD $(ADVSIMDFLAG) ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimdmain.o iutsimd.c testerutil.c -o iutadvsimd -lm  $(LIB)

tester2advsimddp :  ../libm/sleefsimddp.c ../libm/sleefsimdsp.c ../arch/helperadvsimd.h tester2simddp.c testerutil.c
	$(CC) $(CFLAGS) -DENABLE_ADVSIMD $(ADVSIMDFLAG) -DUSEMPFR ../libm/sleefsimddp.c ../libm/sleefsimdsp.c tester2simddp.c testerutil.c -o tester2advsimddp  -lm -lmpfr $(LIB)

tester2advsimdsp : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c ../arch/helperadvsimd.h tester2simdsp.c testerutil.c
	$(CC) $(CFLAGS) -DENABLE_ADVSIMD $(ADVSIMDFLAG) -DUSEMPFR ../libm/sleefsimddp.c ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c -o tester2advsimdsp  -lm -lmpfr $(LIB)

#

tester2vecextdp : ../libm/sleefsimddp.c tester2simddp.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_VECEXT -DUSEMPFR ../libm/sleefsimddp.c tester2simddp.c testerutil.c -lm -lmpfr -o tester2vecextdp $(LIB)

tester2vecextsp : ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_VECEXT -DUSEMPFR ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c -lm -lmpfr -o tester2vecextsp $(LIB)

iutvecext : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimdmain.o iutsimd.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_VECEXT ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimdmain.o iutsimd.c testerutil.c -lm -o iutvecext $(LIB)

#

tester2purecdp : ../libm/sleefsimddp.c tester2simddp.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_PUREC -DUSEMPFR ../libm/sleefsimddp.c tester2simddp.c testerutil.c -lm -lmpfr -o tester2purecdp $(LIB)

tester2purecsp : ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_PUREC -DUSEMPFR ../libm/sleefsimdsp.c tester2simdsp.c testerutil.c -lm -lmpfr -o tester2purecsp $(LIB)

iutpurec : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimdmain.o iutsimd.c testerutil.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_PUREC ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimdmain.o iutsimd.c testerutil.c -lm -o iutpurec $(LIB)

#

../../lib/libsleef.$(DLLSUFFIX) :
	+"$(MAKE)" --directory=../../lib libsleef.$(DLLSUFFIX)

../../include/sleef.h :
	+"$(MAKE)" --directory=../../include sleef.h

#

tester : tester.c testerutil.c
	$(CC) $(CFLAGS) -Wno-unused-result -DUSEMPFR tester.c testerutil.c -lm -lmpfr -o tester $(LIB)

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
	rm -f iut iutsse2 iutsse4 iutavx iutfma4 iutavx2 iutavx2128 iutavx512f iutneon32 iutadvsimd iutdsp128 iutdsp256
	rm -f tester tester2vecextdp tester2vecextsp iutvecext tester2purecdp tester2purecsp iutpurec
	rm -f tester2dp tester2sp tester2qp tester2ld tester2sse2dp tester2sse2sp tester2fma4dp tester2fma4sp tester2avxdp tester2avxsp tester2avx2dp tester2avx2sp tester2avx512fdp tester2avx512fsp tester2sse4dp tester2sse4sp
	rm -f tester2advsimddp tester2advsimdsp
	rm -f bench_s0_10 bench_s0_50 bench_s1_10 bench_s1_50 bench_s2_10 bench_s2_50 bench_a0_10 bench_a0_50
	rm -f *.csv *.xlsx

.PHONY: distclean
distclean : clean
