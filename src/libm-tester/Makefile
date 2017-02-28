include ../../config.mk

CFLAGS+=$(STRICTMATHFLAG) -O -g -I../common -I../arch -I../../include -I../libm -L../../lib

ifeq ($(OS), Darwin)
CFLAGS += -I/opt/local/include -L/opt/local/lib
endif

#

TARGET=tester testeru1 testersp testerspu1 iut
TARGET2=tester2dp tester2sp

ifeq ($(COMPILER), gcc)
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

TARGET += iutsse2 iutavx iutavx2
TARGET2 += tester2sse2dp tester2sse2sp tester2avxdp tester2avxsp tester2avx2dp tester2avx2sp

ifdef ENABLEFMA4
TARGET += iutfma4
TARGET2 += tester2fma4dp tester2fma4sp
endif

ifdef ENABLEAVX512F
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

.PHONY: all
all : $(TARGET) $(TARGET2)

.PHONY: test
test : $(TARGET)
	bash -c ./runtester.sh

#

iut : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iut.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) iut.c -o iut -lsleef -lm

tester2dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2dp.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) tester2dp.c -o tester2dp -lsleef -lm -lmpfr

tester2sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2sp.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) tester2sp.c -o tester2sp -lsleef -lm -lmpfr

tester2ld : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2ld.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) tester2ld.c -o tester2ld -lsleef -lm -lmpfr

tester2qp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2qp.c ../libm/rename.h ../../include/sleef.h
	$(CC) $(CFLAGS) tester2qp.c -o tester2qp -lsleef -lmpfr -lquadmath -lm

#

iutsse2 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c ../libm/renamesse2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE2 -msse2 iutsimd.c -o iutsse2 -lsleef -lm

tester2sse2dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c ../libm/renamesse2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE2 -msse2 tester2simddp.c -o tester2sse2dp -lsleef -lm -lmpfr

tester2sse2sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c ../libm/renamesse2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_SSE2 -msse2 tester2simdsp.c -o tester2sse2sp -lsleef -lm -lmpfr

../libm/renamesse2.h :
	+"$(MAKE)" --directory=../libm renamesse2.h

#

iutavx : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c ../libm/renameavx.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX -mavx iutsimd.c -o iutavx -lsleef -lm

tester2avxdp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c ../libm/renameavx.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX -mavx tester2simddp.c -o tester2avxdp -lsleef -lm -lmpfr

tester2avxsp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c ../libm/renameavx.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX -mavx tester2simdsp.c -o tester2avxsp -lsleef -lm -lmpfr

../libm/renameavx.h :
	+"$(MAKE)" --directory=../libm renameavx.h

#

iutfma4 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c ../libm/renamefma4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_FMA4 -mavx -mfma4 iutsimd.c -o iutfma4 -lsleef -lm

tester2fma4dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c ../libm/renamefma4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_FMA4 -mfma4 tester2simddp.c -o tester2fma4dp -lsleef -lm -lmpfr

tester2fma4sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c ../libm/renamefma4.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_FMA4 -mfma4 tester2simdsp.c -o tester2fma4sp -lsleef -lm -lmpfr

../libm/renamefma4.h :
	+"$(MAKE)" --directory=../libm renamefma4.h

#

iutavx2 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c ../libm/renameavx2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2 $(AVX2FLAG) iutsimd.c -o iutavx2 -lsleef -lm

tester2avx2dp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c ../libm/renameavx2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2 $(AVX2FLAG) tester2simddp.c -o tester2avx2dp -lsleef -lm -lmpfr

tester2avx2sp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c ../libm/renameavx2.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX2 $(AVX2FLAG) tester2simdsp.c -o tester2avx2sp -lsleef -lm -lmpfr

../libm/renameavx2.h :
	+"$(MAKE)" --directory=../libm renameavx2.h

#

iutavx512f : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c ../libm/renameavx512f.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX512F $(AVX512FLAG) iutsimd.c -o iutavx512f -lsleef -lm

tester2avx512fdp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simddp.c ../libm/renameavx512f.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX512F $(AVX512FLAG) tester2simddp.c -o tester2avx512fdp -lsleef -lm -lmpfr

tester2avx512fsp : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h tester2simdsp.c ../libm/renameavx512f.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_AVX512F $(AVX512FLAG) tester2simdsp.c -o tester2avx512fsp -lsleef -lm -lmpfr

../libm/renameavx512f.h :
	+"$(MAKE)" --directory=../libm renameavx512f.h

#

iutneon32 : ../../lib/libsleef.$(DLLSUFFIX) ../../include/sleef.h iutsimd.c ../libm/renameneon32.h ../../include/sleef.h
	$(CC) $(CFLAGS) -DENABLE_NEON32 -mfpu=neon iutsimd.c -o iutneon32 -lsleef -lm


#

iutadvsimd : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c ../arch/helperadvsimd.h
	$(CC) $(CFLAGS) -DENABLE_ADVSIMD $(ADVSIMDFLAG) ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c -o iutadvsimd -lm

tester2advsimddp :  ../libm/sleefsimddp.c ../libm/sleefsimdsp.c ../arch/helperadvsimd.h tester2simddp.c
	$(CC) $(CFLAGS) -DENABLE_ADVSIMD $(ADVSIMDFLAG) ../libm/sleefsimddp.c ../libm/sleefsimdsp.c tester2simddp.c -o tester2advsimddp  -lm -lmpfr

tester2advsimdsp : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c ../arch/helperadvsimd.h tester2simdsp.c
	$(CC) $(CFLAGS) -DENABLE_ADVSIMD $(ADVSIMDFLAG) ../libm/sleefsimddp.c ../libm/sleefsimdsp.c tester2simdsp.c -o tester2advsimdsp  -lm -lmpfr

#

tester2vecextdp : ../libm/sleefsimddp.c tester2simddp.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_VECEXT ../libm/sleefsimddp.c tester2simddp.c -lm -lmpfr -o tester2vecextdp

tester2vecextsp : ../libm/sleefsimdsp.c tester2simdsp.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_VECEXT ../libm/sleefsimdsp.c tester2simdsp.c -lm -lmpfr -o tester2vecextsp

iutvecext : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_VECEXT ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c -lm -o iutvecext

#

tester2purecdp : ../libm/sleefsimddp.c tester2simddp.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_PUREC ../libm/sleefsimddp.c tester2simddp.c -lm -lmpfr -o tester2purecdp

tester2purecsp : ../libm/sleefsimdsp.c tester2simdsp.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_PUREC ../libm/sleefsimdsp.c tester2simdsp.c -lm -lmpfr -o tester2purecsp

iutpurec : ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c
	$(CC) $(CFLAGS) -Wno-attributes -I../libm -DENABLE_PUREC ../libm/sleefsimddp.c ../libm/sleefsimdsp.c iutsimd.c -lm -o iutpurec

#

../../lib/libsleef.$(DLLSUFFIX) :
	+"$(MAKE)" --directory=../../lib libsleef.$(DLLSUFFIX)

../../include/sleef.h :
	+"$(MAKE)" --directory=../../include sleef.h

#

tester : tester.c
	$(CC) $(CFLAGS) -Wno-unused-result tester.c -lm -lmpfr -o tester

testeru1 : testeru1.c
	$(CC) $(CFLAGS) -Wno-unused-result testeru1.c -lm -lmpfr -o testeru1

testersp : testersp.c
	$(CC) $(CFLAGS) -Wno-unused-result testersp.c -lm -lmpfr -o testersp

testerspu1 : testerspu1.c
	$(CC) $(CFLAGS) -Wno-unused-result testerspu1.c -lm -lmpfr -o testerspu1

#

.PHONY: clean
clean :
	rm -f *~ a.out *.obj *.lib *.dll *.exp *.exe *.stackdump
	rm -rf *.dSYM *.dylib
	rm -f *.so *.so.* *.a
	rm -f iut iutsse2 iutavx iutfma4 iutavx2 iutavx512f iutneon32 iutadvsimd
	rm -f tester testeru1 testersp testerspu1 tester2vecextdp tester2vecextsp iutvecext tester2purecdp tester2purecsp iutpurec
	rm -f tester2dp tester2sp tester2qp tester2ld tester2sse2dp tester2sse2sp tester2fma4dp tester2fma4sp tester2avxdp tester2avxsp tester2avx2dp tester2avx2sp tester2avx512fdp tester2avx512fsp

.PHONY: distclean
distclean : clean
