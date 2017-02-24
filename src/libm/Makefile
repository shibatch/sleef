include ../../config.mk

CFLAGS+=$(STRICTMATHFLAG) -O3 -I../common -I../arch

OBJ=sleefdp.o sleefsp.o


ifeq ($(ARCH), aarch64)
OBJ+=sleefdpadvsimd.o
OBJ+=sleefspadvsimd.o
endif

ifdef ENABLEFLOAT80
OBJ+=sleefld.o
endif

ifdef ENABLEFLOAT128
OBJ+=sleefqp.o
endif

ifeq ($(ARCH), x86_64)
OBJ+=sleefdpsse2.o sleefdpavx.o sleefdpavx2.o
OBJ+=sleefspsse2.o sleefspavx.o sleefspavx2.o
ifdef ENABLEFMA4
OBJ+=sleefdpfma4.o
OBJ+=sleefspfma4.o
endif
ifdef ENABLEAVX512F
OBJ+=sleefdpavx512f.o
OBJ+=sleefspavx512f.o
endif

ifeq ($(ARCH), arm)
OBJ+=sleefspneon32.o
endif

endif

.PHONY: all
all : libm.a sleef.h

libm.a : $(OBJ)
	$(FLOCK) libm.a.lock -c 'ar rcs libm.a $(OBJ)'

ifeq ($(ARCH), x86_64)
sleef.h : mkrename
	cp sleeflibm.h.org sleef.h
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename sse2 2 4 __m128d __m128 __m128i __m128i __SSE2__ >> sleef.h
	./mkrename avx 4 8 __m256d __m256 __m128i 'struct { __m128i x, y; }' __AVX__ >> sleef.h
	./mkrename fma4 4 8 __m256d __m256 __m128i 'struct { __m128i x, y; }' __FMA4__ >> sleef.h
	./mkrename avx2 4 8 __m256d __m256 __m128i __m256i __AVX2__ >> sleef.h
	./mkrename avx512f 8 16 __m512d __m512 __m256i __m512i __AVX512F__ >> sleef.h
	echo '#undef IMPORT' >> sleef.h
	echo '#endif' >> sleef.h
	cp sleef.h ../../include
else ifeq ($(ARCH), arm)
sleef.h : mkrename
	cp sleeflibm.h.org sleef.h
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename neon 2 4 - float32x4_t int32x4_t int32x4_t __ARM_NEON__ >> sleef.h
	echo '#undef IMPORT' >> sleef.h
	echo '#endif' >> sleef.h
	cp sleef.h ../../include
else
sleef.h :
	cp sleeflibm.h.org sleef.h
	echo '#undef IMPORT' >> sleef.h
	echo '#endif' >> sleef.h
	cp sleef.h ../../include
endif

#

sleefdp.o : sleefdp.c rename.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DNDEBUG -DDORENAME sleefdp.c -c -o sleefdp.o

sleefsp.o : sleefsp.c rename.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DNDEBUG -DDORENAME sleefsp.c -c -o sleefsp.o

sleefld.o : sleefld.c rename.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DNDEBUG -DDORENAME sleefld.c -c -o sleefld.o

sleefqp.o : sleefqp.c rename.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DNDEBUG -DDORENAME sleefqp.c -c -o sleefqp.o

#

mkrename : mkrename.c
	$(FLOCK) mkrename.c -c '$(CC) $(CFLAGS) -Wall -Wno-unused mkrename.c -o mkrename'

#

sleefdpsse2.o : sleefsimddp.c renamesse2.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -msse2 -msse4.1 sleefsimddp.c -c -o sleefdpsse2.o

sleefspsse2.o : sleefsimdsp.c renamesse2.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -msse2 -msse4.1 sleefsimdsp.c -c -o sleefspsse2.o

renamesse2.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename sse2 2 4 > renamesse2.h

#

sleefdpavx.o : sleefsimddp.c renameavx.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX -mavx sleefsimddp.c -c -o sleefdpavx.o

sleefspavx.o : sleefsimdsp.c renameavx.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX -mavx sleefsimdsp.c -c -o sleefspavx.o

renameavx.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename avx 4 8 > renameavx.h

#

sleefdpfma4.o : sleefsimddp.c renamefma4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_FMA4 -mavx -mfma4 sleefsimddp.c -c -o sleefdpfma4.o

sleefspfma4.o : sleefsimdsp.c renamefma4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_FMA4 -mavx -mfma4 sleefsimdsp.c -c -o sleefspfma4.o

renamefma4.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename fma4 4 8 > renamefma4.h

#

sleefdpavx2.o : sleefsimddp.c renameavx2.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2 $(AVX2FLAG) sleefsimddp.c -c -o sleefdpavx2.o

sleefspavx2.o : sleefsimdsp.c renameavx2.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2 $(AVX2FLAG) sleefsimdsp.c -c -o sleefspavx2.o

renameavx2.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename avx2 4 8 > renameavx2.h

#

sleefdpavx512f.o : sleefsimddp.c renameavx512f.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX512F $(AVX512FLAG) sleefsimddp.c -c -o sleefdpavx512f.o

sleefspavx512f.o : sleefsimdsp.c renameavx512f.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX512F $(AVX512FLAG) sleefsimdsp.c -c -o sleefspavx512f.o

renameavx512f.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename avx512f 8 16 > renameavx512f.h

#

sleefspneon32.o : sleefsimdsp.c renameneon32.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_NEON32 -mfpu=neon sleefsimdsp.c -c -o sleefspneon32.o

renameneon32.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename neon 2 4 > renameneon32.h

#

sleefspadvsimd.o : sleefsimdsp.c  ../arch/helperadvsimd.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) $(ADVSIMDFLAG)  -DENABLE_ADVSIMD  sleefsimdsp.c -c -o sleefspadvsimd.o
sleefdpadvsimd.o : sleefsimddp.c ../arch/helperadvsimd.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) $(ADVSIMDFLAG)  -DENABLE_ADVSIMD  sleefsimddp.c -c -o sleefdpadvsimd.o

#

.PHONY: clean
clean :
	rm -f *~ *.o *.s libm.a sleef.h a.out *.lock
	rm -f *.obj *.lib *.dll *.exp *.exe
	rm -f mkrename renamesse2.h renameavx.h renamefma4.h renameavx2.h renameavx512f.h renameneon32.h renameadvsimd.h
