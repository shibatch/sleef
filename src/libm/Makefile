include ../../config.mk

CFLAGS+=$(STRICTMATHFLAG) -O3 -I../common -I../arch -DNDEBUG

OBJ=sleefdp.o sleefsp.o

ifeq ($(ENABLEFLOAT80), 1)
OBJ+=sleefld.o
endif

ifeq ($(ENABLEFLOAT128), 1)
OBJ+=sleefqp.o
endif

#

ifeq ($(ARCH), aarch64)
OBJ+=sleefdpadvsimd.o sleefspadvsimd.o
OBJ2+=sleefgdpadvsimd.o sleefgspadvsimd.o
endif

ifeq ($(ARCH), arm)
OBJ+=sleefspneon32.o
endif

#

ifeq ($(ARCH), x86_64)
OBJ+=dispsse.o dispavx.o sleefdpsse2.o sleefdpavx.o sleefspsse2.o sleefspavx.o sleefdpsse4.o sleefspsse4.o
OBJ2+=sleefgdpsse2.o sleefgdpavx.o sleefgspsse2.o sleefgspavx.o

ifeq ($(ENABLEAVX2), 1)
OBJ+=sleefdpavx2.o sleefspavx2.o sleefdpavx2128.o sleefspavx2128.o
OBJ2+=sleefgdpavx2.o sleefgspavx2.o
endif

ifdef ENABLEFMA4
OBJ+=sleefdpfma4.o sleefspfma4.o
endif

ifeq ($(ENABLEAVX512F), 1)
OBJ+=sleefdpavx512f.o sleefspavx512f.o
OBJ2+=sleefgdpavx512f.o sleefgspavx512f.o
endif

endif

#

.PHONY: all
all : libm.a libgm.a sleef.h

libm.a : $(OBJ)
	$(FLOCK) libm.a.lock -c 'ar rcs libm.a $(OBJ)'

libgm.a : $(OBJ2)
	$(FLOCK) libgm.a.lock -c 'ar rcs libgm.a $(OBJ2)'

OBJ.txt : $(OBJ)
	rm -rf OBJ.txt
	for i in `echo $(OBJ)`; do ../script/abspath.sh $(@D) $$i >> OBJ.txt; done

OBJ2.txt : $(OBJ2)
	rm -rf OBJ2.txt
	for i in `echo $(OBJ2)`; do ../script/abspath.sh $(@D) $$i >> OBJ2.txt; done

ifeq ($(ARCH), x86_64)
sleef.h : mkrename
	cp sleeflibm.h.org sleef.h
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 __m128d __m128 __m128i __m128i __SSE2__ >> sleef.h
	./mkrename 2 4 __m128d __m128 __m128i __m128i __SSE2__ sse2 >> sleef.h
	./mkrename 2 4 __m128d __m128 __m128i __m128i __SSE2__ sse4 >> sleef.h
	./mkrename 2 4 __m128d __m128 __m128i __m128i __SSE2__ avx2128 >> sleef.h
	./mkrename 4 8 __m256d __m256 __m128i 'struct { __m128i x, y; }' __AVX__ >> sleef.h
	./mkrename 4 8 __m256d __m256 __m128i 'struct { __m128i x, y; }' __AVX__ avx >> sleef.h
	./mkrename 4 8 __m256d __m256 __m128i 'struct { __m128i x, y; }' __AVX__ fma4 >> sleef.h
	./mkrename 4 8 __m256d __m256 __m128i __m256i __AVX__ avx2 >> sleef.h
	./mkrename 8 16 __m512d __m512 __m256i __m512i __AVX512F__ avx512f >> sleef.h
	echo '#undef IMPORT' >> sleef.h
	echo '#endif' >> sleef.h
	cp sleef.h ../../include
else ifeq ($(ARCH), arm)
sleef.h : mkrename
	cp sleeflibm.h.org sleef.h
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 - float32x4_t int32x4_t int32x4_t __ARM_NEON__ neon >> sleef.h
	echo '#undef IMPORT' >> sleef.h
	echo '#endif' >> sleef.h
	cp sleef.h ../../include
else ifeq ($(ARCH), aarch64)
sleef.h : mkrename
	cp sleeflibm.h.org sleef.h
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON advsimd >> sleef.h
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
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME sleefdp.c -c -o sleefdp.o

sleefsp.o : sleefsp.c rename.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME sleefsp.c -c -o sleefsp.o

sleefld.o : sleefld.c rename.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME sleefld.c -c -o sleefld.o

sleefqp.o : sleefqp.c rename.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME sleefqp.c -c -o sleefqp.o

#

mkrename : mkrename.c
	$(FLOCK) mkrename.c -c '$(CC) $(CFLAGS) -Wall -Wno-unused mkrename.c -o mkrename'

mkrename_gnuabi : mkrename_gnuabi.c
	$(FLOCK) mkrename_gnuabi.c -c '$(CC) $(CFLAGS) -Wall -Wno-unused mkrename_gnuabi.c -o mkrename_gnuabi'

##

sleefdpsse2.o : sleefsimddp.c renamesse2.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -msse2 sleefsimddp.c -c -o sleefdpsse2.o

sleefspsse2.o : sleefsimdsp.c renamesse2.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -msse2 sleefsimdsp.c -c -o sleefspsse2.o

renamesse2.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 sse2 > renamesse2.h

#

sleefgdpsse2.o : sleefsimddp.c renamesse2_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -DENABLE_GNUABI -msse2 -msse4.1 sleefsimddp.c -c -o sleefgdpsse2.o

sleefgspsse2.o : sleefsimdsp.c renamesse2_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -DENABLE_GNUABI -msse2 -msse4.1 sleefsimdsp.c -c -o sleefgspsse2.o

renamesse2_gnuabi.h : mkrename_gnuabi
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename'
	./mkrename_gnuabi sse2 b 2 4 _mm128d _mm128 _mm128i _mm128i '__SSE2__' > renamesse2_gnuabi.h

##

sleefdpsse4.o : sleefsimddp.c renamesse4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE4 -msse4.1 sleefsimddp.c -c -o sleefdpsse4.o

sleefspsse4.o : sleefsimdsp.c renamesse4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE4 -msse4.1 sleefsimdsp.c -c -o sleefspsse4.o

renamesse4.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 sse4 > renamesse4.h

##

DISPAVXOPT=
ifeq ($ENABLEFMA4, 1)
DISPAVXOPT+=-DENABLE_FMA4
endif
ifeq ($(ENABLEAVX2), 1)
DISPAVXOPT+=-DENABLE_AVX2
endif

dispavx.c : mkdisp
	cp dispavx.c.org dispavx.c
	./mkdisp 4 8 __m256d __m256 __m128i avx fma4 avx2 >> dispavx.c

dispsse.c : mkdisp
	cp dispsse.c.org dispsse.c
	./mkdisp 2 4 __m128d __m128 __m128i sse2 sse4 avx2128 >> dispsse.c

mkdisp : mkdisp.c
	$(CC) $(CFLAGS) -Wall mkdisp.c -o mkdisp

dispavx.o : dispavx.c sleef.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) $(DISPAVXOPT) -mavx dispavx.c -c -o dispavx.o

dispsse.o : dispsse.c sleef.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) $(DISPAVXOPT) dispsse.c -c -o dispsse.o

##

sleefdpavx.o : sleefsimddp.c renameavx.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX -mavx sleefsimddp.c -c -o sleefdpavx.o

sleefspavx.o : sleefsimdsp.c renameavx.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX -mavx sleefsimdsp.c -c -o sleefspavx.o

renameavx.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 4 8 avx > renameavx.h

#

sleefgdpavx.o : sleefsimddp.c renameavx_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX -DENABLE_GNUABI -mavx sleefsimddp.c -c -o sleefgdpavx.o

sleefgspavx.o : sleefsimdsp.c renameavx_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX -DENABLE_GNUABI -mavx sleefsimdsp.c -c -o sleefgspavx.o

renameavx_gnuabi.h : mkrename_gnuabi
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename'
	./mkrename_gnuabi avx c 4 8 __m256d __m256 __m128i 'struct { __m128i x, y; }' __AVX__ > renameavx_gnuabi.h

##

sleefdpfma4.o : sleefsimddp.c renamefma4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_FMA4 -mavx -mfma4 sleefsimddp.c -c -o sleefdpfma4.o

sleefspfma4.o : sleefsimdsp.c renamefma4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_FMA4 -mavx -mfma4 sleefsimdsp.c -c -o sleefspfma4.o

renamefma4.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 4 8 fma4 > renamefma4.h

##

sleefdpavx2.o : sleefsimddp.c renameavx2.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2 $(AVX2FLAG) sleefsimddp.c -c -o sleefdpavx2.o

sleefspavx2.o : sleefsimdsp.c renameavx2.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2 $(AVX2FLAG) sleefsimdsp.c -c -o sleefspavx2.o

renameavx2.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 4 8 avx2 > renameavx2.h

#

sleefgdpavx2.o : sleefsimddp.c renameavx2_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2 -DENABLE_GNUABI $(AVX2FLAG) sleefsimddp.c -c -o sleefgdpavx2.o

sleefgspavx2.o : sleefsimdsp.c renameavx2_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2 -DENABLE_GNUABI $(AVX2FLAG) sleefsimdsp.c -c -o sleefgspavx2.o

renameavx2_gnuabi.h : mkrename_gnuabi
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename'
	./mkrename_gnuabi avx2 d 4 8 __m256d __m256 __m128i __m256i __AVX2__ > renameavx2_gnuabi.h

##

sleefdpavx2128.o : sleefsimddp.c renameavx2128.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2128 $(AVX2FLAG) sleefsimddp.c -c -o sleefdpavx2128.o

sleefspavx2128.o : sleefsimdsp.c renameavx2128.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2128 $(AVX2FLAG) sleefsimdsp.c -c -o sleefspavx2128.o

renameavx2128.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 avx2128 > renameavx2128.h

##

sleefdpavx512f.o : sleefsimddp.c renameavx512f.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX512F $(AVX512FLAG) sleefsimddp.c -c -o sleefdpavx512f.o

sleefspavx512f.o : sleefsimdsp.c renameavx512f.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX512F $(AVX512FLAG) sleefsimdsp.c -c -o sleefspavx512f.o

renameavx512f.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 8 16 avx512f > renameavx512f.h

#

sleefgdpavx512f.o : sleefsimddp.c renameavx512f_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX512F -DENABLE_GNUABI $(AVX512FLAG) sleefsimddp.c -c -o sleefgdpavx512f.o

sleefgspavx512f.o : sleefsimdsp.c renameavx512f_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX512F -DENABLE_GNUABI $(AVX512FLAG) sleefsimdsp.c -c -o sleefgspavx512f.o

renameavx512f_gnuabi.h : mkrename_gnuabi
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename'
	./mkrename_gnuabi avx512f e 8 16 __m512d __m512 __m256i __m512i __AVX512F__ > renameavx512f_gnuabi.h

##

sleefspneon32.o : sleefsimdsp.c renameneon32.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_NEON32 -mfpu=neon sleefsimdsp.c -c -o sleefspneon32.o

renameneon32.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 neon > renameneon32.h

##

sleefspadvsimd.o : sleefsimdsp.c  ../arch/helperadvsimd.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) $(ADVSIMDFLAG)  -DENABLE_ADVSIMD  sleefsimdsp.c -c -o sleefspadvsimd.o

sleefdpadvsimd.o : sleefsimddp.c ../arch/helperadvsimd.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) $(ADVSIMDFLAG)  -DENABLE_ADVSIMD  sleefsimddp.c -c -o sleefdpadvsimd.o

renameadvsimd.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 advsimd > renameadvsimd.h

#

sleefgdpadvsimd.o : sleefsimddp.c renameadvsimd_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_ADVSIMD -DENABLE_GNUABI $(ADVSIMDFLAG) sleefsimddp.c -c -o sleefgdpadvsimd.o

sleefgspadvsimd.o : sleefsimdsp.c renameadvsimd_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_ADVSIMD -DENABLE_GNUABI $(ADVSIMDFLAG) sleefsimdsp.c -c -o sleefgspadvsimd.o

renameadvsimd_gnuabi.h : mkrename_gnuabi
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename'
	./mkrename_gnuabi advsimd n 2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON > renameadvsimd_gnuabi.h

#

renamedsp128.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 > renamedsp128.h

renamedsp256.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 4 8 > renamedsp256.h

#

.PHONY: clean
clean :
	rm -f *~ *.o *.s libm.a libgm.a sleef.h a.out *.lock OBJ.txt OBJ2.txt
	rm -f *.obj *.lib *.dll *.exp *.exe
	rm -f mkrename renamesse2.h renamesse4.h renameavx.h renamefma4.h renameavx2.h renameavx2128.h renameavx512f.h renameneon32.h renameadvsimd.h renamedsp128.h renamedsp256.h
	rm -f mkdisp dispavx.c dispsse.c
	rm -f mkrename_gnuabi renamesse2_gnuabi.h renameavx_gnuabi.h renamefma4_gnuabi.h renameavx2_gnuabi.h renameavx512f_gnuabi.h renameneon32_gnuabi.h renameadvsimd_gnuabi.h
