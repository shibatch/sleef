include ../../config.mk

CFLAGS+=$(STRICTMATHFLAG) -O3 -I../common -I../arch -DNDEBUG

OBJ=sleefdp.o sleefsp.o
OBJ3=sleefldp.o sleeflsp.o

ifeq ($(ENABLEFLOAT80), 1)
OBJ+=sleefld.o
OBJ3+=sleeflld.o
endif

ifeq ($(ENABLEFLOAT128), 1)
OBJ+=sleefqp.o
OBJ+=sleeflqp.o
endif

#

ifeq ($(ARCH), aarch64)
OBJ+=sleefdpadvsimd.o sleefspadvsimd.o
OBJ2+=sleefgdpadvsimd.o sleefgspadvsimd.o
OBJ3+=sleefldpadvsimd.o sleeflspadvsimd.o
endif

ifeq ($(ARCH), arm)
OBJ+=sleefspneon32.o
endif

#

ifeq ($(ARCH), x86_64)
OBJ+=dispavx.o dispsse.o sleefdpsse2.o sleefspsse2.o sleefdpsse4.o sleefspsse4.o sleefdpavx.o sleefspavx.o
OBJ2+=sleefgdpsse2.o sleefgdpavx.o sleefgspsse2.o sleefgspavx.o
OBJ3+=sleefldpsse2.o sleeflspsse2.o sleefldpsse4.o sleeflspsse4.o sleefldpavx.o sleeflspavx.o

ifdef ENABLEFMA4
OBJ+=sleefdpfma4.o sleefspfma4.o
OBJ3+=sleefldpfma4.o sleeflspfma4.o
endif

ifeq ($(ENABLEAVX2), 1)
OBJ+=sleefdpavx2.o sleefspavx2.o
OBJ2+=sleefgdpavx2.o sleefgspavx2.o
OBJ3+=sleefldpavx2.o sleeflspavx2.o
endif

ifeq ($(ENABLEAVX512F), 1)
OBJ+=sleefdpavx512f.o sleefspavx512f.o
OBJ2+=sleefgdpavx512f.o sleefgspavx512f.o
OBJ3+=sleefldpavx512f.o sleeflspavx512f.o
endif

endif

#

.PHONY: all
all : libm.a libgm.a sleefllvm.a sleef.h

libm.a : $(OBJ)
	$(FLOCK) libm.a.lock -c 'ar rcs libm.a $(OBJ)'

libgm.a : $(OBJ2)
	$(FLOCK) libgm.a.lock -c 'ar rcs libgm.a $(OBJ2)'

sleefllvm.a : $(OBJ3)
	$(FLOCK) sleefllvm.a.lock -c 'ar rcs sleefllvm.a $(OBJ3)'

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

sleefldp.o : sleefdp.c rename_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_LLVMABI sleefdp.c -c -o sleefldp.o

sleeflsp.o : sleefsp.c rename_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_LLVMABI sleefsp.c -c -o sleeflsp.o

sleeflld.o : sleefld.c rename_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_LLVMABI sleefld.c -c -o sleeflld.o

sleeflqp.o : sleefqp.c rename_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_LLVMABI sleefqp.c -c -o sleeflqp.o

#

mkrename : mkrename.c
	$(FLOCK) mkrename.c -c '$(CC) $(CFLAGS) -Wall -Wno-unused mkrename.c -o mkrename'

mkrename_gnuabi : mkrename_gnuabi.c
	$(FLOCK) mkrename_gnuabi.c -c '$(CC) $(CFLAGS) -Wall -Wno-unused mkrename_gnuabi.c -o mkrename_gnuabi'

mkrename_llvm : mkrename_llvm.c
	$(FLOCK) mkrename_llvm.c -c '$(CC) $(CFLAGS) -Wall -Wno-unused mkrename_llvm.c -o mkrename_llvm'

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
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -DENABLE_GNUABI -msse2 sleefsimddp.c -c -o sleefgdpsse2.o

sleefgspsse2.o : sleefsimdsp.c renamesse2_gnuabi.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -DENABLE_GNUABI -msse2 sleefsimdsp.c -c -o sleefgspsse2.o

renamesse2_gnuabi.h : mkrename_gnuabi
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename_gnuabi'
	./mkrename_gnuabi sse2 b 2 4 _mm128d _mm128 _mm128i _mm128i '__SSE2__' > renamesse2_gnuabi.h

#

sleefldpsse2.o : sleefsimddp.c renamesse2_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -DENABLE_LLVMABI -msse2 sleefsimddp.c -c -o sleefldpsse2.o

sleeflspsse2.o : sleefsimdsp.c renamesse2_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -DENABLE_LLVMABI -msse2 sleefsimdsp.c -c -o sleeflspsse2.o

renamesse2_llvm.h : mkrename_llvm
	$(FLOCK) mkrename_llvm.c -c 'echo Acquiring lock for mkrename'
	./mkrename_llvm sse2 2 4 > renamesse2_llvm.h

##

sleefdpsse4.o : sleefsimddp.c renamesse4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE4 -msse2 -msse4.1 sleefsimddp.c -c -o sleefdpsse4.o

sleefspsse4.o : sleefsimdsp.c renamesse4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE4 -msse2 -msse4.1 sleefsimdsp.c -c -o sleefspsse4.o

renamesse4.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 2 4 sse4 > renamesse4.h

#

sleefldpsse4.o : sleefsimddp.c renamesse4_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE4 -DENABLE_LLVMABI -msse2 -msse4.1 sleefsimddp.c -c -o sleefldpsse4.o

sleeflspsse4.o : sleefsimdsp.c renamesse4_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE4 -DENABLE_LLVMABI -msse2 -msse4.1 sleefsimdsp.c -c -o sleeflspsse4.o

renamesse4_llvm.h : mkrename_llvm
	$(FLOCK) mkrename_llvm.c -c 'echo Acquiring lock for mkrename'
	./mkrename_llvm sse4 2 4 > renamesse4_llvm.h

##

dispavx.c : mkdisp
	cp dispavx.c.org dispavx.c
	./mkdisp 4 8 __m256d __m256 __m128i avx fma4 avx2 >> dispavx.c

dispsse.c : mkdisp
	cp dispsse.c.org dispsse.c
	./mkdisp 2 4 __m128d __m128 __m128i sse2 sse4 >> dispsse.c

mkdisp : mkdisp.c
	$(CC) $(CFLAGS) -Wall mkdisp.c -o mkdisp

DISPAVXOPT=
ifdef ENABLEFMA4
DISPAVXOPT+=-DENABLE_FMA4
endif
ifeq ($(ENABLEAVX2), 1)
DISPAVXOPT+=-DENABLE_AVX2
endif

dispavx.o : dispavx.c sleef.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) $(DISPAVXOPT) -mavx dispavx.c -c -o dispavx.o

dispsse.o : dispsse.c sleef.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) dispsse.c -c -o dispsse.o

#

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
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename_gnuabi'
	./mkrename_gnuabi avx c 4 8 __m256d __m256 __m128i 'struct { __m128i x, y; }' __AVX__ > renameavx_gnuabi.h

#

sleefldpavx.o : sleefsimddp.c renameavx_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX -DENABLE_LLVMABI -mavx sleefsimddp.c -c -o sleefldpavx.o

sleeflspavx.o : sleefsimdsp.c renameavx_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX -DENABLE_LLVMABI -mavx sleefsimdsp.c -c -o sleeflspavx.o

renameavx_llvm.h : mkrename_llvm
	$(FLOCK) mkrename_llvm.c -c 'echo Acquiring lock for mkrename_llvm'
	./mkrename_llvm avx 4 8 > renameavx_llvm.h

##

sleefdpfma4.o : sleefsimddp.c renamefma4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_FMA4 -mavx -mfma4 sleefsimddp.c -c -o sleefdpfma4.o

sleefspfma4.o : sleefsimdsp.c renamefma4.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_FMA4 -mavx -mfma4 sleefsimdsp.c -c -o sleefspfma4.o

renamefma4.h : mkrename
	$(FLOCK) mkrename.c -c 'echo Acquiring lock for mkrename'
	./mkrename 4 8 fma4 > renamefma4.h

#

sleefldpfma4.o : sleefsimddp.c renamefma4_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_FMA4 -DENABLE_LLVMABI -mavx -mfma4 sleefsimddp.c -c -o sleefldpfma4.o

sleeflspfma4.o : sleefsimdsp.c renamefma4_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_FMA4 -DENABLE_LLVMABI -mavx -mfma4 sleefsimdsp.c -c -o sleeflspfma4.o

renamefma4_llvm.h : mkrename_llvm
	$(FLOCK) mkrename_llvm.c -c 'echo Acquiring lock for mkrename_llvm'
	./mkrename_llvm fma4 4 8 > renamefma4_llvm.h

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
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename_gnuabi'
	./mkrename_gnuabi avx2 d 4 8 __m256d __m256 __m128i __m256i __AVX2__ > renameavx2_gnuabi.h

#

sleefldpavx2.o : sleefsimddp.c renameavx2_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2 -DENABLE_LLVMABI $(AVX2FLAG) sleefsimddp.c -c -o sleefldpavx2.o

sleeflspavx2.o : sleefsimdsp.c renameavx2_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX2 -DENABLE_LLVMABI $(AVX2FLAG) sleefsimdsp.c -c -o sleeflspavx2.o

renameavx2_llvm.h : mkrename_llvm
	$(FLOCK) mkrename_llvm.c -c 'echo Acquiring lock for mkrename_llvm'
	./mkrename_llvm avx2 4 8 > renameavx2_llvm.h

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
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename_gnuabi'
	./mkrename_gnuabi avx512f e 8 16 __m512d __m512 __m256i __m512i __AVX512F__ > renameavx512f_gnuabi.h

#

sleefldpavx512f.o : sleefsimddp.c renameavx512f_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX512F -DENABLE_LLVMABI $(AVX512FLAG) sleefsimddp.c -c -o sleefldpavx512f.o

sleeflspavx512f.o : sleefsimdsp.c renameavx512f_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_AVX512F -DENABLE_LLVMABI $(AVX512FLAG) sleefsimdsp.c -c -o sleeflspavx512f.o

renameavx512f_llvm.h : mkrename_llvm
	$(FLOCK) mkrename_llvm.c -c 'echo Acquiring lock for mkrename_llvm'
	./mkrename_llvm avx512f 8 16 > renameavx512f_llvm.h

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
	$(FLOCK) mkrename_gnuabi.c -c 'echo Acquiring lock for mkrename_gnuabi'
	./mkrename_gnuabi advsimd n 2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON > renameadvsimd_gnuabi.h

#

sleefldpadvsimd.o : sleefsimddp.c renameadvsimd_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -DENABLE_LLVMABI $(ADVSIMDFLAG) sleefsimddp.c -c -o sleefldpadvsimd.o

sleeflspadvsimd.o : sleefsimdsp.c renameadvsimd_llvm.h
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -DDORENAME -DENABLE_SSE2 -DENABLE_LLVMABI $(ADVSIMDFLAG) sleefsimdsp.c -c -o sleeflspadvsimd.o

renameadvsimd_llvm.h : mkrename_llvm
	$(FLOCK) mkrename_llvm.c -c 'echo Acquiring lock for mkrename'
	./mkrename_llvm advsimd 2 4 > renameadvsimd_llvm.h

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
	rm -f *~ *.o *.s libm.a libgm.a sleefllvm.a sleef.h a.out *.lock OBJ.txt OBJ2.txt
	rm -f *.obj *.lib *.dll *.exp *.exe
	rm -f mkrename renamesse2.h renamesse4.h renameavx.h renamefma4.h renameavx2.h renameavx512f.h renameneon32.h renameadvsimd.h renamedsp128.h renamedsp256.h
	rm -f mkdisp dispavx.c dispsse.c
	rm -f mkrename_gnuabi renamesse2_gnuabi.h renameavx_gnuabi.h renameavx2_gnuabi.h renameavx512f_gnuabi.h renameneon32_gnuabi.h renameadvsimd_gnuabi.h
	rm -f mkrename_llvm renamesse2_llvm.h renamesse4_llvm.h renameavx_llvm.h renamefma4_llvm.h renameavx2_llvm.h renameavx512f_llvm.h renameneon32_llvm.h renameadvsimd_llvm.h
