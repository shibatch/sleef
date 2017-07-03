include ../../config.mk

CFLAGS += -g $(LDFLAGS) -Wall -Wno-unused -Wno-attributes -I../../include -I../common -L../../lib

TARGET = naivetestdp naivetestsp tutorial bench1D

ifdef ($(ENABLEFLOAT80)
TARGET+=naivetestld
endif

.PHONY: all
all : $(TARGET)

.PHONY: test

ifeq ($(OS),MinGW)
test : $(TARGET)
	cp ../../lib/*.dll .
	./naivetestdp 1
	./naivetestdp 2
	./naivetestdp 3
	./naivetestdp 4
	./naivetestdp 5
	./naivetestdp 10
	./naivetestsp 1
	./naivetestsp 2
	./naivetestsp 3
	./naivetestsp 4
	./naivetestsp 5
	./naivetestsp 10
else
test : $(TARGET)
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestdp 1
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestdp 2
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestdp 3
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestdp 4
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestdp 5
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestdp 10
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestsp 1
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestsp 2
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestsp 3
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestsp 4
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestsp 5
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../lib DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../lib ./naivetestsp 10
endif

.PHONY: all
all : $(TARGET)

naivetestdp : naivetest.c ../../lib/libsleef.$(DLLSUFFIX) ../../lib/libsleefdft.$(DLLSUFFIX)
	$(CC) $(CFLAGS) -DBASETYPEID=1 naivetest.c -o naivetestdp -lsleefdft -lsleef -lm 

naivetestsp : naivetest.c ../../lib/libsleef.$(DLLSUFFIX) ../../lib/libsleefdft.$(DLLSUFFIX)
	$(CC) $(CFLAGS) -DBASETYPEID=2 naivetest.c -o naivetestsp -lsleefdft -lsleef  -lm 

naivetestld : naivetest.c ../../lib/libsleef.$(DLLSUFFIX) ../../lib/libsleefdft.$(DLLSUFFIX)
	$(CC) $(CFLAGS) -DBASETYPEID=3 naivetest.c -o naivetestld -lsleefdft -lsleef  -lm 

naivetestqp : naivetest.c ../../lib/libsleef.$(DLLSUFFIX) ../../lib/libsleefdft.$(DLLSUFFIX)
	$(CC) $(CFLAGS) -DBASETYPEID=4 naivetest.c -o naivetestqp -lsleefdft -lsleef  -lm 

pi_fft_sleef : pi_fft.h ../../lib/libsleef.$(DLLSUFFIX) ../../lib/libsleefdft.$(DLLSUFFIX)
	if [ ! -f pi_fft.c ] ; then echo;\
		echo "Please put pi_fft.c in Prof. Takuya Ooura's FFT package.";\
		echo "Run wget http://www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz; tar xvfz fft.tgz fft/sample2/pi_fft.c; mv fft/sample2/pi_fft.c .";echo;echo; false;\
	fi
	cat pi_fft.h pi_fft.c > pi_fft_sleef.c
	$(CC) $(CFLAGS) -O3 pi_fft_sleef.c -lsleefdft -lsleef  -o pi_fft_sleef -lm 

bench1D : bench1D.c ../../lib/libsleef.$(DLLSUFFIX) ../../lib/libsleefdft.$(DLLSUFFIX)
	$(CC) $(CFLAGS) bench1D.c -lsleefdft -lsleef  -o bench1D -lm

bench1Dfftw : bench1D.c
	$(CC) $(CFLAGS) -DUSEFFTW bench1D.c -o bench1Dfftw -lm -lfftw3_threads -lfftw3

tutorial : tutorial.c
	$(CC) $(CFLAGS) tutorial.c -o tutorial -lsleefdft -lsleef -lm

fftwtest1D : fftwtest1D.c
	$(CC) $(CFLAGS) fftwtest1D.c -o fftwtest1D -lsleefdft -lsleef  -lm -lfftw3

fftwtest2D : fftwtest2D.c
	$(CC) $(CFLAGS) fftwtest2D.c -o fftwtest2D -lsleefdft -lsleef  -lm -lfftw3

../../lib/libsleef.$(DLLSUFFIX) :
	+"$(MAKE)" --directory=../../lib libsleef

 ../../lib/libsleefdft.$(DLLSUFFIX) :
	+"$(MAKE)" --directory=../../lib libsleefdft

.PHONY: clean
clean :
	rm -f *~ a.out *.obj *.lib *.dll *.exp *.exe
	rm -rf *.dSYM *.dylib
	rm -f *.so *.so.* *.a
	rm -f naivetestdp naivetestsp naivetestld naivetestqp bench1D bench1Dld bench1Dfftw
	rm -f fftwtest1D fftwtest2D tutorial
	rm -f pi_fft_sleef pi_fft_sleef.c pi.log pi.dat sleefdftplan.txt plan.txt

.PHONY: distclean
distclean : clean
	rm -rf fft.tgz* fft pi_fft.c

