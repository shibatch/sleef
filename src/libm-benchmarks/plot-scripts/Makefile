BENCHSLEEF := $(shell command -v benchsleef 2> /dev/null)
ICCAVAILABLE := $(shell command -v icc 2> /dev/null)

all :
ifndef BENCHSLEEF
	@echo
	@echo Please add the bin directory under build directory to PATH environment variable and run make once again.
	@echo e.g. export PATH='$$'PATH:'`pwd`'/../../../build/bin
	@echo
else
	@echo
	@echo You can start measurement by "'"make measure"'".
ifdef ICCAVAILABLE
	@echo You can start measurement with SVML by "'"make measureSVML"'".
endif
	@echo Then, you can plot the results of measurement by "'"make plot"'".
	@echo	
	@echo You have to install java and gnuplot to do plotting.
	@echo Stop all tasks on the computer before starting measurement.
	@echo
endif

benchsvml128_10.o : benchsvml128.c ../bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml128.c -Wall -I.. -DSVMLULP=1 -fimf-max-error=1.0 -fimf-domain-exclusion=0 -march=core-avx2 -O0 -lm -c -o benchsvml128_10.o

benchsvml128_40.o : benchsvml128.c ../bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml128.c -Wall -I.. -DSVMLULP=4 -fimf-max-error=4.0 -fimf-domain-exclusion=0 -march=core-avx2 -O0 -lm -c -o benchsvml128_40.o

benchsvml256_10.o : benchsvml256.c ../bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml256.c -Wall -I.. -DSVMLULP=1 -fimf-max-error=1.0 -fimf-domain-exclusion=0 -march=core-avx2 -O0 -lm -c -o benchsvml256_10.o

benchsvml256_40.o : benchsvml256.c ../bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml256.c -Wall -I.. -DSVMLULP=4 -fimf-max-error=4.0 -fimf-domain-exclusion=0 -march=core-avx2 -O0 -lm -c -o benchsvml256_40.o

benchsvml512_10.o : benchsvml512.c ../bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml512.c -Wall -I.. -DSVMLULP=1 -fimf-max-error=1.0 -fimf-domain-exclusion=0 -xCOMMON-AVX512 -O0 -lm -c -o benchsvml512_10.o

benchsvml512_40.o : benchsvml512.c ../bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml512.c -Wall -I.. -DSVMLULP=4 -fimf-max-error=4.0 -fimf-domain-exclusion=0 -xCOMMON-AVX512 -O0 -lm -c -o benchsvml512_40.o


benchsvml_10 : benchsvml.c benchsvml128_10.o benchsvml256_10.o benchsvml512_10.o ../bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml.c benchsvml128_10.o benchsvml256_10.o benchsvml512_10.o -Wall -I.. -DSVMLULP=1 -fimf-max-error=1.0 -fimf-domain-exclusion=0 -O0 -march=native -lm -o benchsvml_10

benchsvml_40 : benchsvml.c benchsvml128_40.o benchsvml256_40.o benchsvml512_40.o ../bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml.c benchsvml128_40.o benchsvml256_40.o benchsvml512_40.o -Wall -I.. -DSVMLULP=4 -fimf-max-error=4.0 -fimf-domain-exclusion=0 -O0 -march=native -lm -o benchsvml_40

#

ProcessData.class : ProcessData.java
	javac ProcessData.java

#

measure :
ifndef BENCHSLEEF
	@echo
	@echo Please add the bin directory under build directory to PATH environment variable and run make once again.
	@echo e.g. export PATH='$$'PATH:'`pwd`'/../../../build/bin
	@echo
else
	chmod +x ./measure.sh
	./measure.sh benchsleef
	@echo
	@echo Now, you can plot the results of measurement by "'"make plot"'".
	@echo You can do another measurement by "'"make measure"'".
ifdef ICCAVAILABLE
	@echo You can start another measurement with SVML by "'"make measureSVML"'".
endif
	@echo You can start over by "'"make restart"'".
	@echo
endif

measureSVML : all benchsvml_10 benchsvml_40
	chmod +x ./measure.sh
	./measure.sh ./benchsvml_10 ./benchsvml_40
	@echo
	@echo Now, you can plot the results of measurement by "'"make plot"'".
	@echo You can do another measurement by "'"make measure"'".
ifdef ICCAVAILABLE
	@echo You can start another measurement with SVML by "'"make measureSVML"'".
endif
	@echo You can start over by "'"make restart"'".
	@echo

plot : ProcessData.class counter.txt
	java ProcessData *dptrig*.out
	gnuplot script.out
	mv output.png trigdp.png
	java ProcessData *dpnontrig*.out
	gnuplot script.out
	mv output.png nontrigdp.png
	java ProcessData *sptrig*.out
	gnuplot script.out
	mv output.png trigsp.png
	java ProcessData *spnontrig*.out
	gnuplot script.out
	mv output.png nontrigsp.png
	@echo
	@echo Plotted results are in trigdp.png, nontrigdp.png, trigsp.png and nontrigsp.png.
	@echo

clean :
	rm -f *~ a.out *.so *.so.* *.a *.s *.o
	rm -rf *.dSYM *.dylib
	rm -f *.obj *.lib *.dll *.exp *.exe *.stackdump
	rm -f *.class *.png benchsvml_10 benchsvml_40 *.out counter.txt

restart :
	rm -f *.out counter.txt
