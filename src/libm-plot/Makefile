all : benchsleef
	@echo
	@echo You can start measurement by "'"make measure"'".
	@echo Then, you can plot the results of measurement by "'"make plot"'".
	@echo
	@echo You have to install java and gnuplot to do plotting.
	@echo Stop all tasks on the computer before starting measurement.
	@echo

benchsvml128_10.o : benchsvml128.c bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml128.c -Wall -DSVMLULP=1 -fimf-max-error=1.0 -fimf-domain-exclusion=0 -march=core-avx2 -O0 -lm -c -o benchsvml128_10.o

benchsvml128_40.o : benchsvml128.c bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml128.c -Wall -DSVMLULP=4 -fimf-max-error=4.0 -fimf-domain-exclusion=0 -march=core-avx2 -O0 -lm -c -o benchsvml128_40.o

benchsvml256_10.o : benchsvml256.c bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml256.c -Wall -DSVMLULP=1 -fimf-max-error=1.0 -fimf-domain-exclusion=0 -march=core-avx2 -O0 -lm -c -o benchsvml256_10.o

benchsvml256_40.o : benchsvml256.c bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml256.c -Wall -DSVMLULP=4 -fimf-max-error=4.0 -fimf-domain-exclusion=0 -march=core-avx2 -O0 -lm -c -o benchsvml256_40.o

benchsvml512_10.o : benchsvml512.c bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml512.c -Wall -DSVMLULP=1 -fimf-max-error=1.0 -fimf-domain-exclusion=0 -xCOMMON-AVX512 -O0 -lm -c -o benchsvml512_10.o

benchsvml512_40.o : benchsvml512.c bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml512.c -Wall -DSVMLULP=4 -fimf-max-error=4.0 -fimf-domain-exclusion=0 -xCOMMON-AVX512 -O0 -lm -c -o benchsvml512_40.o


benchsvml_10 : benchsvml.c benchsvml128_10.o benchsvml256_10.o benchsvml512_10.o bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml.c benchsvml128_10.o benchsvml256_10.o benchsvml512_10.o -Wall -DSVMLULP=1 -fimf-max-error=1.0 -fimf-domain-exclusion=0 -O0 -march=native -lm -o benchsvml_10

benchsvml_40 : benchsvml.c benchsvml128_40.o benchsvml256_40.o benchsvml512_40.o bench.h
	-command -v icc >/dev/null 2>&1 && icc benchsvml.c benchsvml128_40.o benchsvml256_40.o benchsvml512_40.o -Wall -DSVMLULP=4 -fimf-max-error=4.0 -fimf-domain-exclusion=0 -O0 -march=native -lm -o benchsvml_40

#

benchsleef : benchsleef.c benchsleef128.o benchsleef256.o benchsleef512.o bench.h
	$(CC) benchsleef.c benchsleef128.o benchsleef256.o benchsleef512.o -Wall -O0 -g -I../../include -L../../lib -Wno-attributes -lm -lsleef -o benchsleef

benchsleef128.o : benchsleef128.c bench.h
	$(CC) benchsleef128.c -Wall -march=native -O0 -g -I../../include -L../../lib -Wno-attributes -c

benchsleef256.o : benchsleef256.c bench.h
	$(CC) benchsleef256.c -Wall -march=native -O0 -g -I../../include -L../../lib -Wno-attributes -c

benchsleef512.o : benchsleef512.c bench.h
	$(CC) benchsleef512.c -Wall -mavx512f -O0 -g -I../../include -L../../lib -Wno-attributes -c

#

ProcessData.class : ProcessData.java
	javac ProcessData.java

#

measure : all
	LD_LIBRARY_PATH=../../lib:$(LD_LIBRARY_PATH) ./measure.sh ./benchsleef
	@echo
	@echo Now, you can plot the results of measurement by "'"make plot"'".
	@echo You can do another measurement by "'"make measure"'".
	@echo You can start over by "'"make restart"'".
	@echo

measureSVML : all benchsvml_10 benchsvml_40
	./measure.sh ./benchsvml_10 ./benchsvml_40
	@echo
	@echo Now, you can plot the results of measurement by "'"make plot"'".
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

clean :
	rm -f *~ *.o *.s *.class *.png benchsvml_10 benchsvml_40 benchsleef

restart :
	rm -f *.out counter.txt

distclean : clean restart
