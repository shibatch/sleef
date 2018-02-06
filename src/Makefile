.PHONY: clean
clean :
	+"$(MAKE)" --directory=./arch clean
	+"$(MAKE)" --directory=./common clean
	+"$(MAKE)" --directory=./gencoef clean
	+"$(MAKE)" --directory=./libm clean
	+"$(MAKE)" --directory=./libm-tester clean
	+"$(MAKE)" --directory=./libm-plot clean
	+"$(MAKE)" --directory=./dft clean
	+"$(MAKE)" --directory=./dft-tester clean
	rm -f *~

.PHONY: distclean
distclean : clean
	+"$(MAKE)" --directory=./dft-tester distclean
	+"$(MAKE)" --directory=./libm-plot distclean
