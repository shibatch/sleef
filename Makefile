.PHONY: clean
clean :
	+"$(MAKE)" --directory=src clean
	rm -f *~ .*~
	rm -f debian/*~
