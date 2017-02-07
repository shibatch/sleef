.PHONY: all
all : gencoef

gencoef : gencoef.c simplexfr.c sp.h dp.h ld.h qp.h
	gcc -O gencoef.c simplexfr.c -o gencoef -lmpfr -lm

.PHONY: clean
clean :
	rm -f gencoef gencoefdp gencoefld a.out *~
	rm -f *.obj *.lib *.dll *.exp *.exe
