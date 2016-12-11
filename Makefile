test :
	cd purec; make test
	cd simd; make testsse2 testavx

clean :
	rm -f *~ *.out
	cd purec; make clean
	cd simd; make clean
	cd tester; make clean
	cd gencoef; make clean
