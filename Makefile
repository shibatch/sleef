test :
	cd java; make test
	cd purec; make test
	cd simd; make testsse2 testavx

clean :
	rm -f *~
	cd java; make clean
	cd purec; make clean
	cd simd; make clean
	cd tester; make clean
