CC=gcc

iut : sleefdp.c sleefsp.c iut.c
	$(CC) -ffp-contract=off -Wall -DNDEBUG sleefdp.c sleefsp.c iut.c -o iut -lm

../tester/tester :
	cd ../tester; make tester

../tester/testeru1 :
	cd ../tester; make testeru1

../tester/testersp :
	cd ../tester; make testersp

../tester/testerspu1 :
	cd ../tester; make testerspu1

tester2 : tester2.c tester2f.c sleefdp.c sleefsp.c
	$(CC) -ffp-contract=off -Wall -DNDEBUG tester2.c sleefdp.c -o tester2 -lm -lmpfr
	$(CC) -ffp-contract=off -Wall -DNDEBUG tester2f.c sleefsp.c -o tester2f -lm -lmpfr

test : iut ../tester/tester ../tester/testeru1 ../tester/testersp ../tester/testerspu1
	../tester/tester ./iut
	../tester/testeru1 ./iut
	../tester/testersp ./iut
	../tester/testerspu1 ./iut

clean :
	rm -f *~ *.o iut tester2 tester2f
