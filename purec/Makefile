CC=gcc

iut : sleefdp.c sleefsp.c iut.c
	$(CC) -Wall -DNDEBUG sleefdp.c sleefsp.c iut.c -o iut -lm

../tester/tester :
	cd ../tester; make tester

../tester/testeru1 :
	cd ../tester; make testeru1

../tester/testersp :
	cd ../tester; make testersp

../tester/testerspu1 :
	cd ../tester; make testerspu1

test : iut ../tester/tester ../tester/testeru1 ../tester/testersp ../tester/testerspu1
	../tester/tester ./iut
	../tester/testeru1 ./iut
	../tester/testersp ./iut
	../tester/testerspu1 ./iut

clean :
	rm -f *~ *.o iut
