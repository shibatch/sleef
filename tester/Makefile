all : tester testersp testeru1 testerspu1

tester : tester.c
	gcc -Wall tester.c -lm -lmpfr -o tester

testeru1 : testeru1.c
	gcc -Wall testeru1.c -lm -lmpfr -o testeru1

testersp : testersp.c
	gcc -Wall testersp.c -lm -lmpfr -o testersp

testerspu1 : testerspu1.c
	gcc -Wall testerspu1.c -lm -lmpfr -o testerspu1

clean :
	rm -f *~ *.o tester testersp testeru1 testerspu1
