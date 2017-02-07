include ../../config.mk
CFLAGS+=-Wall -Wno-unused -Wno-attributes $(SHAREDFLAGS) -fvisibility=hidden -O -I../arch -I../common -I../../include

.PHONY: all
all : common.a arraymap.a

common.a : common.o
	$(FLOCK) common.a.lock -c 'ar rcs common.a common.o'

arraymap.a : arraymap.o
	$(FLOCK) arraymap.a.lock -c 'ar rcs arraymap.a arraymap.o'

arraymap.o : arraymap.c
	$(CC) $(CFLAGS) arraymap.c -c -o arraymap.o

common.o : common.c
	$(CC) $(CFLAGS) common.c -c -o common.o

.PHONY: clean
clean :
	rm -f *~ *.o *.lock *.obj *.lib *.dll *.exp *.exe
	rm -f common.a arraymap.a
