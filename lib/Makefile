include ../config.mk

.PHONY: all
all : libsleef libsleefgnuabi libsleefdft

.PHONY: libsleef
libsleef : libsleef.$(DLLSUFFIX)

.PHONY: libsleefgnuabi
libsleefgnuabi : libsleefgnuabi.$(DLLSUFFIX)

.PHONY: libsleef-dft
libsleefdft : libsleefdft.$(DLLSUFFIX)

#

.PHONY: clean
clean :
	rm -f *~

.PHONY: distclean
distclean : clean
	rm -f *.so *.so.* *.a
	rm -f *.obj *.lib *.dll *.exp *.exe
	rm -rf *.dSYM *.dylib

#

ifeq ($(OS),Linux)
libsleef.so : ../src/libm/libm.a ../src/common/common.a
	$(FLOCK) ../src/libm/libm.a.lock -c '$(CC) -fPIC -shared -o libsleef.so.$(SOVERSION).$(MINORVERSION) -Wl,-soname,libsleef.so.$(SOVERSION) -Wl,--whole-archive ../src/libm/libm.a ../src/common/common.a -Wl,--no-whole-archive -lm'
	ln -f -s libsleef.so.$(SOVERSION).$(MINORVERSION) libsleef.so.$(SOVERSION)
	ln -f -s libsleef.so.$(SOVERSION) libsleef.so

libsleefgnuabi.so : ../src/libm/libgm.a
	$(FLOCK) ../src/libm/libm.a.lock -c '$(CC) -fPIC -shared -o libsleefgnuabi.so.$(SOVERSION).$(MINORVERSION) -Wl,-soname,libsleefgnuabi.so.$(SOVERSION) -Wl,--whole-archive ../src/libm/libgm.a -Wl,--no-whole-archive -lm'
	ln -f -s libsleefgnuabi.so.$(SOVERSION).$(MINORVERSION) libsleefgnuabi.so.$(SOVERSION)
	ln -f -s libsleefgnuabi.so.$(SOVERSION) libsleefgnuabi.so

libsleefdft.so : ../src/dft/dft.a libsleef.so ../src/common/arraymap.a
	$(CC) -fPIC $(OPENMPFLAG) -shared -o libsleefdft.so.$(SOVERSION).$(MINORVERSION) -Wl,-soname,libsleefdft.so.$(SOVERSION) -Wl,--whole-archive ../src/dft/dft.a ../src/common/arraymap.a -Wl,--no-whole-archive -L. -lsleef -lm
	ln -f -s libsleefdft.so.$(SOVERSION).$(MINORVERSION) libsleefdft.so.$(SOVERSION)
	ln -f -s libsleefdft.so.$(SOVERSION) libsleefdft.so

.PHONY: install
install : libsleef.so libsleefgnuabi.so libsleefdft.so
	install -D -t $(DESTDIR)$(PREFIX)/lib libsleef.so.$(SOVERSION).$(MINORVERSION)
	install -D -t $(DESTDIR)$(PREFIX)/lib libsleefgnuabi.so.$(SOVERSION).$(MINORVERSION)
	install -D -t $(DESTDIR)$(PREFIX)/lib libsleefdft.so.$(SOVERSION).$(MINORVERSION)
	ln -f -s libsleef.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleef.so.$(SOVERSION)
	ln -f -s libsleefgnuabi.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleefgnuabi.so.$(SOVERSION)
	ln -f -s libsleefdft.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleefdft.so.$(SOVERSION)
	ln -f -s libsleef.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleef.so
	ln -f -s libsleefgnuabi.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleefgnuabi.so
	ln -f -s libsleefdft.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleefdft.so

.PHONY: uninstall
uninstall :
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleef.so.$(SOVERSION).$(MINORVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleef.so.$(SOVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleef.so
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleefgnuabi.so.$(SOVERSION).$(MINORVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleefgnuabi.so.$(SOVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleefgnuabi.so
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleefdft.so.$(SOVERSION).$(MINORVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleefdft.so.$(SOVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleefdft.so
endif

#

ifeq ($(OS),MinGW)
libsleef.dll : ../src/libm/libm.a ../src/common/common.a
	$(FLOCK) ../src/libm/libm.a.lock -c '$(CC) -shared -o libsleef.dll -Wl,--out-implib,libsleef.a -Wl,--whole-archive ../src/libm/libm.a ../src/common/common.a -Wl,--no-whole-archive -lm'

libsleefgnuabi.dll : ../src/libm/libgm.a
	$(FLOCK) ../src/libm/libgm.a.lock -c '$(CC) -shared -o libsleefgnuabi.dll -Wl,--out-implib,libsleefgnuabi.a -Wl,--whole-archive ../src/libm/libgm.a -Wl,--no-whole-archive -lm'

libsleefdft.dll : ../src/dft/dft.a libsleef.dll ../src/common/arraymap.a
	$(CC) -fPIC $(OPENMPFLAG) -shared -o libsleefdft.dll -Wl,--out-implib,libsleefdft.a -Wl,--whole-archive ../src/dft/dft.a ../src/common/arraymap.a -Wl,--no-whole-archive -L. -lsleef -lm
endif

#

ifeq ($(OS),Darwin)
libsleef.dylib : ../src/libm/libm.a ../src/libm/OBJ.txt ../src/common/common.a
	$(CC) -dynamiclib -current_version 3.1 -compatibility_version 3.1 -fvisibility=hidden `cat ../src/libm/OBJ.txt` ../src/common/common.o -lm -o libsleef.dylib

libsleefgnuabi.dylib : ../src/libm/libm.a ../src/libm/OBJ2.txt ../src/common/common.a
	$(CC) -dynamiclib -current_version 3.1 -compatibility_version 3.1 -fvisibility=hidden `cat ../src/libm/OBJ2.txt` ../src/common/common.o -lm -o libsleefgnuabi.dylib

libsleefdft.dylib : ../src/dft/dft.a libsleef.dylib ../src/common/arraymap.a
	$(CC) $(OPENMPFLAG) -dynamiclib -current_version 3.0 -compatibility_version 3.0 -fvisibility=hidden ../src/dft/*.o ../src/common/arraymap.o libsleef.dylib -lm -o libsleefdft.dylib
endif

#

libsleefstatic.a : ../src/libm/libm.a ../src/common/common.a
	ar -M < libsleefstatic.mri

libsleefgnuabistatic.a : ../src/libm/libgm.a
	ar -M < libsleefgnuabistatic.mri

libsleefdftstatic.a : ../src/dft/dft.a ../src/common/arraymap.a
	ar -M < libsleefdftstatic.mri

../src/libm/libm.a :
	+"$(MAKE)" --directory=../src/libm libm.a

../src/libm/libgm.a :
	+"$(MAKE)" --directory=../src/libm libgm.a

../src/dft/dft.a :
	+"$(MAKE)" --directory=../src/dft dft.a

../src/common/common.a :
	+"$(MAKE)" --directory=../src/common common.a

../src/common/arraymap.a :
	+"$(MAKE)" --directory=../src/common arraymap.a

../src/libm/OBJ.txt :
	+"$(MAKE)" --directory=../src/libm OBJ.txt

../src/libm/OBJ2.txt :
	+"$(MAKE)" --directory=../src/libm OBJ2.txt
