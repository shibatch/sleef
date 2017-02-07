include ../config.mk

.PHONY: all
all : libsleef libsleefdft

.PHONY: libsleef
libsleef : libsleef.$(DLLSUFFIX)

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

libsleefdft.so : ../src/dft/dft.a libsleef.so ../src/common/arraymap.a
	$(CC) -fPIC $(OPENMPFLAG) -shared -o libsleefdft.so.$(SOVERSION).$(MINORVERSION) -Wl,-soname,libsleefdft.so.$(SOVERSION) -Wl,--whole-archive ../src/dft/dft.a ../src/common/arraymap.a -Wl,--no-whole-archive -L. -lsleef -lm
	ln -f -s libsleefdft.so.$(SOVERSION).$(MINORVERSION) libsleefdft.so.$(SOVERSION)
	ln -f -s libsleefdft.so.$(SOVERSION) libsleefdft.so

.PHONY: install
install : libsleef.so libsleefdft.so
	install -D -t $(DESTDIR)$(PREFIX)/lib libsleef.so.$(SOVERSION).$(MINORVERSION)
	install -D -t $(DESTDIR)$(PREFIX)/lib libsleefdft.so.$(SOVERSION).$(MINORVERSION)
	ln -f -s libsleef.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleef.so.$(SOVERSION)
	ln -f -s libsleefdft.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleefdft.so.$(SOVERSION)
	ln -f -s libsleef.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleef.so
	ln -f -s libsleefdft.so.$(SOVERSION).$(MINORVERSION) $(DESTDIR)$(PREFIX)/lib/libsleefdft.so

.PHONY: uninstall
uninstall :
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleef.so.$(SOVERSION).$(MINORVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleef.so.$(SOVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleef.so
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleefdft.so.$(SOVERSION).$(MINORVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleefdft.so.$(SOVERSION)
	rm -f $(DESTDIR)$(PREFIX)/lib/libsleefdft.so
endif

#

ifeq ($(OS),MinGW)
libsleef.dll : ../src/libm/libm.a ../src/common/common.a
	$(FLOCK) ../src/libm/libm.a.lock -c '$(CC) -shared -o libsleef.dll -Wl,--out-implib,libsleef.a -Wl,--whole-archive ../src/libm/libm.a ../src/common/common.a -Wl,--no-whole-archive -lm'

libsleefdft.dll : ../src/dft/dft.a libsleef.dll ../src/common/arraymap.a
	$(CC) -fPIC $(OPENMPFLAG) -shared -o libsleefdft.dll -Wl,--out-implib,libsleefdft.a -Wl,--whole-archive ../src/dft/dft.a ../src/common/arraymap.a -Wl,--no-whole-archive -L. -lsleef -lm
endif

#

ifeq ($(OS),Darwin)
libsleef.dylib : ../src/libm/libm.a ../src/common/common.a
	$(CC) -dynamiclib -current_version 3.0 -compatibility_version 3.0 -fvisibility=hidden ../src/libm/*.o ../src/common/common.o -lm -o libsleef.dylib

libsleefdft.dylib : ../src/dft/dft.a libsleef.dylib ../src/common/arraymap.a
	$(CC) $(OPENMPFLAG) -dynamiclib -current_version 3.0 -compatibility_version 3.0 -fvisibility=hidden ../src/dft/*.o ../src/common/arraymap.o libsleef.dylib -lm -o libsleefdft.dylib
endif

#

libsleefstatic.a : ../src/libm/libm.a ../src/common/common.a
	ar -M < libsleefstatic.mri

libsleefdftstatic.a : ../src/dft/dft.a
	ar -M < libsleefdftstatic.mri

../src/libm/libm.a :
	+"$(MAKE)" --directory=../src/libm libm.a

../src/dft/dft.a :
	+"$(MAKE)" --directory=../src/dft dft.a

../src/common/common.a :
	+"$(MAKE)" --directory=../src/common common.a

../src/common/arraymap.a :
	+"$(MAKE)" --directory=../src/common arraymap.a
