include config.mk

ifeq ($(OS),Windows_NT)
export FLOCK=flock
else
UNAME=$(shell uname -s)
ifeq ($(UNAME),Linux)
export OS=Linux
export FLOCK=flock
endif
ifeq ($(UNAME),Darwin)
export OS=Darwin
export FLOCK=$(shell pwd)/noflock.sh
endif
endif

.PHONY: all
all : libsleef libsleef-dft

.PHONY: libsleef
libsleef :
	+"$(MAKE)" --directory=./lib libsleef
	+"$(MAKE)" --directory=./src/libm-tester

.PHONY: test
test : libsleef
	+"$(MAKE)" --directory=./src/libm-tester test

# TODO: reactivate DFT builds for AArch64
ifneq ($(ARCH),aarch64)
.PHONY: libsleef-dft
libsleef-dft :
	+"$(MAKE)" --directory=./lib libsleefdft
#	+"$(MAKE)" --directory=./src/dft-tester

endif
.PHONY: install
install :
	+"$(MAKE)" --directory=./lib install
	+"$(MAKE)" --directory=./include install

.PHONY: uninstall
uninstall :
	+"$(MAKE)" --directory=./lib uninstall
	+"$(MAKE)" --directory=./include uninstall

.PHONY: clean
clean :
	+"$(MAKE)" --directory=include clean
	+"$(MAKE)" --directory=lib clean
	+"$(MAKE)" --directory=src clean
	rm -f *~
	rm -f debian/*~

.PHONY: distclean
distclean : clean
	+"$(MAKE)" --directory=include distclean
	+"$(MAKE)" --directory=lib distclean
	+"$(MAKE)" --directory=src distclean
#	rm -f debian/debhelper-build-stamp
#	rm -f debian/files
#	rm -f debian/libsleef3.debhelper.log
#	rm -f debian/libsleef3.substvars
#	rm -rf debian/libsleef3 debian/patches debian/.debhelper


