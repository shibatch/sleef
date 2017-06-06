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
export FLOCK=$(shell pwd)/src/script/noflock.sh
endif
endif

ifdef ENABLEGNUABI
.PHONY: all
all : displayVars libsleef libsleefgnuabi libsleef-dft libsleefllvm
else
.PHONY: all
all : displayVars libsleef libsleef-dft libsleefllvm
endif

.PHONY: displayVars
displayVars :
	@echo OS = $(OS)
	@echo ARCH = $(ARCH)
	@echo COMPILER = $(COMPILER)
	@echo ENABLEGNUABI = $(ENABLEGNUABI)
	@echo ENABLE256X86 = $(ENABLE256X86)
	@echo ENABLEAVX2 = $(ENABLEAVX2)
	@echo ENABLEAVX512f = $(ENABLEAVX512F)
	@echo ENABLEFMA4 = $(ENABLEFMA4)
	@echo ENABLEFLOAT80 = $(ENABLEFLOAT80)
	@echo ENABLEFLOAT128 = $(ENABLEFLOAT128)

.PHONY: libsleef
libsleef :
	+"$(MAKE)" --directory=./lib libsleef
#	+"$(MAKE)" --directory=./src/libm-tester

.PHONY: libsleefgnuabi
libsleefgnuabi :
	+"$(MAKE)" --directory=./lib libsleefgnuabi

.PHONY: libsleefllvm
libsleefllvm :
	+"$(MAKE)" --directory=./lib libsleefllvm

.PHONY: test
test : displayVars libsleef
	+"$(MAKE)" --directory=./src/libm-tester test

.PHONY: testdft
testdft : displayVars libsleef
	+"$(MAKE)" --directory=./src/dft-tester test

.PHONY: libsleef-dft
libsleef-dft :
	+"$(MAKE)" --directory=./lib libsleefdft
#	+"$(MAKE)" --directory=./src/dft-tester

.PHONY: testall
testall : displayVars test testdft

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
	rm -f *~ .*~
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


