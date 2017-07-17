export SOVERSION=3
export MINORVERSION=1

export MAXBUTWIDTH=3

ifndef PREFIX
export PREFIX=/usr
endif

ifndef FLOCK
export FLOCK=flock
endif

ifeq ($(OS),Windows_NT)
  ifneq ($(shell $(CC) -v 2>&1 | grep -c "mingw"), 0)
    export OS=MinGW
  endif
  ifneq ($(shell $(CC) -v 2>&1 | grep -c "cygwin"), 0)
    export OS=MinGW
  endif
  ifneq ($(shell $(CC) -v 2>&1 | grep -c "cygnus"), 0)
    export OS=MinGW
  endif
else
  UNAME=$(shell uname -s)
  ifeq ($(UNAME),Linux)
    export OS=Linux
  endif
  ifeq ($(UNAME),Darwin)
    export OS=Darwin
  endif
endif

ifneq ($(shell $(CC) -v 2>&1 | grep -c "clang"), 0)
  export COMPILER=clang
  export ARCH=$(shell $(CC) -v 2>&1 | grep ^Target | sed -e 's/^.* //g' -e 's/-.*//g')
  ifeq ($(shell echo $(ARCH) | grep -q "^arm" && echo "yes"), yes)
    export ARCH=arm
  endif
  export FASTMATHFLAG=-ffast-math
  export STRICTMATHFLAG=-ffp-contract=off
  export ENABLEAVX2=1
  export AVX2FLAG=-mavx2 -mfma
  export AVX512FLAG=-mavx512f
  export WALLFLAGS=-Wall -Wno-shift-negative-value -Wno-unused -Wno-attributes -Wno-unused-command-line-argument
  export NEONFLAGS=--target=arm-linux-gnueabihf -mcpu=cortex-a8
  export CFLAGS=$(WALLFLAGS)
else ifneq ($(shell $(CC) -v 2>&1 | grep -c "icc version"), 0)
  export COMPILER=icc
  export ARCH=x86_64

  export FASTMATHFLAG=-fp-model fast=2
  export STRICTMATHFLAG=-fp-model strict
  export ENABLEAVX2=1
  export AVX2FLAG=-march=core-avx2
  export AVX512FLAG=-xCOMMON-AVX512
  export WALLFLAGS=-fmax-errors=3 -Wall -Wno-unused -Wno-attributes
  export CFLAGS=$(WALLFLAGS) -Qoption,cpp,--extended_float_type -qoverride-limits 
else ifneq ($(shell $(CC) -v 2>&1 | grep -c "gcc version"), 0)
  export COMPILER=gcc
  export ARCH=$(shell $(CC) -v 2>&1 | grep ^Target | sed -e 's/^.* //g' -e 's/-.*//g')

  export FASTMATHFLAG=-ffast-math
  export STRICTMATHFLAG=-ffp-contract=off
  export ENABLEAVX2= $(shell expr `$(CC) -dumpversion | sed 's/\..*//g'` '>=' 5)
  export AVX2FLAG=-mavx2 -mfma
  export AVX512FLAG=-mavx512f
  export ADVSIMDFLAG=-march=armv8-a+simd
  export NEONFLAGS=-mfpu=neon
  export WALLFLAGS=-fmax-errors=3 -Wall -Wno-unused -Wno-attributes -Wno-psabi
  export CFLAGS=$(WALLFLAGS) -std=gnu99
endif

ifeq ($(ARCH),x86_64)
  export X86ARCH=1
endif

ifeq ($(ARCH),i686)
  export X86ARCH=1
endif

export ENABLEAVX=1

ifeq ($(OS),MinGW)
  export OPENMPFLAG=-fopenmp
  export SHAREDFLAGS=-fvisibility=hidden
  export ENABLEAVX=0
  export DLLSUFFIX=dll
else ifeq ($(OS),Darwin)
  ifeq ($(COMPILER),clang)
    export SHAREDFLAGS=-fPIC -fvisibility=hidden
    export ENABLEFLOAT80=1
  else ifeq ($(COMPILER),gcc)
    export OPENMPFLAG=-fopenmp
    export SHAREDFLAGS=-fPIC -fvisibility=hidden -Wa,-q
    export ENABLEFLOAT80=1
    export ENABLEFLOAT128=1
    export CFLAGS+=-DENABLEFLOAT128
  endif

  export DLLSUFFIX=dylib
else ifeq ($(OS),Linux)
  export DLLSUFFIX=so

  ifeq ($(COMPILER),gcc)
    export OPENMPFLAG=-fopenmp
    export SHAREDFLAGS=-fPIC -shared -fvisibility=hidden

    ifeq ($(X86ARCH),1)
      export ENABLEFMA4=1
      export ENABLEAVX512F= $(shell expr `$(CC) -dumpversion | sed 's/\..*//g'` '>=' 5)
      export ENABLEFLOAT80= $(shell expr `$(CC) -dumpversion | sed 's/\..*//g'` '>=' 5)
      export ENABLEFLOAT128= $(shell expr `$(CC) -dumpversion | sed 's/\..*//g'` '>=' 5)
      export ENABLEVECEXT= $(shell expr `$(CC) -dumpversion | sed 's/\..*//g'` '>=' 5)
      ifeq ($(ENABLEFLOAT128), 1)
        export CFLAGS+=-DENABLEFLOAT128
      endif
    endif

    ifeq ($(ARCH),i686)
      export CFLAGS+=-m128bit-long-double -msse2 -mfpmath=sse
      export ENABLEVECEXT=0
    endif

  else ifeq ($(COMPILER),clang)
    export OPENMPFLAG=-fopenmp
    export SHAREDFLAGS=-fPIC -fvisibility=hidden

    export ENABLEFMA4=1
    export ENABLEFLOAT80=1

    ifeq ($(ARCH),i686)
      export CFLAGS+=-msse2 -mfpmath=sse
    endif
  else ifeq ($(COMPILER),icc)
    export OPENMPFLAG=-fopenmp
    export SHAREDFLAGS=-fPIC -shared -fvisibility=hidden 
    export LDFLAGS=-Wl,-rpath=/export/opt/intel/compilers_and_libraries_2017.1.132/linux/compiler/lib/intel64_lin

    export ENABLEAVX512F=1
    export ENABLEFLOAT80=1
    export ENABLEFLOAT128=1
    export CFLAGS+=-DENABLEFLOAT128
  endif
endif

ifeq ($(X86ARCH),1)
  export ENABLEGNUABI=1
endif

ifeq ($(ARCH),aarch64)
  export ENABLEGNUABI=1
endif

ifeq ($(ENABLEAVX), 0)
  export ENABLEAVX512F=0
endif
