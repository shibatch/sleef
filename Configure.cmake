include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckTypeSize)

# PLATFORM DETECTION

if(WIN32)
  set(_TMP_OSTYPE "MSVC")
elseif(MSYS OR MINGW OR CYGWIN)
  set(_TMP_OSTYPE "MinGW")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(_TMP_OSTYPE "Linux")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(_TMP_OSTYPE "Darwin")
endif()

# Cache OS type into a sleef variable and define it in the configuration header
set(SLEEF_OSTYPE ${_TMP_OSTYPE} CACHE INTERNAL
  "The OS 'type' for which SLEEF is configured.")

# Cache machine architecture determined by checking size of a pointer
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(SLEEF_ARCH_64BIT ON CACHE INTERNAL
    "True for 64-bit architecture machines.")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(SLEEF_ARCH_32BIT ON CACHE INTERNAL
    "True for 32-bit architecture machines.")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
  set(SLEEF_ARCH_X86 ON CACHE INTERNAL "True for x86 architecture.")
  set(COMPILER_SUPPORTS_SSE2 1)
endif()

# COMPILER DETECTION

# All variables storing compiler flags should be prefixed with FLAGS_

if(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
  set(FLAGS_WALL "-Wall -Wno-unused -Wno-attributes -Wno-shift-negative-value")
  set(FLAGS_FASTMATH "-ffast-math")
  set(FLAGS_STRICTMATH "-ffp-contract=off")
  set(FLAGS_OPENMP "-fopenmp")
  # Architecture flags for Clang only (TODO: add GNU, ICC, MSVC)
  set(FLAGS_ENABLE_NEON "-mpfu=neon")
  set(FLAGS_ENABLE_ADVSIMD "-march=armv8-a+simd")
  set(FLAGS_ENABLE_SSE2 "-msse2")
  set(FLAGS_ENABLE_AVX "-mavx")
  set(FLAGS_ENABLE_AVX2 "-mavx2;-mfma")
  set(FLAGS_ENABLE_AVX512F "-mavx512f")
  set(FLAGS_ENABLE_FMA4 "-mfma4")
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
  # TODO
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
  # TODO
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAGS_WALL}")

# Always compile sleef with -ffp-contract and log at configuration time
if(SLEEF_SHOW_CONFIG)
  message(STATUS "Using option `${FLAGS_STRICTMATH}` to compile libsleef")
endif(SLEEF_SHOW_CONFIG)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAGS_STRICTMATH}")

# With -fPIC
check_c_compiler_flag("-fPIC" WITH_FPIC)
if (WITH_FPIC)
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif (WITH_FPIC)

# FEATURE DETECTION

CHECK_TYPE_SIZE("long double" LD_SIZE)
if(LD_SIZE GREATER "9")
  set(COMPILER_SUPPORTS_LONG_DOUBLE 1)
endif()

CHECK_C_SOURCE_COMPILES(
  "int main(){ __float128 r = 1;}"
  COMPILER_SUPPORTS_FLOAT128)

set (CMAKE_REQUIRED_FLAGS "${FLAGS_ENABLE_AVX}")
CHECK_C_SOURCE_COMPILES("
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
int main() {
  __m256d r = _mm256_add_pd(_mm256_set1_pd(1), _mm256_set1_pd(2));
}" COMPILER_SUPPORTS_AVX)

set (CMAKE_REQUIRED_FLAGS "${FLAGS_ENABLE_AVX2}")
CHECK_C_SOURCE_COMPILES("
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
int main() {
  __m256i r = _mm256_abs_epi32(_mm256_set1_epi32(1));
}" COMPILER_SUPPORTS_AVX2)

set (CMAKE_REQUIRED_FLAGS "${FLAGS_ENABLE_AVX512F}")
CHECK_C_SOURCE_COMPILES("
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
int main() {
  __m512d r = _mm512_add_pd(_mm512_set1_pd(1), _mm512_set1_pd(2));
}" COMPILER_SUPPORTS_AVX512F)

set (CMAKE_REQUIRED_FLAGS "${FLAGS_ENABLE_FMA4}")
CHECK_C_SOURCE_COMPILES("
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
int main() {
  __m256d r = _mm256_macc_pd(_mm256_set1_pd(1), _mm256_set1_pd(2), _mm256_set1_pd(3));
  }" COMPILER_SUPPORTS_FMA4)
