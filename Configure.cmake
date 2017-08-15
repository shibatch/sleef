include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckTypeSize)

# PLATFORM DETECTION

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
  set(SLEEF_ARCH_X86 ON CACHE INTERNAL "True for x86 architecture.")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(SLEEF_ARCH_AARCH64 ON CACHE INTERNAL "True for Aarch64 architecture.")
  # Aarch64 requires support for advsimdfma4
  set(COMPILER_SUPPORTS_ADVSIMD 1)
endif()

# COMPILER DETECTION

# All variables storing compiler flags should be prefixed with FLAGS_

if(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
  set(FLAGS_WALL "-Wall -Wno-unused -Wno-attributes")
  set(FLAGS_FASTMATH "-ffast-math")
  set(FLAGS_STRICTMATH "-ffp-contract=off")
  set(FLAGS_OPENMP "-fopenmp")

  set(FLAGS_ENABLE_SSE2 "-msse2")
  set(FLAGS_ENABLE_SSE4 "-msse4.1")
  set(FLAGS_ENABLE_AVX "-mavx")
  set(FLAGS_ENABLE_FMA4 "-mfma4")
  set(FLAGS_ENABLE_AVX2 "-mavx2;-mfma")
  set(FLAGS_ENABLE_AVX512F "-mavx512f")

  set(FLAGS_ENABLE_ADVSIMD "-march=armv8-a+simd")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAGS_WALL}")

# Always compile sleef with -ffp-contract and log at configuration time
if(SLEEF_SHOW_CONFIG)
  message(STATUS "Using option `${FLAGS_STRICTMATH}` to compile libsleef")
endif(SLEEF_SHOW_CONFIG)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAGS_STRICTMATH}")

# FEATURE DETECTION

set (CMAKE_REQUIRED_FLAGS ${FLAGS_ENABLE_SSE2})
CHECK_C_SOURCE_COMPILES("
  #if defined(_MSC_VER)
  #include <intrin.h>
  #else
  #include <x86intrin.h>
  #endif
  int main() {
    __m128d r = _mm_mul_pd(_mm_set1_pd(1), _mm_set1_pd(2)); }"
  COMPILER_SUPPORTS_SSE2)

set (CMAKE_REQUIRED_FLAGS ${FLAGS_ENABLE_SSE4})
CHECK_C_SOURCE_COMPILES("
  #if defined(_MSC_VER)
  #include <intrin.h>
  #else
  #include <x86intrin.h>
  #endif
  int main() {
    __m128d r = _mm_floor_sd(_mm_set1_pd(1), _mm_set1_pd(2)); }"
  COMPILER_SUPPORTS_SSE4)

set (CMAKE_REQUIRED_FLAGS ${FLAGS_ENABLE_AVX})
CHECK_C_SOURCE_COMPILES("
  #if defined(_MSC_VER)
  #include <intrin.h>
  #else
  #include <x86intrin.h>
  #endif
  int main() {
    __m256d r = _mm256_add_pd(_mm256_set1_pd(1), _mm256_set1_pd(2));
  }" COMPILER_SUPPORTS_AVX)

set (CMAKE_REQUIRED_FLAGS ${FLAGS_ENABLE_FMA4})
CHECK_C_SOURCE_COMPILES("
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
int main() {
  __m256d r = _mm256_macc_pd(_mm256_set1_pd(1), _mm256_set1_pd(2), _mm256_set1_pd(3));
}" COMPILER_SUPPORTS_FMA4)

set (CMAKE_REQUIRED_FLAGS ${FLAGS_ENABLE_AVX2})
CHECK_C_SOURCE_COMPILES("
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
int main() {
  __m256i r = _mm256_abs_epi32(_mm256_set1_epi32(1));
}" COMPILER_SUPPORTS_AVX2)

set (CMAKE_REQUIRED_FLAGS ${FLAGS_ENABLE_AVX512F})
CHECK_C_SOURCE_COMPILES("
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
void f(void *p) {
  _mm512_loadu_si512(p);
}
int main() {
  __m512i a = _mm512_set1_epi32(1);
  __m512i r = _mm512_andnot_si512(a, a);
}" COMPILER_SUPPORTS_AVX512F)

# Reset used flags
set(CMAKE_REQUIRED_FLAGS)

CHECK_TYPE_SIZE("long double" LD_SIZE)
if(LD_SIZE GREATER "9")
  set(COMPILER_SUPPORTS_LONG_DOUBLE 1)
endif()

CHECK_C_SOURCE_COMPILES("
  int main(){ __float128 r = 1;}"
  COMPILER_SUPPORTS_FLOAT128)
