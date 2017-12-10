include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckTypeSize)

# Check that the library LIB_MPFR is available
find_library(LIB_MPFR mpfr)
find_library(LIBM m)

# The library currently supports the following SIMD architectures
set(SLEEF_SUPPORTED_EXTENSIONS
  SSE2 SSE4 AVX FMA4 AVX2 AVX2128 AVX512F # x86
  ADVSIMD				  # Aarch64
  NEON32				  # Aarch32
  CACHE STRING "List of SIMD architectures supported by libsleef."
  )
set(SLEEF_SUPPORTED_GNUABI_EXTENSIONS 
  SSE2 AVX AVX2 AVX512F ADVSIMD
  CACHE STRING "List of SIMD architectures supported by libsleef for GNU ABI."
)

# Force set default build type if none was specified
# Note: some sleef code requires the optimisation flags turned on
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' (required for full support).")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Function used to generate safe command arguments for add_custom_command
function(command_arguments PROPNAME)
  set(quoted_args "")
  foreach(arg ${ARGN})
    list(APPEND quoted_args "\"${arg}\"" )
  endforeach()
  set(${PROPNAME} ${quoted_args} PARENT_SCOPE)
endfunction()


# PLATFORM DETECTION
if((CMAKE_SYSTEM_PROCESSOR MATCHES "x86") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
  set(SLEEF_ARCH_X86 ON CACHE INTERNAL "True for x86 architecture.")

  set(SLEEF_HEADER_LIST
    SSE_
    SSE2
    SSE4
    AVX_
    AVX
    FMA4
    AVX2
    AVX2128
    AVX512F
  )
  command_arguments(HEADER_PARAMS_SSE_      2 4 __m128d __m128 __m128i __m128i __SSE2__)
  command_arguments(HEADER_PARAMS_SSE2      2 4 __m128d __m128 __m128i __m128i __SSE2__ sse2)
  command_arguments(HEADER_PARAMS_SSE4      2 4 __m128d __m128 __m128i __m128i __SSE2__ sse4)
  command_arguments(HEADER_PARAMS_AVX_      4 8 __m256d __m256 __m128i "struct { __m128i x, y$<SEMICOLON> }" __AVX__)
  command_arguments(HEADER_PARAMS_AVX       4 8 __m256d __m256 __m128i "struct { __m128i x, y$<SEMICOLON> }" __AVX__ avx)
  command_arguments(HEADER_PARAMS_FMA4      4 8 __m256d __m256 __m128i "struct { __m128i x, y$<SEMICOLON> }" __AVX__ fma4)
  command_arguments(HEADER_PARAMS_AVX2      4 8 __m256d __m256 __m128i __m256i __AVX__ avx2)
  command_arguments(HEADER_PARAMS_AVX2128   2 4 __m128d __m128 __m128i __m128i __SSE2__ avx2128)
  command_arguments(HEADER_PARAMS_AVX512F   8 16 __m512d __m512 __m256i __m512i __AVX512F__ avx512f)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(SLEEF_ARCH_AARCH64 ON CACHE INTERNAL "True for Aarch64 architecture.")
  # Aarch64 requires support for advsimdfma4
  set(COMPILER_SUPPORTS_ADVSIMD 1)

  set(SLEEF_HEADER_LIST
    ADVSIMD
  )
  command_arguments(HEADER_PARAMS_ADVSIMD    2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON advsimd)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
  set(SLEEF_ARCH_AARCH32 ON CACHE INTERNAL "True for Aarch32 architecture.")
  set(COMPILER_SUPPORTS_NEON32 1)

  set(SLEEF_HEADER_LIST
    NEON32
  )
  command_arguments(HEADER_PARAMS_NEON32    2 4 - float32x4_t int32x2_t int32x4_t __ARM_NEON__ neon)
endif()

# MKRename arguments per type
command_arguments(RENAME_PARAMS_SSE2           2 4 sse2)
command_arguments(RENAME_PARAMS_SSE4           2 4 sse4)
command_arguments(RENAME_PARAMS_AVX            4 8 avx)
command_arguments(RENAME_PARAMS_FMA4           4 8 fma4)
command_arguments(RENAME_PARAMS_AVX2           4 8 avx2)
command_arguments(RENAME_PARAMS_AVX2128        2 4 avx2128)
command_arguments(RENAME_PARAMS_AVX512F        8 16 avx512f)
command_arguments(RENAME_PARAMS_ADVSIMD        2 4 advsimd)
command_arguments(RENAME_PARAMS_NEON32         2 4 neon)

command_arguments(RENAME_PARAMS_GNUABI_SSE2    sse2 b 2 4 _mm128d _mm128 _mm128i _mm128i __SSE2__)
command_arguments(RENAME_PARAMS_GNUABI_AVX     avx c 4 8 __m256d __m256 __m128i "struct { __m128i x, y$<SEMICOLON> }" __AVX__)
command_arguments(RENAME_PARAMS_GNUABI_AVX2    avx2 d 4 8 __m256d __m256 __m128i __m256i __AVX2__)
command_arguments(RENAME_PARAMS_GNUABI_AVX512F avx512f e 8 16 __m512d __m512 __m256i __m512i __AVX512F__)
command_arguments(RENAME_PARAMS_GNUABI_ADVSIMD advsimd n 2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON)

# COMPILER DETECTION

# Detect CLANG executable path (on both Windows and Linux/OSX)
if(NOT CLANG_EXE_PATH)
  # If the current compiler used by CMAKE is already clang, use this one directly
  if(CMAKE_C_COMPILER MATCHES "clang")
    set(CLANG_EXE_PATH ${CMAKE_C_COMPILER})
  else()
    # Else we may find clang on the path?
    find_program(CLANG_EXE_PATH NAMES clang "clang-5.0" "clang-4.0" "clang-3.9")
  endif()
endif()

# Allow to define the Gcc/Clang here
# As we might compile the lib with MSVC, but generates bitcode with CLANG
# Intel vector extensions.
set(CLANG_FLAGS_ENABLE_SSE2 "-msse2")
set(CLANG_FLAGS_ENABLE_SSE4 "-msse4.1")
set(CLANG_FLAGS_ENABLE_AVX "-mavx")
set(CLANG_FLAGS_ENABLE_FMA4 "-mfma4")
set(CLANG_FLAGS_ENABLE_AVX2 "-mavx2;-mfma")
set(CLANG_FLAGS_ENABLE_AVX2128 "-mavx2;-mfma")
set(CLANG_FLAGS_ENABLE_AVX512F "-mavx512f")
set(CLANG_FLAGS_ENABLE_NEON32 "--target=arm-linux-gnueabihf;-mcpu=cortex-a8")
# Arm AArch64 vector extensions.
set(CLANG_FLAGS_ENABLE_ADVSIMD "-march=armv8-a+simd")

# All variables storing compiler flags should be prefixed with FLAGS_
if(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
  # Always compile sleef with -ffp-contract.
  set(FLAGS_STRICTMATH "-ffp-contract=off")

  # Intel vector extensions.
  foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
    set(FLAGS_ENABLE_${SIMD} ${CLANG_FLAGS_ENABLE_${SIMD}})
  endforeach()

  # Warning flags.
  set(FLAGS_WALL "-Wall -Wno-unused -Wno-attributes")
  if(CMAKE_C_COMPILER_ID MATCHES "GNU")
    # The following compiler option is needed to suppress the warning
    # "AVX vector return without AVX enabled changes the ABI" at
    # src/arch/helpervecext.h:88
    string(CONCAT FLAGS_WALL ${FLAGS_WALL} " -Wno-psabi")
    set(FLAGS_ENABLE_NEON32 "-mfpu=neon")
  endif(CMAKE_C_COMPILER_ID MATCHES "GNU")
elseif(MSVC)
  # Intel vector extensions.
  set(FLAGS_ENABLE_SSE2 /D__SSE2__)
  set(FLAGS_ENABLE_SSE4 /D__SSE2__ /D__SSE3__ /D__SSE4_1__)
  set(FLAGS_ENABLE_AVX  /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /arch:AVX)
  set(FLAGS_ENABLE_FMA4 /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /D__FMA4__ /arch:AVX2)
  set(FLAGS_ENABLE_AVX2 /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /arch:AVX2)
  set(FLAGS_ENABLE_AVX2128 /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /arch:AVX2)
  set(FLAGS_ENABLE_AVX512F /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /D__AVX512F__ /arch:AVX2)
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
  set(FLAGS_ENABLE_SSE2 "-msse2")
  set(FLAGS_ENABLE_SSE4 "-msse4.1")
  set(FLAGS_ENABLE_AVX "-mavx")
  set(FLAGS_ENABLE_AVX2 "-march=core-avx2")
  set(FLAGS_ENABLE_AVX2128 "-march=core-avx2")
  set(FLAGS_ENABLE_AVX512F "-xCOMMON-AVX512")
  set(FLAGS_STRICTMATH "-fp-model strict -Qoption,cpp,--extended_float_type -qoverride-limits")
  set(FLAGS_FASTMATH "-fp-model fast=2 -Qoption,cpp,--extended_float_type -qoverride-limits")
  set(FLAGS_WALL "-fmax-errors=3 -Wall -Wno-unused -Wno-attributes")
endif()

set(SLEEF_C_FLAGS "${FLAGS_WALL} ${FLAGS_STRICTMATH}")

# FEATURE DETECTION

CHECK_TYPE_SIZE("long double" LD_SIZE)
if(LD_SIZE GREATER "9")
  # This is needed to check since internal compiler error occurs with gcc 4.x
  CHECK_C_SOURCE_COMPILES("
  typedef long double vlongdouble __attribute__((vector_size(sizeof(long double)*2)));
  vlongdouble vcast_vl_l(long double d) { return (vlongdouble) { d, d }; }
  int main() { vlongdouble vld = vcast_vl_l(0);
  }" COMPILER_SUPPORTS_LONG_DOUBLE)
endif()

CHECK_C_SOURCE_COMPILES("
  int main() { __float128 r = 1;
  }" COMPILER_SUPPORTS_FLOAT128)

# Detect if sleef supported architectures are also supported by the compiler

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
    __m256d r = _mm256_macc_pd(_mm256_set1_pd(1), _mm256_set1_pd(2), _mm256_set1_pd(3)); }"
  COMPILER_SUPPORTS_FMA4)

set (CMAKE_REQUIRED_FLAGS ${FLAGS_ENABLE_AVX2})
CHECK_C_SOURCE_COMPILES("
  #if defined(_MSC_VER)
  #include <intrin.h>
  #else
  #include <x86intrin.h>
  #endif
  int main() {
    __m256i r = _mm256_abs_epi32(_mm256_set1_epi32(1)); }"
  COMPILER_SUPPORTS_AVX2)

# AVX512F code requires optimisation flags -O3
set (CMAKE_TRY_COMPILE_CONFIGURATION Release)
set (CMAKE_REQUIRED_FLAGS ${FLAGS_ENABLE_AVX512F})
CHECK_C_SOURCE_COMPILES("
  #if defined(_MSC_VER)
  #include <intrin.h>
  #else
  #include <x86intrin.h>
  #endif
  __m512 addConstant(__m512 arg) {
    return _mm512_add_ps(arg, _mm512_set1_ps(1.f));
  }
  int main() {
    __m512i a = _mm512_set1_epi32(1);
    __mmask16 m = _mm512_cmp_epi32_mask(a, a, _MM_CMPINT_EQ);
    __m512i r = _mm512_andnot_si512(a, a); }"
  COMPILER_SUPPORTS_AVX512F)

# AVX2 implies AVX2128
if(COMPILER_SUPPORTS_AVX2)
  set(COMPILER_SUPPORTS_AVX2128 1)
endif()

# Check if compilation with OpenMP really succeeds
# It does not succeed on Travis even though find_package(OpenMP) succeeds.
find_package(OpenMP)
if(OPENMP_FOUND)
  set (CMAKE_REQUIRED_FLAGS "${OpenMP_C_FLAGS}")
  CHECK_C_SOURCE_COMPILES("
  #include <stdio.h>
  int main() {
  int i;
  #pragma omp parallel for
    for(i=0;i < 10;i++) { putchar(0); }
  }"
  COMPILER_SUPPORTS_OPENMP)
endif(OPENMP_FOUND)

# Reset used flags
set(CMAKE_REQUIRED_FLAGS)

# Cache the flags required to compile SLEEF.
string(CONCAT CMAKE_C_FLAGS ${SLEEF_C_FLAGS})

# Check weak aliases are supported.
CHECK_C_SOURCE_COMPILES("
  int f(int a) {
   return a + 2;
  }
  int g(int a) __attribute__((weak, alias(\"f\")));
  int main(void) {
    return g(2);
  }"
  COMPILER_SUPPORTS_WEAK_ALIASES)

##

option(SLEEF_SHOW_ERROR_LOG "Show cmake error log." OFF)
if(SLEEF_SHOW_ERROR_LOG)
  if (EXISTS ${PROJECT_BINARY_DIR}/CMakeFiles/CMakeError.log)
    file(READ ${PROJECT_BINARY_DIR}/CMakeFiles/CMakeError.log FILE_CONTENT)
    message("${FILE_CONTENT}")
  endif()
endif(SLEEF_SHOW_ERROR_LOG)

# Detect if cmake is running on Travis
string(COMPARE NOTEQUAL "" "$ENV{TRAVIS}" RUNNING_ON_TRAVIS)

if (${RUNNING_ON_TRAVIS} AND CMAKE_C_COMPILER_ID MATCHES "Clang")
  set(COMPILER_SUPPORTS_OPENMP FALSE)   # Workaround for https://github.com/travis-ci/travis-ci/issues/8613
  set(COMPILER_SUPPORTS_FLOAT128 FALSE) # Compilation on unroll_0_vecextqp.c does not finish on Travis
endif()
