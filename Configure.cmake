include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckTypeSize)

# Check that the library LIB_MPFR is available
find_library(LIB_MPFR mpfr)

# The library currently supports the following SIMD architectures
set(SLEEF_SUPPORTED_EXTENSIONS
  SSE2 SSE4 AVX FMA4 AVX2 AVX2128 AVX512F # x86
  ADVSIMD				  # Aarch64
  NEON					  # Aarch32
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
  set(HEADER_PARAMS_ADVSIMD    "2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON advsimd")

endif()

# Enable building of the GNU ABI version for x86 and aarch64
if(NOT MSVC)
  if((SLEEF_ARCH_X86 OR SLEEF_ARCH_AARCH64))
    set(SLEEF_ENABLE_GNUABI ON CACHE INTERNAL "Build GNU ABI compatible version.")
  endif()
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

command_arguments(RENAME_PARAMS_GNUABI_SSE2    sse2 b 2 4 _mm128d _mm128 _mm128i _mm128i __SSE2__)
command_arguments(RENAME_PARAMS_GNUABI_AVX     avx c 4 8 __m256d __m256 __m128i "struct { __m128i x, y$<SEMICOLON> }" __AVX__)
command_arguments(RENAME_PARAMS_GNUABI_AVX2    avx2 d 4 8 __m256d __m256 __m128i __m256i __AVX2__)
command_arguments(RENAME_PARAMS_GNUABI_AVX512F avx512f e 8 16 __m512d __m512 __m256i __m512i __AVX512F__)
command_arguments(RENAME_PARAMS_GNUABI_ADVSIMD advsimd n 2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON)

# COMPILER DETECTION

# All variables storing compiler flags should be prefixed with FLAGS_
if(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
  # Always compile sleef with -ffp-contract.
  set(FLAGS_STRICTMATH "-ffp-contract=off")

  # Intel vector extensions.
  set(FLAGS_ENABLE_SSE2 "-msse2")
  set(FLAGS_ENABLE_SSE4 "-msse4.1")
  set(FLAGS_ENABLE_AVX "-mavx")
  set(FLAGS_ENABLE_FMA4 "-mfma4")
  set(FLAGS_ENABLE_AVX2 "-mavx2;-mfma")
  set(FLAGS_ENABLE_AVX2128 "-mavx2;-mfma")
  set(FLAGS_ENABLE_AVX512F "-mavx512f")
  set(FLAGS_ENABLE_NEON "")

  # Arm AArch64 vector extensions.
  set(FLAGS_ENABLE_ADVSIMD "-march=armv8-a+simd")
  
  # Warning flags.
  set(FLAGS_WALL "-Wall -Wno-unused -Wno-attributes")
elseif(MSVC)
  # Intel vector extensions.
  set(FLAGS_ENABLE_SSE2 /D__SSE2__)
  set(FLAGS_ENABLE_SSE4 /D__SSE2__ /D__SSE3__ /D__SSE4_1__)
  set(FLAGS_ENABLE_AVX  /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /arch:AVX)
  set(FLAGS_ENABLE_FMA4 /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /D__FMA4__ /arch:AVX2)
  set(FLAGS_ENABLE_AVX2 /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /arch:AVX2)
  set(FLAGS_ENABLE_AVX2128 /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /arch:AVX2)
  set(FLAGS_ENABLE_AVX512F /D__SSE2__ /D__SSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /arch:AVX2)
endif()

set(SLEEF_C_FLAGS "${FLAGS_WALL} ${FLAGS_STRICTMATH}")

# FEATURE DETECTION

CHECK_TYPE_SIZE("long double" LD_SIZE)
if(LD_SIZE GREATER "9")
  set(COMPILER_SUPPORTS_LONG_DOUBLE 1)
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
  static auto TestCmp = _MM_CMPINT_EQ;
  int main() {
    __m512i a = _mm512_set1_epi32(1);
    __m512i r = _mm512_andnot_si512(a, a); }"
  COMPILER_SUPPORTS_AVX512F)

# AVX2 implies AVX2128
if(COMPILER_SUPPORTS_AVX2)
  set(COMPILER_SUPPORTS_AVX2128 1)
endif()

# Reset used flags
set(CMAKE_REQUIRED_FLAGS)

# Cache the flags required to compile SLEEF.
string(CONCAT CMAKE_C_FLAGS ${SLEEF_C_FLAGS})
