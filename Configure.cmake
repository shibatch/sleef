include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckTypeSize)

# PLATFORM DETECTION

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
  set(SLEEF_ARCH_X86 ON CACHE INTERNAL "True for x86 architecture.")
  set(COMPILER_SUPPORTS_SSE2 1)
endif()

# COMPILER DETECTION

# All variables storing compiler flags should be prefixed with FLAGS_

if(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
  set(FLAGS_WALL "-Wall -Wno-unused -Wno-attributes")
  set(FLAGS_FASTMATH "-ffast-math")
  set(FLAGS_STRICTMATH "-ffp-contract=off")
  set(FLAGS_OPENMP "-fopenmp")

  set(FLAGS_ENABLE_SSE2 "-msse2")
endif()

# Warning flags tuning
# Do we need this for clang; currently get a unknown warning option on Ubuntu
#if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  #set(FLAGS_WALL "${FLAGS_WALL} -Wno-shift-negative-value")
#endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAGS_WALL}")

# Always compile sleef with -ffp-contract and log at configuration time
if(SLEEF_SHOW_CONFIG)
  message(STATUS "Using option `${FLAGS_STRICTMATH}` to compile libsleef")
endif(SLEEF_SHOW_CONFIG)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAGS_STRICTMATH}")

# FEATURE DETECTION

CHECK_TYPE_SIZE("long double" LD_SIZE)
if(LD_SIZE GREATER "9")
  set(COMPILER_SUPPORTS_LONG_DOUBLE 1)
endif()

CHECK_C_SOURCE_COMPILES(
  "int main(){ __float128 r = 1;}"
  COMPILER_SUPPORTS_FLOAT128)
