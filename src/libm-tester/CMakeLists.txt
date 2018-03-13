# Note: We are assuming SLEEF is the CMake root project.
# TODO: Remove constraint: do not use CMAKE_BINARY_DIR and CMAKE_SOURCE_DIR
link_directories(${CMAKE_BINARY_DIR}/lib)                 # libsleef
link_directories(${CMAKE_BINARY_DIR}/src/common)          # common.a
include_directories(${CMAKE_BINARY_DIR}/include)          # sleef.h
include_directories(${CMAKE_SOURCE_DIR}/src/libm)         # rename.h
include_directories(${CMAKE_BINARY_DIR}/src/libm/include) # rename headers

if(NOT LIB_MPFR)
  find_program(TESTER_COMMAND tester)
endif(NOT LIB_MPFR)

find_library(LIBRT rt)
if (NOT LIBRT)
  set(LIBRT "")
endif()

set(CMAKE_C_FLAGS ORG_CMAKE_C_FLAGS)
string(CONCAT CMAKE_C_FLAGS ${SLEEF_C_FLAGS})

#

function(add_test_iut IUT)
  if (LIB_MPFR)
    set(TESTER ${TARGET_TESTER})
  elseif(TESTER_COMMAND)
    set(TESTER ${TESTER_COMMAND})
  endif()
  # When we are crosscompiling using the mkrename* tools from a native
  # build, we use the tester executable from the native build.
  if (CMAKE_CROSSCOMPILING AND NATIVE_BUILD_DIR)
    set(TESTER ${NATIVE_BUILD_DIR}/bin/${TARGET_TESTER})
  endif(CMAKE_CROSSCOMPILING AND NATIVE_BUILD_DIR)
  if (TESTER)
    if (NOT EMULATOR)
      if (SDE_COMMAND)
	set(FLAGS_SDE "--sde" ${SDE_COMMAND})
      else()
	set(FLAGS_SDE)
      endif()
      if (ARMIE_COMMAND)
        set(FLAGS_ARMIE ${ARMIE_COMMAND} -msve-vector-bits=${SVE_VECTOR_BITS})
      else()
        set(FLAGS_ARMIE)
      endif()
      add_test(NAME ${IUT}
	COMMAND ${TESTER} ${FLAGS_SDE} ${FLAGS_ARMIE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${IUT}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    else()
      add_test(NAME ${IUT}
	COMMAND ${TESTER} ${EMULATOR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${IUT}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif()
  endif()
endfunction()

# Compile executable 'iut'
add_executable(${TARGET_IUT} iut.c testerutil.c)
target_compile_definitions(${TARGET_IUT} PRIVATE ${COMMON_TARGET_DEFINITIONS})
target_link_libraries(${TARGET_IUT} ${TARGET_LIBSLEEF}
  ${LIBM} ${LIBRT})
set_target_properties(${TARGET_IUT} PROPERTIES C_STANDARD 99)
add_test_iut(${TARGET_IUT})
set(IUT_LIST ${TARGET_IUT})

set(IUT_SRC iutsimd.c iutsimdmain.c testerutil)

# Add vector extension `iut`s
macro(test_extension SIMD)
  if(COMPILER_SUPPORTS_${SIMD})
    add_executable(${TARGET_IUT${SIMD}} ${IUT_SRC})
    target_compile_options(${TARGET_IUT${SIMD}}
      PRIVATE ${FLAGS_ENABLE_${SIMD}})
    target_compile_definitions(${TARGET_IUT${SIMD}}
      PRIVATE ENABLE_${SIMD}=1 ${COMMON_TARGET_DEFINITIONS})
    target_link_libraries(${TARGET_IUT${SIMD}} ${TARGET_LIBSLEEF}
      ${LIBM} ${LIBRT})

    add_dependencies(${TARGET_IUT${SIMD}} ${TARGET_HEADERS})
    add_dependencies(${TARGET_IUT${SIMD}} ${TARGET_LIBSLEEF})
    set_target_properties(${TARGET_IUT${SIMD}} PROPERTIES C_STANDARD 99)
    add_test_iut(${TARGET_IUT${SIMD}})
    list(APPEND IUT_LIST ${TARGET_IUT${SIMD}})

    if(LIB_MPFR AND NOT ${SIMD} STREQUAL NEON32 AND NOT MINGW)
      # Build tester2 SIMD
      string(TOLOWER ${SIMD} SCSIMD)
      foreach(P dp sp)
	      set(T "tester2${SCSIMD}${P}")
	      add_executable(${T} tester2simd${P}.c testerutil.c)
	      target_compile_options(${T} PRIVATE ${FLAGS_ENABLE_${SIMD}})
	      target_compile_definitions(${T} PRIVATE ENABLE_${SIMD}=1 USEMPFR=1 ${COMMON_TARGET_DEFINITIONS})
	      set_target_properties(${T} PROPERTIES C_STANDARD 99)
	      target_link_libraries(${T} ${TARGET_LIBSLEEF} ${LIB_MPFR} ${LIBM} ${LIBGMP})
	      add_dependencies(${T} ${TARGET_HEADERS})
	      add_dependencies(${T} ${TARGET_LIBSLEEF})
	      if (MPFR_INCLUDE_DIR)
		target_include_directories(${T} PRIVATE ${MPFR_INCLUDE_DIR})
	      endif()
      endforeach()
    endif()
  endif(COMPILER_SUPPORTS_${SIMD})
endmacro(test_extension)

foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
  test_extension(${SIMD})
endforeach()

function(add_gnuabi_compatibility_test SIMD MASKED)
  if (MASKED)
    set(GNUABI_COMPATIBILITY_TEST gnuabi_compatibility_${SIMD}_masked)
  else(MASKED)
    set(GNUABI_COMPATIBILITY_TEST gnuabi_compatibility_${SIMD})
  endif(MASKED)
  add_executable(${GNUABI_COMPATIBILITY_TEST} gnuabi_compatibility.c)
  set_target_properties(${GNUABI_COMPATIBILITY_TEST} PROPERTIES C_STANDARD 99)
  target_compile_options(${GNUABI_COMPATIBILITY_TEST}
    PRIVATE ${FLAGS_ENABLE_${SIMD}})
  if (MASKED)
    target_compile_definitions(${GNUABI_COMPATIBILITY_TEST}
      PRIVATE ENABLE_${SIMD}=1 ${COMMON_TARGET_DEFINITIONS} MASKED_GNUABI=1)
  else(MASKED)
    target_compile_definitions(${GNUABI_COMPATIBILITY_TEST}
      PRIVATE ENABLE_${SIMD}=1 ${COMMON_TARGET_DEFINITIONS})
  endif(MASKED)
  target_link_libraries(${GNUABI_COMPATIBILITY_TEST} ${TARGET_LIBSLEEFGNUABI} ${LIBM})
  # These are linker tests that don't really need to be executed,
  # but seeing them in the report of ctest gives an idea of what
  # has been built for testing.
  if (EMULATOR)
    add_test(NAME ${GNUABI_COMPATIBILITY_TEST}
      COMMAND ${EMULATOR} $<TARGET_FILE:${GNUABI_COMPATIBILITY_TEST}>
      WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  else()
    add_test(NAME ${GNUABI_COMPATIBILITY_TEST}
      COMMAND $<TARGET_FILE:${GNUABI_COMPATIBILITY_TEST}>)
  endif(EMULATOR)
endfunction(add_gnuabi_compatibility_test)

if(ENABLE_GNUABI)
  foreach(SIMD ${SLEEF_SUPPORTED_GNUABI_EXTENSIONS})
    if(COMPILER_SUPPORTS_${SIMD})
      # GNUABI compatibility for the unmasked symbols.
      add_gnuabi_compatibility_test(${SIMD} OFF)
      # GNUABI compatibility for the masked symbols.
      if (MKMASKED_PARAMS_GNUABI_${SIMD}_sp)
	add_gnuabi_compatibility_test(${SIMD} ON)
      endif(MKMASKED_PARAMS_GNUABI_${SIMD}_sp)
    endif (COMPILER_SUPPORTS_${SIMD})
  endforeach(SIMD ${SLEEF_SUPPORTED_GNUABI_EXTENSIONS})
endif(ENABLE_GNUABI)

if (SLEEF_ARCH_X86)
  # iutdsp128
  add_executable(iutdsp128 ${IUT_SRC})
  target_compile_definitions(iutdsp128 PRIVATE ENABLE_DSP128=1 ${COMMON_TARGET_DEFINITIONS})
  target_link_libraries(iutdsp128 ${TARGET_LIBSLEEF} ${LIBM} ${LIBRT})
  add_dependencies(iutdsp128 ${TARGET_HEADERS} ${TARGET_LIBSLEEF})
  add_test_iut(iutdsp128)
  list(APPEND IUT_LIST iutdsp128)

  # iutdsp256
  add_executable(iutdsp256 ${IUT_SRC})
  target_compile_definitions(iutdsp256 PRIVATE ENABLE_DSP256=1 ${COMMON_TARGET_DEFINITIONS})
  target_compile_options(iutdsp256 PRIVATE ${FLAGS_ENABLE_AVX})
  target_link_libraries(iutdsp256 ${TARGET_LIBSLEEF} ${LIBM} ${LIBRT})
  add_dependencies(iutdsp256 ${TARGET_HEADERS} ${TARGET_LIBSLEEF})
  add_test_iut(iutdsp256)
  list(APPEND IUT_LIST iutdsp256)
endif(SLEEF_ARCH_X86)

if(LIB_MPFR AND NOT MINGW)
  # Build tester2 scalar
  foreach(P dp sp)
    set(T "tester2${P}")
    add_executable(${T} tester2${P}.c testerutil.c)
    target_compile_definitions(${T} PRIVATE USEMPFR=1 ${COMMON_TARGET_DEFINITIONS})
    if (MPFR_INCLUDE_DIR)
      target_include_directories(${T} PRIVATE ${MPFR_INCLUDE_DIR})
    endif()

    target_link_libraries(${T} ${TARGET_LIBSLEEF} ${LIB_MPFR} ${LIBM} ${LIBGMP})
    add_dependencies(${T} ${TARGET_HEADERS})
    add_dependencies(${T} ${TARGET_LIBSLEEF})
  endforeach()

  # No test defined with tester2

  # Compile executable 'tester'
  add_host_executable(${TARGET_TESTER} tester.c testerutil.c)
  if (NOT CMAKE_CROSSCOMPILING)
    target_link_libraries(${TARGET_TESTER} ${LIB_MPFR} ${TARGET_LIBSLEEF} ${LIBM} ${LIBGMP})
    target_compile_definitions(${TARGET_TESTER}
      PRIVATE USEMPFR=1 ${COMMON_TARGET_DEFINITIONS})
    target_compile_options(${TARGET_TESTER} PRIVATE -Wno-unused-result)
    set_target_properties(${TARGET_TESTER} PROPERTIES C_STANDARD 99)
    if (MPFR_INCLUDE_DIR)
      target_include_directories(${TARGET_TESTER} PRIVATE ${MPFR_INCLUDE_DIR})
    endif()
  endif()
endif(LIB_MPFR AND NOT MINGW)

# Tests depends on the library
add_dependencies(${TARGET_IUT} ${TARGET_HEADERS})
