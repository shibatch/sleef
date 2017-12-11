# Note: We are assuming SLEEF is the CMake root project.
# TODO: Remove constraint: do not use CMAKE_BINARY_DIR and CMAKE_SOURCE_DIR
link_directories(${CMAKE_BINARY_DIR}/lib)                 # libsleef
link_directories(${CMAKE_BINARY_DIR}/src/common)          # common.a
include_directories(${CMAKE_BINARY_DIR}/include)          # sleef.h
include_directories(${CMAKE_SOURCE_DIR}/src/libm)         # rename.h
include_directories(${CMAKE_BINARY_DIR}/src/libm/include) # rename headers

set(EXTRA_LIBS ${LIBM}
  ${LIB_MPFR}
  ${TARGET_LIBCOMMON_STATIC}
  )

# Compile executable 'iut'
add_executable(${TARGET_IUT} iut.c testerutil.c)
target_link_libraries(${TARGET_IUT} ${TARGET_LIBSLEEF} ${EXTRA_LIBS})
add_test(NAME ${TARGET_IUT}
  COMMAND ${TARGET_TESTER} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET_IUT})

set(IUT_LIST ${TARGET_IUT})

set(IUT_SRC iutsimd.c iutsimdmain.c testerutil)

# Add vector extension `iut`s
macro(test_extension SIMD)
  if(COMPILER_SUPPORTS_${SIMD})
    add_executable(${TARGET_IUT${SIMD}} ${IUT_SRC})
    target_compile_options(${TARGET_IUT${SIMD}}
      PRIVATE ${FLAGS_ENABLE_${SIMD}})
    target_compile_definitions(${TARGET_IUT${SIMD}}
      PRIVATE ENABLE_${SIMD}=1)
    target_link_libraries(${TARGET_IUT${SIMD}} ${TARGET_LIBSLEEF} ${EXTRA_LIBS})

    add_dependencies(${TARGET_IUT${SIMD}} ${TARGET_HEADERS})
    add_dependencies(${TARGET_IUT${SIMD}} ${TARGET_LIBSLEEF})

    add_test(NAME ${TARGET_IUT${SIMD}}
      COMMAND ${TARGET_TESTER} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET_IUT${SIMD}})
    
    list(APPEND IUT_LIST ${TARGET_IUT${SIMD}})

    if(ENABLE_GNUABI)
      if (COMPILER_SUPPORTS_WEAK_ALIASES)
	add_executable(link_finite${SIMD} finite_names_testing.c)
	target_compile_options(link_finite${SIMD}
	  PRIVATE ${FLAGS_ENABLE_${SIMD}})
	target_compile_definitions(link_finite${SIMD}
	  PRIVATE ENABLE_${SIMD}=1)
	target_link_libraries(link_finite${SIMD} ${TARGET_LIBSLEEFGNUABI})
	add_test(NAME link_finite${SIMD}
	  COMMAND link_finite${SIMD})
      endif (COMPILER_SUPPORTS_WEAK_ALIASES)
    endif(ENABLE_GNUABI)

    # Build tester2 SIMD
    string(TOLOWER ${SIMD} SCSIMD)
    foreach(P dp sp)
      set(T "tester2${SCSIMD}${P}")
      add_executable(${T} tester2simd${P}.c testerutil.c)
      target_compile_options(${T} PRIVATE ${FLAGS_ENABLE_${SIMD}})
      target_compile_definitions(${T} PRIVATE ENABLE_${SIMD}=1 USEMPFR=1)
      set_target_properties(${T} PROPERTIES C_STANDARD 99)
      target_link_libraries(${T} ${TARGET_LIBSLEEF} ${EXTRA_LIBS})
      add_dependencies(${T} ${TARGET_HEADERS})
      add_dependencies(${T} ${TARGET_LIBSLEEF})
    endforeach()

  endif(COMPILER_SUPPORTS_${SIMD})
endmacro(test_extension)

foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
  test_extension(${SIMD})
endforeach()

# Build tester2 scalar
foreach(P dp sp)
  set(T "tester2${P}")
  add_executable(${T} tester2${P}.c testerutil.c)
  target_compile_definitions(${T} PRIVATE USEMPFR=1)
  target_link_libraries(${T} ${TARGET_LIBSLEEF} ${EXTRA_LIBS})
  add_dependencies(${T} ${TARGET_HEADERS})
  add_dependencies(${T} ${TARGET_LIBSLEEF})
endforeach()

# No test defined with tester2

# Compile executable 'tester'
add_executable(${TARGET_TESTER} tester.c testerutil.c)
target_link_libraries(${TARGET_TESTER} ${EXTRA_LIBS} ${TARGET_LIBSLEEF})
target_compile_definitions(${TARGET_TESTER}
  PRIVATE USEMPFR=1)
target_compile_options(${TARGET_TESTER} PRIVATE -Wno-unused-result)

# Set C standard requirement (-std=gnu99 for gcc)
set_target_properties(${TARGET_TESTER} ${IUT_LIST} PROPERTIES
  C_STANDARD 99)

# Tests depends on the library
add_dependencies(${TARGET_IUT} ${TARGET_HEADERS})

# Tester3

if (SLEEF_ARCH_X86)
  set(TESTER3_0 AVX2128        __m128d d2 __m128 f4 AVX2    __m256d d4 __m256 f8)
  set(TESTER3_1 SSE2           __m128d d2 __m128 f4 AVX     __m256d d4 __m256 f8)
  set(TESTER3_2 SSE4           __m128d d2 __m128 f4 AVX     __m256d d4 __m256 f8)

  set(TESTER3_ITERATION 0 1 2)
endif(SLEEF_ARCH_X86)

add_library(tester3main_obj OBJECT tester3main.c)

foreach(N ${TESTER3_ITERATION})
  list(GET TESTER3_${N} 0 EXTX)                                    # EXTX is AVX2_SCALAR, for example
  list(GET TESTER3_${N} 1 DPTYPEX)
  list(GET TESTER3_${N} 2 DPTYPESPECX)
  list(GET TESTER3_${N} 3 SPTYPEX)
  list(GET TESTER3_${N} 4 SPTYPESPECX)
  list(GET TESTER3_${N} 5 EXTY)                                    # EXTY is AVX2
  list(GET TESTER3_${N} 6 DPTYPEY)
  list(GET TESTER3_${N} 7 DPTYPESPECY)
  list(GET TESTER3_${N} 8 SPTYPEY)
  list(GET TESTER3_${N} 9 SPTYPESPECY)

  string(TOLOWER ${EXTX} EXTSPECX)                                 # EXTSPECX is avx2_scalar
  string(TOLOWER ${EXTY} EXTSPECY)                                 # EXTSPECY is avx2
  set(EXEFN tester3_${EXTSPECX}_${EXTSPECY})                       # EXEFN is tester3_avx_scalar_avx2

  add_executable(${EXEFN} $<TARGET_OBJECTS:tester3main_obj> tester3.c testerutil.c)
  target_compile_options(${EXEFN} PRIVATE ${FLAGS_ENABLE_${EXTX}} ${FLAGS_ENABLE_${EXTY}})
  target_compile_definitions(${EXEFN} PRIVATE
    DPTYPEX=${DPTYPEX} DPTYPESPECX=${DPTYPESPECX} SPTYPEX=${SPTYPEX} SPTYPESPECX=${SPTYPESPECX} EXTSPECX=${EXTSPECX}
    DPTYPEY=${DPTYPEY} DPTYPESPECY=${DPTYPESPECY} SPTYPEY=${SPTYPEY} SPTYPESPECY=${SPTYPESPECY} EXTSPECY=${EXTSPECY})
  set_target_properties(${EXEFN} PROPERTIES C_STANDARD 99)
  target_link_libraries(${EXEFN} ${TARGET_LIBSLEEF} ${TARGET_LIBCOMMON_STATIC})
  add_dependencies(${EXEFN} ${TARGET_HEADERS} ${TARGET_LIBSLEEF})

  add_test(NAME ${EXEFN} COMMAND ${EXEFN})
endforeach()
