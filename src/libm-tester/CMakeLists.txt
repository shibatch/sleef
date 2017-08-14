# Note: We are assuming SLEEF is the CMake root project.
# TODO: Remove constraint: do not use CMAKE_BINARY_DIR and CMAKE_SOURCE_DIR
link_directories(${CMAKE_BINARY_DIR}/lib)                 # libsleef
include_directories(${CMAKE_BINARY_DIR}/include)          # sleef.h
include_directories(${CMAKE_SOURCE_DIR}/src/libm)         # rename.h
include_directories(${CMAKE_BINARY_DIR}/src/libm/include) # rename headers

set(LIB_SLEEF sleef)
find_library(LIB_MPFR mpfr)
set(EXTRA_LIBS m
  ${LIB_MPFR})

# Compile executable 'iut'
add_executable(${TARGET_IUT} iut.c testerutil.c)
target_link_libraries(${TARGET_IUT} ${LIB_SLEEF} ${EXTRA_LIBS})
set(IUT_LIST ${TARGET_IUT})

# Add vector extension `iut`s
if(COMPILER_SUPPORTS_SSE2)
  add_executable(${TARGET_IUTSSE2} iutsimd.c iutsimdmain.c testerutil.c)
  target_compile_options(${TARGET_IUTSSE2}
    PRIVATE ${FLAGS_ENABLE_SSE2})
  target_compile_definitions(${TARGET_IUTSSE2}
    PRIVATE ENABLE_SSE2=1)
  target_link_libraries(${TARGET_IUTSSE2} ${LIB_SLEEF} ${EXTRA_LIBS})

  list(APPEND IUT_LIST ${TARGET_IUTSSE2})
endif(COMPILER_SUPPORTS_SSE2)

if(COMPILER_SUPPORTS_SSE4)
  add_executable(${TARGET_IUTSSE4} iutsimd.c iutsimdmain.c testerutil.c)
  target_compile_options(${TARGET_IUTSSE4}
    PRIVATE ${FLAGS_ENABLE_SSE4})
  target_compile_definitions(${TARGET_IUTSSE4}
    PRIVATE ENABLE_SSE4=1)
  target_link_libraries(${TARGET_IUTSSE4} ${LIB_SLEEF} ${EXTRA_LIBS})

  list(APPEND IUT_LIST ${TARGET_IUTSSE4})
endif(COMPILER_SUPPORTS_SSE4)

if(COMPILER_SUPPORTS_AVX)
  add_executable(${TARGET_IUTAVX} iutsimd.c iutsimdmain.c testerutil.c)
  target_compile_options(${TARGET_IUTAVX}
    PRIVATE ${FLAGS_ENABLE_AVX})
  target_compile_definitions(${TARGET_IUTAVX}
    PRIVATE ENABLE_AVX=1)
  target_link_libraries(${TARGET_IUTAVX} ${LIB_SLEEF} ${EXTRA_LIBS})

  list(APPEND IUT_LIST ${TARGET_IUTAVX})
endif(COMPILER_SUPPORTS_AVX)

if(COMPILER_SUPPORTS_FMA4)
  add_executable(${TARGET_IUTFMA4} iutsimd.c iutsimdmain.c testerutil.c)
  target_compile_options(${TARGET_IUTFMA4}
    PRIVATE ${FLAGS_ENABLE_AVX} ${FLAGS_ENABLE_FMA4})
  target_compile_definitions(${TARGET_IUTFMA4}
    PRIVATE ENABLE_FMA4=1)
  target_link_libraries(${TARGET_IUTFMA4} ${LIB_SLEEF} ${EXTRA_LIBS})

  list(APPEND IUT_LIST ${TARGET_IUTFMA4})
endif(COMPILER_SUPPORTS_FMA4)

if(COMPILER_SUPPORTS_AVX2)
  add_executable(${TARGET_IUTAVX2} iutsimd.c iutsimdmain.c testerutil.c)
  target_compile_options(${TARGET_IUTAVX2}
    PRIVATE ${FLAGS_ENABLE_AVX2})
  target_compile_definitions(${TARGET_IUTAVX2}
    PRIVATE ENABLE_AVX2=1)
  target_link_libraries(${TARGET_IUTAVX2} ${LIB_SLEEF} ${EXTRA_LIBS})

  list(APPEND IUT_LIST ${TARGET_IUTAVX2})
endif(COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
  add_executable(${TARGET_IUTAVX2128} iutsimd.c iutsimdmain.c testerutil.c)
  target_compile_options(${TARGET_IUTAVX2128}
    PRIVATE ${FLAGS_ENABLE_AVX2})
  target_compile_definitions(${TARGET_IUTAVX2128}
    PRIVATE ENABLE_AVX2128=1)
  target_link_libraries(${TARGET_IUTAVX2128} ${LIB_SLEEF} ${EXTRA_LIBS})

  list(APPEND IUT_LIST ${TARGET_IUTAVX2128})
endif(COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX512F)
  add_executable(${TARGET_IUTAVX512F} iutsimd.c iutsimdmain.c testerutil.c)
  target_compile_options(${TARGET_IUTAVX512F}
    PRIVATE ${FLAGS_ENABLE_AVX512F})
  target_compile_definitions(${TARGET_IUTAVX512F}
    PRIVATE ENABLE_AVX512F=1)
  target_link_libraries(${TARGET_IUTAVX512F} ${LIB_SLEEF} ${EXTRA_LIBS})

  list(APPEND IUT_LIST ${TARGET_IUTAVX512F})
endif(COMPILER_SUPPORTS_AVX512F)

if(COMPILER_SUPPORTS_ADVSIMD)
  # TODO: Remove relative-ness of the source file paths
  add_executable(${TARGET_IUTADVSIMD} iutsimd.c iutsimdmain.c testerutil.c ../libm/sleefsimdsp.c ../libm/sleefsimddp.c)
  target_compile_options(${TARGET_IUTADVSIMD}
    PRIVATE ${FLAGS_ENABLE_ADVSIMD})
  target_compile_definitions(${TARGET_IUTADVSIMD}
    PRIVATE ENABLE_ADVSIMD=1)
  target_link_libraries(${TARGET_IUTADVSIMD} ${LIB_SLEEF} ${EXTRA_LIBS})

  list(APPEND IUT_LIST ${TARGET_IUTADVSIMD})
endif(COMPILER_SUPPORTS_ADVSIMD)

# Compile executable 'tester'
add_executable(${TARGET_TESTER} tester.c testerutil.c)
target_link_libraries(${TARGET_TESTER} ${EXTRA_LIBS} ${LIB_SLEEF})
target_compile_definitions(${TARGET_TESTER}
  PRIVATE USEMPFR=1)
target_compile_options(${TARGET_TESTER} PRIVATE -Wno-unused-result)

# Set C standard requirement (-std=gnu99 for gcc)
set_target_properties(${TARGET_TESTER} ${IUT_LIST} PROPERTIES
  C_STANDARD 99)

# Target 'test-libm' runs the libm tests
add_custom_target(${TARGET_TEST}
  COMMAND ${CMAKE_COMMAND}
    -DTARGET_TESTER="${TARGET_TESTER}"
    -DIUT_LIST="${IUT_LIST}"
    -DLOCATIONS_FILE="${PROJECT_BINARY_DIR}/DefineLocations.cmake"
    -P ${SLEEF_SCRIPT_PATH}/ExecuteTests.cmake
  COMMENT "Running tests..."
  DEPENDS ${TARGET_TESTER} ${IUT_LIST})
# Tests depends on the library
add_dependencies(${TARGET_TEST} ${TARGET_LIBSLEEF})
