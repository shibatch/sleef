# Note: We are assuming SLEEF is the CMake root project.
# TODO: Remove constraint: do not use CMAKE_BINARY_DIR and CMAKE_SOURCE_DIR
link_directories(${CMAKE_BINARY_DIR}/lib)                 # libsleef
link_directories(${CMAKE_BINARY_DIR}/src/common)          # common.a
include_directories(${CMAKE_BINARY_DIR}/include)          # sleef.h
include_directories(${CMAKE_SOURCE_DIR}/src/libm)         # rename.h
include_directories(${CMAKE_BINARY_DIR}/src/libm/include) # rename headers

set(EXTRA_LIBS m
  ${LIB_MPFR}
  ${TARGET_LIBCOMMON_STATIC}
  )

# Compile executable 'iut'
add_executable(${TARGET_IUT} iut.c testerutil.c)
target_link_libraries(${TARGET_IUT} ${TARGET_LIBSLEEF} ${EXTRA_LIBS})
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

    list(APPEND IUT_LIST ${TARGET_IUT${SIMD}})
  endif(COMPILER_SUPPORTS_${SIMD})
endmacro(test_extension)

foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
  test_extension(${SIMD})
endforeach()

# Compile executable 'tester'
add_executable(${TARGET_TESTER} tester.c testerutil.c)
target_link_libraries(${TARGET_TESTER} ${EXTRA_LIBS} ${TARGET_LIBSLEEF})
target_compile_definitions(${TARGET_TESTER}
  PRIVATE USEMPFR=1)
target_compile_options(${TARGET_TESTER} PRIVATE -Wno-unused-result)

# Set C standard requirement (-std=gnu99 for gcc)
set_target_properties(${TARGET_TESTER} ${IUT_LIST} PROPERTIES
  C_STANDARD 99)

# Target 'test-libm' runs the libm tests
add_custom_target(${TARGET_TEST})
foreach(exe_iut ${IUT_LIST})
  add_custom_command(TARGET ${TARGET_TEST}
    COMMAND ${TARGET_TESTER} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${exe_iut}
  )
endforeach()

# Tests depends on the library
add_dependencies(${TARGET_IUT} ${TARGET_HEADERS})
add_dependencies(${TARGET_TEST} ${TARGET_LIBSLEEF} ${TARGET_HEADERS} ${TARGET_IUT})

