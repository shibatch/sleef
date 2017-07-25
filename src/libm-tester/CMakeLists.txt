link_directories(${CMAKE_BINARY_DIR}/lib)
include_directories(${CMAKE_BINARY_DIR}/include)

# This is why a conversation about headers is needed.
# TODO: Find way to avoid relative includes.
include_directories(../libm)

set(LIB_SLEEF sleef)
find_library(LIB_MPFR mpfr)
set(EXTRA_LIBS m
  ${LIB_MPFR})

# Compile executable 'iut'
add_executable(${TARGET_IUT} iut.c testerutil.c)
target_link_libraries(${TARGET_IUT} ${LIB_SLEEF} ${EXTRA_LIBS})
target_compile_options(${TARGET_IUT}
  PRIVATE ${FLAGS_WALL} ${FLAGS_STRICTMATH})

# Compiler executable 'tester'
add_executable(${TARGET_TESTER} tester.c testerutil.c)
target_link_libraries(${TARGET_TESTER} ${EXTRA_LIBS})
target_compile_definitions(${TARGET_TESTER}
  PRIVATE USEMPFR=1)
target_compile_options(${TARGET_TESTER}
  PRIVATE ${FLAGS_WALL} ${FLAGS_STRICTMATH} -Wno-unused-result)

# Target 'test-libm' runs the libm tests
add_custom_target(${TARGET_TEST}
  COMMAND ./tester ./iut
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
add_dependencies(${TARGET_TEST} ${TARGET_TESTER} ${TARGET_IUT} ${TARGET_LIBSLEEF})
