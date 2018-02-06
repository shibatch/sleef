include_directories(${PROJECT_BINARY_DIR}/include)
set(COMMON_TARGET_PROPERTIES C_STANDARD 99)
set_source_files_properties(${CMAKE_BINARY_DIR}/include/sleef.h PROPERTIES GENERATED TRUE)

#

if(SLEEF_ARCH_X86 OR SLEEF_ARCH_AARCH64)
  add_executable(benchsleef benchsleef.c)
  target_link_libraries(benchsleef ${TARGET_LIBSLEEF} ${LIBM})

  add_library(benchsleef128_obj OBJECT benchsleef128.c bench.h ${CMAKE_BINARY_DIR}/include/sleef.h)
  add_dependencies(benchsleef128_obj ${TARGET_HEADERS})
  target_compile_options(benchsleef128_obj PRIVATE ${FLAGS_NO_OPTIMIZE})
  set_target_properties(benchsleef128_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
  target_sources(benchsleef PRIVATE $<TARGET_OBJECTS:benchsleef128_obj>)

  add_library(benchsleef256_obj OBJECT benchsleef256.c bench.h ${CMAKE_BINARY_DIR}/include/sleef.h)
  add_dependencies(benchsleef256_obj ${TARGET_HEADERS})
  set_target_properties(benchsleef256_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
  target_sources(benchsleef PRIVATE $<TARGET_OBJECTS:benchsleef256_obj>)

  add_library(benchsleef512_obj OBJECT benchsleef512.c bench.h ${CMAKE_BINARY_DIR}/include/sleef.h)
  add_dependencies(benchsleef512_obj ${TARGET_HEADERS})
  set_target_properties(benchsleef512_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
  target_sources(benchsleef PRIVATE $<TARGET_OBJECTS:benchsleef512_obj>)

  if(COMPILER_SUPPORTS_AVX)
    target_compile_options(benchsleef128_obj PRIVATE ${FLAGS_NO_OPTIMIZE} ${FLAGS_ENABLE_AVX})
    target_compile_options(benchsleef256_obj PRIVATE ${FLAGS_NO_OPTIMIZE} ${FLAGS_ENABLE_AVX})
  endif()

  if(COMPILER_SUPPORTS_AVX512F)
    target_compile_options(benchsleef512_obj PRIVATE ${FLAGS_NO_OPTIMIZE} ${FLAGS_ENABLE_AVX512F})
  endif()
endif()

