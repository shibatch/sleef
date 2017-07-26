# Helper executable: generates parts of the sleef header file
add_executable(${TARGET_MKRENAME} mkrename.c)
target_compile_options(${TARGET_MKRENAME}
  PRIVATE ${FLAGS_WALL})
target_compile_features(${TARGET_MKRENAME}
  PUBLIC c_variadic_macros)

# TODO: set the parameters for mkrename run for the SIMD implementation
set(MKRENAME_PARAMS "")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/GenerateSleefHeader.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GenerateSleefHeader.cmake @ONLY)

# Add custom command to generate the sleef header at build time
set(SLEEF_INCLUDE_HEADER ${CMAKE_BINARY_DIR}/include/sleef.h)
add_custom_command(
  OUTPUT ${SLEEF_INCLUDE_HEADER}
  COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/GenerateSleefHeader.cmake
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm.h.org
    ${CMAKE_CURRENT_BINARY_DIR}/GenerateSleefHeader.cmake
    ${TARGET_MKRENAME})

# Build main library
add_library(${TARGET_LIBSLEEF}
  SHARED sleefdp.c sleefsp.c ${SLEEF_INCLUDE_HEADER})

if(COMPILER_SUPPORTS_LONG_DOUBLE)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefld.c)
endif(COMPILER_SUPPORTS_LONG_DOUBLE)

if(COMPILER_SUPPORTS_FLOAT128)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefqp.c)
  target_compile_definitions(${TARGET_LIBSLEEF}
    PRIVATE ENABLEFLOAT128=1)
endif(COMPILER_SUPPORTS_FLOAT128)

target_compile_definitions(${TARGET_LIBSLEEF}
  PRIVATE DORENAME=1)
target_compile_options(${TARGET_LIBSLEEF}
  PUBLIC ${FLAGS_WALL} ${FLAGS_STRICTMATH})
# Selects C99 with standard gnu99 (add -std=gnu99) for gcc
target_compile_features(${TARGET_LIBSLEEF}
  PUBLIC c_variadic_macros)
set_target_properties(${TARGET_LIBSLEEF} PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION})

# Always compile sleef with -ffp-contract and log at configuration time
if(SLEEF_SHOW_CONFIG)
  message(STATUS "Using option `${FLAGS_STRICTMATH}` to compile libsleef")
endif(SLEEF_SHOW_CONFIG)

install(FILES ${SLEEF_INCLUDE_HEADER}
  DESTINATION include)
install(TARGETS ${TARGET_LIBSLEEF}
  DESTINATION lib)
