# Helper executable: generates parts of the sleef header file
add_executable(${TARGET_MKRENAME} mkrename.c)
target_compile_options(${TARGET_MKRENAME}
  PRIVATE ${FLAGS_BASIC_C} ${FLAGS_WALL})

# Helper executable: dispatcher
add_executable(${TARGET_MKDISP} mkdisp.c)
target_compile_options(${TARGET_MKDISP}
  PRIVATE ${FLAGS_BASIC_C} ${FLAGS_WALL})

# TODO: set the parameters for mkrename run
set(MKRENAME_PARAMS "")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/GenerateSleefHeader.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GenerateSleefHeader.cmake @ONLY)

# Add sleeh header target
add_custom_target(generate_sleef_header ALL
  DEPENDS ${CMAKE_BINARY_DIR}/include/sleef.h)

# Add custom command to process the sleef header and install it under /include
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/include/sleef.h
  COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/GenerateSleefHeader.cmake
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm.h.org
    ${CMAKE_CURRENT_BINARY_DIR}/GenerateSleefHeader.cmake)

# Install header files in the include directory
add_custom_target(install_headers ALL
  DEPENDS ${CMAKE_BINARY_DIR}/include/rename.h)

# Add custom command to install the rename header under /include
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/include/rename.h
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/rename.h ${CMAKE_BINARY_DIR}/include/rename.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rename.h)

# Build main library
add_library(${TARGET_LIBSLEEF}
  SHARED sleefdp.c sleefsp.c ${CMAKE_BINARY_DIR}/include/sleef.h ${CMAKE_BINARY_DIR}/include/rename.h)

if(ENABLE_LONG_DOUBLE)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefld.c)
endif(ENABLE_LONG_DOUBLE)

if(ENABLE_FLOAT128)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefqp.c)
  target_compile_definitions(${TARGET_LIBSLEEF}
    PRIVATE ENABLEFLOAT128=1)
endif(ENABLE_FLOAT128)

target_compile_definitions(${TARGET_LIBSLEEF}
  PRIVATE DORENAME=1)
target_compile_options(${TARGET_LIBSLEEF}
  PUBLIC ${FLAGS_BASIC_C} ${FLAGS_WALL} ${FLAGS_STRICTMATH})
set_target_properties(${TARGET_LIBSLEEF} PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION})

# Always compile sleef with -ffp-contract and log at configuration time
if(SLEEF_SHOW_CONFIG)
  message(STATUS "Using option `${FLAGS_STRICTMATH}` to compile libsleef")
endif(SLEEF_SHOW_CONFIG)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sleef.h
  DESTINATION include)
install(TARGETS ${TARGET_LIBSLEEF}
  DESTINATION lib)
