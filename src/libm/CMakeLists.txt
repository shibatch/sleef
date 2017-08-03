include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Helper executable: generates parts of the sleef header file
add_executable(${TARGET_MKRENAME} mkrename.c)
target_compile_features(${TARGET_MKRENAME}
  PUBLIC c_variadic_macros)
# Helper executable: dispatcher for the vector extensions
add_executable(${TARGET_MKDISP} mkdisp.c)
target_compile_features(${TARGET_MKDISP}
  PUBLIC c_variadic_macros)

set(RENAME_HEADERS "")
set(TARGET_OBJECTS "")

# Single precision and double precision
set(SIMD_SOURCES sleefsimdsp.c sleefsimddp.c)

if(COMPILER_SUPPORTS_SSE2)
  set(HEADER_SSE2 ${CMAKE_CURRENT_BINARY_DIR}/include/renamesse2.h)
  set(OBJECT_SSE2 "sleefsse2")

  add_library(${OBJECT_SSE2} OBJECT ${SIMD_SOURCES} ${HEADER_SSE2})
  target_compile_options(${OBJECT_SSE2} PRIVATE -msse2)
  target_compile_definitions(${OBJECT_SSE2} PRIVATE ENABLE_SSE2=1 DORENAME=1)
  target_compile_features(${OBJECT_SSE2} PRIVATE c_variadic_macros)

  list(APPEND TARGET_OBJECTS ${OBJECT_SSE2})
  list(APPEND RENAME_HEADERS ${HEADER_SSE2})
endif(COMPILER_SUPPORTS_SSE2)

# Generate the rename headers at build time
add_custom_command(
  OUTPUT ${RENAME_HEADERS}
  COMMAND ${CMAKE_COMMAND}
    -DRENAME_HEADERS="${RENAME_HEADERS}"
    -DCMAKE_MODULE_PATH="${CMAKE_MODULE_PATH}"
    -P ${SLEEF_SCRIPT_PATH}/GenerateRenameHeader.cmake
  DEPENDS
    ${TARGET_MKRENAME})

# Generate the sleef header at build time
set(SLEEF_INCLUDE_HEADER ${CMAKE_BINARY_DIR}/include/sleef.h)
add_custom_command(
  OUTPUT ${SLEEF_INCLUDE_HEADER}
  COMMAND ${CMAKE_COMMAND}
    -DTARGET_MKRENAME="${TARGET_MKRENAME}"
    -DSLEEF_ARCH_X86="${SLEEF_ARCH_X86}"
    -DCMAKE_MODULE_PATH="${CMAKE_MODULE_PATH}"
    -P ${SLEEF_SCRIPT_PATH}/GenerateSleefHeader.cmake
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm.h.org
    ${TARGET_MKRENAME}
    ${TARGET_MKDISP})

# Always run the commands to generate the headers
add_custom_target(headers ALL DEPENDS ${SLEEF_INCLUDE_HEADER} ${RENAME_HEADERS})

# Build main library
add_library(${TARGET_LIBSLEEF}
  SHARED sleefdp.c sleefsp.c
  $<TARGET_OBJECTS:${OBJECT_SSE2}>)
add_dependencies(${TARGET_LIBSLEEF} headers)

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
# Selects C99 with standard gnu99 (add -std=gnu99) for gcc
target_compile_features(${TARGET_LIBSLEEF}
  PUBLIC c_variadic_macros)
set_target_properties(${TARGET_LIBSLEEF}
  PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION})

# Install libsleef and sleef.h
install(FILES ${SLEEF_INCLUDE_HEADER}
  DESTINATION include)
install(TARGETS ${TARGET_LIBSLEEF}
  DESTINATION lib)
