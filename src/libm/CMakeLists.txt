include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Helper executable: generates parts of the sleef header file
add_executable(${TARGET_MKRENAME} mkrename.c)
# Helper executable: dispatcher for the vector extensions
add_executable(${TARGET_MKDISP} mkdisp.c)
# Set C standard requirement (-std=gnu99 for gcc)
set_target_properties(${TARGET_MKRENAME} ${TARGET_MKDISP} PROPERTIES
  C_STANDARD 99)

set(RENAME_HEADERS "")
set(TARGET_OBJECTS "")

# Single precision and double precision
set(SIMD_SOURCES sleefsimdsp.c sleefsimddp.c)

if(COMPILER_SUPPORTS_ADVSIMD)
  set(HEADER_ADVSIMD ${CMAKE_CURRENT_BINARY_DIR}/include/renameadvsimd.h)
  set(OBJECT_ADVSIMD "sleefadvsimd")

  add_library(${OBJECT_ADVSIMD} OBJECT ${SIMD_SOURCES} ${HEADER_ADVSIMD})
  target_compile_definitions(${OBJECT_ADVSIMD} PRIVATE ENABLE_ADVSIMD=1 DORENAME=1)
  target_compile_options(${OBJECT_ADVSIMD} PRIVATE ${FLAGS_ENABLE_ADVSIMD})

  list(APPEND TARGET_OBJECTS ${OBJECT_ADVSIMD})
  list(APPEND RENAME_HEADERS ${HEADER_ADVSIMD})
endif(COMPILER_SUPPORTS_ADVSIMD)

if(COMPILER_SUPPORTS_SSE2)
  set(HEADER_SSE2 ${CMAKE_CURRENT_BINARY_DIR}/include/renamesse2.h)
  set(OBJECT_SSE2 "sleefsse2")

  add_library(${OBJECT_SSE2} OBJECT ${SIMD_SOURCES} ${HEADER_SSE2})
  target_compile_definitions(${OBJECT_SSE2} PRIVATE ENABLE_SSE2=1 DORENAME=1)
  target_compile_options(${OBJECT_SSE2} PRIVATE ${FLAGS_ENABLE_SSE2})

  list(APPEND TARGET_OBJECTS ${OBJECT_SSE2})
  list(APPEND RENAME_HEADERS ${HEADER_SSE2})
endif(COMPILER_SUPPORTS_SSE2)

if(COMPILER_SUPPORTS_SSE4)
  set(HEADER_SSE4 ${CMAKE_CURRENT_BINARY_DIR}/include/renamesse4.h)
  set(OBJECT_SSE4 "sleefsse4")

  add_library(${OBJECT_SSE4} OBJECT ${SIMD_SOURCES} ${HEADER_SSE4})
  target_compile_definitions(${OBJECT_SSE4} PRIVATE ENABLE_SSE4=1 DORENAME=1)
  target_compile_options(${OBJECT_SSE4} PRIVATE ${FLAGS_ENABLE_SSE4})

  list(APPEND TARGET_OBJECTS ${OBJECT_SSE4})
  list(APPEND RENAME_HEADERS ${HEADER_SSE4})
endif(COMPILER_SUPPORTS_SSE4)

if(COMPILER_SUPPORTS_AVX)
  set(HEADER_AVX ${CMAKE_CURRENT_BINARY_DIR}/include/renameavx.h)
  set(OBJECT_AVX "sleefavx")

  add_library(${OBJECT_AVX} OBJECT ${SIMD_SOURCES} ${HEADER_AVX})
  target_compile_definitions(${OBJECT_AVX} PRIVATE ENABLE_AVX=1 DORENAME=1)
  target_compile_options(${OBJECT_AVX} PRIVATE ${FLAGS_ENABLE_AVX})

  list(APPEND TARGET_OBJECTS ${OBJECT_AVX})
  list(APPEND RENAME_HEADERS ${HEADER_AVX})
endif(COMPILER_SUPPORTS_AVX)

set_target_properties(${TARGET_OBJECTS} PROPERTIES
  POSITION_INDEPENDENT_CODE ON   # -fPIC
  C_STANDARD 99)                 # -std=gnu99

# Generate the rename headers at build time
add_custom_command(
  OUTPUT ${RENAME_HEADERS}
  COMMAND ${CMAKE_COMMAND}
    -DRENAME_HEADERS="${RENAME_HEADERS}"
    -DLOCATIONS_FILE="${PROJECT_BINARY_DIR}/DefineLocations.cmake"
    -DTARGET_MKRENAME="${TARGET_MKRENAME}"
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
    -DSLEEF_ARCH_AARCH64="${SLEEF_ARCH_AARCH64}"
    -DLOCATIONS_FILE="${PROJECT_BINARY_DIR}/DefineLocations.cmake"
    -P ${SLEEF_SCRIPT_PATH}/GenerateSleefHeader.cmake
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm.h.org
    ${TARGET_MKRENAME}
    ${TARGET_MKDISP})

# Always run the commands to generate the headers
add_custom_target(headers ALL DEPENDS ${SLEEF_INCLUDE_HEADER} ${RENAME_HEADERS})

# Build main library
add_library(${TARGET_LIBSLEEF} SHARED sleefdp.c sleefsp.c)
add_dependencies(${TARGET_LIBSLEEF} headers)

# Add the object files as sources to the main library via cmake conditional generator expressions
target_sources(${TARGET_LIBSLEEF} PRIVATE $<$<BOOL:${COMPILER_SUPPORTS_SSE2}>:$<TARGET_OBJECTS:${OBJECT_SSE2}>>)
target_sources(${TARGET_LIBSLEEF} PRIVATE $<$<BOOL:${COMPILER_SUPPORTS_SSE4}>:$<TARGET_OBJECTS:${OBJECT_SSE4}>>)
target_sources(${TARGET_LIBSLEEF} PRIVATE $<$<BOOL:${COMPILER_SUPPORTS_AVX}>:$<TARGET_OBJECTS:${OBJECT_AVX}>>)
target_sources(${TARGET_LIBSLEEF} PRIVATE $<$<BOOL:${COMPILER_SUPPORTS_ADVSIMD}>:$<TARGET_OBJECTS:${OBJECT_ADVSIMD}>>)

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
set_target_properties(${TARGET_LIBSLEEF}
  PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION}
  C_STANDARD 99)

# Install libsleef and sleef.h
install(FILES ${SLEEF_INCLUDE_HEADER}
  DESTINATION include)
install(TARGETS ${TARGET_LIBSLEEF}
  DESTINATION lib)
