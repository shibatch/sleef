include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Helper executable: generates parts of the sleef header file
add_executable(${TARGET_MKRENAME} mkrename.c)
add_executable(${TARGET_MKRENAME_GNUABI} mkrename_gnuabi.c)
# Helper executable: dispatcher for the vector extensions
add_executable(${TARGET_MKDISP} mkdisp.c)
# Set C standard requirement (-std=gnu99 for gcc)
set_target_properties(${TARGET_MKRENAME} ${TARGET_MKDISP} PROPERTIES
  C_STANDARD 99)

# Build main library
add_library(${TARGET_LIBSLEEF} SHARED sleefdp.c sleefsp.c)

# Check for different precision support and add sources accordingly
if(COMPILER_SUPPORTS_LONG_DOUBLE)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefld.c)
endif(COMPILER_SUPPORTS_LONG_DOUBLE)

if(COMPILER_SUPPORTS_FLOAT128)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefqp.c)
  target_compile_definitions(${TARGET_LIBSLEEF}
    PRIVATE ENABLEFLOAT128=1)
endif(COMPILER_SUPPORTS_FLOAT128)

set(RENAME_HEADERS "")
set(TARGET_OBJECTS "")
# Single precision and double precision
set(SIMD_SOURCES sleefsimdsp.c sleefsimddp.c)

# Repeat for each architecture: configure header file and compile object file
macro(compile_object SIMD)
  if(COMPILER_SUPPORTS_${SIMD})
    # Need lowercase string for rename header
    string(TOLOWER ${SIMD} vecarch)

    set(OBJECT_${SIMD} "sleef${vecarch}")
    set(HEADER_${SIMD} ${CMAKE_CURRENT_BINARY_DIR}/include/rename${vecarch}.h)

    add_library(${OBJECT_${SIMD}} OBJECT ${SIMD_SOURCES} ${HEADER_${SIMD}})
    target_compile_definitions(${OBJECT_${SIMD}} PRIVATE
      ENABLE_${SIMD}=1
      DORENAME=1)
    target_compile_options(${OBJECT_${SIMD}} PRIVATE
      ${FLAGS_ENABLE_${SIMD}})

    list(APPEND TARGET_OBJECTS ${OBJECT_${SIMD}})
    list(APPEND RENAME_HEADERS ${HEADER_${SIMD}})
  endif(COMPILER_SUPPORTS_${SIMD})
endmacro(compile_object)

# Include symbols for each SIMD architecture (if supported by the platform)
# Note: adds object file as sources via cmake conditional generator expression
foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
  compile_object(${SIMD})
  target_sources(${TARGET_LIBSLEEF} PRIVATE
    $<$<BOOL:${COMPILER_SUPPORTS_${SIMD}}>:$<TARGET_OBJECTS:${OBJECT_${SIMD}}>>)
endforeach()

macro(compile_object_gnuabi SIMD)
  if(COMPILER_SUPPORTS_${SIMD})
    # Need lowercase string for rename header
    string(TOLOWER ${SIMD} vecarch)

    set(OBJECT_${SIMD}_GNUABI "sleefgnuabi${vecarch}")
    set(HEADER_${SIMD}_GNUABI ${CMAKE_CURRENT_BINARY_DIR}/include/rename${vecarch}_gnuabi.h)

    add_library(${OBJECT_${SIMD}_GNUABI} OBJECT ${SIMD_SOURCES} ${HEADER_${SIMD}_GNUABI})
    target_compile_definitions(${OBJECT_${SIMD}_GNUABI} PRIVATE
      ENABLE_${SIMD}=1
      DORENAME=1
      ENABLE_GNUABI=1)
    target_compile_options(${OBJECT_${SIMD}_GNUABI} PRIVATE
      ${FLAGS_ENABLE_${SIMD}})

    list(APPEND TARGET_OBJECTS_GNUABI ${OBJECT_${SIMD}_GNUABI})
    list(APPEND RENAME_HEADERS_GNUABI ${HEADER_${SIMD}_GNUABI})
  endif(COMPILER_SUPPORTS_${SIMD})
endmacro(compile_object_gnuabi)

# Build gnuabi version from just simd object files
if(SLEEF_ENABLE_GNUABI)
  set(SLEEF_SUPPORTED_GNUABI_EXTENSIONS SSE2 AVX AVX2 AVX512F ADVSIMD)
  foreach(SIMD ${SLEEF_SUPPORTED_GNUABI_EXTENSIONS})
    compile_object_gnuabi(${SIMD})
    list(APPEND GNUABI_OBJ_SRC
      $<$<BOOL:${COMPILER_SUPPORTS_${SIMD}}>:$<TARGET_OBJECTS:${OBJECT_${SIMD}_GNUABI}>>)
  endforeach()

  add_library(${TARGET_LIBSLEEFGNUABI} SHARED ${GNUABI_OBJ_SRC})
endif(SLEEF_ENABLE_GNUABI)

# All code that goes into the main library needs to be position independent code
set_target_properties(${TARGET_OBJECTS} ${TARGET_OBJECTS_GNUABI} PROPERTIES
  POSITION_INDEPENDENT_CODE ON   # -fPIC
  C_STANDARD 99)                 # -std=gnu99

target_compile_definitions(${TARGET_LIBSLEEF}
  PRIVATE DORENAME=1)
set_target_properties(${TARGET_LIBSLEEF} ${TARGET_LIBSLEEFGNUABI} PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION}
  C_STANDARD 99)

# Generate the rename headers at build time
add_custom_command(
  OUTPUT ${RENAME_HEADERS} ${RENAME_HEADERS_GNUABI}
  COMMAND ${CMAKE_COMMAND}
    -DRENAME_HEADERS="${RENAME_HEADERS}"
    -DRENAME_HEADERS_GNUABI="${RENAME_HEADERS_GNUABI}"
    -DLOCATIONS_FILE="${PROJECT_BINARY_DIR}/DefineLocations.cmake"
    -DTARGET_MKRENAME="${TARGET_MKRENAME}"
    -DTARGET_MKRENAME_GNUABI="${TARGET_MKRENAME_GNUABI}"
    -P ${SLEEF_SCRIPT_PATH}/GenerateRenameHeader.cmake
  DEPENDS
    ${TARGET_MKRENAME}
    ${TARGET_MKRENAME_GNUABI})

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
add_custom_target(headers ALL DEPENDS
  ${SLEEF_INCLUDE_HEADER}
  ${RENAME_HEADERS}
  ${RENAME_HEADERS_GNUABI})

# Install libsleef and sleef.h
install(FILES ${SLEEF_INCLUDE_HEADER}
  DESTINATION include)
install(TARGETS ${TARGET_LIBSLEEF} ${TARGET_LIBSLEEFGNUABI}
  DESTINATION lib)
