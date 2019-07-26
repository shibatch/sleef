file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/libm)         # dd.h

set(COMMON_TARGET_PROPERTIES C_STANDARD 99)

if(COMPILER_SUPPORTS_FLOAT128)
  list(APPEND COMMON_TARGET_DEFINITIONS ENABLEFLOAT128=1)
endif()

if(COMPILER_SUPPORTS_BUILTIN_MATH)
  list(APPEND COMMON_TARGET_DEFINITIONS ENABLE_BUILTIN_MATH=1)
endif()

if (BUILD_SHARED_LIBS)
  list(APPEND COMMON_TARGET_PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

#

set(QUAD_HEADER_PARAMS_PUREC_SCALAR    1 Sleef_quad1 Sleef_quad2  double      float       uint64_t   int32_t   __STDC__          purec)
set(QUAD_HEADER_PARAMS_PURECFMA_SCALAR 1 Sleef_quad1 Sleef_quad2  double      float       uint64_t   int32_t   FP_FAST_FMA       purecfma)
set(QUAD_HEADER_PARAMS_SSE2            2 Sleef_quad2 Sleef_quad4  __m128d     __m128      __m128i    __m128i   __SSE2__          sse2)
set(QUAD_HEADER_PARAMS_AVX             4 Sleef_quad4 Sleef_quad8  __m256d     __m256      __m256i    __m128i   __AVX__           avx)
set(QUAD_HEADER_PARAMS_FMA4            4 Sleef_quad4 Sleef_quad8  __m256d     __m256      __m256i    __m128i   __AVX__           fma4)
set(QUAD_HEADER_PARAMS_AVX2            4 Sleef_quad4 Sleef_quad8  __m256d     __m256      __m256i    __m128i   __AVX__           avx2)
set(QUAD_HEADER_PARAMS_AVX512F         8 Sleef_quad8 Sleef_quad16 __m512d     __m512      __m512i    __m256i   __AVX512F__       avx512f)
set(QUAD_HEADER_PARAMS_ADVSIMD         2 Sleef_quad2 Sleef_quad4  float64x2_t float32x4_t uint32x4_t int32x2_t __ARM_NEON        advsimd)
set(QUAD_HEADER_PARAMS_SVE             x Sleef_quadx Sleef_quadx  svfloat64_t svfloat32_t svint32_t  svint32_t __ARM_FEATURE_SVE sve)

set(QUAD_RENAME_PARAMS_PUREC_SCALAR    1 purec)
set(QUAD_RENAME_PARAMS_PURECFMA_SCALAR 1 purecfma)
set(QUAD_RENAME_PARAMS_SSE2            2 sse2)
set(QUAD_RENAME_PARAMS_AVX             4 avx)
set(QUAD_RENAME_PARAMS_FMA4            4 fma4)
set(QUAD_RENAME_PARAMS_AVX2            4 avx2)
set(QUAD_RENAME_PARAMS_AVX512F         8 avx512f)
set(QUAD_RENAME_PARAMS_ADVSIMD         2 advsimd)
set(QUAD_RENAME_PARAMS_SVE             x sve)

#

set(CMAKE_C_FLAGS ORG_CMAKE_C_FLAGS)
string(CONCAT CMAKE_C_FLAGS ${SLEEF_C_FLAGS})

# --------------------------------------------------------------------
# sleefquad.h
# --------------------------------------------------------------------
set(SLEEFQUAD_ORG_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/sleefquad_header.h.org)
set(SLEEFQUAD_ORG_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/sleefquad_footer.h.org)
set(SLEEFQUAD_INCLUDE_HEADER ${CMAKE_BINARY_DIR}/include/sleefquad.h)

set(SLEEF_HEADER_COMMANDS "")
list(APPEND SLEEF_HEADER_COMMANDS COMMAND ${CMAKE_COMMAND} -E copy ${SLEEFQUAD_ORG_HEADER} ${SLEEFQUAD_INCLUDE_HEADER})
foreach(SIMD ${SLEEFQUAD_SUPPORTED_EXT})
  if(COMPILER_SUPPORTS_${SIMD})
    list(APPEND SLEEF_HEADER_COMMANDS COMMAND echo Generating sleefquad.h: qmkrename ${QUAD_HEADER_PARAMS_${SIMD}})
    list(APPEND SLEEF_HEADER_COMMANDS COMMAND $<TARGET_FILE:qmkrename> ${QUAD_HEADER_PARAMS_${SIMD}} >> ${SLEEFQUAD_INCLUDE_HEADER})
  endif()
endforeach()

if(MSVC OR MINGW AND WIN32)
  string(REPLACE "/" "\\" sleef_footer_input_file "${SLEEFQUAD_ORG_FOOTER}")
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND type ${sleef_footer_input_file} >> ${SLEEFQUAD_INCLUDE_HEADER})
else()
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND cat ${SLEEFQUAD_ORG_FOOTER} >> ${SLEEFQUAD_INCLUDE_HEADER})
endif()

add_custom_command(OUTPUT ${SLEEFQUAD_INCLUDE_HEADER}
  ${SLEEF_HEADER_COMMANDS}
  DEPENDS
    ${SLEEFQUAD_ORG_HEADER}
    ${SLEEFQUAD_ORG_FOOTER}
    qmkrename
)

# --------------------------------------------------------------------
# qmkrename
# qrenameXXX.h for each vector extension
# --------------------------------------------------------------------
# Helper executable: generates parts of the sleef header file
add_host_executable(qmkrename qmkrename.c)
set_target_properties(qmkrename PROPERTIES ${COMMON_TARGET_PROPERTIES})

set(HEADER_FILES_GENERATED "")
foreach(SIMD ${SLEEFQUAD_SUPPORTED_EXT})
  if(COMPILER_SUPPORTS_${SIMD})
    string(TOLOWER ${SIMD} SIMDLC)
    set(HEADER_${SIMD} ${CMAKE_CURRENT_BINARY_DIR}/include/qrename${SIMDLC}.h)
    list(APPEND HEADER_FILES_GENERATED ${HEADER_${SIMD}})

    # Generate qmkrename commands
    add_custom_command(OUTPUT ${HEADER_${SIMD}}
      COMMAND echo Generating qrename${vecarch}.h: qmkrename ${QUAD_RENAME_PARAMS_${SIMD}}
      COMMAND $<TARGET_FILE:qmkrename> ${QUAD_RENAME_PARAMS_${SIMD}} > ${HEADER_${SIMD}}
      DEPENDS qmkrename
    )
    add_custom_target(qrename${SIMD}.h_generated DEPENDS ${HEADER_${SIMD}})
  endif()
endforeach()

# --------------------------------------------------------------------
# sleefquad_headers
# --------------------------------------------------------------------
add_custom_target(sleefquad_headers ALL
  DEPENDS
    ${SLEEFQUAD_INCLUDE_HEADER}
    ${HEADER_FILES_GENERATED}
)

# --------------------------------------------------------------------
# libsleefquad
# --------------------------------------------------------------------

foreach(SIMD ${SLEEFQUAD_SUPPORTED_EXT})
  if(COMPILER_SUPPORTS_${SIMD})
    string(TOLOWER ${SIMD} SIMDLC)
    set(OBJECT "sleefquad${SIMDLC}_obj")
    add_library(${OBJECT} OBJECT sleefsimdqp.c ${HEADER_${SIMD}})    

    if(COMPILER_SUPPORTS_BUILTIN_MATH)
      target_compile_definitions(${OBJECT} PRIVATE ENABLE_BUILTIN_MATH=1)
    endif()
    target_compile_definitions(${OBJECT} PRIVATE ENABLE_${SIMD}=1 DORENAME=1 ${COMMON_TARGET_DEFINITIONS})

    set_target_properties(${OBJECT} PROPERTIES ${COMMON_TARGET_PROPERTIES})
    add_dependencies(${OBJECT} qrename${SIMD}.h_generated)
    target_compile_options(${OBJECT} PRIVATE ${FLAGS_ENABLE_${SIMD}})

    list(APPEND SLEEFQUAD_OBJECTS $<TARGET_OBJECTS:${OBJECT}>)
  endif()
endforeach()

add_library(sleefquad rempitabqp.c ${SLEEFQUAD_OBJECTS})

set_target_properties(sleefquad PROPERTIES
  VERSION ${SLEEF_VERSION}
  SOVERSION ${SLEEF_SOVERSION}
  ${COMMON_TARGET_PROPERTIES}
)

set_target_properties(sleefquad PROPERTIES ${COMMON_TARGET_PROPERTIES})

if(LIBM AND NOT COMPILER_SUPPORTS_BUILTIN_MATH)
  target_link_libraries(sleefquad ${LIBM})
endif()

# --------------------------------------------------------------------
# Install
# --------------------------------------------------------------------
# Install libsleef and sleef.h
install(FILES ${SLEEFQUAD_INCLUDE_HEADER} DESTINATION include)
install(TARGETS sleefquad DESTINATION lib)
