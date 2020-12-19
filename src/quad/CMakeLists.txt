
# Settings

set(QUAD_HEADER_PARAMS_PUREC_SCALAR    1 Sleef_quad   Sleef_quad2    double      float       uint64_t   int32_t   int64_t   uint64_t   __STDC__          purec)
set(QUAD_HEADER_PARAMS_PURECFMA_SCALAR 1 Sleef_quad   Sleef_quad2    double      float       uint64_t   int32_t   int64_t   uint64_t   FP_FAST_FMA       purecfma)
set(QUAD_HEADER_PARAMS_SSE2            2 Sleef_quadx2 Sleef_quadx2_2 __m128d     __m128      __m128i    __m128i   __m128i   __m128i    __SSE2__          sse2)
set(QUAD_HEADER_PARAMS_AVX             4 Sleef_quadx4 Sleef_quadx4_2 __m256d     __m256      __m256i    __m128i   __m256i   __m256i    __AVX__           avx)
set(QUAD_HEADER_PARAMS_AVX2            4 Sleef_quadx4 Sleef_quadx4_2 __m256d     __m256      __m256i    __m128i   __m256i   __m256i    __AVX__           avx2)
set(QUAD_HEADER_PARAMS_AVX512F         8 Sleef_quadx8 Sleef_quadx8_2 __m512d     __m512      __m512i    __m256i   __m512i   __m512i    __AVX512F__       avx512f)
set(QUAD_HEADER_PARAMS_ADVSIMD         2 Sleef_quadx2 Sleef_quadx2_2 float64x2_t float32x4_t uint32x4_t int32x2_t int64x2_t uint64x2_t __ARM_NEON        advsimd)
set(QUAD_HEADER_PARAMS_SVE             x Sleef_svquad Sleef_svquad_2 svfloat64_t svfloat32_t svint32_t  svint32_t svint64_t svuint64_t __ARM_FEATURE_SVE sve)
set(QUAD_HEADER_PARAMS_VSX             2 Sleef_quadx2 Sleef_quadx2_2 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_UINT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_LONGLONG" "SLEEF_VECTOR_ULONGLONG" __VSX__ vsx)
set(QUAD_HEADER_PARAMS_VSX3            2 Sleef_quadx2 Sleef_quadx2_2 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_UINT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_LONGLONG" "SLEEF_VECTOR_ULONGLONG" __VSX__ vsx3)
set(QUAD_HEADER_PARAMS_VXE             2 Sleef_quadx2 Sleef_quadx2_2 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_ULONGLONG" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_LONGLONG" "SLEEF_VECTOR_ULONGLONG" __VEC__ vxe)
set(QUAD_HEADER_PARAMS_VXE2            2 Sleef_quadx2 Sleef_quadx2_2 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_ULONGLONG" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_LONGLONG" "SLEEF_VECTOR_ULONGLONG" __VEC__ vxe2)

set(QUAD_RENAME_PARAMS_PUREC_SCALAR    1 purec)
set(QUAD_RENAME_PARAMS_PURECFMA_SCALAR 1 purecfma)
set(QUAD_RENAME_PARAMS_SSE2            2 sse2)
set(QUAD_RENAME_PARAMS_AVX             4 avx)
set(QUAD_RENAME_PARAMS_AVX2            4 avx2)
set(QUAD_RENAME_PARAMS_AVX512F         8 avx512f)
set(QUAD_RENAME_PARAMS_ADVSIMD         2 advsimd)
set(QUAD_RENAME_PARAMS_SVE             x sve)
set(QUAD_RENAME_PARAMS_VSX             2 vsx)
set(QUAD_RENAME_PARAMS_VSX3            2 vsx3)
set(QUAD_RENAME_PARAMS_VXE             2 vxe)
set(QUAD_RENAME_PARAMS_VXE2            2 vxe2)
set(QUAD_RENAME_PARAMS_CUDA            1 cuda)

#

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${sleef_SOURCE_DIR}/src/common)

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

set(CMAKE_C_FLAGS "${ORG_CMAKE_C_FLAGS} ${SLEEF_C_FLAGS}")

# --------------------------------------------------------------------
# sleefquad.h
# --------------------------------------------------------------------
set(SLEEFQUAD_ORG_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/sleefquad_header.h.org)
set(SLEEFQUAD_ORG_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/sleefquad_footer.h.org)
set(SLEEFQUAD_INCLUDE_HEADER ${sleef_BINARY_DIR}/include/sleefquad.h)

set(SLEEF_HEADER_COMMANDS "")
list(APPEND SLEEF_HEADER_COMMANDS COMMAND ${CMAKE_COMMAND} -E copy ${SLEEFQUAD_ORG_HEADER} ${SLEEFQUAD_INCLUDE_HEADER})
foreach(SIMD ${SLEEF_SUPPORTED_QUAD_EXTENSIONS})
  if(COMPILER_SUPPORTS_${SIMD})
    list(APPEND SLEEF_HEADER_COMMANDS COMMAND echo Generating sleefquad.h: qmkrename ${QUAD_HEADER_PARAMS_${SIMD}})
    list(APPEND SLEEF_HEADER_COMMANDS COMMAND $<TARGET_FILE:qmkrename> ${QUAD_HEADER_PARAMS_${SIMD}} >> ${SLEEFQUAD_INCLUDE_HEADER})
  endif()
endforeach()

if((MSVC OR MINGW AND WIN32) OR SLEEF_CLANG_ON_WINDOWS)
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
foreach(SIMD ${SLEEF_SUPPORTED_QUAD_EXTENSIONS})
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

# Generate qrenamecuda.h

set(HEADER_CUDA ${CMAKE_CURRENT_BINARY_DIR}/include/qrenamecuda.h)
list(APPEND HEADER_FILES_GENERATED ${HEADER_CUDA})
add_custom_command(OUTPUT ${HEADER_CUDA}
  COMMAND echo Generating qrenamecuda.h: qmkrename ${QUAD_RENAME_PARAMS_CUDA}
  COMMAND $<TARGET_FILE:qmkrename> ${QUAD_RENAME_PARAMS_CUDA} > ${HEADER_CUDA}
  DEPENDS qmkrename
  )
add_custom_target(qrenameCUDA.h_generated DEPENDS ${HEADER_CUDA})

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

foreach(SIMD ${SLEEF_SUPPORTED_QUAD_EXTENSIONS})
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
  PUBLIC_HEADER ${SLEEFQUAD_INCLUDE_HEADER}
  ${COMMON_TARGET_PROPERTIES}
)

set_target_properties(sleefquad PROPERTIES ${COMMON_TARGET_PROPERTIES})

if(LIBM AND ((NOT COMPILER_SUPPORTS_BUILTIN_MATH) OR (SLEEF_ARCH_32BIT AND SLEEF_ARCH_X86)))
  target_link_libraries(sleefquad ${LIBM})
endif()

# --------------------------------------------------------------------
# Inline headers
# --------------------------------------------------------------------

if(BUILD_INLINE_HEADERS)
  if(CMAKE_C_COMPILER_ID MATCHES "Intel")
    message(FATAL_ERROR "BUILD_INLINE_HEADERS is not supported with Intel Compiler")
  endif()

  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/)
  set(INLINE_HEADER_FILES_GENERATED "")

  if (SED_COMMAND)
    foreach(SIMD ${SLEEF_SUPPORTED_QUAD_EXTENSIONS})
      if(COMPILER_SUPPORTS_${SIMD})
	string(TOLOWER ${SIMD} SIMDLC)

	set(INLINE_HEADER_FILE ${PROJECT_BINARY_DIR}/include/sleefquadinline_${SIMDLC}.h)
	add_custom_command(
	  OUTPUT ${INLINE_HEADER_FILE} 

	  COMMAND echo Generating sleefquadinline_${SIMDLC}.h
  
	  # Preprocess sleefsimddp.c with SLEEF_GENHEADER defined, comments are preserved
	  COMMAND "${CMAKE_C_COMPILER}" ${FLAG_PREPROCESS} ${FLAG_PRESERVE_COMMENTS}                        # gcc -E -C 
	    ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/common ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/arch   # -I/sleef/src/common -I/sleef/src/arch
	    ${FLAG_INCLUDE}${CMAKE_CURRENT_BINARY_DIR}/include/                                             # -I/build/src/quad/include
	    ${FLAG_DEFINE}SLEEF_GENHEADER ${FLAG_DEFINE}ENABLE_${SIMD} ${FLAG_DEFINE}DORENAME               # -DSLEEF_GENHEADER -DENABLE_SSE2 -DDORENAME
	    ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimdqp.c > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp1    # /sleef/src/quad/sleefsimdqp.c > /build/quad/sleefSSE2.h.qtmp1

	  # Remove all lines except those begin with "//@"
	  COMMAND ${SED_COMMAND} -n -e "/^\\/\\/@#.*$/p" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp1   # sed -n -e "/^\\/\\/@#.*$/p" /build/src/quad/sleefSSE2.h.qtmp1
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp2                                              # > /build/src/quad/sleefSSE2.h.qtmp2

	  # Remove "//@"
	  COMMAND ${SED_COMMAND} -e "s/^\\/\\/@#/#/g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp2      # sed -e "s/^\\/\\/@#/#/g" /build/src/common/sleefSSE2.h.qtmp2
	    > ${CMAKE_CURRENT_BINARY_DIR}/include/qmacroonly${SIMD}.h                                       # > /build/src/common/include/qmacroonlySSE2.h

	  # Remove lines beginning with "#" so that the resulting file can be preprocessed again.
	  COMMAND ${SED_COMMAND} -e "s/^#.*//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp1            # sed -e "s/^#.*//g" /build/src/quad/sleefSSE2.h.qtmp1
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleefquad${SIMD}.h.c                                              # > /build/src/quad/sleefquadSSE2.h.c

	  # Preprocess the intemediate file again to remove comments
	  COMMAND "${CMAKE_C_COMPILER}" ${FLAG_PREPROCESS}                                                  # gcc -E
	    ${CMAKE_CURRENT_BINARY_DIR}/sleefquad${SIMD}.h.c                                                # /build/src/quad/sleefquadSSE2.h.c
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp3                                              # > /build/src/quad/sleefSSE2.h.qtmp3

	  # Embed version number into the header
	  COMMAND ${SED_COMMAND} -e                                                                         # sed -e
	    "s/%VERSION%/${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}.${SLEEF_VERSION_PATCHLEVEL}/g"       # "s/%VERSION%/3.5.0/g"
	    ${CMAKE_CURRENT_SOURCE_DIR}/sleefquadinline_header.h.org                                        # /sleef/src/quad/sleefquadinline_header.h.org
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp4                                              # > /build/src/quad/sleefSSE2.h.qtmp4

	  # Remove lines beginning with "#"
	  COMMAND ${SED_COMMAND} -e "s/^#.*//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp3            # sed -e "s/^#.*//g" /build/src/quad/sleefSSE2.h.qtmp3
	    >> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp4                                             # >> /build/src/quad/sleefSSE2.h.qtmp4

	  # Replace multiple empty lines with a single empty line
	  COMMAND ${SED_COMMAND} -e "s/^[[:space:]]*$//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp4  # sed -e "s/^[[:space:]]*$//g" /build/src/quad/sleefSSE2.h.qtmp4
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp5                                              # > /build/src/quad/sleefSSE2.h.qtmp5
	  COMMAND ${SED_COMMAND} "/^$/N;/^\\n$/D" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp5          # sed -e "/^$/N;/^\n$/D" /build/src/quad/sleefSSE2.h.qtmp5
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp6                                              # > /build/src/quad/sleefSSE2.h.qtmp6

	  COMMAND $<TARGET_FILE:addSuffix> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp6                 # addSuffix /build/src/quad/sleefSSE2.h.qtmp6
	    ${sleef_SOURCE_DIR}/src/common/keywords.txt "_${SIMDLC}_sleefq"                                 # keywords.txt "_sse2_sleefq"
            > ${PROJECT_BINARY_DIR}/include/sleefquadinline_${SIMDLC}.h                                     # > /build/include/sleefquadinline_sse2.h

	  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimdqp.c ${HEADER_${SIMD}}
	  DEPENDS ${HEADER_${SIMD}} addSuffix
	  VERBATIM
	  )

	list(APPEND INLINE_HEADER_FILES_GENERATED ${INLINE_HEADER_FILE})
      endif(COMPILER_SUPPORTS_${SIMD})
    endforeach()

    # Generate CUDA header file

    set(SIMD CUDA)
    string(TOLOWER ${SIMD} SIMDLC)

    set(INLINE_HEADER_FILE ${PROJECT_BINARY_DIR}/include/sleefquadinline_cuda.h)

    add_custom_command(
      OUTPUT ${INLINE_HEADER_FILE} 

      COMMAND echo Generating sleefquadinline_${SIMDLC}.h
      
      # Preprocess sleefsimddp.c with SLEEF_GENHEADER defined, comments are preserved
      COMMAND "${CMAKE_C_COMPILER}" ${FLAG_PREPROCESS} ${FLAG_PRESERVE_COMMENTS}                        # gcc -E -C 
      ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/common ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/arch     # -I/sleef/src/common -I/sleef/src/arch
      ${FLAG_INCLUDE}${CMAKE_CURRENT_BINARY_DIR}/include/                                               # -I/build/src/quad/include
      ${FLAG_DEFINE}SLEEF_GENHEADER ${FLAG_DEFINE}ENABLE_${SIMD} ${FLAG_DEFINE}DORENAME                 # -DSLEEF_GENHEADER -DENABLE_CUDA -DDORENAME
      ${FLAG_DEFINE}SLEEF_ALWAYS_INLINE=__device__ ${FLAG_DEFINE}SLEEF_INLINE=__device__                # -DSLEEF_ALWAYS_INLINE=__device__ -DSLEEF_INLINE=__device__
      ${FLAG_DEFINE}SLEEF_CONST=__device__ ${FLAG_DEFINE}static=__device__                              # -DSLEEF_CONST=__device__ -Dstatic=__device__
      ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimdqp.c > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp1      # /sleef/src/quad/sleefsimdqp.c > /build/quad/sleefCUDA.h.qtmp1

      # Remove all lines except those begin with "//@"
      COMMAND ${SED_COMMAND} -n -e "/^\\/\\/@#.*$/p" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp1   # sed -n -e "/^\\/\\/@#.*$/p" /build/src/common/sleefCUDA.h.qtmp1
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp2                                                # > /build/src/common/sleefCUDA.h.qtmp2

      # Remove "//@"
      COMMAND ${SED_COMMAND} -e "s/^\\/\\/@#/#/g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp2      # sed -e "s/^\\/\\/@#/#/g" /build/src/common/sleefCUDA.h.qtmp2
      > ${CMAKE_CURRENT_BINARY_DIR}/include/qmacroonly${SIMD}.h                                         # > /build/src/common/include/qmacroonlyCUDA.h

      # Remove lines beginning with "#" so that the resulting file can be preprocessed again.
      COMMAND ${SED_COMMAND} -e "s/^#.*//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp1            # sed -e "s/^#.*//g" /build/src/quad/sleefCUDA.h.qtmp1
      > ${CMAKE_CURRENT_BINARY_DIR}/sleefquad${SIMD}.h.c                                                # > /build/src/quad/sleefquadCUDA.h.c

      # Preprocess the intemediate file again to remove comments
      COMMAND "${CMAKE_C_COMPILER}" ${FLAG_PREPROCESS}                                                  # gcc -E
      ${CMAKE_CURRENT_BINARY_DIR}/sleefquad${SIMD}.h.c                                                  # /build/src/quad/sleefquadCUDA.h.c
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp3                                                # > /build/src/quad/sleefCUDA.h.qtmp3

      # Embed version number into the header
      COMMAND ${SED_COMMAND} -e                                                                         # sed -e
      "s/%VERSION%/${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}.${SLEEF_VERSION_PATCHLEVEL}/g"         # "s/%VERSION%/3.5.0/g"
      ${CMAKE_CURRENT_SOURCE_DIR}/sleefquadinline_cuda_header.h.org                                     # /sleef/src/quad/sleefquadinline_cuda_header.h.org
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp4                                                # > /build/src/quad/sleefCUDA.h.qtmp4

      # Remove lines beginning with "#"
      COMMAND ${SED_COMMAND} -e "s/^#.*//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp3            # sed -e "s/^#.*//g" /build/src/quad/sleefCUDA.h.qtmp3
      >> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp4                                               # >> /build/src/quad/sleefCUDA.h.qtmp4

      # Replace multiple empty lines with a single empty line
      COMMAND ${SED_COMMAND} -e "s/^[[:space:]]*$//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp4  # sed -e "s/^[[:space:]]*$//g" /build/src/quad/sleefCUDA.h.qtmp4
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp5                                                # > /build/src/quad/sleefCUDA.h.qtmp5
      COMMAND ${SED_COMMAND} "/^$/N;/^\\n$/D" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp5          # sed -e "/^$/N;/^\n$/D" /build/src/quad/sleefCUDA.h.qtmp5
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp6                                                # > /build/src/quad/sleefCUDA.h.qtmp6

      # Remove redundant __device__
      COMMAND ${SED_COMMAND} -e "s/__device__ __device__/__device__/g"                                  # sed -e "s/__device__ __device__/__device__/g"
      -e "s/__device__ __device__/__device__/g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp6        # -e "s/__device__ __device__/__device__/g" /build/src/common/sleefCUDA.h.qtmp6
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp7                                                # > /build/src/common/sleefCUDA.h.qtmp7

      COMMAND $<TARGET_FILE:addSuffix> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.qtmp7                 # addSuffix /build/src/quad/sleefCUDA.h.qtmp7
      ${sleef_SOURCE_DIR}/src/common/keywords_cuda.txt "_${SIMDLC}_sleefq"                              # keywords_cuda.txt "_cuda_sleefq"
      > ${PROJECT_BINARY_DIR}/include/sleefquadinline_${SIMDLC}.h                                       # > /build/include/sleefquadinline_cuda.h

      MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimdqp.c ${HEADER_${SIMD}}
      DEPENDS ${HEADER_${SIMD}} addSuffix
      VERBATIM
      )

    list(APPEND INLINE_HEADER_FILES_GENERATED ${INLINE_HEADER_FILE})

    #

    add_custom_target(${TARGET_QINLINE_HEADERS} ALL
      DEPENDS
      ${INLINE_HEADER_FILES_GENERATED}
      )
    install(FILES ${INLINE_HEADER_FILES_GENERATED} DESTINATION include)
  endif(SED_COMMAND)
endif(BUILD_INLINE_HEADERS)

# --------------------------------------------------------------------
# Install
# --------------------------------------------------------------------
# Install libsleef and sleef.h
install(TARGETS sleefquad
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION bin
)
