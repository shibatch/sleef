
# Settings

if (SLEEF_ARCH_X86)
  set(SLEEF_HEADER_LIST
    SSE_
    SSE2
    SSE4
    AVX_
    AVX
    FMA4
    AVX2
    AVX2128
    AVX512F_
    AVX512F
    AVX512FNOFMA
    PUREC_SCALAR
    PURECFMA_SCALAR
  )
elseif(SLEEF_ARCH_AARCH64)
  set(SLEEF_HEADER_LIST
    ADVSIMD_
    ADVSIMD
    ADVSIMDNOFMA
    SVE
    SVENOFMA
    PUREC_SCALAR
    PURECFMA_SCALAR
  )
elseif(SLEEF_ARCH_AARCH32)
  set(SLEEF_HEADER_LIST
    NEON32_
    NEON32
    NEON32VFPV4
    PUREC_SCALAR
    PURECFMA_SCALAR
  )
elseif(SLEEF_ARCH_PPC64)
  set(SLEEF_HEADER_LIST
    VSX_
    VSX
    VSXNOFMA
    VSX3
    VSX3NOFMA
    PUREC_SCALAR
    PURECFMA_SCALAR
  )
elseif(SLEEF_ARCH_S390X)
  set(SLEEF_HEADER_LIST
    VXE_
    VXE
    VXENOFMA
    VXE2
    VXE2NOFMA
    PUREC_SCALAR
    PURECFMA_SCALAR
  )
endif()

# HEADER_PARAMS

command_arguments(HEADER_PARAMS_SSE_            cinz_ 2 4 __m128d __m128 __m128i __m128i __SSE2__)
command_arguments(HEADER_PARAMS_SSE2            cinz_ 2 4 __m128d __m128 __m128i __m128i __SSE2__ sse2)
command_arguments(HEADER_PARAMS_SSE4            cinz_ 2 4 __m128d __m128 __m128i __m128i __SSE2__ sse4)
command_arguments(HEADER_PARAMS_AVX_            cinz_ 4 8 __m256d __m256 __m128i "struct { __m128i x, y$<SEMICOLON> }" __AVX__)
command_arguments(HEADER_PARAMS_AVX             cinz_ 4 8 __m256d __m256 __m128i "struct { __m128i x, y$<SEMICOLON> }" __AVX__ avx)
command_arguments(HEADER_PARAMS_FMA4            finz_ 4 8 __m256d __m256 __m128i "struct { __m128i x, y$<SEMICOLON> }" __AVX__ fma4)
command_arguments(HEADER_PARAMS_AVX2            finz_ 4 8 __m256d __m256 __m128i __m256i __AVX__ avx2)
command_arguments(HEADER_PARAMS_AVX2128         finz_ 2 4 __m128d __m128 __m128i __m128i __SSE2__ avx2128)
command_arguments(HEADER_PARAMS_AVX512F_        finz_ 8 16 __m512d __m512 __m256i __m512i __AVX512F__)
command_arguments(HEADER_PARAMS_AVX512F         finz_ 8 16 __m512d __m512 __m256i __m512i __AVX512F__ avx512f)
command_arguments(HEADER_PARAMS_AVX512FNOFMA    cinz_ 8 16 __m512d __m512 __m256i __m512i __AVX512F__ avx512fnofma)

command_arguments(HEADER_PARAMS_ADVSIMD_        finz_ 2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON)
command_arguments(HEADER_PARAMS_ADVSIMD         finz_ 2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON advsimd)
command_arguments(HEADER_PARAMS_ADVSIMDNOFMA    cinz_ 2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON advsimdnofma)
command_arguments(HEADER_PARAMS_SVE             finz_ x x svfloat64_t svfloat32_t svint32_t svint32_t __ARM_FEATURE_SVE sve)
command_arguments(HEADER_PARAMS_SVENOFMA        cinz_ x x svfloat64_t svfloat32_t svint32_t svint32_t __ARM_FEATURE_SVE svenofma)

command_arguments(HEADER_PARAMS_NEON32_         cinz_ 2 4 - float32x4_t int32x2_t int32x4_t __ARM_NEON__)
command_arguments(HEADER_PARAMS_NEON32          cinz_ 2 4 - float32x4_t int32x2_t int32x4_t __ARM_NEON__ neon)
command_arguments(HEADER_PARAMS_NEON32VFPV4     finz_ 2 4 - float32x4_t int32x2_t int32x4_t __ARM_NEON__ neonvfpv4)

command_arguments(HEADER_PARAMS_VSX_            finz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VSX__)
command_arguments(HEADER_PARAMS_VSX             finz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VSX__ vsx)
command_arguments(HEADER_PARAMS_VSXNOFMA        cinz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VSX__ vsxnofma)
command_arguments(HEADER_PARAMS_VSX3            finz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VSX__ vsx3)
command_arguments(HEADER_PARAMS_VSX3NOFMA       cinz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VSX__ vsx3nofma)

command_arguments(HEADER_PARAMS_VXE_            finz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VEC__)
command_arguments(HEADER_PARAMS_VXE             finz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VEC__ vxe)
command_arguments(HEADER_PARAMS_VXENOFMA        cinz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VEC__ vxenofma)
command_arguments(HEADER_PARAMS_VXE2            finz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VEC__ vxe2)
command_arguments(HEADER_PARAMS_VXE2NOFMA       cinz_ 2 4 "SLEEF_VECTOR_DOUBLE" "SLEEF_VECTOR_FLOAT" "SLEEF_VECTOR_INT" "SLEEF_VECTOR_INT" __VEC__ vxe2nofma)

command_arguments(HEADER_PARAMS_PUREC_SCALAR    cinz_ 1 1 double float int32_t int32_t __STDC__ purec)
command_arguments(HEADER_PARAMS_PURECFMA_SCALAR finz_ 1 1 double float int32_t int32_t FP_FAST_FMA purecfma)

# RENAME_PARAMS

command_arguments(RENAME_PARAMS_SSE2            cinz_ 2 4 sse2)
command_arguments(RENAME_PARAMS_SSE4            cinz_ 2 4 sse4)
command_arguments(RENAME_PARAMS_AVX             cinz_ 4 8 avx)
command_arguments(RENAME_PARAMS_FMA4            finz_ 4 8 fma4)
command_arguments(RENAME_PARAMS_AVX2            finz_ 4 8 avx2)
command_arguments(RENAME_PARAMS_AVX2128         finz_ 2 4 avx2128)
command_arguments(RENAME_PARAMS_AVX512F         finz_ 8 16 avx512f)
command_arguments(RENAME_PARAMS_AVX512FNOFMA    cinz_ 8 16 avx512fnofma)
command_arguments(RENAME_PARAMS_ADVSIMD         finz_ 2 4 advsimd)
command_arguments(RENAME_PARAMS_ADVSIMDNOFMA    cinz_ 2 4 advsimdnofma)
command_arguments(RENAME_PARAMS_NEON32          cinz_ 2 4 neon)
command_arguments(RENAME_PARAMS_NEON32VFPV4     finz_ 2 4 neonvfpv4)
command_arguments(RENAME_PARAMS_VSX             finz_ 2 4 vsx)
command_arguments(RENAME_PARAMS_VSXNOFMA        cinz_ 2 4 vsxnofma)
command_arguments(RENAME_PARAMS_VSX3            finz_ 2 4 vsx3)
command_arguments(RENAME_PARAMS_VSX3NOFMA       cinz_ 2 4 vsx3nofma)
command_arguments(RENAME_PARAMS_VXE             finz_ 2 4 vxe)
command_arguments(RENAME_PARAMS_VXENOFMA        cinz_ 2 4 vxenofma)
command_arguments(RENAME_PARAMS_VXE2            finz_ 2 4 vxe2)
command_arguments(RENAME_PARAMS_VXE2NOFMA       cinz_ 2 4 vxe2nofma)
command_arguments(RENAME_PARAMS_PUREC_SCALAR    cinz_ 1 1 purec)
command_arguments(RENAME_PARAMS_PURECFMA_SCALAR finz_ 1 1 purecfma)
command_arguments(RENAME_PARAMS_CUDA            finz_ 1 1 cuda)

# The vector length parameters in SVE, for SP and DP, are chosen for
# the smallest SVE vector size (128-bit). The name is generated using
# the "x" token of VLA SVE vector functions.
command_arguments(RENAME_PARAMS_SVE             finz_ x x sve)
command_arguments(RENAME_PARAMS_SVENOFMA        cinz_ x x svenofma)

command_arguments(RENAME_PARAMS_GNUABI_SSE2     sse2 b 2 4 _mm128d _mm128 _mm128i _mm128i __SSE2__)
command_arguments(RENAME_PARAMS_GNUABI_AVX      avx c 4 8 __m256d __m256 __m128i "struct { __m128i x, y$<SEMICOLON> }" __AVX__)
command_arguments(RENAME_PARAMS_GNUABI_AVX2     avx2 d 4 8 __m256d __m256 __m128i __m256i __AVX2__)
command_arguments(RENAME_PARAMS_GNUABI_AVX512F  avx512f e 8 16 __m512d __m512 __m256i __m512i __AVX512F__)
command_arguments(RENAME_PARAMS_GNUABI_ADVSIMD  advsimd n 2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON)
# The vector length parameters in SVE, for SP and DP, are chosen for
# the smallest SVE vector size (128-bit). The name is generated using
# the "x" token of VLA SVE vector functions.
command_arguments(RENAME_PARAMS_GNUABI_SVE sve s x x svfloat64_t svfloat32_t svint32_t svint32_t __ARM_SVE)

# ALIAS_PARAMS

command_arguments(ALIAS_PARAMS_AVX512F_DP   8 __m512d __m256i e avx512f)
command_arguments(ALIAS_PARAMS_AVX512F_SP -16 __m512  __m512i e avx512f)

command_arguments(ALIAS_PARAMS_ADVSIMD_DP  2 float64x2_t int32x2_t n advsimd)
command_arguments(ALIAS_PARAMS_ADVSIMD_SP -4 float32x4_t int32x4_t n advsimd)

command_arguments(ALIAS_PARAMS_NEON32_SP  -4 float32x4_t int32x4_t - neon)
command_arguments(ALIAS_PARAMS_NEON32_DP   0)

command_arguments(ALIAS_PARAMS_PUREC_SCALAR_DP     1 double int32_t purec cinz_)
command_arguments(ALIAS_PARAMS_PUREC_SCALAR_SP    -1 float  int32_t purec cinz_)
command_arguments(ALIAS_PARAMS_PURECFMA_SCALAR_DP  1 double int32_t purecfma finz_)
command_arguments(ALIAS_PARAMS_PURECFMA_SCALAR_SP -1 float  int32_t purecfma finz_)

#

set(EXT_ENABLE_ALIAS AVX512F ADVSIMD NEON32)

#

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_C_FLAGS "${ORG_CMAKE_C_FLAGS} ${SLEEF_C_FLAGS}")

# --------------------------------------------------------------------
# sleef.h
# --------------------------------------------------------------------
# File generated for the headers
set(SLEEF_ORG_HEADER ${CMAKE_CURRENT_BINARY_DIR}/sleeflibm_header.h.org)
set(SLEEF_ORG_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm_footer.h.org)
set(SLEEF_INCLUDE_HEADER ${sleef_BINARY_DIR}/include/sleef.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm_header.h.org.in ${SLEEF_ORG_HEADER})

set(SLEEF_HEADER_COMMANDS "")
list(APPEND SLEEF_HEADER_COMMANDS COMMAND ${CMAKE_COMMAND} -E copy ${SLEEF_ORG_HEADER} ${SLEEF_INCLUDE_HEADER})
foreach(SIMD ${SLEEF_HEADER_LIST})
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND echo Generating sleef.h: ${TARGET_MKRENAME} ${HEADER_PARAMS_${SIMD}})
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND $<TARGET_FILE:${TARGET_MKRENAME}> ${HEADER_PARAMS_${SIMD}} >> ${SLEEF_INCLUDE_HEADER})
endforeach()

if((MSVC OR MINGW AND WIN32) OR SLEEF_CLANG_ON_WINDOWS)
  string(REPLACE "/" "\\" sleef_footer_input_file "${SLEEF_ORG_FOOTER}")
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND type ${sleef_footer_input_file} >> ${SLEEF_INCLUDE_HEADER})
else()
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND cat ${SLEEF_ORG_FOOTER} >> ${SLEEF_INCLUDE_HEADER})
endif()

add_custom_command(OUTPUT ${SLEEF_INCLUDE_HEADER}
  ${SLEEF_HEADER_COMMANDS}
  DEPENDS
    ${SLEEF_ORG_HEADER}
    ${SLEEF_ORG_FOOTER}
    ${TARGET_MKRENAME}
)

# --------------------------------------------------------------------
# TARGET_MKRENAME
# renameXXX.h for each SIMD
# --------------------------------------------------------------------
# Helper executable: generates parts of the sleef header file
add_host_executable(${TARGET_MKRENAME} mkrename.c)

# Enable Vector PCS for Advanced SIMD (if supported)
if(FORCE_AAVPCS)
  host_target_AAVPCS_definitions(${TARGET_MKRENAME})
endif()

set(HEADER_FILES_GENERATED "")
foreach(SIMD ${SLEEF_SUPPORTED_LIBM_EXTENSIONS})
  if(COMPILER_SUPPORTS_${SIMD})
    # Need lowercase string for rename header
    string(TOLOWER ${SIMD} vecarch)
    set(OBJECT_${SIMD} "sleef${vecarch}")
    set(OBJECTDET_${SIMD} "sleefdet${vecarch}")
    set(HEADER_${SIMD} ${CMAKE_CURRENT_BINARY_DIR}/include/rename${vecarch}.h)
    list(APPEND HEADER_FILES_GENERATED ${HEADER_${SIMD}})

    # Generate mkrename commands
    add_custom_command(OUTPUT ${HEADER_${SIMD}}
      COMMAND echo Generating rename${vecarch}.h: ${TARGET_MKRENAME} ${RENAME_PARAMS_${SIMD}}
      COMMAND $<TARGET_FILE:${TARGET_MKRENAME}> ${RENAME_PARAMS_${SIMD}} > ${HEADER_${SIMD}}
      DEPENDS ${TARGET_MKRENAME}
    )
    add_custom_target(rename${SIMD}.h_generated DEPENDS ${HEADER_${SIMD}})
  endif()
endforeach()

# Generate renamecuda.h

set(HEADER_CUDA ${CMAKE_CURRENT_BINARY_DIR}/include/renamecuda.h)
list(APPEND HEADER_FILES_GENERATED ${HEADER_CUDA})
add_custom_command(OUTPUT ${HEADER_CUDA}
  COMMAND echo Generating renamecuda.h: ${TARGET_MKRENAME} ${RENAME_PARAMS_CUDA}
  COMMAND $<TARGET_FILE:${TARGET_MKRENAME}> ${RENAME_PARAMS_CUDA} > ${HEADER_CUDA}
  DEPENDS ${TARGET_MKRENAME}
  )
add_custom_target(renameCUDA.h_generated DEPENDS ${HEADER_CUDA})

# --------------------------------------------------------------------
# TARGET_MKRENAME_GNUABI
# renameXXX_gnuabi.h for each SIMD GNU Abi
# --------------------------------------------------------------------
# Helper executable: generates parts of the sleef header file gnu_abi
add_host_executable(${TARGET_MKRENAME_GNUABI} mkrename_gnuabi.c)

set(HEADER_GNUABI_FILES_GENERATED "")
if(ENABLE_GNUABI)
  foreach(SIMD ${SLEEF_SUPPORTED_GNUABI_EXTENSIONS})
    if(COMPILER_SUPPORTS_${SIMD})
      string(TOLOWER ${SIMD} vecarch)
      set(OBJECT_${SIMD}_dp_GNUABI "sleefgnuabi${vecarch}dp")
      set(OBJECT_${SIMD}_sp_GNUABI "sleefgnuabi${vecarch}sp")
      set(HEADER_${SIMD}_GNUABI ${CMAKE_CURRENT_BINARY_DIR}/include/rename${vecarch}_gnuabi.h)
      list(APPEND HEADER_GNUABI_FILES_GENERATED ${HEADER_${SIMD}_GNUABI})

      # Generate mkrename_gnuabi commands
      add_custom_command(OUTPUT ${HEADER_${SIMD}_GNUABI}
        COMMAND echo Generating rename${vecarch}_gnuabi.h: ${TARGET_MKRENAME_GNUABI} ${RENAME_PARAMS_GNUABI_${SIMD}}
        COMMAND $<TARGET_FILE:${TARGET_MKRENAME_GNUABI}> ${RENAME_PARAMS_GNUABI_${SIMD}} > ${HEADER_${SIMD}_GNUABI}
        DEPENDS ${TARGET_MKRENAME_GNUABI}
      )
      # set_source_files_properties(${HEADER_${SIMD}_GNUABI} PROPERTIES GENERATED TRUE)
    endif()
  endforeach()
endif()

# --------------------------------------------------------------------

# TARGET_MKMASKED_GNUABI
add_host_executable(${TARGET_MKMASKED_GNUABI} mkmasked_gnuabi.c)

# maskedXXX_YY_gnuabi.h
if(ENABLE_GNUABI)
  foreach(SIMD ${SLEEF_SUPPORTED_GNUABI_EXTENSIONS})
    if(COMPILER_SUPPORTS_${SIMD} AND MKMASKED_PARAMS_GNUABI_${SIMD}_sp)
      string(TOLOWER ${SIMD} vecarch)

      set(HEADER_GENERATED "")
      foreach(T dp sp)
	set(HEADER_MASKED_${SIMD}_${T}_GNUABI ${CMAKE_CURRENT_BINARY_DIR}/include/masked_${vecarch}_${T}_gnuabi.h)
	list(APPEND HEADER_GENERATED ${HEADER_MASKED_${SIMD}_${T}_GNUABI})

	add_custom_command(OUTPUT ${HEADER_MASKED_${SIMD}_${T}_GNUABI}
          COMMAND echo Generating ${HEADER_MASKED_${SIMD}_${T}_GNUABI}
          COMMAND $<TARGET_FILE:${TARGET_MKMASKED_GNUABI}> ${MKMASKED_PARAMS_GNUABI_${SIMD}_${T}} > ${HEADER_MASKED_${SIMD}_${T}_GNUABI}
          DEPENDS ${TARGET_MKMASKED_GNUABI}
	  )
      endforeach()
      add_custom_target(masked${SIMD}_generated DEPENDS ${HEADER_GENERATED})
    endif()
  endforeach()
endif()

# --------------------------------------------------------------------
# TARGET_HEADERS
# --------------------------------------------------------------------
add_custom_target(${TARGET_HEADERS} ALL
  DEPENDS
    ${SLEEF_INCLUDE_HEADER}            # Output only
    ${HEADER_FILES_GENERATED}          # Output only
    ${HEADER_GNUABI_FILES_GENERATED}   # Output only
)

# --------------------------------------------------------------------
# TARGET_MKALIAS
# --------------------------------------------------------------------
add_host_executable(${TARGET_MKALIAS} mkalias.c)
if(FORCE_AAVPCS)
  host_target_AAVPCS_definitions(${TARGET_MKALIAS})
endif()

# --------------------------------------------------------------------
# TARGET_MKDISP
# --------------------------------------------------------------------
# Helper executable: dispatcher for the vector extensions
add_host_executable(${TARGET_MKDISP} mkdisp.c)

# Set C standard requirement (-std=gnu99 for gcc)
set_target_properties(
  ${TARGET_MKRENAME} ${TARGET_MKRENAME_GNUABI} ${TARGET_MKDISP}
  ${TARGET_MKALIAS} ${TARGET_MKMASKED_GNUABI}
  PROPERTIES C_STANDARD 99
)

# --------------------------------------------------------------------
# TARGET_LIBSLEEF
# --------------------------------------------------------------------
# Build main library

set(COMMON_TARGET_PROPERTIES 
  C_STANDARD 99                  # -std=gnu99
  )

if (BUILD_SHARED_LIBS)
  list(APPEND COMMON_TARGET_PROPERTIES POSITION_INDEPENDENT_CODE ON)   # -fPIC
endif()

if (ENABLE_LTO)
  list(APPEND COMMON_TARGET_PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)  # -flto
endif()

# Original sleef sources
set(STANDARD_SOURCES sleefdp.c sleefsp.c rempitab.c)

# Check for different precision support and add sources accordingly
if(COMPILER_SUPPORTS_LONG_DOUBLE)
  list(APPEND STANDARD_SOURCES sleefld.c)
endif()

add_library(${TARGET_LIBSLEEF} ${STANDARD_SOURCES})
add_dependencies(${TARGET_LIBSLEEF} ${TARGET_HEADERS})
set_target_properties(${TARGET_LIBSLEEF} PROPERTIES
  VERSION ${SLEEF_VERSION}
  SOVERSION ${SLEEF_SOVERSION}
  PUBLIC_HEADER ${SLEEF_INCLUDE_HEADER}
  ${COMMON_TARGET_PROPERTIES}
)

target_compile_definitions(${TARGET_LIBSLEEF}
  PRIVATE DORENAME=1 ${COMMON_TARGET_DEFINITIONS}
)

if(COMPILER_SUPPORTS_FLOAT128)
  # TODO: Not supported for LLVM bitcode gen as it has a specific compilation flags
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefqp.c)
  target_compile_definitions(${TARGET_LIBSLEEF}
    PRIVATE ENABLEFLOAT128=1 ${COMMON_TARGET_DEFINITIONS})
endif()

if(COMPILER_SUPPORTS_BUILTIN_MATH)
  target_compile_definitions(${TARGET_LIBSLEEF} PRIVATE ENABLE_BUILTIN_MATH=1)
endif()

# Compile SIMD versions
# Single precision and double precision

# Include symbols for each SIMD architecture (if supported by the platform)
# Note: adds object file as sources via cmake conditional generator expression
foreach(SIMD ${SLEEF_SUPPORTED_LIBM_EXTENSIONS})
  if(COMPILER_SUPPORTS_${SIMD})
    list(FIND EXT_ENABLE_ALIAS ${SIMD} INDEX_ALIAS)
    string(TOLOWER ${SIMD} SIMDLC)

    if (${SIMD} STREQUAL "NEON32" OR ${SIMD} STREQUAL "NEON32VFPV4")
      set(SIMD_SOURCES sleefsimdsp.c)
    else()
      set(SIMD_SOURCES sleefsimdsp.c sleefsimddp.c)
    endif()

    # Create a library
    add_library(${OBJECT_${SIMD}} OBJECT ${SIMD_SOURCES} ${HEADER_${SIMD}})    
    add_library(${OBJECTDET_${SIMD}} OBJECT ${SIMD_SOURCES} ${HEADER_${SIMD}})    
    target_compile_definitions(${OBJECTDET_${SIMD}} PRIVATE DETERMINISTIC=1)

    if(COMPILER_SUPPORTS_BUILTIN_MATH)
      target_compile_definitions(${OBJECT_${SIMD}} PRIVATE ENABLE_BUILTIN_MATH=1)
      target_compile_definitions(${OBJECTDET_${SIMD}} PRIVATE ENABLE_BUILTIN_MATH=1)
    endif()

    if(ENABLE_ALTDIV)
      target_compile_definitions(${OBJECT_${SIMD}} PRIVATE ENABLE_ALTDIV=1)
      target_compile_definitions(${OBJECTDET_${SIMD}} PRIVATE ENABLE_ALTDIV=1)
    endif()

    if(ENABLE_ALTSQRT)
      target_compile_definitions(${OBJECT_${SIMD}} PRIVATE ENABLE_ALTSQRT=1)
      target_compile_definitions(${OBJECTDET_${SIMD}} PRIVATE ENABLE_ALTSQRT=1)
    endif()

    if (INDEX_ALIAS EQUAL -1)
      target_compile_definitions(${OBJECT_${SIMD}} PRIVATE
	ENABLE_${SIMD}=1
	DORENAME=1
	${COMMON_TARGET_DEFINITIONS}
	)
      target_compile_definitions(${OBJECTDET_${SIMD}} PRIVATE
	ENABLE_${SIMD}=1
	DORENAME=1
	${COMMON_TARGET_DEFINITIONS}
	)
    else()
      add_custom_command(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/alias_${SIMDLC}.h
       	COMMENT "Generating alias_${SIMDLC}.h"
       	COMMAND $<TARGET_FILE:${TARGET_MKALIAS}> ${ALIAS_PARAMS_${SIMD}_SP} >  ${CMAKE_CURRENT_BINARY_DIR}/include/alias_${SIMDLC}.h
       	COMMAND $<TARGET_FILE:${TARGET_MKALIAS}> ${ALIAS_PARAMS_${SIMD}_DP} >> ${CMAKE_CURRENT_BINARY_DIR}/include/alias_${SIMDLC}.h
       	DEPENDS ${TARGET_MKALIAS}
       	)
      add_custom_target(alias_${SIMDLC}.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/include/alias_${SIMDLC}.h)
      add_dependencies(${OBJECT_${SIMD}} alias_${SIMDLC}.h_generated)
      add_dependencies(${OBJECTDET_${SIMD}} alias_${SIMDLC}.h_generated)
      target_compile_definitions(${OBJECT_${SIMD}} PRIVATE
	ENABLE_${SIMD}=1
	DORENAME=1
	${COMMON_TARGET_DEFINITIONS}
	ALIAS_NO_EXT_SUFFIX=\"alias_${SIMDLC}.h\"
	)
      target_compile_definitions(${OBJECTDET_${SIMD}} PRIVATE
	ENABLE_${SIMD}=1
	DORENAME=1
	${COMMON_TARGET_DEFINITIONS}
	ALIAS_NO_EXT_SUFFIX=\"alias_${SIMDLC}.h\"
	)
    endif()

    # Enable Vector PCS for Advanced SIMD (if supported)
    if(FORCE_AAVPCS AND ${SIMD} STREQUAL "ADVSIMD")
      target_compile_definitions(${OBJECT_${SIMD}} PRIVATE
        ENABLE_AAVPCS=1
      )
    endif()

    add_dependencies(${OBJECT_${SIMD}} rename${SIMD}.h_generated)
    add_dependencies(${OBJECTDET_${SIMD}} rename${SIMD}.h_generated)

    set_target_properties(${OBJECT_${SIMD}} PROPERTIES
      ${COMMON_TARGET_PROPERTIES}
      )

    set_target_properties(${OBJECTDET_${SIMD}} PROPERTIES
      ${COMMON_TARGET_PROPERTIES}
      )

    target_compile_options(${OBJECT_${SIMD}} PRIVATE
      ${FLAGS_ENABLE_${SIMD}})

    target_compile_options(${OBJECTDET_${SIMD}} PRIVATE
      ${FLAGS_ENABLE_${SIMD}})

    target_sources(${TARGET_LIBSLEEF} PRIVATE $<TARGET_OBJECTS:${OBJECT_${SIMD}}> $<TARGET_OBJECTS:${OBJECTDET_${SIMD}}>)
  endif(COMPILER_SUPPORTS_${SIMD})
endforeach()

#

if(BUILD_INLINE_HEADERS)
  if(CMAKE_C_COMPILER_ID MATCHES "Intel")
    message(FATAL_ERROR "BUILD_INLINE_HEADERS is not supported with Intel Compiler")
  endif()

  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/)
  set(INLINE_HEADER_FILES_GENERATED "")

  if (SED_COMMAND)
    foreach(SIMD ${SLEEF_SUPPORTED_LIBM_EXTENSIONS})
      if(COMPILER_SUPPORTS_${SIMD})
	string(TOLOWER ${SIMD} SIMDLC)

	set(INLINE_HEADER_FILE ${PROJECT_BINARY_DIR}/include/sleefinline_${SIMDLC}.h)
	add_custom_command(
	  OUTPUT ${INLINE_HEADER_FILE} 

	  COMMAND echo Generating sleefinline_${SIMDLC}.h
  
	  # Preprocess sleefsimddp.c with SLEEF_GENHEADER defined, comments are preserved
	  COMMAND "${CMAKE_C_COMPILER}"  ${FLAG_PREPROCESS} ${FLAG_PRESERVE_COMMENTS}                      # gcc -E -C 
	    ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/common ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/arch  # -I/sleef/src/common -I/sleef/src/arch
	    ${FLAG_INCLUDE}${CMAKE_CURRENT_BINARY_DIR}/include/                                            # -I/build/src/libm/include
	    ${FLAG_DEFINE}SLEEF_GENHEADER ${FLAG_DEFINE}ENABLE_${SIMD} ${FLAG_DEFINE}DORENAME              # -DSLEEF_GENHEADER -DENABLE_SSE2 -DDORENAME
	    ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimddp.c > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp1    # /sleef/src/libm/sleefsimddp.c > /build/libm/sleefSSE2.h.tmp1

	  # Remove all lines except those begin with "//@"
	  COMMAND ${SED_COMMAND} -n -e "/^\\/\\/@#.*$/p" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp1   # sed -n -e "/^\\/\\/@#.*$/p" /build/src/libm/sleefSSE2.h.tmp1
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp2                                              # > /build/src/libm/sleefSSE2.h.tmp2

	  # Remove "//@"
	  COMMAND ${SED_COMMAND} -e "s/^\\/\\/@#/#/g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp2      # sed -e "s/^\\/\\/@#/#/g" /build/src/libm/sleefSSE2.h.tmp2
	    > ${CMAKE_CURRENT_BINARY_DIR}/include/macroonly${SIMD}.h                                       # > /build/src/libm/include/macroonlySSE2.h

	  # Preprocess sleefsimdsp.c with SLEEF_GENHEADER defined. Include macroonly*.h instead of helper*.h.
	  COMMAND "${CMAKE_C_COMPILER}" ${FLAG_PREPROCESS} ${FLAG_PRESERVE_COMMENTS}                       # gcc -E -C 
	    ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/common ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/arch  # -I/sleef/src/common -I/sleef/src/arch
	    ${FLAG_INCLUDE}${CMAKE_CURRENT_BINARY_DIR}/include/                                            # -I/build/src/libm/include
	    ${FLAG_DEFINE}SLEEF_GENHEADER ${FLAG_DEFINE}ENABLE_${SIMD} ${FLAG_DEFINE}DORENAME              # -DSLEEF_GENHEADER -DENABLE_SSE2 -DDORENAME
	    ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimdsp.c >> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp1   # /sleef/src/libm/sleefsimdsp.c >> /build/libm/sleefSSE2.h.tmp1

	  # Remove lines beginning with "#" so that the resulting file can be preprocessed again.
	  COMMAND ${SED_COMMAND} -e "s/^#.*//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp1            # sed -e "s/^#.*//g" /build/src/libm/sleefSSE2.h.tmp1
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.c                                                 # > /build/src/libm/sleefSSE2.h.c

	  # Preprocess the intemediate file again to remove comments
	  COMMAND "${CMAKE_C_COMPILER}" ${FLAG_PREPROCESS}                                                 # gcc -E
	    ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.c                                                   # /build/src/libm/sleefSSE2.h.c
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp3                                              # > /build/src/libm/sleefSSE2.h.tmp3

	  # Embed version number into the header
	  COMMAND ${SED_COMMAND} -e                                                                        # sed -e
	    "s/%VERSION%/${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}.${SLEEF_VERSION_PATCHLEVEL}/g"      # "s/%VERSION%/3.5.0/g"
	    ${CMAKE_CURRENT_SOURCE_DIR}/sleefinline_header.h.org                                           # /sleef/src/libm/sleefinline_header.h.org
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp4                                              # > /build/src/libm/sleefSSE2.h.tmp4

	  # Remove lines beginning with "#"
	  COMMAND ${SED_COMMAND} -e "s/^#.*//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp3            # sed -e "s/^#.*//g" /build/src/libm/sleefSSE2.h.tmp3
	    >> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp4                                             # >> /build/src/libm/sleefSSE2.h.tmp4

	  # Replace multiple empty lines with a single empty line
	  COMMAND ${SED_COMMAND} -e "s/^[[:space:]]*$//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp4  # sed -e "s/^[[:space:]]*$//g" /build/src/libm/sleefSSE2.h.tmp4
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp5                                              # > /build/src/libm/sleefSSE2.h.tmp5
	  COMMAND ${SED_COMMAND} "/^$/N;/^\\n$/D" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp5          # sed -e "/^$/N;/^\n$/D" /build/src/libm/sleefSSE2.h.tmp5
	    > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp6                                              # > /build/src/libm/sleefSSE2.h.tmp6

	  COMMAND $<TARGET_FILE:addSuffix> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp6                 # addSuffix /build/src/libm/sleefSSE2.h.tmp6
	    ${sleef_SOURCE_DIR}/src/common/keywords.txt "_${SIMDLC}_sleef"                                 # keywords.txt "_sse2_sleef"
            > ${PROJECT_BINARY_DIR}/include/sleefinline_${SIMDLC}.h                                        # > /build/include/sleefinline_sse2.h

	  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimddp.c ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimdsp.c ${HEADER_${SIMD}}
	  DEPENDS ${HEADER_${SIMD}} addSuffix
	  VERBATIM
	  )

	list(APPEND INLINE_HEADER_FILES_GENERATED ${INLINE_HEADER_FILE})
      endif(COMPILER_SUPPORTS_${SIMD})
    endforeach()

    # Generate CUDA header file

    set(SIMD CUDA)
    string(TOLOWER ${SIMD} SIMDLC)

    set(INLINE_HEADER_FILE ${PROJECT_BINARY_DIR}/include/sleefinline_cuda.h)

    add_custom_command(
      OUTPUT ${INLINE_HEADER_FILE} 

      COMMAND echo Generating sleefinline_${SIMDLC}.h
      
      # Preprocess sleefsimddp.c with SLEEF_GENHEADER defined, comments are preserved
      COMMAND "${CMAKE_C_COMPILER}"  ${FLAG_PREPROCESS} ${FLAG_PRESERVE_COMMENTS}                      # gcc -E -C 
      ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/common ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/arch    # -I/sleef/src/common -I/sleef/src/arch
      ${FLAG_INCLUDE}${CMAKE_CURRENT_BINARY_DIR}/include/                                              # -I/build/src/libm/include
      ${FLAG_DEFINE}SLEEF_GENHEADER ${FLAG_DEFINE}ENABLE_${SIMD} ${FLAG_DEFINE}DORENAME                # -DSLEEF_GENHEADER -DENABLE_CUDA -DDORENAME
      ${FLAG_DEFINE}SLEEF_ALWAYS_INLINE=__device__ ${FLAG_DEFINE}SLEEF_INLINE=__device__               # -DSLEEF_ALWAYS_INLINE=__device__ -DSLEEF_INLINE=__device__
      ${FLAG_DEFINE}SLEEF_CONST=__device__ ${FLAG_DEFINE}static=__device__                             # -DSLEEF_CONST=__device__ -Dstatic=__device__
      ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimddp.c > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp1      # /sleef/src/libm/sleefsimddp.c > /build/libm/sleefCUDA.h.tmp1

      # Remove all lines except those begin with "//@"
      COMMAND ${SED_COMMAND} -n -e "/^\\/\\/@#.*$/p" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp1   # sed -n -e "/^\\/\\/@#.*$/p" /build/src/libm/sleefCUDA.h.tmp1
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp2                                                # > /build/src/libm/sleefCUDA.h.tmp2

      # Remove "//@"
      COMMAND ${SED_COMMAND} -e "s/^\\/\\/@#/#/g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp2      # sed -e "s/^\\/\\/@#/#/g" /build/src/libm/sleefCUDA.h.tmp2
      > ${CMAKE_CURRENT_BINARY_DIR}/include/macroonly${SIMD}.h                                         # > /build/src/libm/include/macroonlyCUDA.h

      # Preprocess sleefsimdsp.c with SLEEF_GENHEADER defined. Include macroonly*.h instead of helper*.h.
      COMMAND "${CMAKE_C_COMPILER}" ${FLAG_PREPROCESS} ${FLAG_PRESERVE_COMMENTS}                       # gcc -E -C 
      ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/common ${FLAG_INCLUDE}${PROJECT_SOURCE_DIR}/src/arch    # -I/sleef/src/common -I/sleef/src/arch
      ${FLAG_INCLUDE}${CMAKE_CURRENT_BINARY_DIR}/include/                                              # -I/build/src/libm/include
      ${FLAG_DEFINE}SLEEF_GENHEADER ${FLAG_DEFINE}ENABLE_${SIMD} ${FLAG_DEFINE}DORENAME                # -DSLEEF_GENHEADER -DENABLE_CUDA -DDORENAME
      ${FLAG_DEFINE}SLEEF_ALWAYS_INLINE=__device__ ${FLAG_DEFINE}SLEEF_INLINE=__device__               # -DSLEEF_ALWAYS_INLINE=__device__ -DSLEEF_INLINE=__device__
      ${FLAG_DEFINE}SLEEF_CONST=__device__ ${FLAG_DEFINE}static=__device__                             # -DSLEEF_CONST=__device__ -Dstatic=__device__
      ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimdsp.c >> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp1     # /sleef/src/libm/sleefsimdsp.c >> /build/libm/sleefCUDA.h.tmp1

      # Remove lines beginning with "#" so that the resulting file can be preprocessed again.
      COMMAND ${SED_COMMAND} -e "s/^#.*//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp1            # sed -e "s/^#.*//g" /build/src/libm/sleefCUDA.h.tmp1
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.c                                                   # > /build/src/libm/sleefCUDA.h.c

      # Preprocess the intemediate file again to remove comments
      COMMAND "${CMAKE_C_COMPILER}" ${FLAG_PREPROCESS}                                                 # gcc -E
      ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.c                                                     # /build/src/libm/sleefCUDA.h.c
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp3                                                # > /build/src/libm/sleefCUDA.h.tmp3

      # Embed version number into the header
      COMMAND ${SED_COMMAND} -e                                                                        # sed -e
      "s/%VERSION%/${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}.${SLEEF_VERSION_PATCHLEVEL}/g"        # "s/%VERSION%/3.5.0/g"
      ${CMAKE_CURRENT_SOURCE_DIR}/sleefinline_cuda_header.h.org                                        # /sleef/src/libm/sleefinline_header.h.org
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp4                                                # > /build/src/libm/sleefCUDA.h.tmp4

      # Remove lines beginning with "#"
      COMMAND ${SED_COMMAND} -e "s/^#.*//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp3            # sed -e "s/^#.*//g" /build/src/libm/sleefCUDA.h.tmp3
      >> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp4                                               # >> /build/src/libm/sleefCUDA.h.tmp4

      # Replace multiple empty lines with a single empty line
      COMMAND ${SED_COMMAND} -e "s/^[[:space:]]*$//g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp4  # sed -e "s/^[[:space:]]*$//g" /build/src/libm/sleefCUDA.h.tmp4
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp5                                                # > /build/src/libm/sleefCUDA.h.tmp5
      COMMAND ${SED_COMMAND} "/^$/N;/^\\n$/D" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp5          # sed -e "/^$/N;/^\n$/D" /build/src/libm/sleefCUDA.h.tmp5
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp6                                                # > /build/src/libm/sleefCUDA.h.tmp6

      # Remove redundant __device__
      COMMAND ${SED_COMMAND} -e "s/__device__ __device__/__device__/g"                                 # sed -e "s/__device__ __device__/__device__/g"
      -e "s/__device__ __device__/__device__/g" ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp6        # -e "s/__device__ __device__/__device__/g" /build/src/libm/sleefCUDA.h.tmp6
      > ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp7                                                # > /build/src/libm/sleefCUDA.h.tmp7

      COMMAND $<TARGET_FILE:addSuffix> ${CMAKE_CURRENT_BINARY_DIR}/sleef${SIMD}.h.tmp7                 # addSuffix /build/src/libm/sleefCUDA.h.tmp7
      ${sleef_SOURCE_DIR}/src/common/keywords_cuda.txt "_${SIMDLC}_sleef"                              # keywords_cuda.txt "_cuda_sleef"
      > ${PROJECT_BINARY_DIR}/include/sleefinline_${SIMDLC}.h                                          # > /build/include/sleefinline_cuda.h

      MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimddp.c ${CMAKE_CURRENT_SOURCE_DIR}/sleefsimdsp.c ${HEADER_${SIMD}}
      DEPENDS ${HEADER_${SIMD}} addSuffix
      VERBATIM
      )

    list(APPEND INLINE_HEADER_FILES_GENERATED ${INLINE_HEADER_FILE})

    #

    add_custom_target(${TARGET_INLINE_HEADERS} ALL
      DEPENDS
      ${INLINE_HEADER_FILES_GENERATED}
      )
    install(FILES ${INLINE_HEADER_FILES_GENERATED} DESTINATION include)
  endif(SED_COMMAND)
endif(BUILD_INLINE_HEADERS)

# On some systems we need to explicitly link libsleef against libm to
# use some of the math functions used in the scalar code (for example
# sqrt).
if(LIBM AND NOT COMPILER_SUPPORTS_BUILTIN_MATH)
  target_link_libraries(${TARGET_LIBSLEEF} ${LIBM})
endif()

target_sources(${TARGET_LIBSLEEF} PRIVATE $<TARGET_OBJECTS:${TARGET_LIBCOMMON_OBJ}>)

# --------------------------------------------------------------------

if (SLEEF_ARCH_X86)
  # Target dispsse.c

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dispsse.c
    COMMENT "Generating dispsse.c"
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/dispsse.c.org ${CMAKE_CURRENT_BINARY_DIR}/dispsse.c
    COMMAND $<TARGET_FILE:${TARGET_MKDISP}> 2 4 __m128d __m128 __m128i sse2 sse4 avx2128 >> ${CMAKE_CURRENT_BINARY_DIR}/dispsse.c
    DEPENDS ${TARGET_MKDISP}
    )
  add_custom_target(dispsse.c_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/dispsse.c)

  # Target renamedsp128.h

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp128.h
    COMMENT "Generating renamedsp128.h"
    COMMAND $<TARGET_FILE:${TARGET_MKRENAME}> - 2 4 > ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp128.h
    DEPENDS ${TARGET_MKRENAME}
    )
  add_custom_target(renamedsp128.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp128.h)

  # Target dispavx.c

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dispavx.c
    COMMENT "Generating dispavx.c"
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/dispavx.c.org ${CMAKE_CURRENT_BINARY_DIR}/dispavx.c
    COMMAND $<TARGET_FILE:${TARGET_MKDISP}> 4 8 __m256d __m256 __m128i avx fma4 avx2 >> ${CMAKE_CURRENT_BINARY_DIR}/dispavx.c
    DEPENDS ${TARGET_MKDISP}
    )
  add_custom_target(dispavx.c_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/dispavx.c)

  # Target renamedsp256.h

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp256.h
    COMMENT "Generating renamedsp256.h"
    COMMAND $<TARGET_FILE:${TARGET_MKRENAME}> - 4 8 > ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp256.h
    DEPENDS ${TARGET_MKRENAME}
    )
  add_custom_target(renamedsp256.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp256.h)

  # Target dispsse_obj

  if (COMPILER_SUPPORTS_FMA4)
    set(DISPATCHER_DEFINITIONS ${DISPATCHER_DEFINITIONS} ENABLE_FMA4=1)
  endif()

  if (COMPILER_SUPPORTS_AVX2)
    set(DISPATCHER_DEFINITIONS ${DISPATCHER_DEFINITIONS} ENABLE_AVX2=1)
  endif()

  if(COMPILER_SUPPORTS_SSE2)
    add_library(dispsse_obj OBJECT dispsse.c)
    target_compile_options(dispsse_obj PRIVATE ${FLAGS_ENABLE_SSE2})
    set_target_properties(dispsse_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
    target_compile_definitions(dispsse_obj PRIVATE ${COMMON_TARGET_DEFINITIONS} ${DISPATCHER_DEFINITIONS})
    target_include_directories(dispsse_obj PRIVATE ${sleef_BINARY_DIR}/include)
    add_dependencies(dispsse_obj dispsse.c_generated renamedsp128.h_generated ${TARGET_HEADERS})
    target_sources(${TARGET_LIBSLEEF} PRIVATE $<TARGET_OBJECTS:dispsse_obj>)
  endif()

  # Target dispavx_obj

  if(COMPILER_SUPPORTS_AVX)
    add_library(dispavx_obj OBJECT dispavx.c)
    target_compile_options(dispavx_obj PRIVATE ${FLAGS_ENABLE_AVX})
    set_target_properties(dispavx_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
    target_compile_definitions(dispavx_obj PRIVATE ${COMMON_TARGET_DEFINITIONS} ${DISPATCHER_DEFINITIONS})
    target_include_directories(dispavx_obj PRIVATE ${sleef_BINARY_DIR}/include)
    add_dependencies(dispavx_obj dispavx.c_generated renamedsp256.h_generated ${TARGET_HEADERS})
    target_sources(${TARGET_LIBSLEEF} PRIVATE $<TARGET_OBJECTS:dispavx_obj>)
  endif()
endif(SLEEF_ARCH_X86)

if (SLEEF_ARCH_PPC64)
  # Target disppower_128.c

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/disppower_128.c
    COMMENT "Generating disppower_128.c"
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/disppower_128.c.org ${CMAKE_CURRENT_BINARY_DIR}/disppower_128.c
    COMMAND $<TARGET_FILE:${TARGET_MKDISP}> 2 4 SLEEF_VECTOR_DOUBLE SLEEF_VECTOR_FLOAT SLEEF_VECTOR_INT vsx vsx3 >> ${CMAKE_CURRENT_BINARY_DIR}/disppower_128.c
    DEPENDS ${TARGET_MKDISP}
    )
  add_custom_target(disppower_128.c_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/disppower_128.c)

  # Target renamedsp128.h

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp128.h
    COMMENT "Generating renamedsp128.h"
    COMMAND $<TARGET_FILE:${TARGET_MKRENAME}> - 2 4 > ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp128.h
    DEPENDS ${TARGET_MKRENAME}
    )
  add_custom_target(renamedsp128.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp128.h)

  # Target disppower_128_obj

  add_library(disppower_128_obj OBJECT disppower_128.c)
  target_compile_options(disppower_128_obj PRIVATE ${FLAGS_ENABLE_VSX})
  if(COMPILER_SUPPORTS_VSX3)
    set(DISPATCHER_DEFINITIONS ENABLE_VSX3=1)
  endif()
  set_target_properties(disppower_128_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
  target_compile_definitions(disppower_128_obj PRIVATE ${COMMON_TARGET_DEFINITIONS} ${DISPATCHER_DEFINITIONS})
  target_include_directories(disppower_128_obj PRIVATE ${CMAKE_BINARY_DIR}/include)
  add_dependencies(disppower_128_obj disppower_128.c_generated renamedsp128.h_generated ${TARGET_HEADERS})
  target_sources(${TARGET_LIBSLEEF} PRIVATE $<TARGET_OBJECTS:disppower_128_obj>)

  if(COMPILER_SUPPORTS_VSX3)
    add_library(tryvsx3_obj OBJECT tryvsx3.c)
    target_compile_options(tryvsx3_obj PRIVATE ${FLAGS_ENABLE_VSX3})
    set_target_properties(tryvsx3_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
    target_sources(${TARGET_LIBSLEEF} PRIVATE $<TARGET_OBJECTS:tryvsx3_obj>)
  endif()
endif(SLEEF_ARCH_PPC64)

if (SLEEF_ARCH_S390X)
  # Target disps390x_128.c

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/disps390x_128.c
    COMMENT "Generating disps390x_128.c"
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/disps390x_128.c.org ${CMAKE_CURRENT_BINARY_DIR}/disps390x_128.c
    COMMAND $<TARGET_FILE:${TARGET_MKDISP}> 2 4 SLEEF_VECTOR_DOUBLE SLEEF_VECTOR_FLOAT SLEEF_VECTOR_INT vxe vxe2 >> ${CMAKE_CURRENT_BINARY_DIR}/disps390x_128.c
    DEPENDS ${TARGET_MKDISP}
    )
  add_custom_target(disps390x_128.c_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/disps390x_128.c)

  # Target renamedsp128.h

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp128.h
    COMMENT "Generating renamedsp128.h"
    COMMAND $<TARGET_FILE:${TARGET_MKRENAME}> - 2 4 > ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp128.h
    DEPENDS ${TARGET_MKRENAME}
    )
  add_custom_target(renamedsp128.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/include/renamedsp128.h)

  # Target disps390x_128_obj

  add_library(disps390x_128_obj OBJECT disps390x_128.c)
  target_compile_options(disps390x_128_obj PRIVATE ${FLAGS_ENABLE_VXE})
  if(COMPILER_SUPPORTS_VXE2)
    set(DISPATCHER_DEFINITIONS ENABLE_VXE2=1)
  endif()
  set_target_properties(disps390x_128_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
  target_compile_definitions(disps390x_128_obj PRIVATE ${COMMON_TARGET_DEFINITIONS} ${DISPATCHER_DEFINITIONS})
  target_include_directories(disps390x_128_obj PRIVATE ${CMAKE_BINARY_DIR}/include)
  add_dependencies(disps390x_128_obj disps390x_128.c_generated renamedsp128.h_generated ${TARGET_HEADERS})
  target_sources(${TARGET_LIBSLEEF} PRIVATE $<TARGET_OBJECTS:disps390x_128_obj>)

  if(COMPILER_SUPPORTS_VXE2)
    add_library(tryvxe2_obj OBJECT tryvxe2.c)
    target_compile_options(tryvxe2_obj PRIVATE ${FLAGS_ENABLE_VXE2})
    set_target_properties(tryvxe2_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
    target_sources(${TARGET_LIBSLEEF} PRIVATE $<TARGET_OBJECTS:tryvxe2_obj>)
  endif()
endif(SLEEF_ARCH_S390X)

# --------------------------------------------------------------------
# TARGET_LIBSLEEFGNUABI
# Compile SIMD versions for GNU Abi
# --------------------------------------------------------------------
# Build gnuabi version from just simd object files
if(ENABLE_GNUABI)
  set(TARGET_LIBSLEEFGNUABI_OBJECTS "")
  foreach(SIMD ${SLEEF_SUPPORTED_GNUABI_EXTENSIONS})
    if(COMPILER_SUPPORTS_${SIMD})
      # Need lowercase string for rename header
      string(TOLOWER ${SIMD} vecarch)

      foreach(T dp sp)
	add_library(${OBJECT_${SIMD}_${T}_GNUABI} OBJECT sleefsimd${T}.c ${HEADER_${SIMD}_GNUABI})
	target_compile_definitions(${OBJECT_${SIMD}_${T}_GNUABI} PRIVATE
          ENABLE_${SIMD}=1
          DORENAME=1
          ENABLE_GNUABI=1
	  )

        if(FORCE_AAVPCS AND ${SIMD} STREQUAL "ADVSIMD")
          target_compile_definitions(${OBJECT_${SIMD}_${T}_GNUABI} PRIVATE
            ENABLE_AAVPCS=1
          )
        endif()

	if(ENABLE_ALTDIV)
	  target_compile_definitions(${OBJECT_${SIMD}_${T}_GNUABI} PRIVATE ENABLE_ALTDIV=1)
	endif()

	if(ENABLE_ALTSQRT)
	  target_compile_definitions(${OBJECT_${SIMD}_${T}_GNUABI} PRIVATE ENABLE_ALTSQRT=1)
	endif()

	set_target_properties(${OBJECT_${SIMD}_${T}_GNUABI} PROPERTIES
          ${COMMON_TARGET_PROPERTIES}
	  )

	target_compile_options(${OBJECT_${SIMD}_${T}_GNUABI} PRIVATE ${FLAGS_ENABLE_${SIMD}})
	if (COMPILER_SUPPORTS_WEAK_ALIASES)
          target_compile_options(${OBJECT_${SIMD}_${T}_GNUABI} PRIVATE -DENABLE_GNUABI=1)
	endif(COMPILER_SUPPORTS_WEAK_ALIASES)
	list(APPEND TARGET_LIBSLEEFGNUABI_OBJECTS $<TARGET_OBJECTS:${OBJECT_${SIMD}_${T}_GNUABI}>)

	if(MKMASKED_PARAMS_GNUABI_${SIMD}_${T})
	  target_compile_definitions(${OBJECT_${SIMD}_${T}_GNUABI} PRIVATE
	    HEADER_MASKED=\"masked_${vecarch}_${T}_gnuabi.h\")
	  add_dependencies(${OBJECT_${SIMD}_${T}_GNUABI} masked${SIMD}_generated)
	endif()
      endforeach()
    endif(COMPILER_SUPPORTS_${SIMD})
  endforeach()

  # Create library 
  add_library(${TARGET_LIBSLEEFGNUABI} ${TARGET_LIBSLEEFGNUABI_OBJECTS} rempitab.c)
  
  # Library properties
  set_target_properties(${TARGET_LIBSLEEFGNUABI} PROPERTIES
    VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
    SOVERSION ${SLEEF_SOVERSION}
    POSITION_INDEPENDENT_CODE ON   # -fPIC
    C_STANDARD 99                  # -std=gnu99
    LINKER_LANGUAGE C
  )

# On some systems we need to explicitly link libsleefgnuabi against
# libm to use some of the math functions used in the scalar code (for
# example sqrt).
if(LIBM AND NOT COMPILER_SUPPORTS_BUILTIN_MATH)
  target_link_libraries(${TARGET_LIBSLEEFGNUABI} ${LIBM})
endif()
endif(ENABLE_GNUABI)

# --------------------------------------------------------------------
# TARGET_LLVM_BITCODE
# Generate LLVM bitcode
# --------------------------------------------------------------------
if(CLANG_EXE_PATH AND SLEEF_ENABLE_LLVM_BITCODE)
  set(SLEEP_LLVM_BITCODE_INCLUDES "")
  get_property(SLEEP_LLVM_BITCODE_INCLUDES_LIST DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  foreach(INCLUDE_DIRECTORY ${SLEEP_LLVM_BITCODE_INCLUDES_LIST})
    set(SLEEP_LLVM_BITCODE_INCLUDES "${SLEEP_LLVM_BITCODE_INCLUDES} -I ${INCLUDE_DIRECTORY}")
  endforeach()

  separate_arguments(SLEEP_LLVM_BITCODE_INCLUDES_CLANG WINDOWS_COMMAND "${SLEEP_LLVM_BITCODE_INCLUDES}")
  set(SLEEF_CLANG_LLVM_BITCODE_OPTIONS -O3 -S -emit-llvm -D NDEBUG -D DORENAME=1)
  set(LLVM_BITCODE_OUTPUTS "")

  # Generate LLVM bitcode for regular SLEEF
  foreach(STANDARD_SOURCE ${STANDARD_SOURCES})
    get_filename_component(SRC_WITHOUT_EXT ${STANDARD_SOURCE} NAME_WE)
    set(LLVM_BITCODE_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_WITHOUT_EXT}.c)
    set(LLVM_BITCODE_OUTPUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${SRC_WITHOUT_EXT}.ll)
    add_custom_command(OUTPUT ${LLVM_BITCODE_OUTPUT}
      COMMAND ${CLANG_EXE_PATH} ${SLEEF_CLANG_LLVM_BITCODE_OPTIONS} -o ${LLVM_BITCODE_OUTPUT} ${LLVM_BITCODE_INPUT} ${SLEEP_LLVM_BITCODE_INCLUDES_CLANG}
      DEPENDS
        ${LLVM_BITCODE_INPUT}
    )
    list(APPEND LLVM_BITCODE_OUTPUTS ${LLVM_BITCODE_OUTPUT})
  endforeach()

  # Generate LLVM bitcode for SIMD SLEEF
  foreach(SIMD ${SLEEF_SUPPORTED_LIBM_EXTENSIONS})
    if (${SIMD} STREQUAL "NEON32" OR ${SIMD} STREQUAL "NEON32VFPV4")
      set(SIMD_SOURCES sleefsimdsp.c)
    else()
      set(SIMD_SOURCES sleefsimdsp.c sleefsimddp.c)
    endif()

    if(COMPILER_SUPPORTS_${SIMD})
      foreach(SIMD_SOURCE ${SIMD_SOURCES})
        get_filename_component(SIMD_SOURCE_WITHOUT_EXT ${SIMD_SOURCE} NAME_WE)
        set(LLVM_BITCODE_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/${SIMD_SOURCE})
        set(LLVM_BITCODE_OUTPUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${SIMD_SOURCE_WITHOUT_EXT}_${SIMD}.ll)
        add_custom_command(OUTPUT ${LLVM_BITCODE_OUTPUT}
          COMMAND  ${CLANG_EXE_PATH} ${CLANG_FLAGS_ENABLE_${SIMD}} ${SLEEF_CLANG_LLVM_BITCODE_OPTIONS} -D ENABLE_${SIMD}=1 -o ${LLVM_BITCODE_OUTPUT} ${LLVM_BITCODE_INPUT} ${SLEEP_LLVM_BITCODE_INCLUDES_CLANG}
          DEPENDS
            ${LLVM_BITCODE_INPUT}
        )
        list(APPEND LLVM_BITCODE_OUTPUTS ${LLVM_BITCODE_OUTPUT})
      endforeach()
    endif()
  endforeach()

  file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_custom_target(${TARGET_LLVM_BITCODE} ALL
    DEPENDS
      ${LLVM_BITCODE_OUTPUTS}
  )
  add_dependencies(${TARGET_LLVM_BITCODE} ${TARGET_HEADERS})

  install(FILES ${LLVM_BITCODE_OUTPUTS} DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

# --------------------------------------------------------------------
# Install
# --------------------------------------------------------------------
# Install libsleef and sleef.h
install(TARGETS ${TARGET_LIBSLEEF}
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION bin
)
configure_file("sleef.pc.in" "${CMAKE_CURRENT_BINARY_DIR}/sleef.pc" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sleef.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

if(ENABLE_GNUABI)
  install(TARGETS ${TARGET_LIBSLEEFGNUABI}
          LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
          ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
          RUNTIME DESTINATION bin
          )
endif()
