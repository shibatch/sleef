include(${LOCATIONS_FILE})

# Workaround: temporary copy of sleeflibm.h.org
set(SLEEF_ORG_HEADER ${LOCATION_SOURCE_DIR}/src/libm/sleeflibm.h.org)
set(SLEEF_GEN_HEADER ${LOCATION_BINARY_DIR}/include/sleef.h)
configure_file(${SLEEF_ORG_HEADER} ${SLEEF_GEN_HEADER} COPYONLY)

if(OPTION_SHOW_CONFIG)
  message(STATUS "Generating ${SLEEF_GEN_HEADER} from: ${SLEEF_ORG_HEADER}")
endif(OPTION_SHOW_CONFIG)

# Generate header for x86
  set(mkrename_sse_
    2 4 __m128d __m128 __m128i __m128i __SSE2__)
  set(mkrename_sse2
    2 4 __m128d __m128 __m128i __m128i __SSE2__ sse2)
  set(mkrename_sse4
    2 4 __m128d __m128 __m128i __m128i __SSE2__ sse4)
  set(mkrename_avx2128
    2 4 __m128d __m128 __m128i __m128i __SSE2__ avx2128)
  set(mkrename_avx_
    4 8 __m256d __m256 __m128i "struct { __m128i x, y\; }" __AVX__)
  set(mkrename_avx
    4 8 __m256d __m256 __m128i "struct { __m128i x, y\; }" __AVX__ avx)
  set(mkrename_fma4
    4 8 __m256d __m256 __m128i "struct { __m128i x, y\; }" __AVX__ fma4)
  set(mkrename_avx2
    4 8 __m256d __m256 __m128i __m256i __AVX__ avx2)
  set(mkrename_avx512f
    8 16 __m512d __m512 __m256i __m512i __AVX512F__ avx512f)
  set(mkrename_advsimd
    2 4 float64x2_t float32x4_t int32x2_t int32x4_t __ARM_NEON advsimd)

# TODO: Change condition of generation to use a COMPILER_SUPPORTS_* flag
if(SLEEF_ARCH_X86)
  list(APPEND PARAMS_POINTER_LIST
    mkrename_sse_
    mkrename_sse2
    mkrename_sse4
    mkrename_avx2128
    mkrename_avx_
    mkrename_avx
    mkrename_fma4
    mkrename_avx2
    mkrename_avx512f)
elseif(SLEEF_ARCH_AARCH64)
  list(APPEND PARAMS_POINTER_LIST mkrename_advsimd)
endif()

foreach(params_set ${PARAMS_POINTER_LIST})
  execute_process(
      # Note: <string> params_set is a name pointing to a list of parameters
      # Double-derefencing params_set returns a <list> value (set of params)
      COMMAND ${MKRENAME_EXE} ${${params_set}}
      OUTPUT_VARIABLE _TMP_OUTPUT)
    file(APPEND ${SLEEF_GEN_HEADER} "${_TMP_OUTPUT}")

  if(OPTION_SHOW_CONFIG)
    message(STATUS "Running ${MKRENAME_EXE} ${${params_set}}")
  endif(OPTION_SHOW_CONFIG)
endforeach()

file(APPEND ${SLEEF_GEN_HEADER} "#undef IMPORT\n")
file(APPEND ${SLEEF_GEN_HEADER} "#endif\n")
