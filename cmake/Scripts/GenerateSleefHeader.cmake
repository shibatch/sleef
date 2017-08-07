include(${LOCATIONS_FILE})

# Workaround: temporary copy of sleeflibm.h.org
set(SLEEF_ORG_HEADER ${LOCATION_SOURCE_DIR}/src/libm/sleeflibm.h.org)
set(SLEEF_GEN_HEADER ${LOCATION_BINARY_DIR}/include/sleef.h)
configure_file(${SLEEF_ORG_HEADER} ${SLEEF_GEN_HEADER} COPYONLY)

if(OPTION_SHOW_CONFIG)
  message(STATUS "Generating ${SLEEF_GEN_HEADER} from: ${SLEEF_ORG_HEADER}")
endif(OPTION_SHOW_CONFIG)

# Generate header for x86
  set(run1 2 4 __m128d __m128 __m128i __m128i __SSE2__)
  set(run2 2 4 __m128d __m128 __m128i __m128i __SSE2__ sse2)
  set(run3 2 4 __m128d __m128 __m128i __m128i __SSE2__ sse4)
  set(run4 2 4 __m128d __m128 __m128i __m128i __SSE2__ avx2128)
  set(run5 4 8 __m256d __m256 __m128i "struct { __m128i x, y\; }" __AVX__)
  set(run6 4 8 __m256d __m256 __m128i "struct { __m128i x, y\; }" __AVX__ avx)
  set(run7 4 8 __m256d __m256 __m128i "struct { __m128i x, y\; }" __AVX__ fma4)
  set(run8 4 8 __m256d __m256 __m128i __m256i __AVX__ avx2)
  set(run9 8 16 __m512d __m512 __m256i __m512i __AVX512F__ avx512f)

# TODO: Change condition of generation to use a COMPILER_SUPPORTS_* flag
if(SLEEF_ARCH_X86)
  list(APPEND PARAMS_POINTER_LIST run1 run2 run3 run4 run5 run6 run7 run8 run9)
endif()

foreach(params_set ${PARAMS_POINTER_LIST})
  execute_process(
      COMMAND ${LOCATION_RUNTIME_DIR}/${TARGET_MKRENAME} ${${params_set}}
      OUTPUT_VARIABLE _TMP_OUTPUT)
    file(APPEND ${SLEEF_GEN_HEADER} "${_TMP_OUTPUT}\n")

  if(OPTION_SHOW_CONFIG)
    message(STATUS "Running mkrename ${${params_set}}")
  endif(OPTION_SHOW_CONFIG)
endforeach()

file(APPEND ${SLEEF_GEN_HEADER} "#undef IMPORT\n")
file(APPEND ${SLEEF_GEN_HEADER} "#endif\n")
