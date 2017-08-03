include(${LOCATIONS_FILE})

function(run_mkrename string_params result_variable)
  # Convert string in param variable to list to pass to the executable
  string(REPLACE " " ";" list_params ${string_params})
  execute_process(
      COMMAND ${LOCATION_RUNTIME_DIR}/${TARGET_MKRENAME} ${list_params}
      OUTPUT_VARIABLE _TMP_OUTPUT)
  set(${result_variable} ${_TMP_OUTPUT} PARENT_SCOPE)
endfunction()

# Workaround: temporary copy of sleeflibm.h.org
set(SLEEF_ORG_HEADER ${LOCATION_SOURCE_DIR}/src/libm/sleeflibm.h.org)
set(SLEEF_GEN_HEADER ${LOCATION_BINARY_DIR}/include/sleef.h)
configure_file(${SLEEF_ORG_HEADER} ${SLEEF_GEN_HEADER} COPYONLY)

if(OPTION_SHOW_CONFIG)
  message(STATUS "Generating ${SLEEF_GEN_HEADER} from: ${SLEEF_ORG_HEADER}")
endif(OPTION_SHOW_CONFIG)

# Generate header for x86
if(SLEEF_ARCH_X86)
  list(APPEND MKRENAME_PARAMS_LIST
    "2 4 __m128d __m128 __m128i __m128i __SSE2__")
  list(APPEND MKRENAME_PARAMS_LIST
    "2 4 __m128d __m128 __m128i __m128i __SSE2__ sse2")
  list(APPEND MKRENAME_PARAMS_LIST
    "2 4 __m128d __m128 __m128i __m128i __SSE2__ sse4")
endif()

foreach(params_set ${MKRENAME_PARAMS_LIST})
  run_mkrename(${params_set} RESULT_VARIABLE)
  set(OUTPUT "${OUTPUT}\n${RESULT_VARIABLE}")

  if(OPTION_SHOW_CONFIG)
    message(STATUS "Running mkrename ${params_set}")
  endif(OPTION_SHOW_CONFIG)
endforeach()

file(APPEND ${SLEEF_GEN_HEADER} "\n${OUTPUT}\n#undef IMPORT\n#endif")
