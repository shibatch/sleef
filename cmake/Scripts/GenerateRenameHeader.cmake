include(${LOCATIONS_FILE})

string(REPLACE " " ";" RENAME_HEADER_LIST  "${RENAME_HEADERS}")

foreach(rename_header ${RENAME_HEADER_LIST})
  if(${rename_header} MATCHES "renamesse2")
    set(params 2 4 sse2)
  endif()

  execute_process(
    COMMAND ${LOCATION_RUNTIME_DIR}/${TARGET_MKRENAME} ${params}
    OUTPUT_VARIABLE MKRENAME_OUTPUT)
  file(WRITE ${rename_header} "${MKRENAME_OUTPUT}")

  if(OPTION_SHOW_CONFIG)
    message(STATUS "Generating ${rename_header}")
  endif(OPTION_SHOW_CONFIG)
endforeach()

