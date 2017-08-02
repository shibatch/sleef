include(DefineLocations)

string(REPLACE " " ";" RENAME_HEADER_LIST  "${RENAME_HEADERS}")

set(sse2_params 2 4 sse2)

foreach(rename_header ${RENAME_HEADER_LIST})
  if(${rename_header} MATCHES "sse")
    execute_process(
	COMMAND ./mkrename ${sse2_params}
	WORKING_DIRECTORY ${LOCATION_RUNTIME_DIR}
	OUTPUT_VARIABLE MKRENAME_OUTPUT)
    file(WRITE ${rename_header} "${MKRENAME_OUTPUT}")

    if(OPTION_SHOW_CONFIG)
      message(STATUS "Generating ${rename_header}")
    endif(OPTION_SHOW_CONFIG)
  endif()
endforeach()

