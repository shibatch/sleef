include(${LOCATIONS_FILE})

string(REPLACE " " ";" RENAME_HEADER_LIST  "${RENAME_HEADERS}")

foreach(rename_header ${RENAME_HEADER_LIST})
  if(${rename_header} MATCHES "renamesse2.h")
    set(params 2 4 sse2)
  elseif(${rename_header} MATCHES "renamesse4.h")
    set(params 2 4 sse4)
  elseif(${rename_header} MATCHES "renameavx.h")
    set(params 4 8 avx)
  elseif(${rename_header} MATCHES "renamefma4.h")
    set(params 4 8 fma4)
  elseif(${rename_header} MATCHES "renameavx2.h")
    set(params 4 8 avx2)
  elseif(${rename_header} MATCHES "renameavx2128.h")
    set(params 2 4 avx2128)
  elseif(${rename_header} MATCHES "renameavx512f.h")
    set(params 8 16 avx512f)
  elseif(${rename_header} MATCHES "renameadvsimd.h")
    set(params 2 4 advsimd)
  endif()

  execute_process(
    COMMAND ${LOCATION_RUNTIME_DIR}/${TARGET_MKRENAME} ${params}
    OUTPUT_VARIABLE MKRENAME_OUTPUT)
  file(WRITE ${rename_header} "${MKRENAME_OUTPUT}")

  if(OPTION_SHOW_CONFIG)
    message(STATUS "Generating ${rename_header}")
  endif(OPTION_SHOW_CONFIG)
endforeach()

