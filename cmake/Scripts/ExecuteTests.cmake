include(${LOCATIONS_FILE})

string(REPLACE " " ";" EXE_IUT_LIST "${IUT_LIST}")

foreach(exe_iut IN LISTS EXE_IUT_LIST)
  message(STATUS "Running: ${TARGET_TESTER} ${exe_iut}")
  execute_process(
    COMMAND 
    ${LOCATION_RUNTIME_DIR}/${TARGET_TESTER} ${LOCATION_RUNTIME_DIR}/${exe_iut})
endforeach()

