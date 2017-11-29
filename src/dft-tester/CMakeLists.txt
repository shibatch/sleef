# Compiler properties

set(COMMON_TARGET_PROPERTIES 
  C_STANDARD 99                  # -std=gnu99
  )

# Include directories

include_directories(${PROJECT_SOURCE_DIR}/include)        # sleefdft.h
include_directories(${CMAKE_BINARY_DIR}/include)          # sleef.h

# Link directories

link_directories(${CMAKE_BINARY_DIR}/lib)                 # libsleef, libsleefdft


if(MSVC)
  message(STATUS "The testing program for DFT is currently not available with MSVC build - skip building tests dft-tester")
else()
  # Target executable naivetestdp_exe

  set(TARGET_NAIVETESTDP "naivetestdp_exe")
  add_executable(${TARGET_NAIVETESTDP} naivetest.c ${PROJECT_SOURCE_DIR}/include/sleefdft.h)
  add_dependencies(${TARGET_NAIVETESTDP} ${TARGET_HEADERS} ${TARGET_LIBSLEEF} ${TARGET_LIBDFT})
  target_compile_definitions(${TARGET_NAIVETESTDP} PRIVATE ${COMMON_TARGET_DEFINITIONS} BASETYPEID=1)
  target_link_libraries(${TARGET_NAIVETESTDP} ${TARGET_LIBSLEEF} ${TARGET_LIBDFT})
  set_target_properties(${TARGET_NAIVETESTDP} PROPERTIES ${COMMON_TARGET_PROPERTIES})

  # Target executable naivetestsp_exe

  set(TARGET_NAIVETESTSP "naivetestsp_exe")
  add_executable(${TARGET_NAIVETESTSP} naivetest.c ${PROJECT_SOURCE_DIR}/include/sleefdft.h)
  add_dependencies(${TARGET_NAIVETESTSP} ${TARGET_HEADERS} ${TARGET_LIBSLEEF} ${TARGET_LIBDFT})
  target_compile_definitions(${TARGET_NAIVETESTSP} PRIVATE ${COMMON_TARGET_DEFINITIONS} BASETYPEID=2)
  target_link_libraries(${TARGET_NAIVETESTSP} ${TARGET_LIBSLEEF} ${TARGET_LIBDFT})
  set_target_properties(${TARGET_NAIVETESTSP} PROPERTIES ${COMMON_TARGET_PROPERTIES})

  # Test naivetestdp

  add_test(NAME ${TARGET_NAIVETESTDP}_1 COMMAND ${TARGET_NAIVETESTDP} 1)
  add_test(NAME ${TARGET_NAIVETESTDP}_2 COMMAND ${TARGET_NAIVETESTDP} 2)
  add_test(NAME ${TARGET_NAIVETESTDP}_3 COMMAND ${TARGET_NAIVETESTDP} 3)
  add_test(NAME ${TARGET_NAIVETESTDP}_4 COMMAND ${TARGET_NAIVETESTDP} 4)
  add_test(NAME ${TARGET_NAIVETESTDP}_5 COMMAND ${TARGET_NAIVETESTDP} 5)
  add_test(NAME ${TARGET_NAIVETESTDP}_10 COMMAND ${TARGET_NAIVETESTDP} 10)

  # Test naivetestsp
  
  add_test(NAME ${TARGET_NAIVETESTSP}_1 COMMAND ${TARGET_NAIVETESTSP} 1)
  add_test(NAME ${TARGET_NAIVETESTSP}_2 COMMAND ${TARGET_NAIVETESTSP} 2)
  add_test(NAME ${TARGET_NAIVETESTSP}_3 COMMAND ${TARGET_NAIVETESTSP} 3)
  add_test(NAME ${TARGET_NAIVETESTSP}_4 COMMAND ${TARGET_NAIVETESTSP} 4)
  add_test(NAME ${TARGET_NAIVETESTSP}_5 COMMAND ${TARGET_NAIVETESTSP} 5)
  add_test(NAME ${TARGET_NAIVETESTSP}_10 COMMAND ${TARGET_NAIVETESTSP} 10)
endif(MSVC)
