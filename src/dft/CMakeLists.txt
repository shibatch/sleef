# Option MAXBUTWIDTH

set(MAXBUTWIDTH 4 CACHE STRING "Log_2 (Maximum butterfly length) of butterflies")

# Compiler properties

set(CMAKE_C_FLAGS ORG_CMAKE_C_FLAGS)
string(CONCAT CMAKE_C_FLAGS ${DFT_C_FLAGS})

set(COMMON_TARGET_PROPERTIES 
  POSITION_INDEPENDENT_CODE ON   # -fPIC
  C_STANDARD 99                  # -std=gnu99
  )

set(COMMON_TARGET_DEFINITIONS MAXBUTWIDTH=${MAXBUTWIDTH})

if (COMPILER_SUPPORTS_FLOAT128)
  set(COMMON_TARGET_DEFINITIONS ${COMMON_TARGET_DEFINITIONS} ENABLEFLOAT128)
endif(COMPILER_SUPPORTS_FLOAT128)

if (COMPILER_SUPPORTS_LONG_DOUBLE)
  set(COMMON_TARGET_DEFINITIONS ${COMMON_TARGET_DEFINITIONS} ENABLE_LONGDOUBLE)
endif (COMPILER_SUPPORTS_LONG_DOUBLE)

if(COMPILER_SUPPORTS_OPENMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif(COMPILER_SUPPORTS_OPENMP)

# Include directories

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Constants definition

set(LISTSHORTTYPENAME "dp" "sp" "ld" "qp")
set(LISTLONGTYPENAME "double" "float" "longdouble" "Sleef_quad")
set(LISTTYPEID "1" "2" "3" "4")

set(MACRODEF_vecextdp BASETYPEID=1 ENABLE_VECEXT CONFIG=1)
set(CFLAGS_vecextdp ${FLAGS_ENABLE_VECEXT})
set(MACRODEF_vecextsp BASETYPEID=2 ENABLE_VECEXT CONFIG=1)
set(CFLAGS_vecextsp ${FLAGS_ENABLE_VECEXT})
set(MACRODEF_vecextld BASETYPEID=3 ENABLE_VECEXT CONFIG=1)
set(CFLAGS_vecextld ${FLAGS_ENABLE_VECEXT})
set(MACRODEF_vecextqp BASETYPEID=4 ENABLE_VECEXT CONFIG=1)
set(CFLAGS_vecextqp ${FLAGS_ENABLE_VECEXT})
set(MACRODEF_purecdp BASETYPEID=1 ENABLE_PUREC CONFIG=1)
set(CFLAGS_purecdp ${FLAGS_ENABLE_PUREC})
set(MACRODEF_purecsp BASETYPEID=2 ENABLE_PUREC CONFIG=1)
set(CFLAGS_purecsp ${FLAGS_ENABLE_PUREC})
set(MACRODEF_purecld BASETYPEID=3 ENABLE_PUREC CONFIG=1)
set(CFLAGS_purecld ${FLAGS_ENABLE_PUREC})
set(MACRODEF_purecqp BASETYPEID=4 ENABLE_PUREC CONFIG=1)
set(CFLAGS_purecqp ${FLAGS_ENABLE_PUREC})
set(MACRODEF_sse2dp BASETYPEID=1 ENABLE_SSE2 CONFIG=4)
set(CFLAGS_sse2dp ${FLAGS_ENABLE_SSE4})
set(MACRODEF_sse2sp BASETYPEID=2 ENABLE_SSE2 CONFIG=4)
set(CFLAGS_sse2sp ${FLAGS_ENABLE_SSE4})
set(MACRODEF_avxdp BASETYPEID=1 ENABLE_AVX CONFIG=1)
set(CFLAGS_avxdp ${FLAGS_ENABLE_AVX})
set(MACRODEF_avxsp BASETYPEID=2 ENABLE_AVX CONFIG=1)
set(CFLAGS_avxsp ${FLAGS_ENABLE_AVX})
set(MACRODEF_avx2dp BASETYPEID=1 ENABLE_AVX2 CONFIG=1)
set(CFLAGS_avx2dp ${FLAGS_ENABLE_AVX2})
set(MACRODEF_avx2sp BASETYPEID=2 ENABLE_AVX2 CONFIG=1)
set(CFLAGS_avx2sp ${FLAGS_ENABLE_AVX2})
set(MACRODEF_avx512fdp BASETYPEID=1 ENABLE_AVX512F CONFIG=1)
set(CFLAGS_avx512fdp ${FLAGS_ENABLE_AVX512F})
set(MACRODEF_avx512fsp BASETYPEID=2 ENABLE_AVX512F CONFIG=1)
set(CFLAGS_avx512fsp ${FLAGS_ENABLE_AVX512F})
set(MACRODEF_advsimddp BASETYPEID=1 ENABLE_ADVSIMD CONFIG=1)
set(CFLAGS_advsimddp ${FLAGS_ENABLE_ADVSIMD})
set(MACRODEF_advsimdsp BASETYPEID=2 ENABLE_ADVSIMD CONFIG=1)
set(CFLAGS_advsimdsp ${FLAGS_ENABLE_ADVSIMD})
set(MACRODEF_neon32sp BASETYPEID=2 ENABLE_NEON32 CONFIG=1)
set(CFLAGS_neon32sp ${FLAGS_ENABLE_NEON32})

# List all available scalar data types

set(ISALIST_SP purecsp)
set(ISALIST_DP purecdp)

set(LIST_SUPPORTED_FPTYPE 0 1)
if(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
  set(ISALIST_SP vecextsp)
  set(ISALIST_DP vecextdp)
endif(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")

if (COMPILER_SUPPORTS_LONG_DOUBLE)
  set(LIST_SUPPORTED_FPTYPE ${LIST_SUPPORTED_FPTYPE} 2)
  set(ISALIST_QP purecld)
  if(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
    set(ISALIST_LD vecextld)
  endif(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
endif(COMPILER_SUPPORTS_LONG_DOUBLE)

if (COMPILER_SUPPORTS_FLOAT128)
  set(LIST_SUPPORTED_FPTYPE ${LIST_SUPPORTED_FPTYPE} 3)
  set(ISALIST_QP purecqp)
  if(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
    set(ISALIST_QP vecextqp)
  endif(CMAKE_C_COMPILER_ID MATCHES "(GNU|Clang)")
endif(COMPILER_SUPPORTS_FLOAT128)

# List all available vector data types

if (COMPILER_SUPPORTS_SSE4)
  set(ISALIST_SP ${ISALIST_SP} sse2sp)
  set(ISALIST_DP ${ISALIST_DP} sse2dp)
endif(COMPILER_SUPPORTS_SSE4)

if (COMPILER_SUPPORTS_AVX)
  set(ISALIST_SP ${ISALIST_SP} avxsp)
  set(ISALIST_DP ${ISALIST_DP} avxdp)
endif(COMPILER_SUPPORTS_AVX)

if (COMPILER_SUPPORTS_AVX2)
  set(ISALIST_SP ${ISALIST_SP} avx2sp)
  set(ISALIST_DP ${ISALIST_DP} avx2dp)
endif(COMPILER_SUPPORTS_AVX2)

if (COMPILER_SUPPORTS_AVX512F)
  set(ISALIST_SP ${ISALIST_SP} avx512fsp)
  set(ISALIST_DP ${ISALIST_DP} avx512fdp)
endif(COMPILER_SUPPORTS_AVX512F)

if (COMPILER_SUPPORTS_ADVSIMD)
  set(ISALIST_SP ${ISALIST_SP} advsimdsp)
  set(ISALIST_DP ${ISALIST_DP} advsimddp)
endif(COMPILER_SUPPORTS_ADVSIMD)

if (COMPILER_SUPPORTS_NEON32)
  set(ISALIST_SP ${ISALIST_SP} neon32sp)
endif(COMPILER_SUPPORTS_NEON32)

# Target mkunroll

set(TARGET_MKUNROLL "mkunroll")
add_executable(${TARGET_MKUNROLL} EXCLUDE_FROM_ALL mkunroll.c)
set_target_properties(${TARGET_MKUNROLL} PROPERTIES ${COMMON_TARGET_PROPERTIES})

# Target mkdispatch

set(TARGET_MKDISPATCH "mkdispatch")
add_executable(${TARGET_MKDISPATCH} EXCLUDE_FROM_ALL mkdispatch.c)
set_target_properties(${TARGET_MKDISPATCH} PROPERTIES ${COMMON_TARGET_PROPERTIES})

# Target dispatchparam.h

add_custom_command(OUTPUT dispatchparam.h
  COMMENT "Generating dispatchparam.h"
  COMMAND ${TARGET_MKDISPATCH} paramonly ${MAXBUTWIDTH} ${ISALIST_DP} > ${CMAKE_CURRENT_BINARY_DIR}/dispatchparam.h
  DEPENDS ${TARGET_MKDISPATCH}
  )
add_custom_target(dispatchparam.h_generated SOURCES ${CMAKE_CURRENT_BINARY_DIR}/dispatchparam.h)

# Target dispatch*.h

foreach(T ${LIST_SUPPORTED_FPTYPE})
  list(GET LISTSHORTTYPENAME ${T} ST)                       # ST is "dp", for example
  string(TOUPPER ${ST} CST)                                 # CST is "DP"
  list(GET LISTLONGTYPENAME ${T} LT)                        # LT is "double"
  list(GET LISTTYPEID ${T} ID)                              # ID is 1
  
  string(CONCAT S "dispatch" ${ST} ".h")                    # S is dispatchdp.h
  add_custom_command(OUTPUT ${S}
    COMMENT "Generating ${S}"
    COMMAND ${TARGET_MKDISPATCH} ${LT} ${MAXBUTWIDTH} ${ISALIST_${CST}} > ${S}
    DEPENDS ${TARGET_MKDISPATCH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

  string(CONCAT G ${S} "_generated")                        # G is dispatchdp.h_generated
  add_custom_target(${G} SOURCES ${S})
endforeach()

# Target dftcommon.o

add_library(dftcommon_obj OBJECT dftcommon.c dftcommon.h ${CMAKE_CURRENT_BINARY_DIR}/dispatchparam.h ${CMAKE_BINARY_DIR}/include/sleef.h)
add_dependencies(dftcommon_obj ${TARGET_HEADERS} dispatchparam.h_generated)
set_source_files_properties(${CMAKE_BINARY_DIR}/include/sleef.h PROPERTIES GENERATED TRUE)
set_target_properties(dftcommon_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
target_compile_definitions(dftcommon_obj PRIVATE ${COMMON_TARGET_DEFINITIONS})

# Target dft*.o

foreach(T ${LIST_SUPPORTED_FPTYPE})
  list(GET LISTSHORTTYPENAME ${T} ST)                       # ST is "dp", for example
  
  string(CONCAT G "dft" ${ST} "_obj")                       # G is "dftdp_obj"
  string(CONCAT S "dispatch" ${ST} ".h")                    # S is "dispatchdp.h"
  add_library(${G} OBJECT dft.c dftcommon.h ${S})
  string(CONCAT SG ${S} "_generated")                       # SG is "dispatchdp.h_generated"
  add_dependencies(${G} ${SG})
  set_target_properties(${G} PROPERTIES ${COMMON_TARGET_PROPERTIES})
  list(GET LISTTYPEID ${T} ID)                              # ID is 1
  target_compile_definitions(${G} PRIVATE BASETYPEID=${ID} ${COMMON_TARGET_DEFINITIONS})
endforeach()

# Copy unroll0.org to ${CMAKE_CURRENT_BINARY_DIR}

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/unroll0.org
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/unroll0.org ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unroll0.org)

# Target unroll*.c

foreach(T ${LIST_SUPPORTED_FPTYPE})
  list(GET LISTSHORTTYPENAME ${T} ST)                       # ST is "dp", for example
  string(TOUPPER ${ST} CST)                                 # CST is "DP"
  list(GET LISTLONGTYPENAME ${T} LT)                        # LT is "double"

  foreach(E ${ISALIST_${CST}})                              # E is "sse2dp"
    foreach(N 0 1 2 3)
      string(CONCAT UC unroll_ ${N} _ ${E} ".c")            # UC is "unroll_0_sse2dp.c"
      set(UNROLL_TARGET_${CST} ${UNROLL_TARGET_${CST}} ${UC})
      string(CONCAT UCG ${UC} "_generated")                 # UCG is "unroll_0_sse2dp.c_generated"
      add_custom_target(${UCG} SOURCES ${UC} ${CMAKE_CURRENT_SOURCE_DIR}/unroll0.org)
    endforeach()
  endforeach()
  message(STATUS "Unroll target for ${CST} : ${UNROLL_TARGET_${CST}}")

  if(UNROLL_TARGET_${CST})
    add_custom_command(OUTPUT ${UNROLL_TARGET_${CST}}
      COMMENT "Generating ${UNROLL_TARGET_${CST}}"
      COMMAND ${TARGET_MKUNROLL} ${LT} ${ISALIST_${CST}}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${TARGET_MKUNROLL} ${CMAKE_CURRENT_BINARY_DIR}/unroll0.org
      )
  endif()
endforeach()

# Target unroll*.o

foreach(T ${LIST_SUPPORTED_FPTYPE})
  list(GET LISTSHORTTYPENAME ${T} ST)                       # ST is "dp", for example
  string(TOUPPER ${ST} CST)                                 # CST is "DP"
  list(GET LISTLONGTYPENAME ${T} LT)                        # LT is "double"

  foreach(E ${ISALIST_${CST}})                              # E is "sse2dp"
    foreach(N 0 1 2 3)
      string(CONCAT U unroll_ ${N} _ ${E})                  # U is "unroll_0_sse2dp"
      string(CONCAT UG ${U} "_obj")                         # UG is "unroll_0_sse2dp_obj"
      string(CONCAT UC ${U} ".c")                           # UC is "unroll_0_sse2dp.c"
      string(CONCAT UCG ${UC} "_generated")                 # UCG is "unroll_0_sse2dp.c_generated"
      add_library(${UG} OBJECT ${CMAKE_CURRENT_BINARY_DIR}/${UC} ${CMAKE_BINARY_DIR}/include/sleef.h)
      set_target_properties(${UG} PROPERTIES ${COMMON_TARGET_PROPERTIES})
      target_include_directories(${UG} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
      target_compile_definitions(${UG} PRIVATE ${COMMON_TARGET_DEFINITIONS} ${MACRODEF_${E}})
      target_compile_options(${UG} PRIVATE ${CFLAGS_${E}})
      add_dependencies(${UG} ${TARGET_HEADERS} ${UCG})
    endforeach()
  endforeach()
endforeach()

# Target libdft

add_library(${TARGET_LIBDFT} SHARED $<TARGET_OBJECTS:dftcommon_obj>)
target_link_libraries(${TARGET_LIBDFT} ${TARGET_LIBSLEEF} ${TARGET_LIBARRAYMAP_STATIC})

foreach(T ${LIST_SUPPORTED_FPTYPE})
  list(GET LISTSHORTTYPENAME ${T} ST)                       # ST is "dp", for example
  
  string(CONCAT G "dft" ${ST} "_obj")                       # G is "dftdp_obj"
  target_sources(${TARGET_LIBDFT} PRIVATE $<TARGET_OBJECTS:${G}>)
endforeach()

foreach(T ${LIST_SUPPORTED_FPTYPE})
  list(GET LISTSHORTTYPENAME ${T} ST)                       # ST is "dp", for example
  string(TOUPPER ${ST} CST)                                 # CST is "DP"

  foreach(E ${ISALIST_${CST}})                              # E is "sse2dp"
    foreach(N 0 1 2 3)
      string(CONCAT UG unroll_ ${N} _ ${E} "_obj")          # U is "unroll_0_sse2dp_obj"
      target_sources(${TARGET_LIBDFT} PRIVATE $<TARGET_OBJECTS:${UG}>)
    endforeach()
  endforeach()
endforeach()

set_target_properties(${TARGET_LIBDFT} PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION}
  ${COMMON_TARGET_PROPERTIES}
  ) 

# Install

install(FILES ${PROJECT_SOURCE_DIR}/include/sleefdft.h DESTINATION include)
install(TARGETS ${TARGET_LIBDFT} DESTINATION lib)
