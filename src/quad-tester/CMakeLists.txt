link_directories(${sleef_BINARY_DIR}/lib)                 # libsleefquad
link_directories(${sleef_BINARY_DIR}/src/common)          # common.a
include_directories(${sleef_BINARY_DIR}/include)          # sleef.h, sleefquad.h
include_directories(${sleef_SOURCE_DIR}/src/quad)         # qrename.h
include_directories(${sleef_BINARY_DIR}/src/quad/include) # rename headers

if(NOT LIB_MPFR)
  find_program(QTESTER_COMMAND qtester)
endif(NOT LIB_MPFR)

find_library(LIBRT rt)
if (NOT LIBRT)
  set(LIBRT "")
endif()

set(CMAKE_C_FLAGS "${ORG_CMAKE_C_FLAGS} ${SLEEF_C_FLAGS}")

if(COMPILER_SUPPORTS_FLOAT128)
  list(APPEND COMMON_TARGET_DEFINITIONS ENABLEFLOAT128=1)
endif()

#

if(SLEEF_OPENSSL_FOUND)
  # Build tester3printf
  add_executable(tester3printf tester3printf.c)
  add_dependencies(tester3printf sleefquad sleefquad_headers ${TARGET_LIBSLEEF} ${TARGET_HEADERS})
  target_compile_definitions(tester3printf PRIVATE ${COMMON_TARGET_DEFINITIONS})
  set_target_properties(tester3printf PROPERTIES C_STANDARD 99)
  target_link_libraries(tester3printf sleefquad ${TARGET_LIBSLEEF} ${SLEEF_OPENSSL_LIBRARIES})
  target_include_directories(tester3printf PRIVATE ${SLEEF_OPENSSL_INCLUDE_DIR})

  if (SDE_COMMAND)
    add_test(NAME tester3printf COMMAND ${SDE_COMMAND} "--" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tester3printf ${sleef_SOURCE_DIR}/src/quad-tester/hash_printf.txt)
  elseif(EMULATOR)
    add_test(NAME tester3printf COMMAND ${EMULATOR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tester3printf ${sleef_SOURCE_DIR}/src/quad-tester/hash_printf.txt)
  else()
    add_test(NAME tester3printf COMMAND tester3printf ${sleef_SOURCE_DIR}/src/quad-tester/hash_printf.txt)
  endif()
  set_tests_properties(tester3printf PROPERTIES COST 5.0)
endif()

#

function(add_test_iut IUT)
  if (LIB_MPFR)
    set(QTESTER qtester)
  elseif(QTESTER_COMMAND)
    set(QTESTER ${QTESTER_COMMAND})
  endif()
  # When we are crosscompiling using the mkrename* tools from a native
  # build, we use the tester executable from the native build.
  if (CMAKE_CROSSCOMPILING AND NATIVE_BUILD_DIR)
    set(QTESTER ${NATIVE_BUILD_DIR}/bin/qtester)
  endif(CMAKE_CROSSCOMPILING AND NATIVE_BUILD_DIR)
  if (QTESTER)
    if (NOT EMULATOR)
      if (SDE_COMMAND)
	set(FLAGS_SDE "--sde" ${SDE_COMMAND})
      else()
	set(FLAGS_SDE)
      endif()
      if (ARMIE_COMMAND)
        set(FLAGS_ARMIE ${ARMIE_COMMAND} -msve-vector-bits=${SVE_VECTOR_BITS})
      else()
        set(FLAGS_ARMIE)
      endif()
      add_test(NAME ${IUT}
	COMMAND ${QTESTER} ${FLAGS_SDE} ${FLAGS_ARMIE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${IUT}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
      set_tests_properties(${IUT} PROPERTIES COST 0.5)
    else()
      add_test(NAME ${IUT}
	COMMAND ${QTESTER} ${EMULATOR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${IUT}
	WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
      set_tests_properties(${IUT} PROPERTIES COST 0.5)
    endif()
  endif()
endfunction()

# Add vector extension `iut`s
set(IUT_SRC qiutsimd.c qiutsimdmain.c qtesterutil.c)

macro(test_extension SIMD)
  if(COMPILER_SUPPORTS_${SIMD})
    string(TOLOWER ${SIMD} LCSIMD)
    string(CONCAT TARGET_IUT${SIMD} "qiut" ${LCSIMD})

    add_executable(${TARGET_IUT${SIMD}} ${IUT_SRC})
    target_compile_options(${TARGET_IUT${SIMD}}
      PRIVATE ${FLAGS_ENABLE_${SIMD}})
    target_compile_definitions(${TARGET_IUT${SIMD}}
      PRIVATE ENABLE_${SIMD}=1 ${COMMON_TARGET_DEFINITIONS})
    target_link_libraries(${TARGET_IUT${SIMD}} sleefquad ${TARGET_LIBSLEEF} ${LIBM} ${LIBRT})

    add_dependencies(${TARGET_IUT${SIMD}} sleefquad_headers ${TARGET_HEADERS})
    add_dependencies(${TARGET_IUT${SIMD}} sleefquad ${TARGET_LIBSLEEF})
    set_target_properties(${TARGET_IUT${SIMD}} PROPERTIES C_STANDARD 99)
    add_test_iut(${TARGET_IUT${SIMD}})
    list(APPEND IUT_LIST ${TARGET_IUT${SIMD}})

    # The iut programs whose names begin with "qiuti" are the iut for the
    # inline version of quad functions.

    if (BUILD_INLINE_HEADERS AND SED_COMMAND)
      string(CONCAT IUTINAME "qiuti" ${LCSIMD})
      add_executable(${IUTINAME} ${IUT_SRC})
      target_compile_options(${IUTINAME}
	PRIVATE ${FLAGS_ENABLE_${SIMD}})
      target_compile_definitions(${IUTINAME}
	PRIVATE ENABLE_${SIMD}=1 ${COMMON_TARGET_DEFINITIONS}
	USE_INLINE_HEADER="sleefquadinline_${LCSIMD}.h"
	MACRO_ONLY_HEADER="qmacroonly${SIMD}.h"
	SIMD_SUFFIX=_${LCSIMD}_sleefq
	)
      target_include_directories(${IUTINAME} PRIVATE ${PROJECT_BINARY_DIR}/include)
      target_link_libraries(${IUTINAME} ${LIBM} ${LIBRT})
      add_dependencies(${IUTINAME} ${TARGET_QINLINE_HEADERS})
      set_target_properties(${IUTINAME} PROPERTIES C_STANDARD 99)
      add_test_iut(${IUTINAME})
      list(APPEND IUT_LIST ${IUTINAME})
    endif(BUILD_INLINE_HEADERS AND SED_COMMAND)

    if(LIB_MPFR AND NOT MINGW)
      # Build qtester2 SIMD
      string(TOLOWER ${SIMD} SIMDLC)
      set(T "tester2${SIMDLC}qp")
      add_executable(${T} tester2simdqp.c qtesterutil.c)
      target_compile_options(${T} PRIVATE ${FLAGS_ENABLE_${SIMD}})
      target_compile_definitions(${T} PRIVATE ENABLE_${SIMD}=1 USEMPFR=1 ${COMMON_TARGET_DEFINITIONS})
      set_target_properties(${T} PROPERTIES C_STANDARD 99)
      target_link_libraries(${T} sleefquad ${TARGET_LIBSLEEF} ${LIB_MPFR} ${LIBM} ${LIBGMP})
      add_dependencies(${T} sleefquad sleefquad_headers ${TARGET_LIBSLEEF} ${TARGET_HEADERS})
      if (MPFR_INCLUDE_DIR)
	target_include_directories(${T} PRIVATE ${MPFR_INCLUDE_DIR})
      endif()
    endif()
  endif(COMPILER_SUPPORTS_${SIMD})
endmacro(test_extension)

foreach(SIMD ${SLEEF_SUPPORTED_QUAD_EXTENSIONS})
  test_extension(${SIMD})
endforeach()

# Compile executable 'iutcuda'

if (BUILD_INLINE_HEADERS AND SED_COMMAND AND CMAKE_CUDA_COMPILER)
  add_executable(qiutcuda qiutcuda.cu)
  set_target_properties(qiutcuda PROPERTIES LINKER_LANGUAGE CUDA)
  target_compile_options(qiutcuda PRIVATE "--fmad=false")
  add_dependencies(qiutcuda ${TARGET_QINLINE_HEADERS})
  add_test_iut(qiutcuda 1.0)
  list(APPEND IUT_LIST qiutcuda)
endif()

#

if(LIB_MPFR AND NOT MINGW)
  # Compile executable 'qtester'
  add_host_executable(qtester qtester.c qtesterutil.c)
  if (NOT CMAKE_CROSSCOMPILING)
    target_link_libraries(qtester sleefquad ${TARGET_LIBSLEEF} ${LIBM} ${LIB_MPFR} ${LIBGMP})
    target_compile_definitions(qtester PRIVATE USEMPFR=1 ${COMMON_TARGET_DEFINITIONS})
    target_compile_options(qtester PRIVATE -Wno-unused-result)
    set_target_properties(qtester PROPERTIES C_STANDARD 99)
    if (MPFR_INCLUDE_DIR)
      target_include_directories(qtester PRIVATE ${MPFR_INCLUDE_DIR})
    endif()
  endif()
endif(LIB_MPFR AND NOT MINGW)

# BUild qutil

add_executable(qutil qutil.c)
add_dependencies(qutil sleefquad sleefquad_headers ${TARGET_LIBSLEEF} ${TARGET_HEADERS})
target_compile_definitions(qutil PRIVATE ${COMMON_TARGET_DEFINITIONS})
set_target_properties(qutil PROPERTIES C_STANDARD 99)
target_link_libraries(qutil sleefquad)
