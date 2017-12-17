# Note: We are assuming SLEEF is the CMake root project.
# TODO: Remove constraint: do not use CMAKE_BINARY_DIR and CMAKE_SOURCE_DIR
link_directories(${CMAKE_BINARY_DIR}/lib)                 # libsleef
link_directories(${CMAKE_BINARY_DIR}/src/common)          # common.a
include_directories(${CMAKE_BINARY_DIR}/include)          # sleef.h
include_directories(${CMAKE_SOURCE_DIR}/src/libm)         # rename.h
include_directories(${CMAKE_BINARY_DIR}/src/libm/include) # rename headers

# Check if tester.exe is already there
if (MSVC AND EXISTS ${PROJECT_SOURCE_DIR}/src/libm-tester/tester.exe)
  set(TESTER_AVAILABLE 1)
  add_custom_target(copy_tester_exe ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/src/libm-tester/tester.exe ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Copying tester.exe to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    DEPENDS ${TARGET_IUT})
  if (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE)
    add_custom_target(copy_tester_exe ALL
      COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/src/libm-tester/tester.exe ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}
      COMMENT "Copying tester.exe to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}"
      DEPENDS ${TARGET_IUT})
  endif()
  if (CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG)
    add_custom_target(copy_tester_exe ALL
      COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/src/libm-tester/tester.exe ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}
      COMMENT "Copying tester.exe to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}"
      DEPENDS ${TARGET_IUT})
  endif()
endif()

if(LIB_MPFR)
  set(TESTER_AVAILABLE 1)
endif(LIB_MPFR)

# Compile executable 'iut'
add_executable(${TARGET_IUT} iut.c testerutil.c)
target_link_libraries(${TARGET_IUT} ${TARGET_LIBSLEEF} ${TARGET_LIBCOMMON_STATIC})
set_target_properties(${TARGET_IUT} PROPERTIES C_STANDARD 99)
if (TESTER_AVAILABLE)
  add_test(NAME ${TARGET_IUT}
    COMMAND ${TARGET_TESTER} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET_IUT})
endif()

set(IUT_SRC iutsimd.c iutsimdmain.c testerutil)

# Add vector extension `iut`s
macro(test_extension SIMD)
  if(COMPILER_SUPPORTS_${SIMD})
    add_executable(${TARGET_IUT${SIMD}} ${IUT_SRC})
    target_compile_options(${TARGET_IUT${SIMD}}
      PRIVATE ${FLAGS_ENABLE_${SIMD}})
    target_compile_definitions(${TARGET_IUT${SIMD}}
      PRIVATE ENABLE_${SIMD}=1)
    target_link_libraries(${TARGET_IUT${SIMD}} ${TARGET_LIBSLEEF} ${TARGET_LIBCOMMON_STATIC})

    add_dependencies(${TARGET_IUT${SIMD}} ${TARGET_HEADERS})
    add_dependencies(${TARGET_IUT${SIMD}} ${TARGET_LIBSLEEF})

    if (TESTER_AVAILABLE)
      add_test(NAME ${TARGET_IUT${SIMD}}
	COMMAND ${TARGET_TESTER} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET_IUT${SIMD}})
    endif(TESTER_AVAILABLE)

    set_target_properties(${TARGET_IUT${SIMD}} PROPERTIES C_STANDARD 99)

    if(ENABLE_GNUABI)
      if (COMPILER_SUPPORTS_WEAK_ALIASES)
	add_executable(link_finite${SIMD} finite_names_testing.c)
	target_compile_options(link_finite${SIMD}
	  PRIVATE ${FLAGS_ENABLE_${SIMD}})
	target_compile_definitions(link_finite${SIMD}
	  PRIVATE ENABLE_${SIMD}=1)
	target_link_libraries(link_finite${SIMD} ${TARGET_LIBSLEEFGNUABI})
	add_test(NAME link_finite${SIMD}
	  COMMAND link_finite${SIMD})
      endif (COMPILER_SUPPORTS_WEAK_ALIASES)
    endif(ENABLE_GNUABI)

    if(LIB_MPFR AND NOT ${SIMD} STREQUAL NEON32)
      # Build tester2 SIMD
      string(TOLOWER ${SIMD} SCSIMD)
      foreach(P dp sp)
	set(T "tester2${SCSIMD}${P}")
	add_executable(${T} tester2simd${P}.c testerutil.c)
	target_compile_options(${T} PRIVATE ${FLAGS_ENABLE_${SIMD}})
	target_compile_definitions(${T} PRIVATE ENABLE_${SIMD}=1 USEMPFR=1)
	set_target_properties(${T} PROPERTIES C_STANDARD 99)
	target_link_libraries(${T} ${TARGET_LIBSLEEF} ${LIB_MPFR} ${TARGET_LIBCOMMON_STATIC})
	add_dependencies(${T} ${TARGET_HEADERS})
	add_dependencies(${T} ${TARGET_LIBSLEEF})
      endforeach()
    endif()
      
  endif(COMPILER_SUPPORTS_${SIMD})
endmacro(test_extension)

foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
  test_extension(${SIMD})
endforeach()

if(LIB_MPFR)
  # Build tester2 scalar
  foreach(P dp sp)
    set(T "tester2${P}")
    add_executable(${T} tester2${P}.c testerutil.c)
    target_compile_definitions(${T} PRIVATE USEMPFR=1)
    target_link_libraries(${T} ${TARGET_LIBSLEEF} ${LIB_MPFR} ${TARGET_LIBCOMMON_STATIC})
    add_dependencies(${T} ${TARGET_HEADERS})
    add_dependencies(${T} ${TARGET_LIBSLEEF})
  endforeach()

  # No test defined with tester2

  # Compile executable 'tester'
  add_executable(${TARGET_TESTER} tester.c testerutil.c)
  target_link_libraries(${TARGET_TESTER} ${LIB_MPFR} ${TARGET_LIBSLEEF})
  target_compile_definitions(${TARGET_TESTER}
    PRIVATE USEMPFR=1)
  target_compile_options(${TARGET_TESTER} PRIVATE -Wno-unused-result)
  set_target_properties(${TARGET_TESTER} PROPERTIES C_STANDARD 99)
elseif(MSVC)
  
endif(LIB_MPFR)
  
# Set C standard requirement (-std=gnu99 for gcc)

# Tests depends on the library
add_dependencies(${TARGET_IUT} ${TARGET_HEADERS})
