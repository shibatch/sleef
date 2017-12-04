file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# --------------------------------------------------------------------
# sleef.h
# --------------------------------------------------------------------
# File generated for the headers
set(SLEEF_ORG_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm_header.h.org)
set(SLEEF_ORG_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm_footer.h.org)
set(SLEEF_INCLUDE_HEADER ${CMAKE_BINARY_DIR}/include/sleef.h)

set(SLEEF_HEADER_COMMANDS "")
list(APPEND SLEEF_HEADER_COMMANDS COMMAND ${CMAKE_COMMAND} -E copy ${SLEEF_ORG_HEADER} ${SLEEF_INCLUDE_HEADER})
foreach(SIMD ${SLEEF_HEADER_LIST})
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND echo Generating sleef.h: ${TARGET_MKRENAME} ${HEADER_PARAMS_${SIMD}})
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND ${TARGET_MKRENAME} ${HEADER_PARAMS_${SIMD}} >> ${SLEEF_INCLUDE_HEADER})
endforeach()

if(MSVC)
  string(REPLACE "/" "\\" sleef_footer_input_file "${SLEEF_ORG_FOOTER}")
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND type ${sleef_footer_input_file} >> ${SLEEF_INCLUDE_HEADER})
else()
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND cat ${SLEEF_ORG_FOOTER} >> ${SLEEF_INCLUDE_HEADER})
endif()

add_custom_command(OUTPUT ${SLEEF_INCLUDE_HEADER}
  ${SLEEF_HEADER_COMMANDS}
  DEPENDS 
    ${SLEEF_ORG_HEADER}  
    ${SLEEF_ORG_FOOTER}
    ${TARGET_MKRENAME}
)

# --------------------------------------------------------------------
# TARGET_MKRENAME
# renameXXX.h for each SIMD
# --------------------------------------------------------------------
# Helper executable: generates parts of the sleef header file
add_executable(${TARGET_MKRENAME} mkrename.c)

set(HEADER_FILES_GENERATED "")
foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
  if(COMPILER_SUPPORTS_${SIMD})
    # Need lowercase string for rename header
    string(TOLOWER ${SIMD} vecarch)
    set(OBJECT_${SIMD} "sleef${vecarch}")
    set(HEADER_${SIMD} ${CMAKE_CURRENT_BINARY_DIR}/include/rename${vecarch}.h)
    list(APPEND HEADER_FILES_GENERATED ${HEADER_${SIMD}})

    # Generate mkrename commands
    add_custom_command(OUTPUT ${HEADER_${SIMD}}
      COMMAND echo Generating rename${vecarch}.h: ${TARGET_MKRENAME} ${RENAME_PARAMS_${SIMD}}
      COMMAND ${TARGET_MKRENAME} ${RENAME_PARAMS_${SIMD}} > ${HEADER_${SIMD}}
      DEPENDS ${TARGET_MKRENAME}
    )
    # set_source_files_properties(${HEADER_${SIMD}} PROPERTIES GENERATED TRUE)
  endif()
endforeach()

# --------------------------------------------------------------------
# TARGET_MKRENAME_GNUABI
# renameXXX_gnuabi.h for each SIMD GNU Abi
# --------------------------------------------------------------------
# Helper executable: generates parts of the sleef header file gnu_abi
add_executable(${TARGET_MKRENAME_GNUABI} mkrename_gnuabi.c)

set(HEADER_GNUABI_FILES_GENERATED "")
if(ENABLE_GNUABI)
  foreach(SIMD ${SLEEF_SUPPORTED_GNUABI_EXTENSIONS})
    if(COMPILER_SUPPORTS_${SIMD})
      string(TOLOWER ${SIMD} vecarch)
      set(OBJECT_${SIMD}_GNUABI "sleefgnuabi${vecarch}")    
      set(HEADER_${SIMD}_GNUABI ${CMAKE_CURRENT_BINARY_DIR}/include/rename${vecarch}_gnuabi.h)  
      list(APPEND HEADER_GNUABI_FILES_GENERATED ${HEADER_${SIMD}_GNUABI})

      # Generate mkrename_gnuabi commands
      add_custom_command(OUTPUT ${HEADER_${SIMD}_GNUABI}
        COMMAND echo Generating rename${vecarch}_gnuabi.h: ${TARGET_MKRENAME_GNUABI} ${RENAME_PARAMS_GNUABI_${SIMD}}  
        COMMAND ${TARGET_MKRENAME_GNUABI} ${RENAME_PARAMS_GNUABI_${SIMD}} > ${HEADER_${SIMD}_GNUABI}
        DEPENDS ${TARGET_MKRENAME_GNUABI}      
      )
      # set_source_files_properties(${HEADER_${SIMD}_GNUABI} PROPERTIES GENERATED TRUE)
    endif()
  endforeach()
endif()

# --------------------------------------------------------------------
# TARGET_HEADERS
# --------------------------------------------------------------------
add_custom_target(${TARGET_HEADERS} ALL
  DEPENDS 
    ${SLEEF_INCLUDE_HEADER}            # Output only
    ${HEADER_FILES_GENERATED}          # Output only
    ${HEADER_GNUABI_FILES_GENERATED}   # Output only
)

# --------------------------------------------------------------------
# TARGET_MKDISP
# --------------------------------------------------------------------
# Helper executable: dispatcher for the vector extensions
add_executable(${TARGET_MKDISP} mkdisp.c)

# Set C standard requirement (-std=gnu99 for gcc)
set_target_properties(
  ${TARGET_MKRENAME} ${TARGET_MKRENAME_GNUABI} ${TARGET_MKDISP} PROPERTIES
  C_STANDARD 99
)

# --------------------------------------------------------------------
# TARGET_LIBSLEEF
# --------------------------------------------------------------------
# Build main library

set(COMMON_TARGET_PROPERTIES 
POSITION_INDEPENDENT_CODE ON   # -fPIC
C_STANDARD 99                  # -std=gnu99
)

# Original sleef sources
set(STANDARD_SOURCES sleefdp.c sleefsp.c)

# Check for different precision support and add sources accordingly
if(COMPILER_SUPPORTS_LONG_DOUBLE)
  list(APPEND STANDARD_SOURCES sleefld.c)
endif()

add_library(${TARGET_LIBSLEEF} SHARED ${STANDARD_SOURCES})
set_target_properties(${TARGET_LIBSLEEF} PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION}
  ${COMMON_TARGET_PROPERTIES}
)

target_compile_definitions(${TARGET_LIBSLEEF} 
  PRIVATE DORENAME=1
)

if(COMPILER_SUPPORTS_FLOAT128)
  # TODO: Not supported for LLVM bitcode gen as it has a specific compilation flags
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefqp.c)
  target_compile_definitions(${TARGET_LIBSLEEF}
    PRIVATE ENABLEFLOAT128=1)
endif()

# Compile SIMD versions
# Single precision and double precision
set(SIMD_SOURCES sleefsimdsp.c sleefsimddp.c)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
  set(SIMD_SOURCES sleefsimdsp.c)
endif()

# Include symbols for each SIMD architecture (if supported by the platform)
# Note: adds object file as sources via cmake conditional generator expression
foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
  if(COMPILER_SUPPORTS_${SIMD})
    # Create a library
    add_library(${OBJECT_${SIMD}} OBJECT ${SIMD_SOURCES} ${HEADER_${SIMD}})    
    target_compile_definitions(${OBJECT_${SIMD}} PRIVATE
      ENABLE_${SIMD}=1
      DORENAME=1
    )

    set_target_properties(${OBJECT_${SIMD}} PROPERTIES
      ${COMMON_TARGET_PROPERTIES}
    )    

    target_compile_options(${OBJECT_${SIMD}} PRIVATE
      ${FLAGS_ENABLE_${SIMD}})

    target_sources(${TARGET_LIBSLEEF} PRIVATE $<TARGET_OBJECTS:${OBJECT_${SIMD}}>)
  endif(COMPILER_SUPPORTS_${SIMD})
endforeach()

# On some systems we need to explicitly link libsleef against libm to
# use some of the math functions used in the scalar code (for example
# sqrt).
if(LIBM)
  target_link_libraries(${TARGET_LIBSLEEF} ${LIBM})
endif(LIBM)


# --------------------------------------------------------------------
# TARGET_LIBSLEEFGNUABI
# Compile SIMD versions for GNU Abi
# --------------------------------------------------------------------
# Build gnuabi version from just simd object files
if(ENABLE_GNUABI)
  set(TARGET_LIBSLEEFGNUABI_OBJECTS "")
  foreach(SIMD ${SLEEF_SUPPORTED_GNUABI_EXTENSIONS})
    if(COMPILER_SUPPORTS_${SIMD})
      # Need lowercase string for rename header
      string(TOLOWER ${SIMD} vecarch)

      add_library(${OBJECT_${SIMD}_GNUABI} OBJECT ${SIMD_SOURCES} ${HEADER_${SIMD}_GNUABI})
      target_compile_definitions(${OBJECT_${SIMD}_GNUABI} PRIVATE
        ENABLE_${SIMD}=1
        DORENAME=1
        ENABLE_GNUABI=1
      )

      set_target_properties(${OBJECT_${SIMD}_GNUABI} PROPERTIES
        ${COMMON_TARGET_PROPERTIES}
      )    

      target_compile_options(${OBJECT_${SIMD}_GNUABI} PRIVATE ${FLAGS_ENABLE_${SIMD}})
      if (COMPILER_SUPPORTS_WEAK_ALIASES)
        target_compile_options(${OBJECT_${SIMD}_GNUABI} PRIVATE -DENABLE_GNUABI=1)
      endif(COMPILER_SUPPORTS_WEAK_ALIASES)
      list(APPEND TARGET_LIBSLEEFGNUABI_OBJECTS $<TARGET_OBJECTS:${OBJECT_${SIMD}_GNUABI}>)
    endif(COMPILER_SUPPORTS_${SIMD})
  endforeach()

  # Create library 
  add_library(${TARGET_LIBSLEEFGNUABI} SHARED ${TARGET_LIBSLEEFGNUABI_OBJECTS})
  
  # Library properties
  set_target_properties(${TARGET_LIBSLEEFGNUABI} PROPERTIES
    VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
    SOVERSION ${SLEEF_SOVERSION}
    POSITION_INDEPENDENT_CODE ON   # -fPIC
    C_STANDARD 99                  # -std=gnu99
  ) 

# On some systems we need to explicitly link libsleefgnuabi against
# libm to use some of the math functions used in the scalar code (for
# example sqrt).
if(LIBM)
  target_link_libraries(${TARGET_LIBSLEEFGNUABI} ${LIBM})
endif(LIBM)
endif(ENABLE_GNUABI)

# --------------------------------------------------------------------
# TARGET_LLVM_BITCODE
# Generate LLVM bitcode
# --------------------------------------------------------------------
if(CLANG_EXE_PATH AND SLEEF_ENABLE_LLVM_BITCODE)
  set(SLEEP_LLVM_BITCODE_INCLUDES "") 
  get_property(SLEEP_LLVM_BITCODE_INCLUDES_LIST DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  foreach(INCLUDE_DIRECTORY ${SLEEP_LLVM_BITCODE_INCLUDES_LIST})
    set(SLEEP_LLVM_BITCODE_INCLUDES "${SLEEP_LLVM_BITCODE_INCLUDES} -I ${INCLUDE_DIRECTORY}")
  endforeach()

  separate_arguments(SLEEP_LLVM_BITCODE_INCLUDES_CLANG WINDOWS_COMMAND "${SLEEP_LLVM_BITCODE_INCLUDES}")
  set(SLEEF_CLANG_LLVM_BITCODE_OPTIONS -O3 -S -emit-llvm -D NDEBUG -D DORENAME=1)
  set(LLVM_BITCODE_OUTPUTS "")
  
  # Generate LLVM bitcode for regular SLEEF
  foreach(STANDARD_SOURCE ${STANDARD_SOURCES})
    get_filename_component(SRC_WITHOUT_EXT ${STANDARD_SOURCE} NAME_WE)
    set(LLVM_BITCODE_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_WITHOUT_EXT}.c)
    set(LLVM_BITCODE_OUTPUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${SRC_WITHOUT_EXT}.ll)
    add_custom_command(OUTPUT ${LLVM_BITCODE_OUTPUT}
      COMMAND ${CLANG_EXE_PATH} ${SLEEF_CLANG_LLVM_BITCODE_OPTIONS} -o ${LLVM_BITCODE_OUTPUT} ${LLVM_BITCODE_INPUT} ${SLEEP_LLVM_BITCODE_INCLUDES_CLANG}
      DEPENDS
        ${LLVM_BITCODE_INPUT}
    )
    list(APPEND LLVM_BITCODE_OUTPUTS ${LLVM_BITCODE_OUTPUT})
  endforeach()

  # Generate LLVM bitcode for SIMD SLEEF
  foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
    if(COMPILER_SUPPORTS_${SIMD})
      foreach(SIMD_SOURCE ${SIMD_SOURCES})
        get_filename_component(SIMD_SOURCE_WITHOUT_EXT ${SIMD_SOURCE} NAME_WE)
        set(LLVM_BITCODE_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/${SIMD_SOURCE})
        set(LLVM_BITCODE_OUTPUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${SIMD_SOURCE_WITHOUT_EXT}_${SIMD}.ll)
        add_custom_command(OUTPUT ${LLVM_BITCODE_OUTPUT}
          COMMAND  ${CLANG_EXE_PATH} ${CLANG_FLAGS_ENABLE_${SIMD}} ${SLEEF_CLANG_LLVM_BITCODE_OPTIONS} -D ENABLE_${SIMD}=1 -o ${LLVM_BITCODE_OUTPUT} ${LLVM_BITCODE_INPUT} ${SLEEP_LLVM_BITCODE_INCLUDES_CLANG}
          DEPENDS
            ${LLVM_BITCODE_INPUT}
        )
        list(APPEND LLVM_BITCODE_OUTPUTS ${LLVM_BITCODE_OUTPUT})
      endforeach()
    endif()
  endforeach()

  file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_custom_target(${TARGET_LLVM_BITCODE} ALL
    DEPENDS
      ${LLVM_BITCODE_OUTPUTS}
  )
  add_dependencies(${TARGET_LLVM_BITCODE} ${TARGET_HEADERS})

  install(FILES ${LLVM_BITCODE_OUTPUTS} DESTINATION lib)
endif()

# --------------------------------------------------------------------
# Install
# --------------------------------------------------------------------
# Install libsleef and sleef.h
install(FILES ${SLEEF_INCLUDE_HEADER} DESTINATION include)
install(TARGETS ${TARGET_LIBSLEEF} DESTINATION lib)

if(ENABLE_GNUABI)
  install(TARGETS ${TARGET_LIBSLEEFGNUABI} DESTINATION lib)
endif()
