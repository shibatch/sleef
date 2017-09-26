file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# --------------------------------------------------------------------
# sleef.h
# --------------------------------------------------------------------
# File generated for the headers
set(SLEEF_ORG_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm.h.org)
set(SLEEF_INCLUDE_HEADER ${CMAKE_BINARY_DIR}/include/sleef.h)

set(SLEEF_HEADER_COMMANDS "")
list(APPEND SLEEF_HEADER_COMMANDS COMMAND cmake -E copy ${SLEEF_ORG_HEADER} ${SLEEF_INCLUDE_HEADER})
foreach(SIMD ${SLEEF_HEADER_LIST})
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND echo Generating sleef.h: ${TARGET_MKRENAME} ${HEADER_PARAMS_${SIMD}})
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND ${TARGET_MKRENAME} ${HEADER_PARAMS_${SIMD}} >> ${SLEEF_INCLUDE_HEADER})
endforeach()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/sleef_footer.h "#undef IMPORT\n#endif\n")
if(MSVC)
  string(REPLACE "/" "\\" sleef_footer_input_file "${CMAKE_CURRENT_BINARY_DIR}/sleef_footer.h")
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND type ${sleef_footer_input_file} >> ${SLEEF_INCLUDE_HEADER})
else()
  list(APPEND SLEEF_HEADER_COMMANDS COMMAND cat ${CMAKE_CURRENT_BINARY_DIR}/sleef_footer.h >> ${SLEEF_INCLUDE_HEADER})
endif()

add_custom_command(OUTPUT ${SLEEF_INCLUDE_HEADER}
  ${SLEEF_HEADER_COMMANDS}
  DEPENDS 
    ${SLEEF_ORG_HEADER}  
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
if(SLEEF_ENABLE_GNUABI)
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
add_library(${TARGET_LIBSLEEF} SHARED sleefdp.c sleefsp.c)

set(COMMON_TARGET_PROPERTIES 
  POSITION_INDEPENDENT_CODE ON   # -fPIC
  C_STANDARD 99                  # -std=gnu99
)

set_target_properties(${TARGET_LIBSLEEF} PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION}
  ${COMMON_TARGET_PROPERTIES}
)



target_compile_definitions(${TARGET_LIBSLEEF} 
  PRIVATE DORENAME=1
)

# Check for different precision support and add sources accordingly
if(COMPILER_SUPPORTS_LONG_DOUBLE)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefld.c)
endif()

if(COMPILER_SUPPORTS_FLOAT128)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefqp.c)
  target_compile_definitions(${TARGET_LIBSLEEF}
    PRIVATE ENABLEFLOAT128=1)
endif()

# Compile SIMD versions
# Single precision and double precision
set(SIMD_SOURCES sleefsimdsp.c sleefsimddp.c)

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

# --------------------------------------------------------------------
# TARGET_LIBSLEEFGNUABI
# Compile SIMD versions for GNU Abi
# --------------------------------------------------------------------
# Build gnuabi version from just simd object files
if(SLEEF_ENABLE_GNUABI)
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
endif(SLEEF_ENABLE_GNUABI)

# --------------------------------------------------------------------
# Install
# --------------------------------------------------------------------
# Install libsleef and sleef.h
install(FILES ${SLEEF_INCLUDE_HEADER} DESTINATION include)
install(TARGETS ${TARGET_LIBSLEEF} DESTINATION lib)

if(SLEEF_ENABLE_GNUABI)
  install(TARGETS ${TARGET_LIBSLEEFGNUABI} DESTINATION lib)
endif()
