include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Helper executable: generates parts of the sleef header file
add_executable(${TARGET_MKRENAME} mkrename.c)
add_executable(${TARGET_MKRENAME_GNUABI} mkrename_gnuabi.c)
# Helper executable: dispatcher for the vector extensions
add_executable(${TARGET_MKDISP} mkdisp.c)
# Set C standard requirement (-std=gnu99 for gcc)
set_target_properties(
  ${TARGET_MKRENAME} ${TARGET_MKRENAME_GNUABI} ${TARGET_MKDISP} PROPERTIES
  C_STANDARD 99)

# Custom target to build headers
add_custom_target(${TARGET_HEADERS})

# Build main library
# Single precision and double precision
set(SLEEF_SOURCES sleefdp.c sleefsp.c)
add_library(${TARGET_LIBSLEEF} SHARED ${SLEEF_SOURCES})
add_dependencies(${TARGET_LIBSLEEF} ${TARGET_HEADERS})

# Check for different precision support and add sources accordingly
if(COMPILER_SUPPORTS_LONG_DOUBLE)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefld.c)
  set(SLEEF_SOURCES ${SLEEF_SOURCES} sleefld.c)
endif(COMPILER_SUPPORTS_LONG_DOUBLE)

if(COMPILER_SUPPORTS_FLOAT128)
  # TODO: can't add sleefqp.c to SLEEF_SOURCES as it requires special compile definitions (for LLVM bitcode gen)
  target_sources(${TARGET_LIBSLEEF} PRIVATE sleefqp.c)
  target_compile_definitions(${TARGET_LIBSLEEF}
    PRIVATE ENABLEFLOAT128=1)
endif(COMPILER_SUPPORTS_FLOAT128)

# --------------------------------------------------------------------
# Compile SIMD versions, generate renameXXX.h files
# --------------------------------------------------------------------
set(TARGET_OBJECTS "")
# Single precision and double precision
set(SIMD_SOURCES sleefsimdsp.c sleefsimddp.c)

# Repeat for each architecture: configure header file and compile object file
macro(compile_object SIMD)
  if(COMPILER_SUPPORTS_${SIMD})
    # Need lowercase string for rename header
    string(TOLOWER ${SIMD} vecarch)

    set(OBJECT_${SIMD} "sleef${vecarch}")
    set(HEADER_${SIMD} ${CMAKE_CURRENT_BINARY_DIR}/include/rename${vecarch}.h)

    # Ensure that the renameXXX.h file exists so that add_custom_command mkrename will work
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/)
    file(WRITE ${HEADER_${SIMD}} "")
    
    # Create a library
    add_library(${OBJECT_${SIMD}} OBJECT ${SIMD_SOURCES} ${HEADER_${SIMD}})    
    target_compile_definitions(${OBJECT_${SIMD}} PRIVATE
      ENABLE_${SIMD}=1
      DORENAME=1)

    add_dependencies(${OBJECT_${SIMD}} ${TARGET_HEADERS})

    target_compile_options(${OBJECT_${SIMD}} PRIVATE
      ${FLAGS_ENABLE_${SIMD}})

    # Generate mkrename commands
    add_custom_command(TARGET ${TARGET_HEADERS} 
      COMMAND echo Generating rename${vecarch}.h: ${TARGET_MKRENAME} ${RENAME_PARAMS_${SIMD}}
      COMMAND ${TARGET_MKRENAME} ${RENAME_PARAMS_${SIMD}} > ${HEADER_${SIMD}}
    )

    # Store the generate object/headers
    list(APPEND TARGET_OBJECTS ${OBJECT_${SIMD}})
  endif(COMPILER_SUPPORTS_${SIMD})
endmacro(compile_object)

# Include symbols for each SIMD architecture (if supported by the platform)
# Note: adds object file as sources via cmake conditional generator expression
foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
  compile_object(${SIMD})
  target_sources(${TARGET_LIBSLEEF} PRIVATE
    $<$<BOOL:${COMPILER_SUPPORTS_${SIMD}}>:$<TARGET_OBJECTS:${OBJECT_${SIMD}}>>)
endforeach()

# --------------------------------------------------------------------
# Compile SIMD versions for GNU Abi, generate renameXXX_gnuabi.h files
# --------------------------------------------------------------------
macro(compile_object_gnuabi SIMD)
  if(COMPILER_SUPPORTS_${SIMD})
    # Need lowercase string for rename header
    string(TOLOWER ${SIMD} vecarch)

    set(OBJECT_${SIMD}_GNUABI "sleefgnuabi${vecarch}")
    set(HEADER_${SIMD}_GNUABI ${CMAKE_CURRENT_BINARY_DIR}/include/rename${vecarch}_gnuabi.h)

    # Ensure that the renameXXX.h file exists so that add_custom_command mkrename will work
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/)
    file(WRITE ${HEADER_${SIMD}_GNUABI} "")

    add_library(${OBJECT_${SIMD}_GNUABI} OBJECT ${SIMD_SOURCES} ${HEADER_${SIMD}_GNUABI})
    target_compile_definitions(${OBJECT_${SIMD}_GNUABI} PRIVATE
      ENABLE_${SIMD}=1
      DORENAME=1
      ENABLE_GNUABI=1)
    target_compile_options(${OBJECT_${SIMD}_GNUABI} PRIVATE
      ${FLAGS_ENABLE_${SIMD}})

    add_dependencies(${OBJECT_${SIMD}_GNUABI} ${TARGET_HEADERS})

    # Generate mkrename_gnuabi commands
    add_custom_command(TARGET ${TARGET_HEADERS} 
      COMMAND echo Generating rename${vecarch}_gnuabi.h: ${TARGET_MKRENAME_GNUABI} ${RENAME_PARAMS_GNUABI_${SIMD}}  
      COMMAND ${TARGET_MKRENAME_GNUABI} ${RENAME_PARAMS_GNUABI_${SIMD}} > ${HEADER_${SIMD}_GNUABI}
    )

    list(APPEND TARGET_OBJECTS_GNUABI ${OBJECT_${SIMD}_GNUABI})
  endif(COMPILER_SUPPORTS_${SIMD})
endmacro(compile_object_gnuabi)

# Build gnuabi version from just simd object files
if(SLEEF_ENABLE_GNUABI)
  set(SLEEF_SUPPORTED_GNUABI_EXTENSIONS SSE2 AVX AVX2 AVX512F ADVSIMD)
  foreach(SIMD ${SLEEF_SUPPORTED_GNUABI_EXTENSIONS})
    compile_object_gnuabi(${SIMD})
    list(APPEND GNUABI_OBJ_SRC
      $<$<BOOL:${COMPILER_SUPPORTS_${SIMD}}>:$<TARGET_OBJECTS:${OBJECT_${SIMD}_GNUABI}>>)
  endforeach()

  add_library(${TARGET_LIBSLEEFGNUABI} SHARED ${GNUABI_OBJ_SRC})

  set_target_properties(${TARGET_LIBSLEEFGNUABI} PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION}
  C_STANDARD 99)
  
  add_dependencies(${TARGET_LIBSLEEFGNUABI} ${TARGET_HEADERS})
  
endif(SLEEF_ENABLE_GNUABI)

# All code that goes into the main library needs to be position independent code
set_target_properties(${TARGET_OBJECTS} ${TARGET_OBJECTS_GNUABI} PROPERTIES
  POSITION_INDEPENDENT_CODE ON   # -fPIC
  C_STANDARD 99)                 # -std=gnu99

target_compile_definitions(${TARGET_LIBSLEEF}
  PRIVATE DORENAME=1)
set_target_properties(${TARGET_LIBSLEEF} PROPERTIES
  VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
  SOVERSION ${SLEEF_SOVERSION}
  C_STANDARD 99)

# --------------------------------------------------------------------
# Generate include/sleef.h header file
# --------------------------------------------------------------------
set(SLEEF_ORG_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/sleeflibm.h.org)
set(SLEEF_GEN_HEADER ${CMAKE_BINARY_DIR}/include/sleef.h)

add_custom_command(TARGET ${TARGET_HEADERS} 
  COMMAND cmake -E copy ${SLEEF_ORG_HEADER} ${SLEEF_GEN_HEADER}
)

foreach(SIMD ${SLEEF_HEADER_LIST})
  add_custom_command(TARGET ${TARGET_HEADERS} 
    COMMAND echo Generating sleef.h: ${TARGET_MKRENAME} ${HEADER_PARAMS_${SIMD}}
    COMMAND ${TARGET_MKRENAME} ${HEADER_PARAMS_${SIMD}} >> ${SLEEF_GEN_HEADER}
  )
endforeach()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/sleef_footer.h "#undef IMPORT\n#endif\n")
if(MSVC)
  string(REPLACE "/" "\\" sleef_footer_input_file "${CMAKE_CURRENT_BINARY_DIR}/sleef_footer.h")
  add_custom_command(TARGET ${TARGET_HEADERS} 
    COMMAND type ${sleef_footer_input_file} >> ${SLEEF_GEN_HEADER}
  )
else()
  add_custom_command(TARGET ${TARGET_HEADERS} 
    COMMAND cat ${CMAKE_CURRENT_BINARY_DIR}/sleef_footer.h >> ${SLEEF_GEN_HEADER}
  )
endif()

# --------------------------------------------------------------------
# Generate LLVM bitcode
# --------------------------------------------------------------------
if(CLANG_EXE_PATH AND SLEEF_ENABLE_LLVM_BITCODE)
  set(SLEEF_LLVM_BITCODE_INCLUDES "") 
  get_property(SLEEF_LLVM_BITCODE_INCLUDES_LIST DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  foreach(INCLUDE_DIRECTORY ${SLEEF_LLVM_BITCODE_INCLUDES_LIST})
    set(SLEEF_LLVM_BITCODE_INCLUDES "${SLEEF_LLVM_BITCODE_INCLUDES} -I ${INCLUDE_DIRECTORY}")
  endforeach()

  separate_arguments(SLEEF_LLVM_BITCODE_INCLUDES_CLANG WINDOWS_COMMAND "${SLEEF_LLVM_BITCODE_INCLUDES}")
  file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_custom_target(${TARGET_LLVM_BITCODE} ALL)
  add_dependencies(${TARGET_LLVM_BITCODE} ${TARGET_HEADERS})

  set(SLEEF_CLANG_LLVM_BITCODE_OPTIONS -O3 -S -emit-llvm -D NDEBUG -D DORENAME=1)

  # Generate LLVM bitcode for regular SLEEF
  foreach(SLEEF_SRC ${SLEEF_SOURCES})
    get_filename_component(SRC_WITHOUT_EXT ${SLEEF_SRC} NAME_WE)
    add_custom_command(TARGET ${TARGET_LLVM_BITCODE}
      COMMAND ${CLANG_EXE_PATH} ${SLEEF_CLANG_LLVM_BITCODE_OPTIONS} -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${SRC_WITHOUT_EXT}.ll ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_WITHOUT_EXT}.c ${SLEEF_LLVM_BITCODE_INCLUDES_CLANG}
    )
  endforeach()

  # Generate LLVM bitcode for SIMD SLEEF
  foreach(SIMD ${SLEEF_SUPPORTED_EXTENSIONS})
    if(COMPILER_SUPPORTS_${SIMD})
      foreach(SIMD_SOURCE ${SIMD_SOURCES})
        get_filename_component(SIMD_SOURCE_WITHOUT_EXT ${SIMD_SOURCE} NAME_WE)
        add_custom_command(TARGET ${TARGET_LLVM_BITCODE}
          COMMAND  ${CLANG_EXE_PATH} ${CLANG_FLAGS_ENABLE_${SIMD}} ${SLEEF_CLANG_LLVM_BITCODE_OPTIONS} -D ENABLE_${SIMD}=1 -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${SIMD_SOURCE_WITHOUT_EXT}_${SIMD}.ll ${CMAKE_CURRENT_SOURCE_DIR}/${SIMD_SOURCE} ${SLEEF_LLVM_BITCODE_INCLUDES_CLANG}
        )
      endforeach()
    endif()
  endforeach()
endif()

# Install libsleef and sleef.h
install(FILES ${SLEEF_INCLUDE_HEADER}
  DESTINATION include)
if(SLEEF_ENABLE_GNUABI)
  install(TARGETS ${TARGET_LIBSLEEF} ${TARGET_LIBSLEEFGNUABI}
  DESTINATION lib)
else()
  install(TARGETS ${TARGET_LIBSLEEF} DESTINATION lib)
endif(SLEEF_ENABLE_GNUABI)
