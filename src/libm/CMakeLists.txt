set(LIBSLEEF_SRCS sleefdp.c sleefsp.c)

if (ENABLE_LONG_DOUBLE)
  set(LIBSLEEF_SRCS ${LIBSLEEF_SRCS} sleefld.c)
endif(ENABLE_LONG_DOUBLE)

if (ENABLE_FLOAT128)
  set(LIBSLEEF_SRCS ${LIBSLEEF_SRCS} sleefqp.c)
endif(ENABLE_FLOAT128)

add_library("${TARGET_LIBSLEEF}" SHARED ${LIBSLEEF_SRCS})

# Always compile sleef with -ffp-contract and log at configuration time
message(STATUS "Using option `${STRICTMATHFLAG}` to compile libsleef")
target_compile_options(${TARGET_LIBSLEEF} PUBLIC ${STRICTMATHFLAG})

# CMake supports soname-based version number encoding on platforms supporting soname natively
# libsleef version number currently follows the MAJOR version of the SLEEF project
set_target_properties(${TARGET_LIBSLEEF} PROPERTIES VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR} SOVERSION ${SLEEF_SOVERSION})
