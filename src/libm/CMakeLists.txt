set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG")

set(LIBSLEEF_SRCS sleefdp.c sleefsp.c)

add_library("${TARGET_LIBSLEEF}" SHARED ${LIBSLEEF_SRCS})

# Always compile sleef with -ffp-contract and log at configuration time
message(STATUS "Using option `${STRICTMATHFLAG}` to compile libsleef")
target_compile_options(${TARGET_LIBSLEEF} PUBLIC ${STRICTMATHFLAG})

# Shared library versioning
set_target_properties(${TARGET_LIBSLEEF} PROPERTIES VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR} SOVERSION ${SLEEF_SOVERSION})
