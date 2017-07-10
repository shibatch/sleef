set(LIBSLEEF_SRCS sleefdp.c sleefsp.c)
add_library("${TARGET_LIBSLEEF}" SHARED ${LIBSLEEF_SRCS})
target_compile_options(${TARGET_LIBSLEEF} PUBLIC -ffp-contract=off)
