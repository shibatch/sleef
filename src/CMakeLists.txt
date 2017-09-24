include_directories("common")
include_directories("arch")

add_subdirectory("libm")
# We might not have MPFR installed, so skipt it if not found instead of crashing
if(LIB_MPFR)
    add_subdirectory("libm-tester")
else()
    message(STATUS "The library MPFR used for testing was not found - skipping building tests libm-tester")
endif()
add_subdirectory("common")
