include_directories("common")
include_directories("arch")

add_subdirectory("libm")
# We might not have MPFR installed, so skip it if not found and report a message
if(LIB_MPFR)
    add_subdirectory("libm-tester")
else()
    if(MSVC)
        message(STATUS "The library MPFR is currently not available with MSVC build - skip building tests libm-tester")
    else()
        message(FATAL_ERROR "The library MPFR is required for testing (e.g install `sudo apt-get install libmpfr-dev`)")
    endif()
endif()
add_subdirectory("common")

add_subdirectory("dft")
add_subdirectory("dft-tester")
