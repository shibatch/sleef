include_directories("common")
include_directories("arch")

add_subdirectory("libm")
add_subdirectory("libm-tester")
add_subdirectory("common")

if (BUILD_DFT AND NOT MINGW)
  add_subdirectory("dft")
  add_subdirectory("dft-tester")
endif()
