cmake_minimum_required(VERSION 3.4.3)
include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(libsleef
  GIT_REPOSITORY https://github.com/shibatch/sleef
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib
)

include_directories(${CMAKE_BINARY_DIR}/contrib/include)
link_directories(${CMAKE_BINARY_DIR}/contrib/lib)

add_executable(hellox86 hellox86.c)
add_dependencies(hellox86 libsleef)
target_link_libraries(hellox86 sleef)

#

option(SLEEF_BUILD_DFT_TUTORIAL "Build DFT tutorial" OFF)
if (SLEEF_BUILD_DFT_TUTORIAL)
  add_executable(dfttutorial tutorial.c)
  add_dependencies(dfttutorial libsleef)
  find_library(LIBM m)
  target_link_libraries(dfttutorial sleef sleefdft ${LIBM})
endif()
