cmake_minimum_required(VERSION 3.4.3)
project(MyProject)
include(ExternalProject)
find_package(Git REQUIRED)

option(SLEEF_BUILD_DFT_TUTORIAL "Build DFT tutorial" OFF)

ExternalProject_Add(libsleef
  GIT_REPOSITORY https://github.com/shibatch/sleef
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib -DSLEEF_BUILD_DFT=${SLEEF_BUILD_DFT_TUTORIAL} -DSLEEF_BUILD_SCALAR_LIB=ON -DSLEEF_BUILD_TESTS=OFF
)

include_directories(${CMAKE_BINARY_DIR}/contrib/include)
link_directories(${CMAKE_BINARY_DIR}/contrib/lib)

add_executable(hellox86 hellox86.c)
add_dependencies(hellox86 libsleef)
target_link_libraries(hellox86 sleef)

#

if (SLEEF_BUILD_DFT_TUTORIAL)
  add_executable(dfttutorial tutorial.c)
  add_dependencies(dfttutorial libsleef)
  find_library(LIBM m)
  target_link_libraries(dfttutorial sleef sleefdft sleefscalar ${LIBM} -fopenmp)
endif()
