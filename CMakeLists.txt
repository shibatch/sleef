# See doc/build-with-cmake.md for instructions on how to build Sleef.
cmake_minimum_required(VERSION 3.4.3)

set(SLEEF_VERSION_MAJOR 3)
set(SLEEF_VERSION_MINOR 0)

project(SLEEF
	VERSION ${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}
	LANGUAGES C CXX)

message(STATUS "Configuring build for ${PROJECT_NAME}-version${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}")

# We like to have a documented index of all targets in the project. The
# variables listed below carry the names of the targets defined throughout
# the project.

# Library defined in src/libm/CMakeLists.txt via command add_library
# Outputs libsleef.so
set(TARGET_LIBSLEEF "sleef")


add_subdirectory("src")
