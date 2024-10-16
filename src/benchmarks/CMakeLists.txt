# CMakeList.txt : Microbenchmarks with google bench
cmake_minimum_required (VERSION 3.11)

project ("SLEEF Microbenchmarks")

find_package(Threads)
# Apply CMake options in Installation guide in https://github.com/google/benchmark
include(ExternalProject)
find_package(Git REQUIRED)
ExternalProject_Add(googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG origin/main
  CMAKE_ARGS -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON
             -DCMAKE_BUILD_TYPE=Release
             -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/googlebench
             -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
)
include_directories(${CMAKE_BINARY_DIR}/googlebench/include)
link_directories(${CMAKE_BINARY_DIR}/googlebench/lib)

# include headers
include_directories(${sleef_BINARY_DIR}/include)   # sleef.h
# include libs
link_directories(${sleef_BINARY_DIR}/lib)          # libsleef


set(EXTRA_CFLAGS -Wall -O2 -Wno-attributes)
set(BENCH_SRC_FILE "benchsleef.cpp" "benchmark_callers.hpp" "benchmark_templates.hpp" "gen_input.hpp" "type_defs.hpp")
set(BENCH_PROPERTIES C_STANDARD 99 CXX_STANDARD 17)
set(BENCH_LIBS benchmark sleef Threads::Threads) # Link Google Benchmark and sleef to the project

# Add source to this project's executable.
add_executable (benchsleef128 ${BENCH_SRC_FILE})
set_target_properties(benchsleef128 PROPERTIES ${BENCH_PROPERTIES})
target_compile_options(benchsleef128 PRIVATE ${EXTRA_CFLAGS} -march=native)
target_link_libraries(benchsleef128 ${BENCH_LIBS})
add_dependencies(benchsleef128 googlebenchmark)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
	add_executable (benchsleef256 ${BENCH_SRC_FILE})
	set_target_properties(benchsleef256 PROPERTIES ${BENCH_PROPERTIES})
	target_compile_options(benchsleef256 PRIVATE ${EXTRA_CFLAGS} "-march=native" "-DARCH_VECT_LEN=256")
	target_link_libraries(benchsleef256 ${BENCH_LIBS})
	add_dependencies(benchsleef256 googlebenchmark)

	add_executable (benchsleef512 ${BENCH_SRC_FILE})
	set_target_properties(benchsleef512 PROPERTIES ${BENCH_PROPERTIES})
	target_compile_options(benchsleef512 PRIVATE ${EXTRA_CFLAGS} "-mavx512f" "-DARCH_VECT_LEN=512")
	target_link_libraries(benchsleef512 ${BENCH_LIBS})
	add_dependencies(benchsleef512 googlebenchmark)
endif()