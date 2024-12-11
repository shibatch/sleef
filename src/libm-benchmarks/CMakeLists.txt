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
set(BENCH_HPP_SRC_FILES "benchmark_callers.hpp" "benchmark_templates.hpp" "gen_input.hpp" "type_defs.hpp")
set(BENCH_PROPERTIES C_STANDARD 99 CXX_STANDARD 17)
set(GOOGLE_BENCH_LIBS benchmark Threads::Threads) # Link Google Benchmark to the project

# Add source to this project's executable.
add_executable (benchsleef128 "benchsleef.cpp" ${BENCH_HPP_SRC_FILES})
set_target_properties(benchsleef128 PROPERTIES ${BENCH_PROPERTIES})
target_compile_options(benchsleef128 PRIVATE ${EXTRA_CFLAGS} -march=native)
target_link_libraries(benchsleef128 sleef ${GOOGLE_BENCH_LIBS})
add_dependencies(benchsleef128 googlebenchmark)

if(SLEEF_BUILD_BENCH_REF)
	add_executable (benchlibm128 "benchlibm.cpp" ${BENCH_HPP_SRC_FILES})
	set_target_properties(benchlibm128 PROPERTIES ${BENCH_PROPERTIES})
	target_compile_options(benchlibm128 PRIVATE ${EXTRA_CFLAGS} -march=native "-DBENCH_LIBM")
	target_link_libraries(benchlibm128 ${GOOGLE_BENCH_LIBS})
	add_dependencies(benchlibm128 googlebenchmark)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
	add_executable (benchsleef256 "benchsleef.cpp" ${BENCH_HPP_SRC_FILES})
	set_target_properties(benchsleef256 PROPERTIES ${BENCH_PROPERTIES})
	target_compile_options(benchsleef256 PRIVATE ${EXTRA_CFLAGS} "-march=native" "-DARCH_VECT_LEN=256")
	target_link_libraries(benchsleef256 sleef ${GOOGLE_BENCH_LIBS})
	add_dependencies(benchsleef256 googlebenchmark)

	add_executable (benchsleef512 "benchsleef.cpp" ${BENCH_HPP_SRC_FILES})
	set_target_properties(benchsleef512 PROPERTIES ${BENCH_PROPERTIES})
	target_compile_options(benchsleef512 PRIVATE ${EXTRA_CFLAGS} "-mavx512f" "-DARCH_VECT_LEN=512")
	target_link_libraries(benchsleef512 sleef ${GOOGLE_BENCH_LIBS})
	add_dependencies(benchsleef512 googlebenchmark)
endif()