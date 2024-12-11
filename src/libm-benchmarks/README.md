<h2 id="benchmark">Benchmarking tool</h2>

This tool uses the [googlebench](https://github.com/google/benchmark) framework to benchmark SLEEF
functions.
It is integrated with SLEEF via CMake.
In order to build this tool automatically when SLEEF is
built, pass the `-DSLEEF_BUILD_BENCH=ON` CMake option when
setting up the build directory:
```sh
cmake -S . -B build -DSLEEF_BUILD_BENCH=ON
```
After building SLEEF:
```sh
cmake --build build -j
```
in `build/bin` folder you will find an executable named
benchsleef128.
Run this executable with `./build/bin/benchsleef128` in
order to obtain microbenchmarks for the functions in the project.
A filter option can also be provided to the executable.
This feature in inherited from googlebench, and takes
a regular expression, and executes only the benchmarks
whose name matches the regular expression.
The set of all the benchmarks available can be obtained
when running the benchmark tool when no filter is set
and corresponds to all the benchmarks listed in
`benchsleef.cpp`.
```sh
# Examples:
# * This will benchmark Sleef_sinf_u10 on all intervals enabled in the tool.
./build/bin/benchsleef128 --benchmark_filter=sinf_u10
# * This will benchmark all single precision sin functions (scalar, vector and sve if available):
./build/bin/benchsleef128 --benchmark_filter=sinf
# * This will benchmark all single precision vector functions:
./build/bin/benchsleef128 --benchmark_filter=vectorf
```
Note: all corresponds to all functions available in SLEEF and enabled in the benchmarks in this context.

This tool also supports multiple output formats, a feature
also inherited from googlebench framework.
The output formats available are `console`(default), `json` and `csv`.
```sh
# Examples:
# * This will print output in the terminal in json format:
./build/bin/benchsleef128 --benchmark_format=json
```

<h3 id="benchmark">Benchmarking on aarch64</h3>
If you're running SLEEF on a machine with SVE support the executable generated will have SVE benchmarks
available for functions specified in `benchsleef.cpp`.
<h3 id="benchmark">Benchmarking on x86</h3>
If you're running SLEEF on an x86 machine, two extra
executables may be built (according to feature detection):

```sh
  ./build/bin/benchsleef256
  ./build/bin/benchsleef512
```

These will benchmark 256bit and 512bit vector implementations
for vector functions respectively.
Note these executables can also be used to benchmark scalar
functions.

<h3 id="benchmark">Benchmarking libm</h3>
This tool can also benchmark libm functions listed in 
`benchlibm.cpp`.
To enable this, pass on extra build option `-DSLEEF_BUILD_BENCH_REF=ON`
(on top of `-DSLEEF_BUILD_BENCH` from before).
This will produce a new script `./build/bin/benchlibm128`.
You can interact with this in a similar fashion as described for 
`./build/bin/benchsleef128`.

<h3 id="benchmark">Maintenance</h3>
Some functions are still not enabled in the benchmarks.
In order to add a function which uses the types already
declared in `type_defs.hpp`, add a benchmark entry using
the macros declared in `benchmark_callers.hpp`.
These macros have been designed to group benchmarking
patterns observed in the previous benchmarking system,
and minimize the number of lines of code while preserving
readability as much as possible.

Examples:

(1) If a scalar float lower ulp precision version of
log1p gets implemented at some point in SLEEF one could
add benchmarks for it by adding a line to `sleefbench.cpp`:
```cpp
BENCH(Sleef_log10f_u35, scalarf, <min>, <max>)
```
This line can be repeated to provide benchmarks on
multiple intervals.

(2) If the double precision of the function above gets
implemented as well then, we can simply add:
```cpp
BENCH_SCALAR(log10, u35, <min>, <max>)
```
which would be equivalent to adding:
```cpp
BENCH(Sleef_log10f_u35, scalarf, <min>, <max>)
BENCH(Sleef_log10_u35, scalard, <min>, <max>)
```
If the function you want to add does not use the types in
`type_defs.hpp`, extend this file with the types required
(and ensure type detection is implemented correctly).
Most likely you will also have to make some changes to
`gen_input.hpp`:
* Add adequate declaration for `vector_len`:
```cpp
template <> const inline int vector_len<new_type> = *;
```
* and add adequate template specialization for `gen_input()`:
```cpp
template <> newtype gen_input (double lo, double hi)
{ your implementation }
```
<h3 id="benchmark">Note</h3>
This tool can also be built as a standalone project.
From `sleef/src/libm-benchmarks` directory, run:
```sh
cmake -S . -B build -Dsleef_BINARY_DIR=<build_dir>
cmake --build build -j
./build/benchsleef128
```
