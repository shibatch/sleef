---
layout: default
title: Discrete Fourier Transform
parent: References
permalink: /2-references/dft
---

<h1>DFT library reference</h1>

<h2>Table of contents</h2>

* [Tutorial](#tutorial)
* [Function reference](#reference)

<h2 id="tutorial">Tutorial</h2>

I now explain how to use this DFT library by referring to an example source
code shown below. [This source code](../../src/tutorial.c) is included in the
distribution package under src/dft-tester directory.

```c
// gcc tutorial.c -lsleef -lsleefdft -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>

#include "sleef.h"
#include "sleefdft.h"

#define THRES 1e-4

typedef double complex cmpl;

cmpl omega(double n, double kn) {
  return cexp((-2 * M_PI * _Complex_I / n) * kn);
}

void forward(cmpl *ts, cmpl *fs, int len) {
  for(int k=0;k<len;k++) {
    fs[k] = 0;
    for(int n=0;n<len;n++) fs[k] += ts[n] * omega(len, n*k);
  }
}

int main(int argc, char **argv) {
  int n = 256;
  if (argc == 2) n = 1 <&lt; atoi(argv[1]);

  SleefDFT_setPlanFilePath("plan.txt", NULL, SLEEF_PLAN_AUTOMATIC);

  double *sx = (double *)Sleef_malloc(n*2 * sizeof(double));
  double *sy = (double *)Sleef_malloc(n*2 * sizeof(double));

  struct SleefDFT *p = SleefDFT_double_init1d(n, sx, sy, SLEEF_MODE_FORWARD);

  if (p == NULL) {
    printf("SleefDFT initialization failed\n");
    exit(-1);
  }

  cmpl *ts = (cmpl *)malloc(sizeof(cmpl)*n);
  cmpl *fs = (cmpl *)malloc(sizeof(cmpl)*n);

  for(int i=0;i<n;i++) {
    ts[i] =
    (2.0 * (rand() / (double)RAND_MAX) - 1) * 1.0 +
    (2.0 * (rand() / (double)RAND_MAX) - 1) * _Complex_I;

  sx[(i*2+0)] = creal(ts[i]);
  sx[(i*2+1)] = cimag(ts[i]);
  }

  forward(ts, fs, n);

  SleefDFT_double_execute(p, NULL, NULL);

  int success = 1;

  for(int i=0;i<n;i++) {
    if ((fabs(sy[(i*2+0)] - creal(fs[i])) > THRES) ||
        (fabs(sy[(i*2+1)] - cimag(fs[i])) > THRES)) {
      success = 0;
    }
  }

  printf("%s\n", success ? "OK" : "NG");

  free(fs); free(ts);
  Sleef_free(sy); Sleef_free(sx);

  SleefDFT_dispose(p);

  exit(success);
}
```
<p style="text-align:center;">
Fig. 4.1: <a href="../../src/tutorial.c">Test code for DFT subroutines</a>
</p>

As shown in the first line, you can compile the source code with the following
command, after you install the library.

```sh
gcc tutorial.c -lsleef -lsleefdft -lm
```

This program takes one integer argument `n`. It executes forward complex
transform with size `2^n` using a naive transform and the library. If the two
results match, it prints OK.

For the first execution, this program takes a few seconds to finish. This is
because the library measures computation speed with many different
configurations to find the best execution plan. The best plan is saved to
"plan.txt", as specified in line 28. Later executions will finish instantly as
the library reads the plan from this file. Instead of specifying the file name
in the program, the file can be specified by `SLEEFDFTPLAN` environment
variable. Instead of constructing or loading a plan, the library can estimate a
modestly good configuration, if `SLEEF_MODE_ESTIMATE` flag is specified at line
30.

This library executes transforms using the most suitable SIMD instructions
available on the computer, in addition to multi-threading. In order to make the
computation efficient, the library requires the input and output arrays to be
aligned to some boundaries so that the data can be accessed with SIMD
instructions. By using `Sleef_malloc`, as seen in line 37 and 38, this
alignment is ensured. Memory allocated with `Sleef_malloc` has to be freed with
`Sleef_free`, as seen in line 68.  When a transform is executed, you need to
pass the pointer returned by `Sleef_malloc`. You can allocate an aligned memory
region yourself, and pass the pointer to the library.

The real and imaginary parts of the `k`th number are stored in (`2k`)-th and
(`2k+1`)-th elements of the input and output array, respectively. At line 54,
the transform is executed by the library. You can specify the same array as the
input and output.

Under `../src/dft-tester` directory, there are other examples showing how to
execute transforms in a way that you get equivalent results to other libraries.

<h2 id="reference">Function reference</h2>

### `Sleef_malloc`

Allocate aligned memory

```c
#include <stdlib.h>
#include <sleef.h>

void * Sleef_malloc (size_t z);
```

Link with `-lsleef`.

`Sleef_malloc` allocates `z` bytes of aligned memory region, and return the
pointer to that region. The returned pointer points an address that can be
accessed by all SIMD load and store instructions available on that computer.
Memory regions allocated by `Sleef_malloc` have to be freed with `Sleef_free`.

### `Sleef_free`

```c
#include <stdlib.h>
#include <sleef.h>

void Sleef_free (void *ptr);
```

Link with `-lsleef`.

A memory region pointed by `ptr` that is allocated by `Sleef_malloc` can be
freed with `Sleef_free`.

### `SleefDFT_setPlanFilePath`

Set the file path for storing execution plans

```c
#include <stdint.h>
#include <sleefdft.h>

void SleefDFT_setPlanFilePath (const char *path, const char *arch, uint64_t mode);
```

Link with `-lsleefdft -lsleef`.

File name for storing execution plan can be specified by this function. If NULL
is specified as `path`, the file name is read from SLEEFDFTPLAN environment
variable. A string for identifying system micro architecture can be also given.
The library will automatically detect the marchitecture if NULL is given as
`arch`. Management options for the plan file can be specified by the `mode`
parameter, as shown below.

<p style="text-align:center;">
Table 4.2: Mode flags for `SleefFT_setPlanFilePath`
</p>

| Flag                   | Meaning           |
|:----------------------:|:------------------|
| `SLEEF_PLAN_AUTOMATIC` | Execution plans are automatically loaded and saved. Plans are generated if it does not exist. |
| `SLEEF_PLAN_READONLY`  | Execution plans are automatically loaded, but not saved. |
| `SLEEF_PLAN_RESET`     | Existing execution plans are reset and constructed from the beginning. |

### `SleefDFT_double_init1d`
### `SleefDFT_float_init1d`

Initialize the tables for 1D transform

```c
#include <stdint.h>
#include <sleefdft.h>

struct SleefDFT * SleefDFT_double_init1d(uint32_t n, const double *in, double *out, uint64_t mode);
struct SleefDFT * SleefDFT_float_init1d(uint32_t n, const float *in, float *out, uint64_t mode);
```

Link with `-lsleefdft -lsleef`.

These functions generates and initializes the tables that is used for 1D
transform, and returns the pointer. Size of transform can be specified by `n`.
Currently, power-of-two sizes can be only specified. The list of the flags that
can be passed to `mode` is shown below.

<p style="text-align:center;">
Table 4.3: Mode flags for `SleefDFT_double_init`
</p>

| Flag                   | Meaning           |
|:----------------------:|:------------------|
| `SLEEF_MODE_FORWARD`   | Tables are initialized for forward transforms. |
| `SLEEF_MODE_BACKWARD`  | Tables are initialized for backward transforms. |
| `SLEEF_MODE_COMPLEX`   | Tables are initialized for complex transforms. |
| `SLEEF_MODE_REAL`      | Tables are initialized for real transforms. |
| `SLEEF_MODE_ALT`       | Tables are initialized for alternative real transforms. |
| `SLEEF_MODE_ESTIMATE`  | Execution plans are estimated. |
| `SLEEF_MODE_MEASURE`   | Execution plans are measured when they are needed. |
| `SLEEF_MODE_VERBOSE`   | Messages are displayed. |
| `SLEEF_MODE_NO_MT`     | Multithreading will be disabled in the computation for transforms. |

These functions return a pointer to the data that is used for 1D DFT
computation, or NULL if an error occurred.

### `SleefDFT_double_init2d`
### `SleefDFT_float_init2d`

Initialize the tables for 2D transform

```c
#include <stdint.h>
#include <sleefdft.h>

struct SleefDFT * SleefDFT_double_init2d(uint32_t n, uint32_t m, const double *in, double *out, uint64_t mode);
struct SleefDFT * SleefDFT_float_init2d(uint32_t n, uint32_t m, const float *in, float *out, uint64_t mode);
```

Link with `-lsleefdft -lsleef`.

These functions generates and initilizes the tables that is used for 2D
transform, and returns the pointer. Size of transform can be specified by `n`.
Currently, power-of-two sizes can be only specified. The list of the flags that
can be passed to `mode` is shown below.

These functions return a pointer to the data that is used for 2D DFT
computation, or NULL if an error occurred.

### `SleefDFT_double_execute`
### `SleefDFT_float_execute`

Execute a transform

```c
#include <stdint.h>
#include <sleefdft.h>

void SleefDFT_double_execute(struct SleefDFT *ptr, const double *in, double *out);
void SleefDFT_float_execute(struct SleefDFT *ptr, const float *in, float *out);
```

Link with `-lsleefdft -lsleef`.

`ptr` is a pointer to the plan. `in` and `out` must be pointers returned from
`Sleef_malloc` function. You can specify the same pointer to `in` and `out`.

### `SleefDFT_dispose`

Dispose the tables for transforms

```c
#include <stdint.h>
#include <sleefdft.h>

void SleefDFT_dispose(struct SleefDFT *ptr);
```

Link with `-lsleefdft -lsleef`.

This function frees a plan returned by `SleefDFT_double_init1d`,
`SleefDFT_double_init2d`, or `SleefDFT_float_init2d` functions.

