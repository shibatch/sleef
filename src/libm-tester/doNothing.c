#if (defined(__GNUC__) || defined(__CLANG__)) && (defined(__i386__) || defined(__x86_64__))
#include <x86intrin.h>
#endif

#if (defined(_MSC_VER))
#include <intrin.h>
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "sleef.h"

double doNothing1_1_double(double x) {}
__m128d doNothing1_1_m128d(__m128d x) {}
__m256d doNothing1_1_m256d(__m256d x) {}
double doNothing1_2_double(double x, double y) {}
__m128d doNothing1_2_m128d(__m128d x, __m128d y) {}
__m256d doNothing1_2_m256d(__m256d x, __m256d y) {}
Sleef_double2 doNothing2_1_double(double x) {}
Sleef___m128d_2 doNothing2_1_m128d(__m128d x) {}
Sleef___m256d_2 doNothing2_1_m256d(__m256d x) {}
