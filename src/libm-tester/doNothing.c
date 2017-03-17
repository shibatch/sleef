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

double doNothing1_1_double(double x) { return  0; }
__m128d doNothing1_1_m128d(__m128d x) { return _mm_set1_pd(0); }
__m256d doNothing1_1_m256d(__m256d x) { return _mm256_set1_pd(0); }
double doNothing1_2_double(double x, double y) { return 0; }
__m128d doNothing1_2_m128d(__m128d x, __m128d y) { return _mm_set1_pd(0); }
__m256d doNothing1_2_m256d(__m256d x, __m256d y) {  return _mm256_set1_pd(0); }
Sleef_double2 doNothing2_1_double(double x) {
  Sleef_double2 d = { 0, 0 };
  return d;
}
Sleef___m128d_2 doNothing2_1_m128d(__m128d x) {
  Sleef___m128d_2 d = { _mm_set1_pd(0), _mm_set1_pd(0) };
  return d;
}
Sleef___m256d_2 doNothing2_1_m256d(__m256d x) {
  Sleef___m256d_2 d = { _mm256_set1_pd(0), _mm256_set1_pd(0) };
  return d;
}
