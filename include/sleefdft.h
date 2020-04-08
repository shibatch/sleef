#ifndef __SLEEFDFT_H__
#define __SLEEFDFT_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdlib.h>
#include <stdint.h>

#define SLEEF_MODE_FORWARD     (0 <<  0)
#define SLEEF_MODE_BACKWARD    (1 <<  0)

#define SLEEF_MODE_COMPLEX     (0 <<  1)
#define SLEEF_MODE_REAL        (1 <<  1)

#define SLEEF_MODE_ALT         (1 <<  2)
#define SLEEF_MODE_FFTWCOMPAT  (1 <<  3)

#define SLEEF_MODE_DEBUG       (1 << 10)
#define SLEEF_MODE_VERBOSE     (1 << 11)
#define SLEEF_MODE_NO_MT       (1 << 12)

#define SLEEF_MODE_ESTIMATE    (1 << 20)
#define SLEEF_MODE_MEASURE     (2 << 20)

#if (defined(__MINGW32__) || defined(__MINGW64__) || defined(__CYGWIN__) || defined(_MSC_VER)) && !defined(SLEEF_STATIC_LIBS)
#ifdef IMPORT_IS_EXPORT
#define IMPORT __declspec(dllexport)
#else // #ifdef IMPORT_IS_EXPORT
#define IMPORT __declspec(dllimport)
#if (defined(_MSC_VER))
#pragma comment(lib,"sleefdft.lib")
#endif // #if (defined(_MSC_VER))
#endif // #ifdef IMPORT_IS_EXPORT
#else // #if (defined(__MINGW32__) || defined(__MINGW64__) || defined(__CYGWIN__) || defined(_MSC_VER)) && !defined(SLEEF_STATIC_LIBS)
#define IMPORT
#endif // #if (defined(__MINGW32__) || defined(__MINGW64__) || defined(__CYGWIN__) || defined(_MSC_VER)) && !defined(SLEEF_STATIC_LIBS)

IMPORT struct SleefDFT *SleefDFT_double_init1d(uint32_t n, const double *in, double *out, uint64_t mode);
IMPORT struct SleefDFT *SleefDFT_double_init2d(uint32_t n, uint32_t m, const double *in, double *out, uint64_t mode);
IMPORT void SleefDFT_double_execute(struct SleefDFT *ptr, const double *in, double *out);

IMPORT struct SleefDFT *SleefDFT_float_init1d(uint32_t n, const float *in, float *out, uint64_t mode);
IMPORT struct SleefDFT *SleefDFT_float_init2d(uint32_t n, uint32_t m, const float *in, float *out, uint64_t mode);
IMPORT void SleefDFT_float_execute(struct SleefDFT *ptr, const float *in, float *out);

IMPORT struct SleefDFT *SleefDFT_longdouble_init1d(uint32_t n, const long double *in, long double *out, uint64_t mode);
IMPORT struct SleefDFT *SleefDFT_longdouble_init2d(uint32_t n, uint32_t m, const long double *in, long double *out, uint64_t mode);
IMPORT void SleefDFT_longdouble_execute(struct SleefDFT *ptr, const long double *in, long double *out);

#if defined(ENABLEFLOAT128) && !defined(Sleef_quad2_DEFINED)
#define Sleef_quad2_DEFINED
typedef __float128 Sleef_quad;
typedef struct {
  Sleef_quad x, y;
} Sleef_quad2;
#endif

#if defined(Sleef_quad2_DEFINED)
IMPORT struct SleefDFT *SleefDFT_quad_init1d(uint32_t n, const Sleef_quad *in, Sleef_quad *out, uint64_t mode);
IMPORT struct SleefDFT *SleefDFT_quad_init2d(uint32_t n, uint32_t m, const Sleef_quad *in, Sleef_quad *out, uint64_t mode);
IMPORT void SleefDFT_quad_execute(struct SleefDFT *ptr, const Sleef_quad *in, Sleef_quad *out);
#endif

IMPORT void SleefDFT_dispose(struct SleefDFT *ptr);

IMPORT void SleefDFT_setPath(struct SleefDFT *ptr, char *pathStr);

//

IMPORT void SleefDFT_setPlanFilePath(const char *path, const char *arch, uint64_t mode);

#define SLEEF_PLAN_AUTOMATIC 0
#define SLEEF_PLAN_READONLY (1 << 0)
#define SLEEF_PLAN_RESET (1 << 1)
#define SLEEF_PLAN_BUILDALLPLAN (1 << 2)
#define SLEEF_PLAN_NOLOCK (1 << 3)
#define SLEEF_PLAN_MEASURE (1 << 29)
#define SLEEF_PLAN_REFERTOENVVAR (1 << 30)

#undef IMPORT

#ifdef __cplusplus
}
#endif

#endif // #ifndef __SLEEFDFT_H__
