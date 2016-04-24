#include <stdint.h>

// ******** SSE2 ********

#ifdef ENABLE_SSE2
#include <immintrin.h>

#define VECTLENDP 2
#define VECTLENSP 4

typedef __m128d vdouble;
typedef __m128i vint;

typedef __m128 vfloat;
typedef __m128i vint2;

static vdouble vloadu(double *p) { return _mm_loadu_pd(p); }
static void vstoreu(double *p, vdouble v) { _mm_storeu_pd(p, v); }

static vfloat vloaduf(float *p) { return _mm_loadu_ps(p); }
static void vstoreuf(float *p, vfloat v) { _mm_storeu_ps(p, v); }

static vint2 vloadui2(int32_t *p) { return (vint2)_mm_loadu_si128((__m128i *)p); }
static void vstoreui2(int32_t *p, vint2 v) { _mm_storeu_si128((__m128i *)p, (__m128i)v); }

#define ENABLE_DP
#define ENABLE_SP
#endif


// ******** AVX ********

#if defined(ENABLE_AVX) || defined(ENABLE_FMA4)
#include <immintrin.h>

#define VECTLENDP 4
#define VECTLENSP 8

typedef __m256d vdouble;
typedef __m128i vint;

typedef __m256 vfloat;
typedef struct {
  vint x, y;
} vint2;

static vdouble vloadu(double *p) { return _mm256_loadu_pd(p); }
static void vstoreu(double *p, vdouble v) { return _mm256_storeu_pd(p, v); }

static vfloat vloaduf(float *p) { return _mm256_loadu_ps(p); }
static void vstoreuf(float *p, vfloat v) { return _mm256_storeu_ps(p, v); }

static vint2 vloadui2(int32_t *p) {
  vint2 r;
  r.x = _mm_loadu_si128((__m128i *) p     );
  r.y = _mm_loadu_si128((__m128i *)(p + 4));
  return r;
}

static void vstoreui2(int32_t *p, vint2 v) {
  _mm_storeu_si128((__m128i *) p     , v.x);
  _mm_storeu_si128((__m128i *)(p + 4), v.y);  
}

#define ENABLE_DP
#define ENABLE_SP
#endif


// ******** AVX2 ********

#ifdef ENABLE_AVX2
#include <immintrin.h>

#define VECTLENDP 4
#define VECTLENSP 8

typedef __m256d vdouble;
typedef __m128i vint;

typedef __m256 vfloat;
typedef __m256i vint2;

static vdouble vloadu(double *p) { return _mm256_loadu_pd(p); }
static void vstoreu(double *p, vdouble v) { return _mm256_storeu_pd(p, v); }

static vfloat vloaduf(float *p) { return _mm256_loadu_ps(p); }
static void vstoreuf(float *p, vfloat v) { return _mm256_storeu_ps(p, v); }

static vint2 vloadui2(int32_t *p) { return _mm256_loadu_si256((__m256i const *)p); }
static void vstoreui2(int32_t *p, vint2 v) { return _mm256_storeu_si256((__m256i *)p, v); }

#define ENABLE_DP
#define ENABLE_SP
#endif


// ******** ARM NEON ********

#ifdef ENABLE_NEON
#include <arm_neon.h>

#define VECTLENDP 2
#define VECTLENSP 4

//typedef __m128d vdouble;
typedef int32x4_t vint;
typedef uint32x4_t vmask;

typedef float32x4_t vfloat;
typedef int32x4_t vint2;

//static vdouble vloadu(double *p) { return _mm_loadu_pd(p); }
//static void vstoreu(double *p, vdouble v) { _mm_storeu_pd(p, v); }

static vfloat vloaduf(float *p) { return vld1q_f32(p); }
static void vstoreuf(float *p, vfloat v) { vst1q_f32(p, v); }

static vint2 vloadui2(int32_t *p) { return (vint2)vld1q_s32(p); }
static void vstoreui2(int32_t *p, vint2 v) { vst1q_s32(p, v); }

#define ENABLE_SP
#endif


#ifdef ENABLE_DP
typedef struct {
  vdouble x, y;
} vdouble2;

vdouble xldexp(vdouble x, vint q);
vint xilogb(vdouble d);

vdouble xsin(vdouble d);
vdouble xcos(vdouble d);
vdouble2 xsincos(vdouble d);
vdouble xtan(vdouble d);
vdouble xasin(vdouble s);
vdouble xacos(vdouble s);
vdouble xatan(vdouble s);
vdouble xatan2(vdouble y, vdouble x);
vdouble xlog(vdouble d);
vdouble xexp(vdouble d);
vdouble xpow(vdouble x, vdouble y);

vdouble xsinh(vdouble d);
vdouble xcosh(vdouble d);
vdouble xtanh(vdouble d);
vdouble xasinh(vdouble s);
vdouble xacosh(vdouble s);
vdouble xatanh(vdouble s);

vdouble xcbrt(vdouble d);

vdouble xexp2(vdouble a);
vdouble xexp10(vdouble a);
vdouble xexpm1(vdouble a);
vdouble xlog10(vdouble a);
vdouble xlog1p(vdouble a);

vdouble xsin_u1(vdouble d);
vdouble xcos_u1(vdouble d);
vdouble2 xsincos_u1(vdouble d);
vdouble xtan_u1(vdouble d);
vdouble xasin_u1(vdouble s);
vdouble xacos_u1(vdouble s);
vdouble xatan_u1(vdouble s);
vdouble xatan2_u1(vdouble y, vdouble x);
vdouble xlog_u1(vdouble d);
vdouble xcbrt_u1(vdouble d);
#endif

//

#ifdef ENABLE_SP
typedef struct {
  vfloat x, y;
} vfloat2;

vfloat xldexpf(vfloat x, vint2 q);

vfloat xsinf(vfloat d);
vfloat xcosf(vfloat d);
vfloat2 xsincosf(vfloat d);
vfloat xtanf(vfloat d);
vfloat xasinf(vfloat s);
vfloat xacosf(vfloat s);
vfloat xatanf(vfloat s);
vfloat xatan2f(vfloat y, vfloat x);
vfloat xlogf(vfloat d);
vfloat xexpf(vfloat d);
vfloat xcbrtf(vfloat s);
vfloat xsqrtf(vfloat s);

vfloat xpowf(vfloat x, vfloat y);
vfloat xsinhf(vfloat x);
vfloat xcoshf(vfloat x);
vfloat xtanhf(vfloat x);
vfloat xasinhf(vfloat x);
vfloat xacoshf(vfloat x);
vfloat xatanhf(vfloat x);
vfloat xexp2f(vfloat a);
vfloat xexp10f(vfloat a);
vfloat xexpm1f(vfloat a);
vfloat xlog10f(vfloat a);
vfloat xlog1pf(vfloat a);

vfloat xsinf_u1(vfloat d);
vfloat xcosf_u1(vfloat d);
vfloat2 xsincosf_u1(vfloat d);
vfloat xtanf_u1(vfloat d);
vfloat xasinf_u1(vfloat s);
vfloat xacosf_u1(vfloat s);
vfloat xatanf_u1(vfloat s);
vfloat xatan2f_u1(vfloat y, vfloat x);
vfloat xlogf_u1(vfloat d);
vfloat xcbrtf_u1(vfloat s);
#endif
