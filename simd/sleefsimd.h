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

static vint vloadui(int32_t *p) { return (vint)_mm_loadu_si128((__m128i *)p); }
static void vstoreui(int32_t *p, vint v) { _mm_storeu_si128((__m128i *)p, (__m128i)v); }

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
typedef struct { __m128i x, y; } vint2;

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

static vint vloadui(int32_t *p) { return (vint)_mm_loadu_si128((__m128i *)p); }
static void vstoreui(int32_t *p, vint v) { _mm_storeu_si128((__m128i *)p, (__m128i)v); }

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

static vint vloadui(int32_t *p) { return (vint)_mm_loadu_si128((__m128i *)p); }
static void vstoreui(int32_t *p, vint v) { _mm_storeu_si128((__m128i *)p, (__m128i)v); }

#define ENABLE_DP
#define ENABLE_SP
#endif


// ******** AVX512F ********

#ifdef ENABLE_AVX512F
#include <immintrin.h>

#define VECTLENDP 8
#define VECTLENSP 16

typedef __m512d vdouble;
typedef __m256i vint;

typedef __m512 vfloat;
typedef __m512i vint2;

static vdouble vloadu(double *p) { return _mm512_loadu_pd(p); }
static void vstoreu(double *p, vdouble v) { return _mm512_storeu_pd(p, v); }

static vfloat vloaduf(float *p) { return _mm512_loadu_ps(p); }
static void vstoreuf(float *p, vfloat v) { return _mm512_storeu_ps(p, v); }

static vint2 vloadui2(int32_t *p) { return _mm512_loadu_si512((__m512i const *)p); }
static void vstoreui2(int32_t *p, vint2 v) { return _mm512_storeu_si512((__m512i *)p, v); }

static vint vloadui(int32_t *p) { return _mm256_loadu_si256((__m256i const *)p); }
static void vstoreui(int32_t *p, vint v) { return _mm256_storeu_si256((__m256i *)p, v); }

#define ENABLE_DP
#define ENABLE_SP
#endif


// ******** ARM NEON AArch32 ********

#ifdef ENABLE_NEON32
#include <arm_neon.h>

#define VECTLENDP 2
#define VECTLENSP 4

//typedef __m128d vdouble;
typedef int32x4_t vint;

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


// ******** ARM NEON AArch64 ********

#ifdef ENABLE_NEON64
#include <arm_neon.h>

#define VECTLENDP 2
#define VECTLENSP 4

//typedef __m128d vdouble;
typedef int32x4_t vint;

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


// ******** Clang Exntended Vector ********

#ifdef ENABLE_CLANGVEC
#define VECTLENDP 8
#define VECTLENSP (VECTLENDP*2)

typedef double vdouble __attribute__((ext_vector_type(VECTLENDP)));
typedef int32_t vint __attribute__((ext_vector_type(VECTLENDP)));

typedef float vfloat __attribute__((ext_vector_type(VECTLENDP*2)));
typedef int32_t vint2 __attribute__((ext_vector_type(VECTLENDP*2)));

static vdouble vloadu(double *p) {
  vdouble vd;
  for(int i=0;i<VECTLENDP;i++) vd[i] = p[i];
  return vd;
}
static void vstoreu(double *p, vdouble v) {
  for(int i=0;i<VECTLENDP;i++) p[i] = v[i];
}

static vfloat vloaduf(float *p) {
  vfloat vf;
  for(int i=0;i<VECTLENSP;i++) vf[i] = p[i];
  return vf;
}
static void vstoreuf(float *p, vfloat v) {
  for(int i=0;i<VECTLENSP;i++) p[i] = v[i];
}

static vint2 vloadui2(int32_t *p) {
  vint2 vi;
  for(int i=0;i<VECTLENSP;i++) vi[i] = p[i];
  return vi;
}
static void vstoreui2(int32_t *p, vint2 v) {
  for(int i=0;i<VECTLENSP;i++) p[i] = v[i];
}

static vint vloadui(int32_t *p) {
  vint vi;
  for(int i=0;i<VECTLENDP;i++) vi[i] = p[i];
  return vi;
}
static void vstoreui(int32_t *p, vint v) {
  for(int i=0;i<VECTLENDP;i++) p[i] = v[i];
}

#define ENABLE_DP
#define ENABLE_SP
#endif


////

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
vint2 xilogbf(vfloat d);

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
