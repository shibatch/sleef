typedef struct {
  double x, y;
} double2;

typedef struct {
  float x, y;
} float2;

double xsin(double d);
double xcos(double d);
double2 xsincos(double d);
double xtan(double d);
double xasin(double s);
double xacos(double s);
double xatan(double s);
double xatan2(double y, double x);
double xlog(double d);
double xexp(double d);
double xldexp(double x, int q);
int xilogb(double d);

double xpow(double x, double y);
double xsinh(double x);
double xcosh(double x);
double xtanh(double x);
double xasinh(double x);
double xacosh(double x);
double xatanh(double x);

double xfma(double x, double y, double z);
double xsqrt(double d);
double xcbrt(double d);

double xexp2(double a);
double xexp10(double a);
double xexpm1(double a);
double xlog10(double a);
double xlog1p(double a);

double xsin_u1(double d);
double xcos_u1(double d);
double2 xsincos_u1(double d);
double xtan_u1(double d);
double xasin_u1(double s);
double xacos_u1(double s);
double xatan_u1(double s);
double xatan2_u1(double y, double x);
double xlog_u1(double d);
double xexp_u1(double d);
double xpow_u1(double x, double y);
double xsinh_u1(double x);
double xcosh_u1(double x);
double xtanh_u1(double x);
double xasinh_u1(double x);
double xacosh_u1(double x);
double xatanh_u1(double x);
double xexp2_u1(double a);
double xexp10_u1(double a);
double xexpm1_u1(double a);
double xlog10_u1(double a);
double xlog1p_u1(double a);
double xcbrt_u1(double d);

float xsinf(float d);
float xcosf(float d);
float2 xsincosf(float d);
float xtanf(float d);
float xasinf(float s);
float xacosf(float s);
float xatanf(float s);
float xatan2f(float y, float x);
float xlogf(float d);
float xexpf(float d);
float xcbrtf(float d);
float xldexpf(float x, int q);
int xilogbf(float d);

float xpowf(float x, float y);
float xsinhf(float x);
float xcoshf(float x);
float xtanhf(float x);
float xasinhf(float x);
float xacoshf(float x);
float xatanhf(float x);
float xexp2f(float a);
float xexp10f(float a);
float xexpm1f(float a);
float xlog10f(float a);
float xlog1pf(float a);

float xsinf_u1(float d);
float xcosf_u1(float d);
float2 xsincosf_u1(float d);
float xtanf_u1(float d);
float xasinf_u1(float s);
float xacosf_u1(float s);
float xatanf_u1(float s);
float xatan2f_u1(float y, float x);
float xlogf_u1(float d);
float xcbrtf_u1(float d);
