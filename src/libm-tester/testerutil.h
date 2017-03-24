#define DENORMAL_DBL_MIN (4.9406564584124654418e-324)
#define POSITIVE_INFINITY INFINITY
#define NEGATIVE_INFINITY (-INFINITY)

#define DENORMAL_FLT_MIN (1.4012984643248170709e-45f)
#define POSITIVE_INFINITYf ((float)INFINITY)
#define NEGATIVE_INFINITYf (-(float)INFINITY)

#define M_PIf ((float)M_PI)

int enableFlushToZero;
double flushToZero(double y);

int isnumber(double x);
int isPlusZero(double x);
int isMinusZero(double x);
int xisnan(double x);
double sign(double d);

int isnumberf(float x);
int isPlusZerof(float x);
int isMinusZerof(float x);
int xisnanf(float x);
float signf(float d);

int cmpDenormdp(double x, mpfr_t fry);
double countULPdp(double d, mpfr_t c);
double countULP2dp(double d, mpfr_t c);

int cmpDenormsp(float x, mpfr_t fry);
double countULPsp(float d, mpfr_t c);
double countULP2sp(float d, mpfr_t c);

void mpfr_sinpi(mpfr_t ret, mpfr_t arg, mpfr_rnd_t rnd);
void mpfr_cospi(mpfr_t ret, mpfr_t arg, mpfr_rnd_t rnd);
