#include <vecintrin.h>

__vector float sleef_cpuidtmp0;
__vector int sleef_cpuidtmp1;

void sleef_tryVXE2() {
  sleef_cpuidtmp0 = vec_float(sleef_cpuidtmp1);
}
