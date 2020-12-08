#include <altivec.h>

__vector double sleef_cpuidtmp0;
__vector unsigned long long sleef_cpuidtmp1, sleef_cpuidtmp2;

void sleef_tryVSX3() {
  sleef_cpuidtmp0 = vec_insert_exp(sleef_cpuidtmp1, sleef_cpuidtmp2);
}
