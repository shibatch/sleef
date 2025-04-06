#include <stdio.h>
#include <stdarg.h>

#include "tlfloat/tlfloat.h"

#include "sleef.h"
#include "misc.h"

typedef Sleef_quad vargquad;

EXPORT vargquad Sleef_strtoq(const char *str, const char **endptr) {
  return tlfloat_strtoq(str, endptr);
}

EXPORT int Sleef_vfprintf(FILE *fp, const char *fmt, va_list ap) {
  return tlfloat_vfprintf(fp, fmt, ap);
}

EXPORT int Sleef_vsnprintf(char *str, size_t size, const char *fmt, va_list ap) {
  return tlfloat_vsnprintf(str, size, fmt, ap);
}

EXPORT int Sleef_fprintf(FILE *fp, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = tlfloat_vfprintf(fp, fmt, ap);
  va_end(ap);
  return ret;
}

EXPORT int Sleef_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = tlfloat_vfprintf(stdout, fmt, ap);
  va_end(ap);
  return ret;
}

EXPORT int Sleef_snprintf(char *str, size_t size, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = tlfloat_vsnprintf(str, size, fmt, ap);
  va_end(ap);
  return ret;
}
