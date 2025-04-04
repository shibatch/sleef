#include "tlfloat/tlfloat.hpp"

using namespace tlfloat;

extern "C" {
#include "misc.h"

  EXPORT double Sleef_fma_internal(const double x, const double y, const double z) {
    return (double)fma(Double(x), Double(y), Double(z));
  }

  EXPORT float Sleef_fmaf_internal(const float x, const float y, const float z) {
    return (float)fma(Float(x), Float(y), Float(z));
  }
}
