//   Copyright Naoki Shibata and contributors 2010 - 2023.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

template<typename T>
static double countULP(T ot, const T& oc,
		       const int nbmant, const T& fltmin, const T& fltmax,
		       const bool checkSignedZero=false, const double abound=0.0) {
  if (isnan_(oc) && isnan_(ot)) return 0;
  if (isnan_(oc) || isnan_(ot)) return 10001;
  if (isinf_(oc) && !isinf_(ot)) return INFINITY;

  const T halffltmin = mul_(fltmin, T(0.5));
  const bool ciszero = fabs_(oc) < halffltmin, cisinf = fabs_(oc) > fltmax;

  if (cisinf && isinf_(ot) && signbit_(oc) == signbit_(ot)) return 0;
  if (ciszero && ot != 0) return 10000;
  if (checkSignedZero && ciszero && ot == 0 && signbit_(oc) != signbit_(ot)) return 10002;

  double v = 0;
  if (isinf_(ot) && !isinf_(oc)) {
    ot = copysign_(fltmax, ot);
    v = 1;
  }

  const int ec = ilogb_(oc);

  auto e = fabs_(oc - ot);
  if (e < abound) return 0;

  return double(div_(e, fmax_(ldexp_(T(1), ec + 1 - nbmant), fltmin))) + v;
}
