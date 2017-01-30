//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if defined (__GNUC__) || defined (__INTEL_COMPILER) || defined (__clang__)
#ifdef INFINITY
#undef INFINITY
#endif

#ifdef NAN
#undef NAN
#endif

#define NAN __builtin_nan("")
#define NANf __builtin_nanf("")
#define INFINITY __builtin_inf()
#define INFINITYf __builtin_inff()
#else

#include <bits/nan.h>
#include <bits/inf.h>

#endif
