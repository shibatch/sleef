#if defined(__MINGW32__) || defined(__MINGW64__) || defined(_MSC_VER)
#include <Windows.h>
#endif

#include <mutex>
#include "misc.h"
#include "compat.h"

using namespace std;

extern "C" {
NOEXPORT int Sleef_cpuSupportsExt_internal(void (*tryExt)(), int *cache);
}

static void sighandler(int signum) { LONGJMP(sigjmp, 1); }

NOEXPORT int Sleef_cpuSupportsExt_internal(void (*tryExt)(), int *cache) {
  if (*cache != -1) return *cache;

  static mutex mtx;

  unique_lock<mutex> lock(mtx);

  typedef void (*sighandler_t)(int);
  sighandler_t org = signal(SIGILL, sighandler);

  if (SETJMP(sigjmp) == 0) {
    (*tryExt)();
    *cache = 1;
  } else {
    *cache = 0;
  }

  signal(SIGILL, org);
  return *cache;
}
