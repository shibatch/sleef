#include <mutex>
#include "compat.h"

using namespace std;

extern "C" {
int Sleef_cpuSupportsExt_internal(void (*tryExt)(), int *cache);
}

static void sighandler(int signum) { LONGJMP(sigjmp, 1); }

int Sleef_cpuSupportsExt_internal(void (*tryExt)(), int *cache) {
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
