#if !(defined(__MINGW32__) || defined(__MINGW64__) || defined(_MSC_VER))
#include <unistd.h>
#include <sys/types.h>
#include <sys/file.h>

static void FLOCK(FILE *fp) { flock(fileno(fp), LOCK_EX); }
static void FUNLOCK(FILE *fp) { flock(fileno(fp), LOCK_UN); }
static void FTRUNCATE(FILE *fp, off_t z) {
  if (ftruncate(fileno(fp), z))
    ;
}
static FILE *OPENTMPFILE() { return tmpfile(); }
static void CLOSETMPFILE(FILE *fp) { fclose(fp); }
#else
#include <Windows.h>
#include <io.h>

static void FLOCK(FILE *fp) { }
static void FUNLOCK(FILE *fp) { }
static void FTRUNCATE(FILE *fp, long z) {
  fseek(fp, 0, SEEK_SET);
  SetEndOfFile((HANDLE)_get_osfhandle(_fileno(fp)));
}
static FILE *OPENTMPFILE() { return fopen("tmpfile.txt", "w+"); }
static void CLOSETMPFILE(FILE *fp) {
  fclose(fp);
  remove("tmpfile.txt");
}
#endif
