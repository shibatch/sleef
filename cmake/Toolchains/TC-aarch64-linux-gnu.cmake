set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR "aarch64")

set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu/"
  "/home/travis/xcompile-extra-install")

# Compilers to use
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-gcc++)

# Emulator to use
set(CMAKE_EMULATOR "/usr/bin/qemu-aarch64-static;-L;/usr/aarch64-linux-gnu")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
