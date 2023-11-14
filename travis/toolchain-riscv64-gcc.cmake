set(CMAKE_CROSSCOMPILING    TRUE)
set(CMAKE_SYSTEM_NAME       "Linux")
set(CMAKE_SYSTEM_PROCESSOR  "riscv64")

find_program(CMAKE_C_COMPILER NAMES riscv64-unknown-linux-gnu-gcc-12 riscv64-unknown-linux-gnu-gcc-11 riscv64-unknown-linux-gnu-gcc)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
