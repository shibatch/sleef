# Compiler properties

set(COMMON_TARGET_PROPERTIES 
  POSITION_INDEPENDENT_CODE ON   # -fPIC
  C_STANDARD 99                  # -std=gnu99
  )

# Target TARGET_LIBCOMMON_STATIC

add_library(${TARGET_LIBCOMMON_STATIC} STATIC common.c)
set_target_properties(${TARGET_LIBCOMMON_STATIC} PROPERTIES ${COMMON_TARGET_PROPERTIES})

# Target common.o

add_library(common_obj OBJECT common.c)
set_target_properties(common_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})

# Target arraymap.o

add_library(arraymap_obj OBJECT arraymap.c)
set_target_properties(arraymap_obj PROPERTIES ${COMMON_TARGET_PROPERTIES})
