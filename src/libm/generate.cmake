  # This file is designed to replace the custom binaries used to generate header files

  # This func list is used for all generation
  set(FUNCLIST 
  "sin\;35\;0\;0\;0"
  "cos\;35\;0\;0\;0"
  "sincos\;35\;0\;2\;0"
  "tan\;35\;0\;0\;0"
  "asin\;35\;0\;0\;0"
  "acos\;35\;0\;0\;0"
  "atan\;35\;0\;0\;0"
  "atan2\;35\;0\;1\;0"
  "log\;35\;0\;0\;0"
  "cbrt\;35\;0\;0\;0"
  "sin\;10\;1\;0\;0"
  "cos\;10\;1\;0\;0"
  "sincos\;10\;1\;2\;0"
  "tan\;10\;1\;0\;0"
  "asin\;10\;1\;0\;0"
  "acos\;10\;1\;0\;0"
  "atan\;10\;1\;0\;0"
  "atan2\;10\;1\;1\;0"
  "log\;10\;1\;0\;0"
  "cbrt\;10\;1\;0\;0"
  "exp\;10\;0\;0\;0"
  "pow\;10\;0\;1\;0"
  "sinh\;10\;0\;0\;0"
  "cosh\;10\;0\;0\;0"
  "tanh\;10\;0\;0\;0"
  "sinh\;35\;3\;0\;0"
  "cosh\;35\;3\;0\;0"
  "tanh\;35\;3\;0\;0"

  "asinh\;10\;0\;0\;0"
  "acosh\;10\;0\;0\;0"
  "atanh\;10\;0\;0\;0"
  "exp2\;10\;0\;0\;0"
  "exp10\;10\;0\;0\;0"
  "expm1\;10\;0\;0\;0"
  "log10\;10\;0\;0\;0"
  "log2\;10\;0\;0\;0"
  "log1p\;10\;0\;0\;0"
  "sincospi\;5\;2\;2\;0"
  "sincospi\;35\;3\;2\;0"
  "sinpi\;5\;2\;0\;0"
  "cospi\;5\;2\;0\;0"
  "ldexp\;-1\;0\;3\;0"
  "ilogb\;-1\;0\;4\;0"

  "fma\;-1\;0\;5\;0"
  "sqrt\;-1\;0\;0\;0"
  "sqrt\;5\;2\;0\;1"
  "sqrt\;35\;3\;0\;0"
  "hypot\;5\;2\;1\;0"
  "hypot\;35\;3\;1\;0"
  "fabs\;-1\;0\;0\;0"
  "copysign\;-1\;0\;1\;0"
  "fmax\;-1\;0\;1\;0"
  "fmin\;-1\;0\;1\;0"
  "fdim\;-1\;0\;1\;0"
  "trunc\;-1\;0\;0\;0"
  "floor\;-1\;0\;0\;0"
  "ceil\;-1\;0\;0\;0"
  "round\;-1\;0\;0\;0"
  "rint\;-1\;0\;0\;0"
  "nextafter\;-1\;0\;1\;0"
  "frfrexp\;-1\;0\;0\;0"
  "expfrexp\;-1\;0\;4\;0"
  "fmod\;-1\;0\;1\;0"
  "modf\;-1\;0\;6\;0"

  "lgamma\;10\;1\;0\;0"
  "tgamma\;10\;1\;0\;0"
  "erf\;10\;1\;0\;0"
  "erfc\;15\;4\;0\;0"
  
  "getInt\;-1\;0\;7\;1"
  "getPtr\;-1\;0\;8\;1"
)

MACRO(SLEEF_MKALIAS FILENAME FORCE_AAVPCS VEC_WIDTH_QUOTES VEC_FP_TYPE_QUOTES VEC_INT_TYPE_QUOTES MANGLED_ISA)
  # Names come in with extra quotes from list
  string(REPLACE "\"" "" VEC_WIDTH ${VEC_WIDTH_QUOTES})
  string(REPLACE "\"" "" VEC_FP_TYPE ${VEC_FP_TYPE_QUOTES})
  string(REPLACE "\"" "" VEC_INT_TYPE ${VEC_INT_TYPE_QUOTES})
  # Get length of ARGN which always seems to exist
  string(LENGTH "${ARGN}" ARGCOUNT)
  if (ARGCOUNT)
    string(REPLACE "\"" "" EXTENTION ${ARGN})
  endif()

  # Determine type and write header
  if (${VEC_WIDTH} LESS 0)
    set(FPTYPE 1)
    set(TYPESPEC "f")
    # Drop the leading character, which is a negative sign
    string(SUBSTRING "${VEC_WIDTH}" 1 -1 width)
    set(STR "#ifdef __SLEEFSIMDSP_C__\n")
  else()
    set(FPTYPE 0)
    set(TYPESPEC "d")
    set(width "${VEC_WIDTH}")
    set(STR "#ifdef __SLEEFSIMDDP_C__\n")
  endif()

  # This is the current condition on which AAVPCS is set
  if (FORCE_AAVPCS AND NOT CMAKE_CROSSCOMPILING)
    if ("${EXTENTION}" STREQUAL "advsimd")
      set(VECTORCC " __attribute__((aarch64_vector_pcs))")
    endif()
  endif()

  if(FPTYPE)
    set(RETURNTYPES ${VEC_FP_TYPE} ${VEC_FP_TYPE} "vfloat2" ${VEC_FP_TYPE} ${VEC_INT_TYPE} ${VEC_FP_TYPE} "vfloat2" "int" "void *")
  else()
    set(RETURNTYPES ${VEC_FP_TYPE} ${VEC_FP_TYPE} "vdouble2" ${VEC_FP_TYPE} ${VEC_INT_TYPE} ${VEC_FP_TYPE} "vdouble2" "int" "void *")
  endif()

  set(ARGTYPEZERO ${VEC_FP_TYPE} "${VEC_FP_TYPE}, ${VEC_FP_TYPE}" ${VEC_FP_TYPE} "${VEC_FP_TYPE}, ${VEC_INT_TYPE}" ${VEC_FP_TYPE} "${VEC_FP_TYPE}, ${VEC_FP_TYPE}, ${VEC_FP_TYPE}" ${VEC_FP_TYPE} "int" "int")
  set(ARGTYPEONE "${VEC_FP_TYPE} a0" "${VEC_FP_TYPE} a0, ${VEC_FP_TYPE} a1" "${VEC_FP_TYPE} a0" "${VEC_FP_TYPE} a0, ${VEC_INT_TYPE} a1" "${VEC_FP_TYPE} a0" "${VEC_FP_TYPE} a0, ${VEC_FP_TYPE} a1, ${VEC_FP_TYPE} a2" "${VEC_FP_TYPE} a0" "int a0" "int a0")
  set(ARGTYPETWO "a0" "a0, a1" "a0" "a0, a1" "a0" "a0, a1, a2" "a0" "a0" "a0")


  string(APPEND STR "#ifdef ENABLE_ALIAS\n")
  

  if (EXTENTION)
    foreach(funcresult ${FUNCLIST})
      list(GET funcresult 0 name)
      list(GET funcresult 1 ulp)
      list(GET funcresult 3 funcType)
      list(GET RETURNTYPES ${funcType} returnstring)
      list(GET ARGTYPEZERO ${funcType} firstarg)
      # This section is quite complex
      set(ULPVAL "")
      set(ULPVALUNDER "")
      if (${ulp} GREATER 0) # CMake 3.7 required for GREATER_EQUAL
        string(LENGTH ${ulp} ULPLEN)
        if (${ULPLEN} LESS 2)
          set(ulp "0${ulp}")
        endif()
        set(ULPVAL "u${ulp}")
        set(ULPVALUNDER "_${ULPVAL}")
      endif()
      string(APPEND STR "EXPORT CONST ${returnstring} Sleef_${name}${TYPESPEC}${width}${ULPVALUNDER}(${firstarg}) __attribute__((alias(\"Sleef_${name}${TYPESPEC}${width}_${ULPVAL}${EXTENTION}\"))) ${VECTORCC}\;\n")
    endforeach()
    string(APPEND STR "\n")
  endif()
  string(APPEND STR "#else // #ifdef ENABLE_ALIAS\n")

  if (EXTENTION)
    foreach(funcresult ${FUNCLIST})
      list(GET funcresult 0 name)
      list(GET funcresult 1 ulp)
      list(GET funcresult 3 funcType)
      list(GET RETURNTYPES ${funcType} returnstring)
      list(GET ARGTYPEONE ${funcType} secondarg)
      list(GET ARGTYPETWO ${funcType} thirdarg)
      # This section is quite complex
      set(ULPVAL "")
      set(ULPVALUNDER "")
      if (${ulp} GREATER 0) # CMake 3.7 required for GREATER_EQUAL
        string(LENGTH ${ulp} ULPLEN)
        if (${ULPLEN} LESS 2)
          set(ulp "0${ulp}")
        endif()
        set(ULPVAL "u${ulp}")
        set(ULPVALUNDER "_${ULPVAL}")
      endif()
      string(APPEND STR "EXPORT CONST ${returnstring} ${VECTORCC} Sleef_${name}${TYPESPEC}${width}${ULPVALUNDER}(${secondarg}) { return Sleef_${name}${TYPESPEC}${width}_${ULPVAL}${EXTENTION}(${thirdarg})\; }\n")
    endforeach()
    string(APPEND STR "\n")
  endif()

  # Write footer
  string(APPEND STR "#endif // #ifdef ENABLE_ALIAS\n")
  if (FPTYPE)
    string(APPEND STR "#endif // #ifdef __SLEEFSIMDSP_C__\n")
  else()
    string(APPEND STR "#endif // #ifdef __SLEEFSIMDDP_C__\n")
  endif()

  file(APPEND ${FILENAME} ${STR})
ENDMACRO(SLEEF_MKALIAS)
