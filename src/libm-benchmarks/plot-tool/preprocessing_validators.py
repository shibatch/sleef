import re

# Set of options for which this result preprocessing and graph plotting
# tool should work.
compilers = ["gcc14", "llvm17"]
architectures = ["aarch64", "x86"]
variants = ["scalar", "vector128", "vector256", "vector512", "sve"]
precisions = ["single", "double"]
accuracies = ["u10", "u35"]
libraries = ["sleef", "libm"]

# Validation Functions
# Ensure fields in dataframe belong to options defined above.
def valid_compiler(compiler):
    assert compiler in compilers
    return compiler


def valid_architecture(architecture):
    assert architecture in architectures
    return architecture


def valid_variant(variant):
    assert variant in variants
    return variant


def valid_precision(precision):
    assert precision in precisions
    return precision


def valid_library(library):
    assert library in libraries
    return library


def valid_accuracy(accuracy):
    assert accuracy in accuracies
    return accuracy


# This function takes a give results filename and checks if it obbeys
# the naming convention:  "results-<library>-<compiler>-<architecture>.csv"
# Also checks if the options in the name are supported.
# If it passes the checks, it returns the relevant component of the filename.
# Example of a valid filename: "results-libm-gcc14-aarch64.csv"
def filename_validator(result_filename):
    result_filename = result_filename.split("/")[-1]
    filename_components = result_filename.split("-")
    assert len(filename_components) == 4
    assert filename_components[0] == "results"
    library = valid_library(filename_components[1])
    compiler = valid_compiler(filename_components[2])
    architecture_extension = filename_components[3].split(".")
    assert len(architecture_extension) == 2
    architecture = valid_architecture(architecture_extension[0])
    extension = architecture_extension[1]
    assert extension == "csv"
    return library, architecture, compiler


# This function takes a benchmark label and extrapolates the name of the library
# the function belongs to.
# Example "MB_Sleef_tandx_u35sve_sved_0_6.28" -> sleef
def extract_lib(fun_name):
    return valid_library(fun_name.split("_")[1].lower())


# This function takes a benchmark label and extrapolates the name of the math function
# (independent of what vector extension, interval etc...) the routine captures.
# Example "MB_Sleef_tandx_u35sve_sved_0_6.28" -> tan
def extract_fun_name(fun_name):
    # Pass 1: Split by _
    #    MB_Sleef_tandx_u35sve_sved_0_6.28 --> [MB, Sleef,tandx,u35sve,sved,0,6.28].
    #    The element with index 2 contains the name we are interested: tandx
    formatted_fun_name1 = fun_name.split("_")[2]
    # Pass 2: Split by make sure we ignore the sufix usually present in last 3 characters
    #    tandx --> [tan, '']
    #    discard the last elemnt of this list
    #    we discard the last element instad of keeping the first element in case the function
    #    contains f and d in the name.
    #    Also, this could be simplified if reverse split with regular experessions split was
    #    supported with python
    suffix_reg_exp = "fx|dx|f\d+|d\d+|f|d"
    formatted_fun_name2 = re.split(suffix_reg_exp, formatted_fun_name1)
    if len(formatted_fun_name2) == 1:
        return formatted_fun_name2[0]
    else:
        return "".join(formatted_fun_name2[:-1])


# This function takes a benchmark label and extrapolates the interval used to benchmark
# the routine in question.
# Example "MB_Sleef_tandx_u35sve_sved_0_6.28" -> (0, 6.28)
def extract_interval(fun_name):
    interval = fun_name.split("_")
    return "[" + interval[-2] + "," + interval[-1] + "]"


# This function takes a benchmark label and extrapolates the extension of the routine
# Should be scalar, sve or vector
# Example "MB_Sleef_tandx_u35sve_sved_0_6.28" -> sve
def extract_variant(fun_name):
    variant_w_precision = fun_name.split("_")[4]
    variant = re.split("f|d", variant_w_precision)
    return valid_variant("".join(variant))


# This function takes a benchmark label and extrapolates if its a single precision or
# double precision routine
# Example "MB_Sleef_tandx_u35sve_sved_0_6.28" -> double precision
# Works assuming extension names dont have f or d. If that becomes the case,
# use regular expressions
def extract_precision(fun_name):
    extension_w_precision = fun_name.split("_")[4]
    precision = None
    if "f" in extension_w_precision:
        precision = "single"
    if "d" in extension_w_precision:
        precision = "double"
    return valid_precision(precision)


# This function takes a benchmark label and extrapolates if its a single precision or
# double precision routine
# Example "MB_Sleef_tandx_u35sve_sved_0_6.28" -> double precision
def extract_ulp(fun_name):
    return valid_accuracy(fun_name.split("_")[3][:3])


# This function takes a benchmark label and extrapolates if its a single precision or
# double precision routine
# Example "MB_Sleef_tandx_u35sve_sved_0_6.28" -> double precision
def seconds_to_nanoseconds(seconds):
    return seconds * 1e9


########################################################
######################## TESTING #######################
# python3 preprocessing_validators.py                  #
########################################################
if __name__ == "__main__":
    # Testing of functions above:
    print(extract_lib("MB_Sleef_tandx_u35sve_sved_0_6.28"))
    print(extract_lib("MB_libm_sin_scalard_0_1e28"))
    print(extract_fun_name("MB_Sleef_tandx_u35sve_sved_0_6.28"))
    print(extract_interval("MB_Sleef_tandx_u35sve_sved_0_1e+38"))
    print(extract_interval("MB_Sleef_atan2d2_u10_vectord128_-10_10"))
    print(extract_variant("MB_Sleef_tandx_u35sve_sved_0_1e+38"))
    print(extract_variant("MB_Sleef_log10d2_u10_vectord128_0_1e100"))
    print(extract_variant("MB_Sleef_tandx_u35sve_sved_0_1e+38"))
    print(extract_ulp("MB_Sleef_tandx_u35sve_sved_0_1e+38"))
    print(extract_ulp("MB_Sleef_tandx_u35sve_sved_0_1e+38"))
    print(extract_fun_name("MB_libm_tan_u10_scalarf_0_1e+6"))
    print(extract_interval("MB_libm_tan_u10_scalarf_0_1e+6"))
    print(extract_variant("MB_libm_tan_u10_scalarf_0_1e+6"))
    print(extract_ulp("MB_libm_tan_u10_scalarf_0_1e+6"))
    filename_validator("results-libm-gcc14-aarch64.csv")
