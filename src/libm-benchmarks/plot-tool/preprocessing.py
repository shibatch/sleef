import preprocessing_validators as ppv
import pandas as pd

# Convert raw results into organized dataframe
#   1. Drop empty columns
#   2. Reformulate columns in dataframe as (example):
#         Fun: sin
#         Variant: scalar:
#         Precision: double
#         Interval: [-6.28, 6.28]
#         ULP: u10
#         Total Time: 5ns
#         Throughput: 5ns
#         Compiler: gcc14
#         Architecture: aarch64
#         Library: sleef
def raw_to_df(results_file):
    library, architecture, compiler = ppv.filename_validator(results_file)
    raw_df = pd.read_csv(results_file)
    raw_df.dropna(how="all", axis=1, inplace=True)
    intermediate_df = pd.DataFrame({})
    intermediate_df["Function"] = raw_df["name"].apply(ppv.extract_fun_name)
    intermediate_df["Variant"] = raw_df["name"].apply(ppv.extract_variant)
    intermediate_df["Precision"] = raw_df["name"].apply(ppv.extract_precision)
    intermediate_df["Interval"] = raw_df["name"].apply(ppv.extract_interval)
    intermediate_df["ULP"] = raw_df["name"].apply(ppv.extract_ulp)
    intermediate_df["Total Time"] = raw_df["real_time"]
    intermediate_df["Throughput"] = raw_df["NSperEl"].apply(ppv.seconds_to_nanoseconds)
    intermediate_df["Compiler"] = compiler
    intermediate_df["Architecture"] = architecture
    intermediate_df["Library"] = library
    return intermediate_df


# Filter entries that contain a fixed precision, accuracy and variant.
# If keep_lower_interval is True, then it will also only keep one interval per
# function. If the intervals are sorted in the results provided (recommended),
# then it will only keep the lowest intervals per function.
def filter_results(raw_df, precision, accuracy, variant, keep_lower_interval=False):
    filtered_df = raw_df
    filtered_df = filtered_df[filtered_df["Precision"] == precision]
    filtered_df = filtered_df[filtered_df["ULP"] == accuracy]
    filtered_df = filtered_df[filtered_df["Variant"] == variant]
    if keep_lower_interval:
        filtered_df.drop_duplicates(subset="Function", keep="first", inplace=True)
    # Resetting index is important,
    # otherwise they remain with same indexes as in original dataframe
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


########################################################
######################## TESTING #######################
# python3 preprocessing.py                         #
########################################################
if __name__ == "__main__":
    libm_gcc14_aarch64_df = raw_to_df("results-libm-gcc14-aarch64.csv")
    sleef_gcc14_aarch64_df = raw_to_df("results-sleef-gcc14-aarch64.csv")
    print(libm_gcc14_aarch64_df.head(3))
    print(sleef_gcc14_aarch64_df.head(3))
    libm_gcc14_aarch64_scalar_double_u10_df = filter_results(
        libm_gcc14_aarch64_df, precision="double", accuracy="u10", variant="scalar"
    )
    print(libm_gcc14_aarch64_scalar_double_u10_df.head(3))

    sleef_gcc14_aarch64_scalar_double_u10_df = filter_results(
        sleef_gcc14_aarch64_df, precision="double", accuracy="u10", variant="scalar"
    )
    print(sleef_gcc14_aarch64_scalar_double_u10_df.head(3))
