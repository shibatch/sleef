#!/usr/bin/env python3

# Python Libraries Imports
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import os

# Auxiliar File Imports
import preprocessing_validators as ppv
import preprocessing as pp
import plotter as pl

# This function orchestrates everything: from data processing to graph plotting.
# It is organized as follows:
# 1. Define command line arguments
# 2. Validate command line arguments
# 3. Processing (Stage 1): Convert raw results into organized dataframe
# 4. Processing (Stage 2): Filter results
# 5. Generate graph plot over filtered results
# 6. Save image
# Example command line invocation:
# ./src/libm-benchmarks/plot-tool/plot_results.py
#             --result-file <path to result file>
#             --reference-file <path to reference result file>
#             -v scalar vector128 sve -p double -a u35 -d -y "Throughput Ratio" -o graphs


def main():
    # Define command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--result-file",
        type=Path,
        required=True,
        help="File with benchmark results (csv format)",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        required=False,
        help="File with reference benchmark results (csv format)",
    )
    parser.add_argument(
        "-y",
        "--y-axis",
        choices=[
            "Total Time",
            "Total Time Ratio",
            "Throughput",
            "Throughput Ratio",
            "Throughput Speedup",
        ],
        default="Throughput",
        help="Quantity tracked by y axis",
    )
    parser.add_argument(
        "-v",
        "--variant",
        nargs="+",
        choices=["scalar", "vector128", "vector256", "vector512", "sve"],
        required=True,
        help="Which variant to plot",
    )
    parser.add_argument(
        "-m",
        "--machine",
        required=True,
        help="Which machine did the benchmarks occured on",
    )
    parser.add_argument(
        "-p",
        "--precision",
        choices=["double", "single"],
        required=True,
        help="Which precision to plot",
    )
    parser.add_argument(
        "-a",
        "--accuracy",
        choices=["u10", "u35"],
        required=True,
        help="Which accuracy to plot",
    )
    parser.add_argument(
        "-d",
        "--drop-intervals",
        action="store_true",
        help="Keep one interval per function (if intervals are sorted will keep lowest interval)",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        required=False,
        help="Directory to save output",
    )
    args = parser.parse_args()

    # Validate command line arguments
    results_filename = str(args.result_file)
    ref_results_filename = str(args.reference_file)
    library, architecture, compiler = ppv.filename_validator(results_filename)

    # Convert raw results into organized dataframe
    sleef_df_raw = pp.raw_to_df(results_filename)
    precision = ppv.valid_precision(args.precision)
    accuracy = ppv.valid_accuracy(args.accuracy)

    # Filter results by variant, precision, accuracy
    # One dataframe per variant
    filtered_dfs = []
    for v in args.variant:
        variant = ppv.valid_variant(v)
        filtered_df = pp.filter_results(
            sleef_df_raw,
            precision=precision,
            accuracy=accuracy,
            variant=variant,
            keep_lower_interval=args.drop_intervals,
        )
        filtered_dfs.append(filtered_df)

    # If reference provided, repeat similar process
    ref_filtered_df = pd.DataFrame({"A": []})
    if args.reference_file:
        library_ref, architecture_ref, compiler_ref = ppv.filename_validator(
            ref_results_filename
        )
        assert (
            architecture == architecture_ref
            and compiler == compiler_ref
            and library != library_ref
        )
        # Convert raw results into organized dataframe
        ref_df_raw = pp.raw_to_df(ref_results_filename)
        # Filter results by variant, precision, accuracy
        # Note: for now we fix u10 scalar routines in the reference library (ie libm) for comparison.
        ref_filtered_df = pp.filter_results(
            ref_df_raw,
            precision=precision,
            accuracy="u10",
            variant="scalar",
            keep_lower_interval=args.drop_intervals,
        )

    # Plot results
    graph_plot = pl.plot_graph(
        filtered_dfs,
        ref_df=ref_filtered_df,
        y_col=args.y_axis,
        saving_title=f"graph-{precision}-{accuracy}-{compiler}-{architecture}-{args.machine}",
    )

    if not args.output_directory.is_dir():
        os.mkdir(args.output_directory)

    graph_plot.write_image(
        f"{args.output_directory}/graph-{precision}-{accuracy}-{compiler}-{architecture}.png",
        format="png",
    )
    return


if __name__ == "__main__":
    main()
