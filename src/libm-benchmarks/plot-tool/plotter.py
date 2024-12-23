import plotly.graph_objects as go
import pandas as pd


def get_legend(variant, arch):
    if arch == "aarch64":
        if variant == "vector128":
            return "AdvSIMD"
        if variant == "sve":
            return "SVE 256bit"
    if arch == "x86":
        if variant == "vector128":
            return "SSE"
        if variant == "vector256":
            return "AVX2"
        if variant == "vector512":
            return "AVX512"
    return variant


# Given a filtered dataframe, it extracts the data necessary in this dataframe
# for the graph we want to build:
# Information for the x axis: Function and interval data
# Information for the y axis: Performance values (Total Time and Throughput)
# The headers will be changed so that they contain the metric and the variant they relate to
# Example: Total Time - sleef scalar
# This is convenient for the graphing step, where the first part can be used to further filtering
# the dataframe, and the latter part will be used as legends for the plots in the graph.
def extract_coordinates(filtered_df, extra_legend=""):
    graph_df = pd.DataFrame({})
    # x axis
    graph_df["Fun-Interval"] = filtered_df["Function"] + filtered_df["Interval"]
    variant = filtered_df.iloc[0]["Variant"]
    arch = filtered_df.iloc[0]["Architecture"]
    # y axis
    legend = (
        f'{filtered_df.iloc[0]["Library"]} {get_legend(variant, arch)} {extra_legend}'
    )
    graph_df["Throughput - " + legend] = filtered_df["Throughput"]
    graph_df["Total Time - " + legend] = filtered_df["Total Time"]
    graph_df = graph_df.set_index("Fun-Interval")
    return graph_df


# Given a dataframe with all the information necessary to fill x and y values, so that
# we can plot a performance graph.
# The y_col value determines the performance metric that will be shown in the y axis in the graph
# It will also be used to further filter and process the dataframes.
def plot_graph_from_coordinates(
    coordinates_df, y_col="Throughput", graph_title="graph-pr-acc-comp-arch"
):
    # y_col can be Throughput, Total Time, Throughput Ratio or Total Time Ratio.
    # In the coordinates_df, the values don't show ratios, which means they only
    # contain Total Time and Throughput information. We further filter this
    # dataframe so that it only shows the quantity we need (Total Time or Throughput)
    # according to the first word in of y_col
    coordinates_df = coordinates_df.filter(like=y_col.split()[0], axis=1)

    ratio = "ratio" in y_col.lower()
    speedup = "speedup" in y_col.lower()
    if ratio:
        # The program will fail here (as expected) if reference not provided
        # In order to divide all the columns by the reference, we convert the reference
        # into series and divide all columns in this dataframe by the ref series.
        # A trick to convert a column in a dataframe into series is transposing it
        # and then applying iloc function to it
        ref_df = coordinates_df.filter(regex="ref").T.iloc[0]
        coordinates_df = pd.DataFrame(
            coordinates_df.values / ref_df.values[:, None],
            index=coordinates_df.index,
            columns=coordinates_df.columns,
        )
    elif speedup:
        # The program will fail here (as expected) if reference not provided
        # In order to divide all the columns by the reference, we convert the reference
        # into series and divide all columns in this dataframe by the ref series.
        # A trick to convert a column in a dataframe into series is transposing it
        # and then applying iloc function to it
        ref_df = coordinates_df.filter(regex="ref").T.iloc[0]
        coordinates_df = pd.DataFrame(
            ref_df.values[:, None] / coordinates_df.values,
            index=coordinates_df.index,
            columns=coordinates_df.columns,
        )

    # fix naming in y axis by adding units (ratio does not have units)
    elif "throughput" in y_col.lower():
        y_col = f"{y_col} (ns/el)"
    elif "total time" in y_col.lower():
        y_col = f"{y_col} (ns)"

    x_vector = coordinates_df.index
    fig = go.Figure()
    for (columnName, columnData) in coordinates_df.items():
        # In ratio mode, ref is just an horizontal line y=1.0
        # In the coordinates dataframe, the columns headers are expected to be filled
        # in a way that they contain what metric the column contains (Total Time or Throughput),
        # and what variant the results belong to (sleef scalar, sleef sve ...), so they look like
        # Throughput - sleef scalar for example.
        # The first part was used earlier for the y_axis naming
        # We use the second part after "-" for the legends title (in this case example
        # would be "sleef scalar")
        legend = columnName.split("-")[1]
        if "ref" in columnName and (ratio or speedup):
            fig.add_trace(
                go.Scatter(
                    x=x_vector,
                    y=[1 for x in x_vector],
                    name=legend,
                    line=dict(width=2, dash="dash"),
                    mode="lines",
                )
            )
            continue
        fig.add_trace(go.Bar(name=legend, x=x_vector, y=columnData))

    # Configure Title
    # The graphtitle parameter passed on to this function should take the
    # following format graph-{precision}-{accuracy}-{compiler}-{architecture}-{machine}
    _, precision, accuracy, _, _, machine = graph_title.split("-")
    long_acc = {"u10": "1ULP", "u35": "3.5ULP"}[accuracy]
    fig.update_layout(
        title=f"Comparison between system libm (GLIBC 2.35) and SLEEF performance<br>for {precision} precision {long_acc} functions on {machine}",
        barmode="group",
        xaxis_title="function name and interval",
        yaxis_title=y_col,
        legend_title="Variant",
        width=800,
        height=600,
    )
    return fig


# Given an array of filtered dataframes (and potentially a similar style reference dataframe)
# it merges all of them in a single dataframe with the information necessary to build the graph.
# This dataframe has "Function Interval" information as indexes, and each column will
# correspond to a performance quantity per variant (Example: Total Time - scalar)
def plot_graph(
    filtered_df_array,
    ref_df,
    y_col="Throughput (ns)",
    saving_title="graph-pr-acc-comp-arch",
):
    coordinates_df_array = [extract_coordinates(df) for df in filtered_df_array]

    # Use outer join operation ("pd.concat") to merge variant dataframes, as we are
    # interested in showing performance in all functions supported.
    # We substitute the resulting nan values by 0.
    graph_df = pd.concat(coordinates_df_array, axis=1)
    if graph_df.isna().any(axis=None):
        print("Warning: join resulted in nan values")
        graph_df = graph_df.fillna(0)

    if not ref_df.empty:
        # Use left join result dataframe with the reference dataframe, as we are interested
        # in comparing performance with the functions present in the result dataframe, so
        # any other function that is not present there, we are not interested to present
        # in output graph.
        # (No nan values should be produced)
        coordinates_df_ref = extract_coordinates(ref_df, extra_legend="(ref)")
        graph_df = pd.concat([graph_df, coordinates_df_ref], axis=1).reindex(
            graph_df.index
        )
    return plot_graph_from_coordinates(graph_df, y_col=y_col, graph_title=saving_title)
