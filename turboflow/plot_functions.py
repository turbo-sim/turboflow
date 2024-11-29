import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

def load_data(filename):
    """
    Load performance data from an Excel file.

    This function imports an Excel file containing performance parameters. It reads the data into a dictionary
    named 'performance_data' where each key corresponds to a sheet name in the Excel file, and each value is
    a pandas DataFrame containing the data from that sheet. The data is rounded to avoid precision loss when
    loading from Excel.

    The excel file must contain the following sheets:

        - `operation point`
        - `plane`
        - `cascade`
        - `overall`

    Parameters
    ----------
    filename : str
        The name of the Excel file containing performance parameters.

    Returns
    -------
    dict
        A dictionary containing performance data, where keys are sheet names and values are DataFrames.

    """

    # Read excel file
    performance_data = pd.read_excel(
        filename,
        sheet_name=[
            "operation point",
            "plane",
            "cascade",
            # "stage",
            "overall",
        ],
    )

    # Round off to ignore precision loss by loading data from excel
    for key, df in performance_data.items():
        performance_data[key] = df.round(10)

    return performance_data


def plot_lines(
    performance_data,
    x_key,
    y_keys,
    subsets=None,
    fig=None,
    ax=None,
    xlabel="",
    ylabel="",
    title="",
    labels=None,
    filename=None,
    outdir="figures",
    stack=False,
    color_map="viridis",
    colors=None,
    close_fig=False,
    save_figs=False,
    linestyles=None,
    markers = None,
    save_formats=[".png"],
    legend_loc="best",
):
    """
    Plot lines from performance data.

    This function plots lines from performance data. It supports plotting multiple lines on the same axes
    with customizable labels, colors, and linestyles. The resulting plot can be saved to a file if desired.

    Parameters
    ----------
    performance_data : DataFrame
        The performance data to plot.
    x_key : str
        Name of the column in performance_data to be used as x-axis values.
    y_keys : list of str
        Names of the columns in performance_data to be plotted.
    subsets : list, optional
        Name and value of subsets to plot from performance_data. First instance should be a string representing the column name, while the
        remaining elemnts represet the values defining the subset. Default is None.
    fig : matplotlib.figure.Figure, optional
        An existing figure object. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        An existing axes object. If None, a new axes is created.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    title : str, optional
        The title of the plot.
    labels : list of str, optional
        Labels for the plotted lines. Default is None.
    filename : str, optional
        The filename for saving the figure. Default is None.
    outdir : str, optional
        The directory where figures should be saved. Default is 'figures'.
    stack : bool, optional
        Whether to stack the plotted lines. Default is False.
    color_map : str, optional
        The colormap used for the lines. Default is 'viridis'.
    colors : list of str or colors, optional
        Colors for the plotted lines. Default is None.
    close_fig : bool, optional
        Whether to close the figure after plotting. Default is False.
    save_figs : bool, optional
        Whether to save the figure. Default is False.
    linestyles : list of str, optional
        Linestyles for the plotted lines. Default is None.
    save_formats : list of str, optional
        File formats for saving the figure. Default is ['.png'].
    legend_loc : str, optional
        Location for the legend. Default is 'best'.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    matplotlib.axes.Axes
        The axes object.

    """

    # Create figure if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Specify title and axes labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Get data
    if subsets == None:
        x = get_lines(performance_data, x_key)
        y = get_lines(performance_data, y_keys)
    else:
        x = get_lines(performance_data, x_key, subsets=subsets)
        y = get_lines(performance_data, y_keys, subsets=subsets)

    # Get colors
    if not isinstance(colors, (list,np.ndarray)):
        colors = plt.get_cmap(color_map)(np.linspace(0.2, 1.0, len(y)))

    # Get labels
    if labels == None:
        if subsets is not None and len(subsets) >2:
            labels = [f"{subsets[0]} = {subsets[i+1]}" for i in range(len(subsets) - 1)]
        elif len(y_keys) > 1:
            labels = [f"{y_keys[i]}" for i in range(len(y))]
        else:
            labels = y_keys

    # Get linestyles
    if linestyles == None:
        linestyles = ["-"] * len(y)
    if markers == None:
        markers = ["none"] * len(y)

    # Plot figure
    if subsets is not None and len(y_keys) == 1:
        for i in range(len(y)):
            ax.plot(x[i], y[i], label=labels[i], color=colors[i], linestyle=linestyles[i], marker = markers[i])
    elif subsets is not None and len(subsets) == 2:
        if stack == True:
            ax.stackplot(x[0], y, labels=labels, colors=colors)
            # Add edges by overlaying lines
            y_arrays = [series.values for series in y]
            cumulative_y = np.cumsum(y_arrays, axis=0)
            for i, series in enumerate(y_arrays):
                if i == 0:
                    ax.plot(x[0], series, color="black", linewidth=0.5)
                ax.plot(x[0], cumulative_y[i], color="black", linewidth=0.5)
        else:
            for i in range(len(y)):
                ax.plot(
                    x[0], y[i], label=labels[i], color=colors[i], linestyle=linestyles[i], marker = markers[i]
                )
    elif subsets is None:
        if stack == True:
            ax.stackplot(x, y, labels=labels, colors=colors)
            # Add edges by overlaying lines
            y_arrays = [series.values for series in y]
            cumulative_y = np.cumsum(y_arrays, axis=0)
            for i, series in enumerate(y_arrays):
                if i == 0:
                    ax.plot(x, series, color="black", linewidth=0.5)
                ax.plot(x, cumulative_y[i], color="black", linewidth=0.5)
        else:
            for i in range(len(y)):
                ax.plot(x, y[i], label=labels[i], color=colors[i], linestyle=linestyles[i], marker = markers[i])
    else:
        raise Exception("x_keys and y_keys are not well defined")

    # Set margins to zero
    ax.margins(x=0.01, y=0.01)

    # Add legend
    if len(y) > 1:
        ax.legend(loc=legend_loc)
    fig.tight_layout(pad=1, w_pad=None, h_pad=None)

    # Create output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Save figure
    if save_figs:
        base_filename = os.path.join(outdir, filename)
        savefig_in_formats(fig, base_filename, formats=save_formats)

    if close_fig:
        plt.close(fig)

    return fig, ax


def get_lines(performance_data, column_name, subsets=None):
    """
    Retrieve lines of data from the specified column in `performance_data`.

    This function returns a list of array from the specified column(s) of the performance data.
    If no subset is specified, it returns lines covering all rows in `performance_data`. If a subset
    is specified, it returns lines covering only the rows that match the specified subset.

    Parameters
    ----------
    performance_data : DataFrame
        The DataFrame containing the performance data.
    column_name : str or list of str
        The name(s) of the column(s) from which to retrieve data.
    subsets : list, optional
        Name and value of subsets to get data from performance_data. First instance should be a string representing the column name, while the
        remaining elemnts represet the values defining the subset. Default is None.

    Returns
    -------
    list
        A list of arrays from the specified column(s) of performance_data.

    """

    # Get lines covering all rows in performance_data
    if subsets == None:
        # Get single line
        if isinstance(column_name, str):
            return get_column(performance_data, column_name)
        # Get several columns
        else:
            lines = []
            for column in column_name:
                lines.append(get_column(performance_data, column))
            return lines

    subsets[1:] = [round(value, 10) for value in subsets[1:]]
    # Get lines covering given subset
    lines = []
    for val in subsets[1:]:
        if isinstance(column_name, str):
            indices = get_subset(performance_data, subsets[0], val)
            lines.append(get_column(performance_data, column_name)[indices])
        else:
            for column in column_name:
                indices = get_subset(performance_data, subsets[0], val)
                lines.append(get_column(performance_data, column)[indices])

    return lines


def find_column(performance_data, column_name):
    """
    Find the sheet containing the specified column in `performance_data`.

    This function searches through all sheets in `performance_data` to find the one containing the specified column.
    If the column is found, it returns the name of the sheet. If the column is not found, it raises an exception.

    Parameters
    ----------
    performance_data : dict
        A dictionary containing sheets of performance data, where keys are sheet names and values are DataFrames.
    column_name : str
        The name of the column to find.

    Returns
    -------
    str
        The name of the sheet containing the specified column.

    Raises
    ------
    Exception
        If the specified column is not found in any sheet of performance_data.

    """

    for key in performance_data.keys():
        if any(element == column_name for element in performance_data[key].columns):
            return key

    raise Exception(f"Could not find column {column_name} in performance_data")


def get_column(performance_data, column_name):
    """
    Retrieve a column of data from `performance_data`.

    This function retrieves the specified column of data from `performance_data` by finding the sheet containing
    the column using the `find_column` function. It then returns the column as a pandas Series.

    Parameters
    ----------
    performance_data : dict
        A dictionary containing sheets of performance data, where keys are sheet names and values are DataFrames.
    column_name : str
        The name of the column to retrieve.

    Returns
    -------
    pandas.Series
        The column of data corresponding to column_name.

    """
    sheet = find_column(performance_data, column_name)

    return performance_data[sheet][column_name]


def get_subset(performance_data, column_name, row_value):
    """
    Retrieve the index of rows in `performance_data` where `column_name` equals `row_value`.

    This function retrieves the index of rows in `performance_data` where the specified column (`column_name`)
    equals the specified value (`row_value`). It first finds the sheet containing the column using the
    `find_column` function, then returns the index of rows where the column has the specified value.

    Parameters
    ----------
    performance_data : dict
        A dictionary containing sheets of performance data, where keys are sheet names and values are DataFrames.
    column_name : str
        The name of the column to search for `row_value`.
    row_value : object
        The value to match in `column_name`.

    Returns
    -------
    pandas.Index
        The index of rows where `column_name` equals `row_value`.

    """
    sheet = find_column(performance_data, column_name)

    return performance_data[sheet][
        performance_data[sheet][column_name] == row_value
    ].index


def savefig_in_formats(
    fig, path_without_extension, formats=[".png", ".svg", ".pdf", ".eps"]
):
    """
    Save a given matplotlib figure in multiple file formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to be saved.
    path_without_extension : str
        The full path to save the figure excluding the file extension.
    formats : list of str, optional
        A list of string file extensions to specify which formats the figure should be saved in.
        Default is ['.png', '.svg', '.pdf', ".eps"].

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> save_fig_in_formats(fig, "/path/to/figure/filename")

    This will save the figure as "filename.png", "filename.svg", and "filename.pdf" in the "/path/to/figure/" directory.
    """
    for ext in formats:
        fig.savefig(f"{path_without_extension}{ext}", bbox_inches="tight")


def plot_axial_radial_plane(geometry):
    r"""
    Plot the turbine geometry in an axial-radial plane.

    This function plots the turbine geometry in an axial-radial plane. It takes the turbine geometry data
    as input, including the radii at the inner and outer hub and tip, the number of cascades, and the axial
    chord lengths. It then constructs and displays the plot, which represents the turbine blades in the
    axial-radial plane.

    Parameters
    ----------
    geometry : dict
        A dictionary containing turbine geometry data including:

        - `radius_hub_in` (array-like) : Inner hub radii at each cascade.
        - `radius_hub_out` (array-like) : Outer hub radii at each cascade.
        - `radius_tip_in` (array-like) : Inner tip radii at each cascade.
        - `radius_tip_out` (array-like) : Outer tip radii at each cascade.
        - `number_of_cascades` (int) : Number of cascades in the turbine.
        - `axial_chord` (array-like) : Axial chord lengths at each cascade.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    matplotlib.axes.Axes
        The generated axes.

    """

    # Load data
    radius_hub_in = geometry["radius_hub_in"]
    radius_hub_out = geometry["radius_hub_out"]
    radius_tip_in = geometry["radius_tip_in"]
    radius_tip_out = geometry["radius_tip_out"]
    number_of_cascades = geometry["number_of_cascades"]
    axial_chord = geometry["axial_chord"]
    tip_clearance = geometry["tip_clearance"]

    # Define unspecified geometrical parameters
    ct = 0.05  # Thickness of casing relative to maximum tip radius
    cl = 1.2  # Length of casing to turbine
    cs = 0.1  # Axial spacing between cascades relative to max axial chord

    # Get x-points
    dx = cs * max(axial_chord)
    x = np.array([])
    for i in range(len(axial_chord)):
        x = np.append(x, np.sum(axial_chord[0 : i + 1]) + i * dx)
        x = np.append(x, np.sum(axial_chord[0 : i + 1]) + (i + 1) * dx)
    x[1:] = x[0:-1]
    x[0] = 0

    # Get tip and hub points
    y_hub = np.array(
        [val for pair in zip(radius_hub_in, radius_hub_out) for val in pair]
    )
    y_tip = np.array(
        [val for pair in zip(radius_tip_in, radius_tip_out) for val in pair]
    )

    # Initialize figure and axis object
    fig, ax = plt.subplots()

    # Plot upper casing
    tip_clearance_extended = np.array(
        [val for pair in zip(tip_clearance, tip_clearance) for val in pair]
    )  # Extend tip clearance vector
    casing_thickness = max(y_tip) * ct
    x_casing = x
    x_casing = np.insert(x_casing, 0, x[0] - (cl - 1) * x[-1] / 2)
    x_casing = np.append(x_casing, x[-1] + (cl - 1) * x[-1] / 2)

    y_casing_lower = y_tip + tip_clearance_extended
    y_casing_lower = np.insert(y_casing_lower, 0, y_casing_lower[0])
    y_casing_lower = np.append(y_casing_lower, y_casing_lower[-1])
    y_casing_upper = np.ones(len(x_casing)) * (max(y_tip) + casing_thickness)

    ax.plot(x_casing, y_casing_lower, "k")
    ax.plot(x_casing, y_casing_upper, "k")
    ax.plot([x_casing[0], x_casing[0]], [y_casing_lower[0], y_casing_upper[0]], "k")
    ax.plot([x_casing[-1], x_casing[-1]], [y_casing_lower[-1], y_casing_upper[-1]], "k")
    ax.fill_between(x_casing, y_casing_lower, y_casing_upper, color="0.5")

    # Plot lower casing
    tip_clearance_extended = np.array(
        [
            val
            for pair in zip(np.flip(tip_clearance), np.flip(tip_clearance))
            for val in pair
        ]
    )  # Extend tip clearance vector
    y_casing_upper = y_hub - tip_clearance_extended
    y_casing_upper = np.insert(y_casing_upper, 0, y_casing_upper[0])
    y_casing_upper = np.append(y_casing_upper, y_casing_upper[-1])
    y_casing_lower = y_casing_upper - casing_thickness

    ax.plot(x_casing, y_casing_lower, "k")
    ax.plot(x_casing, y_casing_upper, "k")
    ax.plot([x_casing[0], x_casing[0]], [y_casing_lower[0], y_casing_upper[0]], "k")
    ax.plot([x_casing[-1], x_casing[-1]], [y_casing_lower[-1], y_casing_upper[-1]], "k")
    ax.fill_between(x_casing, y_casing_lower, y_casing_upper, color="0.5")

    # Plot cascades
    for i in range(number_of_cascades):
        ax.plot(x[i * 2 : (i + 1) * 2], y_tip[i * 2 : (i + 1) * 2], "k")  # Plot tip
        ax.plot(x[i * 2 : (i + 1) * 2], y_hub[i * 2 : (i + 1) * 2], "k")  # Plot hub
        ax.plot(
            [x[i * 2 : (i + 1) * 2][0], x[i * 2 : (i + 1) * 2][0]],
            [y_hub[i * 2 : (i + 1) * 2][0], y_tip[i * 2 : (i + 1) * 2][0]],
            "k",
        )  # Plot inlet vertical line
        ax.plot(
            [x[i * 2 : (i + 1) * 2][1], x[i * 2 : (i + 1) * 2][1]],
            [y_hub[i * 2 : (i + 1) * 2][1], y_tip[i * 2 : (i + 1) * 2][1]],
            "k",
        )  # Plot exit vertical line
        ax.fill_between(
            x[i * 2 : (i + 1) * 2],
            y_hub[i * 2 : (i + 1) * 2],
            y_tip[i * 2 : (i + 1) * 2],
            color="1",
        )  # Fill cascade with color

    # Plot axis of rotation
    plt.annotate(
        "",
        xy=(x[-1], 0),
        xytext=(0, 0),
        arrowprops=dict(facecolor="black", arrowstyle="-|>"),
    )

    ax.grid(False)
    ax.tick_params(which="both", direction="out", right=False)
    ax.tick_params(which="minor", left=False)
    # ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_xlabel("Axis of rotation")
    ax.set_ylabel("Radius")
    ax.set_aspect("equal")
    lims = ax.get_ylim()
    ax.set_ylim([0, lims[-1]])
    plt.show()

    return fig, ax

def plot_velocity_triangle_stage(plane,
                                 fig = None,
                                 ax = None,
                                 color = None,
                                 linestyle = None,
                                 fontsize = 12,
                                 hs = 15,
                                 start_y = 0,
                                 start_x = 0,
                                 dy = 0.1,
                                 dx = 0):

    """
    Plot velocity triangles for each stage of a turbine.

    Parameters
    ----------
    plane : pandas.DataFrame
        A DataFrame containing velocity components for each plane of the turbine.
        The DataFrame should have the following keys:
        - 'v_m': Meridional velocities.
        - 'v_t': Tangential velocities.
        - 'w_m': Meridional velocities of relative motion.
        - 'w_t': Tangential velocities of relative motion.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the velocity triangle plot.

    matplotlib.axes._subplots.AxesSubplot
        The matplotlib axis object containing the velocity triangle plot.

    """

    # Get indices of inlet and exit of each rotor plane
    indices = np.array(
        [val for pair in zip(plane.index[2::4], plane.index[3::4]) for val in pair]
    )
    data = plane.iloc[indices]

    # Load variables
    v_ms = data["v_m"].values
    v_ts = data["v_t"].values
    w_ms = data["w_m"].values
    w_ts = data["w_t"].values

    # Get maximum an minimum tangential velocities
    max_w_t = max(np.array([val for pair in zip(v_ts, w_ts) for val in pair]))
    min_w_t = min(np.array([val for pair in zip(v_ts, w_ts) for val in pair]))

    # initialize figure and axis
    if fig == None:
        fig, ax = plt.subplots()

    # Calculate number of stages
    number_of_stages = int(np.floor(len(plane)/4))

    # Plot stage velocity triangles
    for i in range(number_of_stages):
        fig, ax = plot_velocity_triangle(fig, ax, start_x, start_y, w_ts[2*i], v_ts[2*i], v_ms[2*i], w_ms[2*i], hs = hs, fontsize = fontsize, lines = "total", color = color, linestyle = linestyle, cascade = 's')
        fig, ax = plot_velocity_triangle(fig, ax, start_x, start_y, w_ts[2*i + 1], v_ts[2*i+ 1], v_ms[2*i+ 1], w_ms[2*i+ 1], hs = hs, fontsize = fontsize, lines = "total", color = color, linestyle = linestyle, cascade = 'r')

        start_x += dx
        start_y += dy

    # Set plot options
    ax.grid(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig, ax

def plot_velocity_triangle(fig, ax, start_x, start_y, w_t, v_t, v_m, w_m, hs = 10, fontsize = 12, lines = "all", color = None, linestyle = None, cascade = ''):

    # Plot triangle
    if lines == "all":
        ax.plot([start_x, start_x], [start_y, start_y - v_m], color=color, linestyle=linestyle)  # Plot meridional velocity
        ax.plot([start_x, v_t], [start_y - v_m, start_y - v_m], color=color, linestyle=linestyle)  # Plot absolute tangential velocity
        ax.plot([start_x, w_t], [start_y - v_m, start_y - v_m], color=color, linestyle=linestyle)  # Plot relative tangential velocity
        ax.plot([start_x, v_t], [start_y, start_y - v_m], color=color, linestyle = linestyle)  # Plot abolsute velocity
        ax.plot([start_x, w_t], [start_y, start_y - w_m], color=color, linestyle = linestyle)  # Plot relative velocity
        ax.plot([w_t, v_t], [start_y - w_m, start_y - w_m], color = color, linestyle = linestyle)  # Plot blade speed
    elif lines == "total":
        ax.plot([start_x, v_t], [start_y, start_y - v_m], color=color, linestyle = linestyle)  # Plot abolsute velocity
        ax.plot([start_x, w_t], [start_y, start_y - w_m], color=color, linestyle = linestyle)  # Plot relative velocity
        ax.plot([w_t, v_t], [start_y - w_m, start_y - w_m], color = color, linestyle = linestyle)  # Plot blade speed
    elif lines == "absolute":
        ax.plot([start_x, start_x], [start_y, start_y - v_m], color=color, linestyle=linestyle)  # Plot meridional velocity
        ax.plot([start_x, v_t], [start_y - v_m, start_y - v_m], color=color, linestyle=linestyle)  # Plot absolute tangential velocity
        ax.plot([start_x, v_t], [start_y, start_y - v_m], color=color, linestyle = linestyle)  # Plot abolsute velocity
    elif lines == "relative":
        ax.plot([start_x, start_x], [start_y, start_y - v_m], color=color, linestyle=linestyle)  # Plot meridional velocity
        ax.plot([start_x, w_t], [start_y - v_m, start_y - v_m], color=color, linestyle=linestyle)  # Plot relative tangential velocity
        ax.plot([start_x, w_t], [start_y, start_y - w_m], color=color, linestyle = linestyle)  # Plot relative velocity
        ax.plot([w_t, v_t], [start_y - w_m, start_y - w_m], color = color, linestyle = linestyle)  # Plot blade speed

    # Arrows
    plt.annotate(
        "",
        xy=(v_t / 2, (start_y + start_y - v_m) / 2),
        xytext=(start_x, start_y),
        arrowprops=dict( color = color, arrowstyle="-|>", linestyle = "none"),
    )  # Arrow for absolute velocity
    plt.annotate(
                "",
                xy=(w_t / 2, (start_y + start_y - w_m) / 2),
                xytext=(start_x, start_y),
                arrowprops=dict(color = color, arrowstyle="-|>", linestyle = "none"),
            )  # Arrow for relative velocity
    plt.annotate(
                "",
                xy=((w_t + v_t) / 2, start_y - w_m),
                xytext=(w_t, start_y - w_m),
                arrowprops=dict(color = color, arrowstyle="-|>", linestyle = "none"),
            )  # Arrow for blade speed
    
    ax.set_aspect('equal')

    return fig, ax

def plot_velocity_triangles_planes(plane):
    """
    Plot velocity triangles for each plane of a turbine.

    Parameters
    ----------
    plane : pandas.DataFrame
        A DataFrame containing velocity components for each plane of the turbine.
        The DataFrame should have the following keys:
        - 'v_m': Meridional velocities.
        - 'v_t': Tangential velocities.
        - 'w_m': Meridional velocities of relative motion.
        - 'w_t': Tangential velocities of relative motion.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the velocity triangle plot.

    matplotlib.axes._subplots.AxesSubplot
        The matplotlib axis object containing the velocity triangle plot.

    """

    # Load variables
    v_ms = plane["v_m"]
    v_ts = plane["v_t"]
    w_ms = plane["w_m"]
    w_ts = plane["w_t"]

    # Get maximum an minimum tangential velocities
    max_w_t = max(np.array([val for pair in zip(v_ts, w_ts) for val in pair]))
    min_w_t = min(np.array([val for pair in zip(v_ts, w_ts) for val in pair]))

    # Nameing of planes (for marking)
    plane_name = ["inlet", "exit"]

    # Tuning options
    dx = 0.1  # Spacing between triangles
    hs = 10  # Horizontal spacing between line and symbol (absolute)
    start = 0  # Starting point in y direction
    fontsize = 12  # Fontsize

    # Get rotor indexes
    i_rotor = np.array(
        [val for pair in zip(plane.index[2::4], plane.index[3::4]) for val in pair]
    )

    # initialize figure and axis
    fig, ax = plt.subplots()

    # Plot velocity triangles
    for i in range(len(plane)):

        v_m = v_ms[i]
        v_t = v_ts[i]

        w_m = w_ms[i]
        w_t = w_ts[i]

        # Define cascade name
        cascade = "Stator"  # Changed if rotor (see below)

        # Plot absolute velocity
        ax.plot(
            [0, 0], [start, start - v_m], color="k", linestyle=":"
        )  # Plot meridional velocity
        ax.plot(
            [0, v_t], [start - v_m, start - v_m], color="k", linestyle=":"
        )  # Plot tangntial velocity
        ax.plot([0, v_t], [start, start - v_m], "k")  # Plot velocity
        x_mid = v_t / 2
        y_mid = (start + start - v_m) / 2
        plt.annotate(
            "",
            xy=(x_mid, y_mid),
            xytext=(0, start),
            arrowprops=dict(facecolor="black", arrowstyle="-|>"),
        )  # Arrow for absolute velocity
        sign = -1 if x_mid < 0 else 1
        ax.text(
            x_mid + hs * sign,
            y_mid,
            f"$v$",
            fontsize=fontsize,
            ha="center",
            va="bottom",
            color="k",
        )  # Symbol for absolute velocity

        # Plot relative velocity
        if i in i_rotor:

            cascade = "Rotor"
            ax.plot(
                [0, w_t], [start - w_m, start - w_m], color="k", linestyle=":"
            )  # Plot tangntial velocity
            ax.plot([0, w_t], [start, start - w_m], color="k")  # Plot velocity
            ax.plot([w_t, v_t], [start - w_m, start - w_m], "k")  # Plot blade speed
            x_mid = w_t / 2
            y_mid = (start + start - w_m) / 2
            plt.annotate(
                "",
                xy=(x_mid, y_mid),
                xytext=(0, start),
                arrowprops=dict(facecolor="black", arrowstyle="-|>"),
            )  # Arrow for relative velocity
            sign = -1 if x_mid < 0 else 1
            ax.text(
                x_mid + hs * np.sign(x_mid),
                y_mid,
                f"$w$",
                fontsize=fontsize,
                ha="center",
                va="bottom",
                color="k",
            )  # Symbol for relative velocity
            x_mid = (w_t + v_t) / 2
            y_mid = start - w_m
            plt.annotate(
                "",
                xy=(x_mid, y_mid),
                xytext=(w_t, start - w_m),
                arrowprops=dict(facecolor="k", arrowstyle="-|>"),
            )  # Arrow for blade speed
            ax.text(
                x_mid,
                y_mid,
                f"$u$",
                fontsize=fontsize,
                ha="center",
                va="bottom",
                color="k",
            )  # Symbol for blade speed

        # Add explanatory text
        if abs(v_t) >= abs(w_t):
            tangential_component = v_t
        else:
            tangential_component = w_t
        if tangential_component < 0:
            ax.text(
                max_w_t / 2,
                start - v_m / 2,
                f"{cascade} {plane_name[i%2]}",
                fontsize=fontsize,
                ha="center",
                va="center",
                color="k",
            )
        else:
            ax.text(
                min_w_t / 2,
                start - v_m / 2,
                f"{cascade} {plane_name[i%2]}",
                fontsize=fontsize,
                ha="center",
                va="center",
                color="k",
            )

        # Plot line diciding the different triangles
        ax.plot(
            [min_w_t, max_w_t], [start - w_m - dx * v_m, start - w_m - dx * v_m], "k:"
        )

        # Update starting point for the next velocity triangle
        start = start - w_m - 2 * dx * v_m

    # Set plot options
    ax.grid(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.show()

    return fig, ax

import matplotlib.pyplot as plt

def combine_multiple_figures(figures, labels=None, markers=None, linestyles=None, colors=None, y_labels = None):
    """
    Combine multiple matplotlib figures into a single figure with custom labels, markers, linestyles, and colors.

    This function extracts the lines from all axes of the given figures 
    and plots them onto a new figure. The resulting figure contains 
    the plots from all corresponding axes in the input figures, with 
    custom labels, markers, linestyles, and colors applied to each line.

    Parameters
    ----------
    figures : list of matplotlib.figure.Figure
        The list of figures to be combined.
    labels : list of str, optional
        The list of labels. 
        If not provided, default labels 'fig1', 'fig2', etc. will be used.
    markers : list of str, optional
        The list of markers to use for the lines from each figure. If not 
        provided, default markers will be used.
    linestyles : list of str, optional
        The list of linestyles to use for the lines from each figure. If not 
        provided, default linestyles will be used.
    colors : list of str, optional
        The list of colors to use for the lines from each figure. If not 
        provided, default colors will be used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The new figure containing the combined plots.
    axes : list of matplotlib.axes.Axes
        The list of axes objects in the new figure.

    Example
    -------
    >>> fig1, ax1 = plt.subplots(2, 1)
    >>> ax1[0].plot([1, 2, 3], [1, 4, 9], label='Plot 1')
    >>> ax1[1].plot([1, 2, 3], [1, 2, 3], label='Plot 2')
    >>> fig2, ax2 = plt.subplots(2, 1)
    >>> ax2[0].plot([1, 2, 3], [9, 4, 1], label='Plot 3')
    >>> ax2[1].plot([1, 2, 3], [3, 2, 1], label='Plot 4')
    >>> fig3, ax3 = plt.subplots(2, 1)
    >>> ax3[0].plot([1, 2, 3], [2, 3, 4], label='Plot 5')
    >>> fig, axes = combine_multiple_figures(
    >>>     [fig1, fig2, fig3], 
    >>>     labels=['Figure 1', 'Figure 2', 'Figure 3'],
    >>>     markers=['o', 's', 'd'],
    >>>     linestyles=['-', '--', ':'],
    >>>     colors=['blue', 'green', 'red']
    >>> )
    >>> plt.show()
    """
    if labels is None:
        labels = [f'fig{i+1}' for i in range(len(figures))]
    
    if markers is None:
        markers = [None] * len(figures)
    
    if linestyles is None:
        linestyles = [None] * len(figures)
    
    if colors is None:
        colors = [None] * len(figures)

    if len(labels) != len(figures):
        raise ValueError("The number of labels must match the number of figures.")
    if len(markers) != len(figures):
        raise ValueError("The number of markers must match the number of figures.")
    if len(linestyles) != len(figures):
        raise ValueError("The number of linestyles must match the number of figures.")
    if len(colors) != len(figures):
        raise ValueError("The number of colors must match the number of figures.")

    # Determine the maximum number of axes across all figures
    max_num_axes = max(len(fig.get_axes()) for fig in figures)

    # Set y_labels
    if y_labels is None:
        y_labels = [""]*max_num_axes
    
    if len(y_labels) != max_num_axes:
        raise ValueError("The number of y-axis labels must match the number of axes.")

    # Create a new figure with the same number of subplots as the maximum number of axes
    fig, axes = plt.subplots(nrows=max_num_axes, ncols=1, squeeze=False)

    # Flatten axes to make iteration easier
    axes = axes.flatten()

    # Combine plots from all figures
    for i in range(max_num_axes):
        for j, (fig, label, marker, linestyle, color) in enumerate(zip(figures, labels, markers, linestyles, colors)):
            axes_list = fig.get_axes()
            if i < len(axes_list):
                for line in axes_list[i].get_lines():
                    axes[i].plot(
                        line.get_xdata(), 
                        line.get_ydata(), 
                        label=label, 
                        marker=marker, 
                        linestyle=linestyle, 
                        color=color
                    )

        # axis options
        axes[i].legend()
        axes[i].set_ylabel(y_labels[i])

    return fig, axes


def plot_optimization_results(initial_guess, lower_bounds, upper_bounds, final_solutions, 
                              markers=None, colors=None, fig=None, ax=None, labels = None, variable_names = None, fontsize = 12, markersizes = 6):
    """
    Plots the relative change from the initial guess for each variable,
    along with the bounds as an error bar style plot.

    Parameters
    ----------
    initial_guess : np.ndarray
        Array of initial guesses for the design variables.
    lower_bounds : list of float
        List of lower bounds for each design variable.
    upper_bounds : list of float
        List of upper bounds for each design variable.
    final_solutions : list of np.ndarray
        List of arrays where each array represents a set of final optimized design variables.
    markers : list of str, optional
        List of markers to use for each set of final solutions. Default is 'o'.
    colors : list of str, optional
        List of colors to use for each set of final solutions. Default is 'b', 'g', 'r', etc.
    fig : matplotlib.figure.Figure, optional
        Matplotlib Figure object. If not provided, a new figure will be created.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object. If not provided, a new axes will be created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The Axes object containing the plot.

    Examples
    --------
    >>> initial_guess = np.array([1.0, 2.0, 3.0])
    >>> lower_bounds = [0.5, 1.0, 2.0]
    >>> upper_bounds = [1.5, 3.0, 4.0]
    >>> final_solutions = [np.array([1.1, 2.5, 2.8]), np.array([0.9, 2.1, 3.2])]
    >>> fig, ax = plot_optimization_results(initial_guess, lower_bounds, upper_bounds, final_solutions)
    >>> plt.show()
    """

    num_vars = len(initial_guess)
    
    # Convert lower and upper bounds to numpy arrays
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Calculate the relative change for each solution and the bounds
    relative_changes = [(solution - initial_guess) / initial_guess for solution in final_solutions]
    relative_lower_bounds = (lower_bounds - initial_guess) / initial_guess
    relative_upper_bounds = (upper_bounds - initial_guess) / initial_guess

    # Default markers and colors if not provided
    if markers is None:
        markers = ['o'] * len(final_solutions)
    if not isinstance(markersizes, (list,np.ndarray)):
        markersizes = [markersizes]*len(final_solutions)
    if colors is None:
        colors = plt.cm.tab10.colors[:len(final_solutions)]  # Use default color cycle
    
    # Create fig and ax if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Set labels 
    if labels == None:
        labels = [f'Solution {i+1}' for i in range(len(final_solutions))]

    # Plotting the error bar for bounds as relative changes
    ax.errorbar(range(num_vars), 
                (relative_upper_bounds + relative_lower_bounds) / 2, 
                yerr=[(relative_upper_bounds - relative_lower_bounds) / 2, 
                      (relative_upper_bounds - relative_lower_bounds) / 2],
                fmt=' ', color='gray', alpha=0.5, capsize = 6)

    # Plot each final solution
    for i, rel_change in enumerate(relative_changes):
        ax.plot(range(num_vars), rel_change, linestyle = 'none', marker=markers[i], markersize=markersizes[i], label=labels[i], color=colors[i])

    # Adding labels and titles
    ax.set_ylabel('Relative Change from Initial Guess', fontsize = fontsize)
    ax.legend(fontsize = fontsize)
    ax.grid(False)
    ax.minorticks_off()

    # Set custom tick labels if provided
    if variable_names is not None:
        ax.set_xticks(range(num_vars))
        ax.set_xticklabels(variable_names, rotation=75, fontsize = fontsize)

    return fig, ax

import matplotlib.pyplot as plt

def plot_convergence_process(data, plot_together=True, colors=None, linestyles=None, labels=None):
    """
    Plots the convergence process for multiple optimization algorithms.
    
    Parameters
    ----------
    data : dict
        A dictionary where each key is the algorithm name and the value is another dictionary
        with the following keys:
        - 'objective' : list
            List of objective values at each iteration.
        - 'constraint_violation' : list
            List of constraint violations at each iteration.
        - 'iterations' : list
            List of iteration counts corresponding to the objective values and constraint violations.
        
    plot_together : bool, optional
        If True, plots the objective value and constraint violation in separate subplots in the same figure.
        If False, plots them in separate figures. Default is True.
        
    colors : list, optional
        List of colors for the lines, one for each algorithm. If None, default colors are used.
        
    linestyles : list, optional
        List of linestyles for the lines, one for each algorithm. If None, default linestyles are used.
        
    labels : list, optional
        List of labels for the legend, one for each algorithm. If None, the dictionary keys are used as labels.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
        
    ax : matplotlib.axes.Axes or list of Axes
        The axes object(s) containing the plots. If `plot_together=True`, a list of two axes is returned.
        Otherwise, a single axes object is returned.
    
    Notes
    -----
    The function creates either one or two subplots depending on the `plot_together` parameter.
    It also customizes the plots using the provided colors, linestyles, and labels.
    
    Example
    -------
    >>> data = {
    >>>     'Algorithm 1': {
    >>>         'objective': [10, 8, 6, 5, 3],
    >>>         'constraint_violation': [1, 0.8, 0.5, 0.3, 0],
    >>>         'iterations': [1, 2, 3, 4, 5],
    >>>     },
    >>>     'Algorithm 2': {
    >>>         'objective': [12, 9, 7, 4, 2],
    >>>         'constraint_violation': [1.2, 0.7, 0.6, 0.2, 0.1],
    >>>         'iterations': [1, 2, 3, 4, 5],
    >>>     }
    >>> }
    >>> fig, ax = plot_convergence_process(data, plot_together=True, colors=['blue', 'green'], linestyles=['-', '--'], labels=['Algo 1', 'Algo 2'])
    """
    
    if colors is None:
        colors = [None] * len(data)
    if linestyles is None:
        linestyles = ['-'] * len(data)
    if labels is None:
        labels = list(data.keys())

    if plot_together:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
        ax = [ax1, ax2]
        ax[0].set_xlabel('Iterations')

    for idx, (algo, algo_data) in enumerate(data.items()):
        iterations = algo_data['iterations']
        objective = algo_data['objective']
        constraint_violation = algo_data['constraint_violation']

        ax[0].plot(iterations, objective, color=colors[idx], linestyle=linestyles[idx], label=f'{labels[idx]}')
        ax[1].plot(iterations, constraint_violation, color=colors[idx], linestyle=linestyles[idx], label=f'{labels[idx]}')

    ax[0].set_ylabel('Objective Value')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Constraint Violation')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()

    return fig, fig2, ax










