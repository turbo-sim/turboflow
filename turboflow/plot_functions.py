import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename, sheet_names = None):
    """
    Load performance data from an Excel file.

    This function imports an Excel file containing performance parameters. It reads the data into a dictionary
    named 'performance_data' where each key corresponds to a sheet name in the Excel file, and each value is
    a pandas DataFrame containing the data from that sheet. The data is rounded to avoid precision loss when
    loading from Excel.

    Parameters
    ----------
    filename : str
        The name of the Excel file containing performance parameters.
    sheet_names : list
        List of the sheets to be exported. Default None extract all worksheets.

    Returns
    -------
    dict
        A dictionary containing performance data, where keys are sheet names and values are DataFrames.

    """

    # Read excel file
    performance_data = pd.read_excel(
        filename,
        sheet_name=sheet_names
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
        colors = plt.get_cmap(color_map)(np.linspace(0.2, 0.8, len(y)))

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
                                 label = '',
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
        fig, ax = plot_velocity_triangle(fig, ax, start_x, start_y, w_ts[2*i], v_ts[2*i], v_ms[2*i], w_ms[2*i], hs = hs, fontsize = fontsize, lines = "total", color = color, linestyle = linestyle, cascade = 's', label = label)
        fig, ax = plot_velocity_triangle(fig, ax, start_x, start_y, w_ts[2*i + 1], v_ts[2*i+ 1], v_ms[2*i+ 1], w_ms[2*i+ 1], hs = hs, fontsize = fontsize, lines = "total", color = color, linestyle = linestyle, cascade = 'r', label = '')

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

def plot_velocity_triangle(fig, ax, start_x, start_y, w_t, v_t, v_m, w_m, hs = 10, fontsize = 12, lines = "all", color = None, linestyle = None, cascade = '', label = ''):

    # Plot triangle
    if lines == "all":
        ax.plot([start_x, start_x], [start_y, start_y - v_m], color=color, linestyle=linestyle, label = label)  # Plot meridional velocity
        ax.plot([start_x, v_t], [start_y - v_m, start_y - v_m], color=color, linestyle=linestyle)  # Plot absolute tangential velocity
        ax.plot([start_x, w_t], [start_y - v_m, start_y - v_m], color=color, linestyle=linestyle)  # Plot relative tangential velocity
        ax.plot([start_x, v_t], [start_y, start_y - v_m], color=color, linestyle = linestyle)  # Plot abolsute velocity
        ax.plot([start_x, w_t], [start_y, start_y - w_m], color=color, linestyle = linestyle)  # Plot relative velocity
        ax.plot([w_t, v_t], [start_y - w_m, start_y - w_m], color = color, linestyle = linestyle)  # Plot blade speed
    elif lines == "total":
        ax.plot([start_x, v_t], [start_y, start_y - v_m], color=color, linestyle = linestyle, label = label)  # Plot abolsute velocity
        ax.plot([start_x, w_t], [start_y, start_y - w_m], color=color, linestyle = linestyle)  # Plot relative velocity
        ax.plot([w_t, v_t], [start_y - w_m, start_y - w_m], color = color, linestyle = linestyle)  # Plot blade speed
    elif lines == "absolute":
        ax.plot([start_x, start_x], [start_y, start_y - v_m], color=color, linestyle=linestyle, label = label)  # Plot meridional velocity
        ax.plot([start_x, v_t], [start_y - v_m, start_y - v_m], color=color, linestyle=linestyle)  # Plot absolute tangential velocity
        ax.plot([start_x, v_t], [start_y, start_y - v_m], color=color, linestyle = linestyle)  # Plot abolsute velocity
    elif lines == "relative":
        ax.plot([start_x, start_x], [start_y, start_y - v_m], color=color, linestyle=linestyle, label = label)  # Plot meridional velocity
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
        colors = plt.get_cmap('magma')(np.linspace(0.2, 1.0, len(final_solutions)))  # Use default color cycle
    
    # Create fig and ax if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

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
    ax.set_ylabel('Relative change from initial guess', fontsize = fontsize)
    ax.legend(fontsize = fontsize)
    ax.grid(False)
    ax.minorticks_off()

    # Set custom tick labels if provided
    if variable_names is not None:
        ax.set_xticks(range(num_vars))
        ax.set_xticklabels(variable_names, rotation=75, fontsize = fontsize)

    return fig, ax

def plot_convergence_process(data, plot_together=True, colors=None, linestyles=None, labels=None, linewidth = 1.5, fontsize = 12):
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        figs = [fig]
        ax = [ax1, ax2]
        ax[0].set_xticklabels([])
    else:
        fig1, ax1 = plt.subplots(1, 1, figsize=(14, 6))
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
        figs = [fig1, fig2]
        ax = [ax1, ax2]
        ax[0].set_xlabel('Iterations')

    for idx, (algo, algo_data) in enumerate(data.items()):
        iterations = algo_data['iterations']
        objective = algo_data['objective']
        constraint_violation = algo_data['constraint_violation']

        ax[0].plot(iterations, objective, color=colors[idx], linestyle=linestyles[idx], label=f'{labels[idx]}', linewidth = linewidth)
        ax[1].plot(iterations, constraint_violation, color=colors[idx], linestyle=linestyles[idx], label=f'{labels[idx]}', linewidth = linewidth)

    ax[0].set_ylabel('Objective value', fontsize = fontsize)
    ax[0].grid(True)
    ax[0].legend(fontsize = fontsize)

    ax[1].set_xlabel('Iterations', fontsize = fontsize)
    ax[1].set_ylabel('Constraint violation', fontsize = fontsize)
    ax[1].grid(True)
    if not plot_together:
        ax[1].legend(fontsize = fontsize)

    ax[0].tick_params(labelsize=fontsize)
    ax[1].tick_params(labelsize=fontsize)
    plt.tight_layout()

    return figs, ax


def NURBS(P, p, U, u, W):
    # ...
    u = np.atleast_1d(u)
    W = np.atleast_1d(W)

    Nt = len(u)
    dim, nn = P.shape  # nn = n+1 control points
    n = nn - 1
    m = n + p + 1      # total B-spline segments for initial p=0

    # ** We now allocate with 'm' columns instead of 'nn': **
    N = np.zeros((Nt, m))

    # First step (p=0)
    for i in range(m):
        N[:, i] = ((u >= U[i]) & (u <= U[i+1])).astype(float)

    current_p = 1
    while m > (n + 1):
        m -= 1
        N_bis = np.zeros((Nt, m))

        for i in range(m):
            denom1 = (U[i + current_p] - U[i])
            denom2 = (U[i + current_p + 1] - U[i + 1])
            
            if denom1 != 0:
                n1 = ((u - U[i]) / denom1) * N[:, i]
            else:
                n1 = 0.0

            if denom2 != 0:
                n2 = ((U[i + current_p + 1] - u) / denom2) * N[:, i + 1]
            else:
                n2 = 0.0

            n1 = np.nan_to_num(n1)
            n2 = np.nan_to_num(n2)
            N_bis[:, i] = n1 + n2

        N = N_bis
        current_p += 1

    # At this point, N has shape (Nt, n+1).
    # Compute rational basis
    W_mat = np.tile(W, (Nt, 1))
    numerator = N * W_mat
    denominator = np.sum(numerator, axis=1, keepdims=True)
    denominator = np.where(denominator == 0.0, 1e-14, denominator)
    R = numerator / denominator

    # Evaluate final curve in R^dim
    r = R @ P.T  # shape (Nt, dim)

    return r, R, N



def plot_blade_profile(blade_parameters):
    """
    Python version of the MATLAB function [x, y] = plot_blade_profile(blade_parameters).

    Given a dictionary `blade_parameters` with:
      chord, stagger, theta_in, theta_out,
      wedge_in, wedge_out, radius_in, radius_out,
      d1, d2, d3, d4
    it computes the shape of a 2D blade profile (x, y) by connecting
    four NURBS segments:
      (1) Leading edge arc (between points 1 and 2)
      (2) Spline from point 2 to point 3
      (3) Trailing edge arc (between points 3 and 4)
      (4) Spline from point 4 back to point 1
    
    Returns:
    --------
    x, y : 1D numpy arrays
        The concatenated coordinates of the blade in the plane.
    """

    # -- 1) Unpack parameters --
    chord      = blade_parameters["chord"]        # Blade chord
    stagger    = blade_parameters["stagger"]      # Stagger angle
    theta_in   = blade_parameters["theta_in"]     # Inlet camber angle
    theta_out  = blade_parameters["theta_out"]    # Outlet camber angle
    wedge_in   = blade_parameters["wedge_in"]     # Leading edge wedge semiangle
    wedge_out  = blade_parameters["wedge_out"]    # Trailing edge wedge semiangle
    radius_in  = blade_parameters["radius_in"]    # Leading edge radius
    radius_out = blade_parameters["radius_out"]   # Trailing edge radius
    d1         = blade_parameters["d1"]
    d2         = blade_parameters["d2"]
    d3         = blade_parameters["d3"]
    d4         = blade_parameters["d4"]

    # -- 2) Points 1, 2, 3, 4 (the 'corners' of the blade) --
    a1 = theta_in + wedge_in
    x1 = radius_in*(1 - np.sin(a1))
    y1 = radius_in*(    np.cos(a1))

    a2 = theta_in - wedge_in
    x2 = radius_in*(1 + np.sin(a2))
    y2 = radius_in*(    -np.cos(a2))

    a3 = theta_out + wedge_out
    x3 = chord*np.cos(stagger) - radius_out*(1 - np.sin(a3))
    y3 = chord*np.sin(stagger) - radius_out*(    np.cos(a3))

    a4 = theta_out - wedge_out
    x4 = chord*np.cos(stagger) - radius_out*(1 + np.sin(a4))
    y4 = chord*np.sin(stagger) + radius_out*(    np.cos(a4))

    # ---------------------------
    # -- Segment 1: arc 1 -> 2 --
    # ---------------------------
    P0 = np.array([x1, y1])
    P2 = np.array([x2, y2])
    # Normal direction
    n = P2 - P0
    n = np.array([-n[1], n[0]])  # rotate 90 deg
    n = n / np.linalg.norm(n)
    a  = wedge_in
    R  = radius_in
    # Middle control point
    P1 = (P0 + P2)/2.0 - R*np.cos(a)/np.tan(a)*n

    # Combine into a matrix of shape (dim, n+1)
    # Here: P0, P1, P2 => 3 control points for a 2nd-order NURBS
    ctrl_pts = np.column_stack([P0, P1, P2])

    # Weights
    W = np.ones(3)
    # The second control point gets weight = sin(a)
    W[1] = np.sin(a)

    # Parameter vector
    u = np.linspace(0, 1, 100)
    p = 2  # Quadratic
    # Clamped knot vector: p+1 zeros, n-p equispaced, p+1 ones
    # n = 2 => number of segments = 2, so for a Bezier we do
    # U = [0,0, ..., 1,1], length = n+p+2 = 2+2+2=6
    # Typically for a quadratic Bezier: [0,0,0,1,1,1]
    U = np.concatenate([np.zeros(p), np.linspace(0,1, (3-p)), np.ones(p)])
    # But strictly we want 3 control points => n=2 => n-p+2=2-2+2=2
    # so U = [0,0, 1,1, 1,1]? We'll mimic the exact MATLAB approach:
    # The MATLAB code does: U = [zeros(1,p), linspace(0,1,n-p+2), ones(1,p)]
    # For n=2, p=2 => n-p+2=2 => U = [0,0, 0..1, 1,1] => [0,0,0,1,1,1]
    U = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    # Evaluate NURBS
    r12, _, _ = NURBS(ctrl_pts, p, U, u, W)

    # -------------------------------------------------
    # -- Segment 2: from point 2 -> point 3 (cubic)  --
    # -------------------------------------------------
    P0 = np.array([x2, y2])
    P1 = P0 + d2*chord*np.array([np.cos(a2), np.sin(a2)])
    P3 = np.array([x3, y3])
    P2 = P3 - d3*chord*np.array([np.cos(a3), np.sin(a3)])

    ctrl_pts = np.column_stack([P0, P1, P2, P3])
    W = np.ones(4)
    u = np.linspace(0, 1, 100)
    p = 3  # cubic
    # For a cubic Bezier with 4 control points, we have n=3 => n-p+2=3-3+2=2
    # so the knot vector length is 4+3+1=8, typically [0,0,0,0,1,1,1,1]
    U = np.array([0,0,0,0,1,1,1,1], dtype=float)

    r23, _, _ = NURBS(ctrl_pts, p, U, u, W)

    # -----------------------------
    # -- Segment 3: arc 3 -> 4   --
    # -----------------------------
    P0 = np.array([x3, y3])
    P2 = np.array([x4, y4])
    # Normal direction
    n = P2 - P0
    n = np.array([-n[1], n[0]])
    n = n / np.linalg.norm(n)
    a  = wedge_out
    R  = radius_out
    P1 = (P0 + P2)/2.0 - R*np.cos(a)/np.tan(a)*n

    ctrl_pts = np.column_stack([P0, P1, P2])
    W = np.ones(3)
    W[1] = np.sin(a)
    u = np.linspace(0, 1, 100)
    p = 2
    U = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    r34, _, _ = NURBS(ctrl_pts, p, U, u, W)

    # ------------------------------------------------
    # -- Segment 4: from point 4 -> point 1 (cubic) --
    # ------------------------------------------------
    # We re-use the same approach as segment 2 (cubic)
    P0 = np.array([x4, y4])
    P1 = P0 - d1*chord*np.array([np.cos(a4), np.sin(a4)])
    P3 = np.array([x1, y1])
    P2 = P3 + d4*chord*np.array([np.cos(a1), np.sin(a1)])

    ctrl_pts = np.column_stack([P0, P1, P2, P3])
    W = np.ones(4)
    u = np.linspace(0, 1, 100)
    p = 3
    U = np.array([0,0,0,0,1,1,1,1], dtype=float)

    r41, _, _ = NURBS(ctrl_pts, p, U, u, W)

    # -- Combine segments --
    r = np.vstack([r12, r23, r34, r41])
    x = r[:, 0]
    y = r[:, 1]

    return x, y


def plot_view_axial_tangential(problem_structure, 
                               fig = None, 
                               ax = None, 
                               repeats = 3,
                               color = 'k', 
                               linestyle = "-", 
                               label = ""):
    """
    Optional: Python version of the MATLAB function plot_view_axial_tangential.
    Demonstrates how a multi-stage turbine might be visualized in an axial-tangential view,
    using repeated calls to plot_blade_profile for rotor/stator rows.

    turbine_data is a dictionary expected to have structure:
       turbine_data['cascade'] = list of dicts, each with
         {
           's': ...,         # pitch
           'cs': ...,        # spacing between cascades
           'c': ...,         # chord
           'b': ...,         # axial spacing
           'stagger': ...,
           'theta_in': ...,
           'theta_out': ...,
           't_max': ...,
           't_te': ...,
         }
       turbine_data['overall'] = {
           'axial_length': ...,
           'n_stages': ...,
       }
       turbine_data['stage'] = {
           'R': [reaction_values],
       }

    This function is included mainly to show how one might replicate
    the high-level plotting. It references internal design logic
    that is specific to your original MATLAB code.
    """

    # Load data from problem structure
    geometry = problem_structure.geometry
    results = problem_structure.results

    # Extract data from turbine_data
    s = geometry['pitch']
    cs = geometry['axial_chord']
    c = geometry['chord']
    stagger = geometry['stagger_angle']
    theta_in = geometry['leading_edge_angle']
    theta_out = geometry['gauging_angle']
    t_max = geometry['maximum_thickness']
    t_te = geometry['trailing_edge_thickness']
    n_cascades = geometry['number_of_cascades']
    reaction = np.array(results['stage']['reaction'].values)  # one per stage
    
    dx = 0.1  # Axial spacing between cascades relative to max axial chord
    dx = dx * max(cs)
    L = np.sum(cs) + np.sum(dx*np.ones(n_cascades-1))

    # Create figure
    if fig == None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks([])
        ax.set_xlim([0.0, L*1.1])
        ax.grid(False)
        ax.tick_params(axis='x', which='both', top=False, labeltop=False)
        ax.set_xlabel("Axial direction")

    # Now actually plot the blades
    x_0 = 0.0
    for j in range(n_cascades):
        # Build a dictionary of blade parameters
        if reaction[j//2] < 0.25 and ((j+1) % 2 == 0):
            # "Impulse blade" parameters for rotor
            blade_params = {
                'chord': c[j],
                'stagger': stagger[j]*np.pi/180,
                'theta_in': theta_in[j]*np.pi/180,
                'theta_out': theta_out[j]*np.pi/180,
                'wedge_in': 10.0 * np.pi/180,
                'wedge_out': 10.0 * np.pi/180,
                'radius_in': 1.0 * t_te[j],
                'radius_out': 1.0 * t_te[j],
                'd1': 0.60,
                'd2': 0.40,
                'd3': 0.40,
                'd4': 0.60
            }
        else:
            # "Reaction blade" parameters
            blade_params = {
                'chord': c[j],
                'stagger': stagger[j]*np.pi/180,
                'theta_in': theta_in[j]*np.pi/180,
                'theta_out': theta_out[j]*np.pi/180,
                'wedge_in': 10.0 * np.pi/180,
                'wedge_out': 7.5 * np.pi/180,
                'radius_in': 0.50 * t_max[j],
                'radius_out': 1.0 * t_te[j],
                'd1': 0.30,
                'd2': 0.30,
                'd3': 0.30,
                'd4': 0.30
            }

        # Get blade coordinates
        bx, by = plot_blade_profile(blade_params)

        # Plot multiple blade passages in the tangential direction
        for k in range(repeats):
            sign = -(-1) ** k
            # sign = 1
            if j == 0 and k == 0:
                ax.plot(bx + x_0, by + s[j]*((1+k)//2), color = color, linestyle = linestyle, label  = label, linewidth=1.0)
            else:
                ax.plot(bx + x_0, by + sign*s[j]*((1+k)//2), color = color, linestyle = linestyle, label='_nolegend_', linewidth=1.0)

        # Move x_0 to the next cascade
        x_0 += dx + cs[j]

        fig.tight_layout()
    
    plt.show()

    return fig, ax 







