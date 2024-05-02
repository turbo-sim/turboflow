import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colormap = "Reds"


def load_data(filename):
    # Import file that contains all performance parameteres
    # The resulting variable 'performance_data' will be a dictionary where keys are sheet names and values are DataFrames
    performance_data = pd.read_excel(
        filename, sheet_name=["operation point", "plane", "cascade", 
                              #"stage",
                              "overall"]
    )
    # Round off to ignore precision loss by loading data from excel
    for key, df in performance_data.items():
        performance_data[key] = df.round(10)

    return performance_data


def plot_lines_on_subset(
    performance_data,
    x_name,
    column_names,
    subset,
    fig=None,
    ax=None,
    xlabel="",
    ylabel="",
    title=None,
    close_fig=True,
    stack=False,
):
    # Plot each parameter in column_names as a function of x_name in the given subset
    # subset can only be one subset

    subset[1:] = [round(value, 10) for value in subset[1:]]

    if len(subset) > 2:
        raise Exception(
            "Only one subset (i.e. (key, value)) is accepted for this function"
        )

    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title == None:
        ax.set_title(f"{subset[0]} = {subset[1]}")
    else:
        ax.set_title(title)

    y = get_lines(performance_data, column_names, subsets=subset)
    x = get_lines(performance_data, x_name, subsets=subset)

    colors = plt.get_cmap(colormap)(np.linspace(0.4, 1, len(column_names)))
    if stack == True:
        ax.stackplot(x[0], y, labels=column_names, colors=colors)
    else:
        for i in range(len(y)):
            ax.plot(x[0], y[i], label=f"{column_names[i]}", color=colors[i])

    ax.legend()
    fig.tight_layout(pad=1, w_pad=None, h_pad=None)

    if close_fig == True:
        plt.close(fig)

    return fig, ax


def plot_lines(
    performance_data,
    x_key,
    y_keys,
    fig=None,
    ax=None,
    xlabel="",
    ylabel="",
    title="",
    stack=False,
    filename=None,
    outdir="figures",
    color_map="viridis",
    close_fig=False,
    save_figs=False,
    linestyles = None,
    save_formats=['.png'],
    legend_loc='best',
):
    """
    Plot lines from performance data.

    Parameters
    ----------
    performance_data : DataFrame
        The performance data to plot.
    x_name : str
        Name of the column in data to be used as x-axis values.
    column_names : list of str
        Names of the columns in data to be plotted.
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
    close_fig : bool, optional
        Whether to close the figure after plotting. Default is True.
    save_figs : bool, optional
        Whether to save the figure. Default is False.
    figures_folder : str, optional
        The directory where figures should be saved. Default is 'figures'.
    datetime_format : str, optional
        Format for datetime string in the saved filename. Default is '%Y-%m-%d_%H-%M-%S'.
    colormap : str, optional
        The colormap used for the lines. Default is 'viridis'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
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
    x = get_lines(performance_data, x_key)
    y = get_lines(performance_data, y_keys)
    
    # Plot lines according to colormap and linestyles
    colors = plt.get_cmap(color_map)(np.linspace(0.2, 1.0, len(y)))
    if not isinstance(linestyles, (list, np.ndarray)):
        linestyles = ['-']*len(y)
    if stack == True:
        ax.stackplot(x, y, labels=y_keys, colors=colors)

        # Add edges by overlaying lines
        
        y_arrays = [series.values for series in y]
        cumulative_y = np.cumsum(y_arrays, axis=0)
        for i, series in enumerate(y_arrays):
            if i == 0:
                ax.plot(x, series, color='black', linewidth=0.5)
            ax.plot(x, cumulative_y[i], color='black', linewidth=0.5)

    else:
        for i in range(len(y)):
            ax.plot(x, y[i], label=f"{y_keys[i]}", color=colors[i], linestyle = linestyles[i])

    # Set margins to zero
    ax.margins(x=0.01, y=0.01)

    # Add legend
    if len(y) > 1:
        ax.legend(loc=legend_loc, fontsize=9)
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


def plot_line(
    performance_data,
    x_key,
    y_key,
    fig=None,
    ax=None,
    xlabel="",
    ylabel="",
    title="",
    color = None,
    linestyle = None,
    close_fig=True,
):
    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

    if not isinstance(color, (list, np.ndarray)):
        color = 'k'

    if not linestyle:
        linestyle = '-'

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    y = get_lines(performance_data, y_key)
    x = get_lines(performance_data, x_key)

    ax.plot(x, y, color = color, linestyle = linestyle, linewidth = 1.25)

    fig.tight_layout(pad=1, w_pad=None, h_pad=None)

    if close_fig == True:
        plt.close(fig)

    return fig, ax


def plot_subsets(
    performance_data,
    x,
    y,
    subsets,
    fig=None,
    ax=None,
    xlabel="",
    ylabel="",
    title="",
    colors = None,
    linestyles = None, 
    labels = None, 
    close_fig=True,
):
    # Plot variable y as a function of x on the subset defined by subsets
    # *Subset is a tuple where first element is the key of a column in performance data
    # The subsequent values defines each subset of performance data

    subsets[1:] = [round(value, 10) for value in subsets[1:]]

    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

    if not colors:
            colors = plt.get_cmap(colormap)(np.linspace(0.2, 1, len(subsets) - 1))
    if not linestyles:
        linestyles = ['-']*len(subsets[1:])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    y = get_lines(performance_data, y, subsets=subsets)
    x = get_lines(performance_data, x, subsets=subsets)

    for i in range(len(y)):
        ax.plot(x[i], y[i], label=f"{subsets[0]} = {subsets[i+1]}", linestyle = linestyles[i], color=colors[i])

    if not labels == None:
        ax.legend(labels)
    else:
        ax.legend()

    fig.tight_layout(pad=1, w_pad=None, h_pad=None)

    if close_fig == True:
        plt.close(fig)

    return fig, ax


def get_lines(performance_data, column_name, subsets=None):
    # Return an array of parameter column_name in performance_data
    # subset is a tuple where the first value determines the parameter, while the subsequent values
    # determines the values of the parameter that defines the subset

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

def plot_error(
    values_exp, 
    values_sim,
    fig=None,
    ax=None,
    title="",
    color = None,
    label = None,
    marker = None, 
    close_fig=True,
):
    
    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))

    # Plot model value vs measured value
    ax.plot(values_exp, values_sim, color = color, marker = marker, label = label, linestyle = 'none')

    return fig, ax


def find_column(performance_data, column_name):
    # Function for column named column_name in all sheets in performance_data

    for key in performance_data.keys():
        if any(element == column_name for element in performance_data[key].columns):
            return key

    raise Exception(f"Could not find column {column_name} in performance_data")


def get_column(performance_data, column_name):
    sheet = find_column(performance_data, column_name)

    return performance_data[sheet][column_name]


def get_subset(performance_data, column_name, row_value):
    sheet = find_column(performance_data, column_name)

    return performance_data[sheet][
        performance_data[sheet][column_name] == row_value
    ].index


def savefig_in_formats(fig, path_without_extension, formats=[".png", ".svg", ".pdf", ".eps"]):
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
        Default is ['.png', '.svg', '.pdf'].

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
    
    radius_hub_in = geometry["radius_hub_in"]
    radius_hub_out = geometry["radius_hub_out"]
    radius_tip_in = geometry["radius_tip_in"]
    radius_tip_out = geometry["radius_tip_out"]
    number_of_cascades = geometry["number_of_cascades"]
    axial_chord = geometry["axial_chord"]
    
    
    dx = 0.05*max(axial_chord)
    x = np.array([])
    for i in range(len(axial_chord)):
        x = np.append(x, np.sum(axial_chord[0:i+1]) + i*dx)
        x = np.append(x, np.sum(axial_chord[0:i+1]) + (i+1)*dx)
    x[1:] = x[0:-1]
    x[0] = 0

    y_hub = np.array([val for pair in zip(radius_hub_in, radius_hub_out) for val in pair])
    y_tip = np.array([val for pair in zip(radius_tip_in, radius_tip_out) for val in pair])

    fig, ax = plt.subplots()
    colors = ["0.5", "0.8"]
    for i in range(number_of_cascades):    
        ax.plot(x[i*2:(i+1)*2], y_tip[i*2:(i+1)*2], 'k')
        ax.plot(x[i*2:(i+1)*2],  y_hub[i*2:(i+1)*2], 'k')
        ax.plot([x[i*2:(i+1)*2][0], x[i*2:(i+1)*2][0]], [y_hub[i*2:(i+1)*2][0], y_tip[i*2:(i+1)*2][0]], 'k')
        ax.plot([x[i*2:(i+1)*2][1], x[i*2:(i+1)*2][1]], [y_hub[i*2:(i+1)*2][1], y_tip[i*2:(i+1)*2][1]], 'k')
        
        ax.fill_between(x[i*2:(i+1)*2], y_hub[i*2:(i+1)*2], y_tip[i*2:(i+1)*2], color = colors[i%2])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Axial direction")
    ax.set_ylabel("Radial direction")
    plt.show()
    
    return fig, ax
