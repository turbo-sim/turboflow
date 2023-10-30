import os
import yaml
import logging
from datetime import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from cycler import cycler


def get_cascades_data(filename):
    """
    Retrieve cascades data from a YAML configuration file and process the geometry data.

    This function reads a YAML configuration file to obtain cascades data, then converts 
    the geometry values to numpy arrays. It also initializes `fixed_params` and `overall` 
    dictionaries within the cascades data. String expressions in the YAML file representing
    numerical values (e.g., "np.pi/180*45") are evaluated to actual numerical values.

    Parameters
    ----------
    filename : str
        The path to the YAML configuration file containing cascades data.

    Returns
    -------
    cascades_data : dictionary with fields:
        - "geometry": A dictionary where each key corresponds to a geometry parameter and its associated value is a numpy array.
        - "fixed_params": An empty dictionary, initialized for future usage.
        - "overall": An empty dictionary, initialized for future usage.

    """
    cascades_data = read_configuration_file(filename)
    cascades_data["geometry"] = {key: np.asarray(value) for key, value in cascades_data["geometry"].items()}
    cascades_data["fixed_params"] = {}
    cascades_data["overall"] = {}
    return cascades_data


def read_configuration_file(filename):
    """
    Read and postprocess a YAML configuration file.

    This function reads the specified YAML configuration file and postprocesses the data, 
    particularly converting string representations of numerical expressions into their 
    actual numerical values.

    Parameters
    ----------
    filename : str
        Path to the YAML configuration file to be read.

    Returns
    -------
    dict
        A dictionary containing the postprocessed data from the configuration file.

    Notes
    -----
    The function uses `postprocess_config` to evaluate string expressions found in the 
    YAML data. For example, a YAML entry like "value: "np.pi/2"" will be converted to 
    its numerical equivalent.
    """
    with open(filename, 'r') as file:
        return postprocess_config(yaml.safe_load(file))


def postprocess_config(config):
    """
    Postprocesses the YAML configuration data by converting string values that represent
    numerical expressions to actual numerical values.

    This function helps ensure that configuration parameters, which may be specified as
    strings in the YAML file (e.g., "np.pi/180*45"), are correctly evaluated to numerical
    values.

    Parameters
    ----------
        config : dict
            The configuration data loaded from a YAML file.

    Returns
    ----------
        data : dict
            The postprocessed configuration data with numerical values.

    """
    def convert_to_numbers(data):
        if isinstance(data, dict):
            return {key: convert_to_numbers(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [convert_to_numbers(item) for item in data]
        elif isinstance(data, str):
            try:
                return eval(data)
            except (NameError, SyntaxError):
                return data
        else:
            return data

    return convert_to_numbers(config)


def print_dict(data, indent=0):
    """
    Recursively prints nested dictionaries with indentation.

    Parameters
    ----------
    data : dict
        The dictionary to print. It can contain nested dictionaries as values.
    indent : int, optional
        The initial level of indentation for the keys of the dictionary, by default 0.
        It controls the number of spaces before each key.

    Returns
    -------
    None

    Examples
    --------
    >>> data = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    >>> print_dict(data)
    a: 1
    b:
        c: 2
        d:
            e: 3
    """
    
    for key, value in data.items():
        print('    ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print('')
            print_dict(value, indent+1)
        else:
            print(value)


def print_boundary_conditions(BC):
    column_width = 25  # Adjust this to your desired width
    print("-" * 80)
    print(" Operating point: ")
    print("-" * 80)
    print(f" {'Fluid: ':<{column_width}} {BC['fluid_name']:<}")
    print(f" {'Flow angle in: ':<{column_width}} {BC['alpha_in']*180/np.pi:<.2f} deg")
    print(f" {'Total temperature in: ':<{column_width}} {BC['T0_in']-273.15:<.2f} degC")
    print(f" {'Total pressure in: ':<{column_width}} {BC['p0_in']/1e5:<.3f} bar")
    print(f" {'Static pressure out: ':<{column_width}} {BC['p_out']/1e5:<.3f} bar")
    print(f" {'Angular speed: ':<{column_width}} {BC['omega']*60/2/np.pi:<.1f} RPM")
    print("-" * 80)
    print()

def create_logger(name, path=None, use_datetime=True):
    """
    Creates and configures a logging object for recording logs during program execution.

    Parameters
    ----------
    name : str
        Name of the log file. Allows for differentiation when analyzing logs from different components or runs of a program.
    path : str, optional
        Specifies the directory where the log files will be saved. By default, a directory named "logs"
        will be created in the current working directory (cwd).
    use_datetime : bool, optional
        Determines whether the log filename should have a unique datetime identifier appended. Default is True.

    Returns
    -------
    logger : object
        Configured logger object.

    Notes
    -----
    - By default, the function sets the log level to `INFO`, which means the logger will handle
      messages of level `INFO` and above (like `ERROR`, `WARNING`, etc.). The log entries will contain
      the timestamp, log level, and the actual log message.
    - When `use_datetime=True`, each log file will have a unique datetime identifier. This ensures
      traceability, avoids overwriting previous logs, allows chronological ordering of log files, and
      handles concurrency in multi-instance environments.
    """

    # Define logs directory if it is not provided
    if path is None:
        path = os.path.join(os.getcwd(), "logs")

    # Create logs directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Define file name and path
    if use_datetime:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(path, f"{name}_{current_time}.log")
    else:
        log_filename = os.path.join(path, f"{name}.log")

    # Set logger configuration
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create logger object
    logger = logging.getLogger()

    return logger


def set_plot_options(
    fontsize=13,
    grid=True,
    major_ticks=True,
    minor_ticks=True,
    margin=0.05,
    color_order="matlab",
):
    """
    Set plot options for creating publication-quality figures using Matplotlib.

    Parameters
    ----------
    fontsize : int, optional
        Font size for text elements in the plot. Default is 13.
    grid : bool, optional
        Whether to show grid lines on the plot. Default is True.
    major_ticks : bool, optional
        Whether to show major ticks. Default is True.
    minor_ticks : bool, optional
        Whether to show minor ticks. Default is True.
    margin : float, optional
        Margin size for axes. Default is 0.05.
    color_order : str, optional
        Color order to be used for plot lines. Options include "python" and "matlab". Default is "matlab".

    Notes
    -----
    This function updates the internal Matplotlib settings to better align with standards for publication-quality figures.
    Features include improved font selections, tick marks, grid appearance, and color selections.
    """

    if isinstance(color_order, str):
        if color_order.lower() == "python":
            color_order = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]

        elif color_order.lower() == "matlab":
            color_order = [
                "#0072BD",
                "#D95319",
                "#EDB120",
                "#7E2F8E",
                "#77AC30",
                "#4DBEEE",
                "#A2142F",
            ]

    # Define dictionary of custom settings
    rcParams = {
        "text.usetex": False,
        "font.size": fontsize,
        "font.style": "normal",
        "font.family": "serif",  # 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
        "font.serif": ["Times New Roman"],  # ['times new roman', 'cmr10']
        "mathtext.fontset": "stix",  # ["stix", 'cm']
        "axes.edgecolor": "black",
        "axes.linewidth": 1.25,
        "axes.titlesize": fontsize,
        "axes.titleweight": "normal",
        "axes.titlepad": fontsize * 1.4,
        "axes.labelsize": fontsize,
        "axes.labelweight": "normal",
        "axes.labelpad": fontsize,
        "axes.xmargin": margin,
        "axes.ymargin": margin,
        # "axes.zmargin": margin,
        "axes.grid": grid,
        "axes.grid.axis": "both",
        "axes.grid.which": "major",
        "axes.prop_cycle": cycler(color=color_order),
        "grid.alpha": 0.5,
        "grid.color": "black",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "legend.borderaxespad": 1,
        "legend.borderpad": 0.6,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.labelcolor": "black",
        "legend.labelspacing": 0.3,
        "legend.fancybox": True,
        "legend.fontsize": fontsize - 2,
        "legend.framealpha": 1.00,
        "legend.handleheight": 0.7,
        "legend.handlelength": 1.25,
        "legend.handletextpad": 0.8,
        "legend.markerscale": 1.0,
        "legend.numpoints": 1,
        "lines.linewidth": 1.25,
        "lines.markersize": 5,
        "lines.markeredgewidth": 1.25,
        "lines.markerfacecolor": "white",
        "xtick.direction": "in",
        "xtick.labelsize": fontsize - 1,
        "xtick.bottom": major_ticks,
        "xtick.top": major_ticks,
        "xtick.major.size": 6,
        "xtick.major.width": 1.25,
        "xtick.minor.size": 3,
        "xtick.minor.width": 0.75,
        "xtick.minor.visible": minor_ticks,
        "ytick.direction": "in",
        "ytick.labelsize": fontsize - 1,
        "ytick.left": major_ticks,
        "ytick.right": major_ticks,
        "ytick.major.size": 6,
        "ytick.major.width": 1.25,
        "ytick.minor.size": 3,
        "ytick.minor.width": 0.75,
        "ytick.minor.visible": minor_ticks,
        "savefig.dpi": 500,
    }

    # Update the internal Matplotlib settings dictionary
    mpl.rcParams.update(rcParams)


def print_installed_fonts():
    """
    Print the list of fonts installed on the system.

    This function identifies and prints all available fonts for use in Matplotlib.
    """
    fonts = mpl.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font in sorted(fonts):
        print(font)


def print_rc_parameters():
    """
    Print the current rcParams used by Matplotlib.

    This function provides a quick overview of the active configuration parameters within Matplotlib.
    """
    params = mpl.rcParams
    for key, value in params.items():
        print(f"{key}: {value}")
    # with open("rcParams_default.txt", 'w') as file:
    #     for key, value in params.items():
    #         file.write(f"{key}: {value}\n")


def _create_sample_plot():
    # Create figure
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.gca()
    ax.set_title(r"$f(x) = \cos(2\pi x + \phi)$")
    ax.set_xlabel("$x$ axis")
    ax.set_ylabel("$y$ axis")
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlim([0, 1])
    ax.set_ylim([-1.15, 1.15])
    # ax.set_xticks([])
    # ax.set_yticks([])

    # Plot the function
    x = np.linspace(0, 1, 200)
    phi_array = np.arange(0, 1, 0.2) * np.pi
    for phi in phi_array:
        ax.plot(x, np.cos(2 * np.pi * x + phi), label=f"$\phi={phi/np.pi:0.2f}\pi$")

    # Create legend
    ax.legend(loc="lower right", ncol=1)

    # Adjust PAD
    fig.tight_layout(pad=1.00, w_pad=None, h_pad=None)

    # Display figure
    plt.show()
