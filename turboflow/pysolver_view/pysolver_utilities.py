import os
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from cycler import cycler
from datetime import datetime

COLORS_PYTHON = [
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

COLORS_MATLAB = [
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
]


def set_plot_options(
    fontsize=13,
    grid=False,
    major_ticks=True,
    minor_ticks=True,
    margin=0.05,
    color_order="matlab",
    linewidth=1.25,
):
    """
    Set options for creating publication-quality figures using Matplotlib.

    This function updates the internal Matplotlib settings to better align with standards for publication-quality figures.
    Features include improved font selections, tick marks, grid appearance, and color selections.

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
        "lines.linewidth": linewidth,
        "lines.markersize": 4,
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


def print_rc_parameters(filename=None):
    """
    Print the current rcParams used by Matplotlib or write to file if provided.

    This function provides a quick overview of the active configuration parameters within Matplotlib.
    """
    params = mpl.rcParams
    for key, value in params.items():
        print(f"{key}: {value}")

    if filename:
        with open(filename, "w") as file:
            for key, value in params.items():
                file.write(f"{key}: {value}\n")


def savefig_in_formats(fig, path_without_extension, formats=[".png", ".svg"], dpi=500):
    """
    Save a given Matplotlib figure in multiple file formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to be saved.
    path_without_extension : str
        The full path to save the figure excluding the file extension.
    formats : list of str, optional
        A list of string file extensions to specify which formats the figure should be saved in.
        Supported formats are ['.png', '.svg', '.pdf', '.eps'].
        Default is ['.png', '.svg'].
    dpi : int, optional
        The resolution in dots per inch with which to save the image. This parameter affects only PNG files.
        Default is 500.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> save_fig_in_formats(fig, "/path/to/figure/filename")

    This will save the figure as "filename.png", "filename.svg", and "filename.pdf" in the "/path/to/figure/" directory.
    """

    # Validate export formats
    allowed_formats = {".png", ".svg", ".pdf", ".eps"}
    invalid_formats = set(formats) - allowed_formats
    if invalid_formats:
        raise ValueError(
            f"Unsupported file formats: {', '.join(invalid_formats)}. Allowed formats: {', '.join(allowed_formats)}"
        )

    for ext in formats:
        if ext == ".png":
            # Save PNG files with specified DPI
            fig.savefig(f"{path_without_extension}{ext}", bbox_inches="tight", dpi=dpi)
        else:
            # Save in vector formats without DPI setting
            fig.savefig(f"{path_without_extension}{ext}", bbox_inches="tight")


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


def print_dict(data, indent=0, return_output=False):
    """
    Recursively prints nested dictionaries with indentation or returns the formatted string.

    Parameters
    ----------
    data : dict
        The dictionary to print.
    indent : int, optional
        The initial level of indentation for the keys of the dictionary, by default 0.
    return_output : bool, optional
        If True, returns the formatted string instead of printing it.

    Returns
    -------
    str or None
        The formatted string representation of the dictionary if return_output is True, otherwise None.
    """
    lines = []
    for key, value in data.items():
        line = "    " * indent + str(key) + ": "
        if isinstance(value, dict):
            if value:
                lines.append(line)
                lines.append(print_dict(value, indent + 1, True))
            else:
                lines.append(line + "{}")
        else:
            lines.append(line + str(value))

    output = "\n".join(lines)
    if return_output:
        return output
    else:
        print(output)


def print_object(obj):
    """
    Prints all attributes and methods of an object, sorted alphabetically.

    - Methods are identified as callable and printed with the prefix 'Method: '.
    - Attributes are identified as non-callable and printed with the prefix 'Attribute: '.

    Parameters
    ----------
    obj : object
        The object whose attributes and methods are to be printed.

    Returns
    -------
    None
        This function does not return any value. It prints the attributes and methods of the given object.

    """
    # Retrieve all attributes and methods
    attributes = dir(obj)

    # Sort them alphabetically, case-insensitive
    sorted_attributes = sorted(attributes, key=lambda x: x.lower())

    # Iterate over sorted attributes
    for attr in sorted_attributes:
        # Retrieve the attribute or method from the object
        attribute_or_method = getattr(obj, attr)

        # Check if it is callable (method) or not (attribute)
        if callable(attribute_or_method):
            print(f"Method: {attr}")
        else:
            print(f"Attribute: {attr}")
