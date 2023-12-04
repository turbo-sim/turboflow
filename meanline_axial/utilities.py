import os
import re
import yaml
import time
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numbers

from datetime import datetime
from numbers import Number
from collections.abc import Iterable
from cycler import cycler



def is_float(element: any) -> bool:
    """
    Check if the given element can be converted to a float.

    Parameters
    ----------
    element : any
        The element to be checked.

    Returns
    -------
    bool
        True if the element can be converted to a float, False otherwise.
    """
    
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def is_numeric(value):
    """
    Check if a value is a numeric type, including both Python and NumPy numeric types.

    This function checks if the given value is a numeric type (int, float, complex) 
    in the Python standard library or NumPy, while explicitly excluding boolean types.

    Parameters
    ----------
    value : any type
        The value to be checked for being a numeric type.

    Returns
    -------
    bool
        Returns True if the value is a numeric type (excluding booleans), 
        otherwise False.
    """
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        return True
    if isinstance(value, (np.int_, np.float_, np.complex_)):
        return True
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.number) and not np.issubdtype(value.dtype, np.bool_)
    return False



def wait_for_file(file_path, timeout=None, poll_interval=0.1):
    """
    Wait until the specified file is created.

    This function is used to wait until a Fluent transcript file is created.

    Parameters
    ----------
    file_path : str
        Path to the file to wait for.
    timeout : float, optional
        Maximum time to wait in seconds. If None, waits indefinitely.
    poll_interval : int, optional
        Time interval between checks in seconds.

    Returns
    -------
    bool
        True if the file was found, False otherwise (only if timeout is set).
    """
    start_time = time.time()
    while True:
        if os.path.exists(file_path):
            return True
        if timeout is not None and time.time() - start_time > timeout:
            raise FileNotFoundError(f"Timeout waiting for file: {file_path}")
        time.sleep(poll_interval)


def savefig_in_formats(fig, path_without_extension, formats=['.png', '.svg', '.pdf']):
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





def validate_keys(checked_dict, required_keys, allowed_keys=None):
    """
    Validate the presence of required keys and check for any unexpected keys in a dictionary.

    Give required keys and allowed keys to have complete control
    Give required keys twice to check that the list of keys is necessary and sufficient
    Give only required keys to allow all extra additional key

    Parameters
    ----------
    checked_dict : dict
        The dictionary to be checked.
    required_keys : set
        A set of keys that are required in the dictionary.
    allowed_keys : set
        A set of keys that are allowed in the dictionary.

    Raises
    ------
    ConfigurationError
        If either required keys are missing or unexpected keys are found.
    """

    # Convert input lists to sets for set operations
    checked_keys = set(checked_dict.keys())
    required_keys = set(required_keys)

    # Set allowed_keys to all present keys if not provided
    if allowed_keys is None:
        allowed_keys = checked_keys
    else:
        allowed_keys = set(allowed_keys)

    # Check for extra and missing keys
    missing_keys = required_keys - checked_keys
    extra_keys = checked_keys - allowed_keys

    # Prepare error messages
    error_messages = []
    if missing_keys:
        error_messages.append(f"Missing required keys: {missing_keys}")
    if extra_keys:
        error_messages.append(f"Found unexpected keys: {extra_keys}")

    # Raise combined error if there are any issues
    if error_messages:
        raise DictionaryValidationError("; ".join(error_messages))


class DictionaryValidationError(Exception):
    """Exception raised for errors in the configuration options."""

    def __init__(self, message, key=None, value=None):
        self.message = message
        self.key = key
        self.value = value
        super().__init__(self._format_message())

    def _format_message(self):
        if self.key is not None and self.value is not None:
            return f"{self.message} Key: '{self.key}', Value: {self.value}"
        return self.message


def convert_numpy_to_python(data, precision=10):
    """
    Recursively converts numpy arrays, scalars, and other numpy types to their Python counterparts
    and rounds numerical values to the specified precision.

    Parameters:
    - data: The numpy data to convert.
    - precision: The decimal precision to which float values should be rounded.

    Returns:
    - The converted data with all numpy types replaced by native Python types and float values rounded.
    """

    if data is None:
        return None

    if isinstance(data, dict):
        return {k: convert_numpy_to_python(v, precision) for k, v in data.items()}

    elif isinstance(data, list):
        return [convert_numpy_to_python(item, precision) for item in data]

    elif isinstance(data, np.ndarray):
        # If the numpy array has more than one element, it is iterable.
        if data.ndim > 0:
            return [convert_numpy_to_python(item, precision) for item in data.tolist()]
        else:
            # This handles the case of a numpy array with a single scalar value.
            return convert_numpy_to_python(data.item(), precision)

    elif isinstance(
        data,
        (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64),
    ):
        return int(data.item())

    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        return round(float(data.item()), precision)

    elif isinstance(data, np.bool_):
        return bool(data.item())

    elif isinstance(data, (np.str_, np.unicode_)):
        return str(data.item())

    # This will handle Python built-in types and other types that are not numpy.
    elif isinstance(data, (float, int, str, bool)):
        if isinstance(data, float):
            return round(data, precision)
        return data

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def compare_contents_or_files(file_or_content_1, file_or_content_2):
    """
    Compare the content of two inputs, which can be either file paths or strings.

    This function accepts two arguments. Each argument can be:
    1. A file path pointing to a file containing text content.
    2. A string containing text content directly.

    If the argument is a file path that exists, the function reads its content.
    If the argument is a string, it's directly used for comparison.

    Parameters
    ----------
    file_or_content1 : str
        First input which can be a file path or string content.
    file_or_content2 : str
        Second input which can be a file path or string content.

    Returns
    -------
    bool
        True if the contents of the two inputs are identical, False otherwise.

    Examples
    --------
    >>> content_same("path/to/file1.txt", "path/to/file2.txt")
    True
    >>> content_same("Hello, world!", "path/to/file_with_hello_world_content.txt")
    True
    >>> content_same("Hello, world!", "Goodbye, world!")
    False
    """
    # If the first argument is a filepath and it exists, read its content
    if os.path.exists(file_or_content_1):
        with open(file_or_content_1, "r") as f1:
            file_or_content_1 = f1.read()

    # If the second argument is a filepath and it exists, read its content
    if os.path.exists(file_or_content_2):
        with open(file_or_content_2, "r") as f2:
            file_or_content_2 = f2.read()

    return file_or_content_1 == file_or_content_2




def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame with multiple rows and columns into a single-row DataFrame with each column
    renamed to include the original row index as a suffix.

    Parameters:
    - df (pd.DataFrame): The original DataFrame to be flattened.

    Returns:
    - pd.DataFrame: A single-row DataFrame with (n×m) columns.

    Example usage:
    >>> original_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> flattened_df = flatten_dataframe(original_df)
    >>> print(flattened_df)

    Note: Row indices start at 1 for the suffix in the column names.
    """

    # Stack the DataFrame to create a MultiIndex Series
    stacked_series = df.stack()

    # Create new column names by combining the original column names with their row index
    new_index = [f"{var}_{index+1}" for index, var in stacked_series.index]

    # Assign the new index to the stacked Series
    stacked_series.index = new_index

    # Convert the Series back to a DataFrame and transpose it to get a single row
    single_row_df = stacked_series.to_frame().T

    return single_row_df


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
        print("    " * indent + str(key) + ":", end=" ")
        if isinstance(value, dict):
            print("")
            print_dict(value, indent + 1)
        else:
            print(value)


def print_boundary_conditions(BC):
    column_width = 25  # Adjust this to your desired width
    print("-" * 80)
    print(" Operating point: ")
    print("-" * 80)
    print(f" {'Fluid: ':<{column_width}} {BC['fluid_name']:<}")
    print(f" {'Flow angle in: ':<{column_width}} {BC['alpha_in']:<.2f} deg")
    print(f" {'Total temperature in: ':<{column_width}} {BC['T0_in']-273.15:<.2f} degC")
    print(f" {'Total pressure in: ':<{column_width}} {BC['p0_in']/1e5:<.3f} bar")
    print(f" {'Static pressure out: ':<{column_width}} {BC['p_out']/1e5:<.3f} bar")
    print(f" {'Angular speed: ':<{column_width}} {BC['omega']*60/2/np.pi:<.1f} RPM")
    print("-" * 80)
    print()


def print_operation_points(operation_points):
    """
    Prints a summary table of operation points scheduled for simulation.

    This function takes a list of operation point dictionaries, formats them
    according to predefined specifications, applies unit conversions where
    necessary, and prints them in a neatly aligned table with headers and units.

    Parameters
    ----------
    - operation_points (list of dict): A list where each dictionary contains
      key-value pairs representing operation parameters and their corresponding
      values.

    Notes
    -----
    - This function assumes that all necessary keys exist within each operation
      point dictionary.
    - The function directly prints the output; it does not return any value.
    - Unit conversions are hardcoded and specific to known parameters.
    - If the units of the parameters change or if different parameters are added,
      the unit conversion logic and `field_specs` need to be updated accordingly.
    """
    length = 80
    index_width = 8
    output_lines = [
        "-" * length,
        " Summary of operation points scheduled for simulation",
        "-" * length,
    ]

    # Configuration for each field with specified width and decimal places
    field_specs = {
        "fluid_name": {"name": "Fluid", "unit": "", "width": 8},
        "alpha_in": {"name": "angle_in", "unit": "[deg]", "width": 10, "decimals": 1},
        "T0_in": {"name": "T0_in", "unit": "[degC]", "width": 12, "decimals": 2},
        "p0_in": {"name": "p0_in", "unit": "[kPa]", "width": 12, "decimals": 2},
        "p_out": {"name": "p_out", "unit": "[kPa]", "width": 12, "decimals": 2},
        "omega": {"name": "omega", "unit": "[RPM]", "width": 12, "decimals": 0},
    }

    # Create formatted header and unit strings using f-strings and field widths
    header_str = f"{'Index':>{index_width}}"  # Start with "Index" header
    unit_str = f"{'':>{index_width}}"  # Start with empty string for unit alignment

    for spec in field_specs.values():
        header_str += f" {spec['name']:>{spec['width']}}"
        unit_str += f" {spec['unit']:>{spec['width']}}"

    # Append formatted strings to the output lines
    output_lines.append(header_str)
    output_lines.append(unit_str)

    # Unit conversion functions
    def convert_units(key, value):
        if key == "T0_in":  # Convert Kelvin to Celsius
            return value - 273.15
        elif key == "omega":  # Convert rad/s to RPM
            return (value * 60) / (2 * np.pi)
        elif key == "alpha_in":  # Convert radians to degrees
            return np.degrees(value)
        elif key in ["p0_in", "p_out"]:  # Pa to kPa
            return value / 1e3
        return value

    # Process and format each operation point
    for index, op_point in enumerate(operation_points, start=1):
        row = [f"{index:>{index_width}}"]
        for key, spec in field_specs.items():
            value = convert_units(key, op_point[key])
            if isinstance(value, float):
                # Format floats with the specified width and number of decimal places
                row.append(f"{value:>{spec['width']}.{spec['decimals']}f}")
            else:
                # Format strings to the specified width without decimals
                row.append(f"{value:>{spec['width']}}")
        output_lines.append(" ".join(row))  # Ensure spaces between columns

    output_lines.append("-" * length)  # Add a closing separator line

    # Join the lines and print the output
    formatted_output = "\n".join(output_lines)

    for line in output_lines:
        print(line)
    return formatted_output


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
    grid=True,
    major_ticks=True,
    minor_ticks=True,
    margin=0.05,
    color_order="matlab",
):
    """Set plot options for publication-quality figures"""

    if isinstance(color_order, str):
        if color_order.lower() == "default":
            color_order = COLORS_PYTHON

        elif color_order.lower() == "matlab":
            color_order = COLORS_MATLAB

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
        "axes.zmargin": margin,
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


def find_latest_results_file(results_path, prefix="performance_analysis_"):
    # Retrieve all files that match the given prefix and extension .xlsx
    files = sorted(
        [
            f
            for f in os.listdir(results_path)
            if f.startswith(prefix) and f.endswith(".xlsx")
        ]
    )

    # Return the first item from the sorted list, which is the latest file
    if files:
        return os.path.join(results_path, files[-1])
    else:
        raise FileNotFoundError(
            f"No Excel files found in directory '{results_path}' with prefix '{prefix}'"
        )


def ensure_iterable(obj):
    """
    Ensure that an object is iterable. If the object is already an iterable
    (except for strings, which are not treated as iterables in this context),
    it will be returned as is. If the object is not an iterable, or if it is
    a string, it will be placed into a list to make it iterable.

    Parameters
    ----------
    obj : any type
        The object to be checked and possibly converted into an iterable.

    Returns
    -------
    Iterable
        The original object if it is an iterable (and not a string), or a new
        list containing the object if it was not iterable or was a string.

    Examples
    --------
    >>> ensure_iterable([1, 2, 3])
    [1, 2, 3]
    >>> ensure_iterable('abc')
    ['abc']
    >>> ensure_iterable(42)
    [42]
    >>> ensure_iterable(np.array([1, 2, 3]))
    array([1, 2, 3])
    """
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return obj
    else:
        return [obj]


def add_string_to_keys(input_dict, suffix):
    """
    Add a suffix to each key in the input dictionary.

    Parameters
    ----------
        input_dict (dict): The input dictionary.
        suffix (str): The string to add to each key.

    Returns
    -------
        dict: A new dictionary with modified keys.

    Examples
    --------
        >>> input_dict = {'a': 1, 'b': 2, 'c': 3}
        >>> add_string_to_keys(input_dict, '_new')
        {'a_new': 1, 'b_new': 2, 'c_new': 3}
    """
    return {f"{key}{suffix}": value for key, value in input_dict.items()}


def extract_timestamp(filename):
    """
    Extract the timestamp from the filename.

    Parameters
    ----------
    filename : str
        The filename containing the timestamp.

    Returns
    -------
    str
        The extracted timestamp.
    """
    # Regular expression to match the timestamp pattern in the filename
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filename)
    if match:
        return match.group(0)
    return ""


def print_simulation_summary(solvers):
    """
    Print a formatted footer summarizing the performance of all operation points.

    This function processes a list of solver objects to provide a summary of the performance
    analysis calculations. It calculates and displays the number of successful points and a summary of
    simulation tme statistics. Additionally, it lists the indices of failed operation points, if any.

    The function is robust against solvers that failed and lack certain attributes like 'elapsed_time'.
    In such cases, these solvers are included in the count of failed operation points, but not in the
    calculation time statistics.

    Parameters
    ----------
    solvers : list
        A list of solver objects. Each solver object should contain attributes related to the
        calculation of an operation point, such as 'elapsed_time' and the 'solution' status.

    """

    # Initialize times list and track failed points
    times = []
    failed_points = []

    for i, solver in enumerate(solvers):
        # Check if the solver is not None and has the required attribute
        if solver and hasattr(solver, "elapsed_time"):
            times.append(solver.elapsed_time)
            if not solver.solution.success:
                failed_points.append(i)
        else:
            # Handle failed solver or missing attributes
            failed_points.append(i)

    # Convert times to a numpy array for calculations
    times = np.asarray(times)
    total_points = len(solvers)

    # Define footer content
    width = 80
    separator = "-" * width
    lines_to_output = [
        "",
        separator,
        "Final summary of performance analysis calculations".center(width),
        separator,
        f" Simulation successful for {total_points - len(failed_points)} out of {total_points} points",
    ]

    # Add failed points message only if there are failed points
    if failed_points:
        lines_to_output.append(
            f"Failed operation points: {', '.join(map(str, failed_points))}"
        )

    # Add time statistics only if there are valid times
    if times.size > 0:
        lines_to_output.extend(
            [
                f" Average calculation time per operation point: {np.mean(times):.3f} seconds",
                f" Minimum calculation time of all operation points: {np.min(times):.3f} seconds",
                f" Maximum calculation time of all operation points: {np.max(times):.3f} seconds",
                f" Total calculation time for all operation points: {np.sum(times):.3f} seconds",
            ]
        )
    else:
        lines_to_output.append(" No valid calculation times available.")

    lines_to_output.append(separator)
    lines_to_output.append("")

    # Display to stdout
    for line in lines_to_output:
        print(line)


def check_lists_match(list1, list2):
    """
    Check if two lists contain the exact same elements, regardless of their order.

    Parameters
    ----------
    list1 : list
        The first list for comparison.
    list2 : list
        The second list for comparison.

    Returns
    -------
    bool
        Returns True if the lists contain the exact same elements, regardless of their order.
        Returns False otherwise.

    Examples
    --------
    >>> check_lists_match([1, 2, 3], [3, 2, 1])
    True

    >>> check_lists_match([1, 2, 3], [4, 5, 6])
    False

    """
    # Convert the lists to sets to ignore order
    list1_set = set(list1)
    list2_set = set(list2)

    # Check if the sets are equal (contain the same elements)
    return list1_set == list2_set


def isclose_significant_digits(a, b, significant_digits):
    """
    Check if two floating-point numbers are close based on a specified number of significant digits.

    Parameters
    ----------
    a : float
        The first number to compare.
    b : float
        The second number to compare.
    sig_digits : int
        The number of significant digits to use for the comparison.

    Returns
    -------
    bool
        True if numbers are close up to the specified significant digits, False otherwise.
    """
    format_spec = f".{significant_digits - 1}e"
    return format(a, format_spec) == format(b, format_spec)

def fill_array_with_increment(n):
    
    """
    Fill an array of length 'n' with values that sum to 1,
    where each value is different but has the same increment
    between neighboring values.
    
    Parameters
    ----------
    n : int
        Length of the array.
    
    Returns
    -------
    array : ndarray
        Array of length 'n' filled with values incrementing
        by a constant factor, resulting in a sum of 1.
    """
    
    if n <= 0:
        return []

    increment = 1.0 / (n+1)
    array = [increment * (i+1) for i in range(n)]
    array /= np.sum(array)

    return array





def render_and_evaluate(expression, data):
    """
    Render variables prefixed with '$' in an expression and evaluate the resulting expression.

    This function processes an input string `expr`, identifying all occurrences of variables
    indicated by a leading '$' symbol. Each such variable is resolved to its value from the
    provided `context` dictionary. The expression with all variables resolved is then evaluated
    and the result is returned.

    This function is useful to render strings defined in a YAML configuration file to values 
    that are calculated within the code and stored in a dicitonary.

    Parameters
    ----------
    expr : str
        The expression string containing variables to be rendered. Variables in the
        expression are expected to be prefixed with a '$' symbol.
    data : dict
        A dictionary containing variables and their corresponding values. These variables
        are used to render values in the expression.

    Returns
    -------
    The result of evaluating the rendered expression. The type of the result depends on the
    expression.

    Notes
    -----
    - `pattern`: A regular expression pattern used to identify variables within the expression.
      Variables are expected to be in the format `$variableName`, potentially with dot-separated
      sub-properties (e.g., `$variable.property`).

    - `replace_with_value`: An inner function that takes a regex match object and returns
      the value of the variable from `context`. `match.group(1)` returns the first captured
      group from the matched text, which in this case is the variable name excluding the
      leading '$' symbol. For example, in `$variableName`, `match.group(1)` would return
      `variableName`.

    - The function uses Python's `eval` for evaluation, which should be used cautiously as
      it can execute arbitrary code. Ensure that the context and expressions are from a trusted
      source.
    """
    # Pattern to find $variable expressions
    pattern = re.compile(r"\$(\w+(\.\w+)*)")

   # Function to replace each match with its resolved value
    def replace_with_value(match):
        nested_key = match.group(1)
        try:
            return str(render_nested_value(nested_key, data))
        except KeyError:
            raise KeyError(f"Variable '{nested_key}' not found in the provided data context.")
        
    try:
        # Replace all $variable with their actual values
        resolved_expr = pattern.sub(replace_with_value, expression)

        # Check if any unresolved variables remain
        if '$' in resolved_expr:
            raise ValueError(f"Unresolved variable in expression: '{resolved_expr}'")

        # Now evaluate the expression
        return eval(resolved_expr, data)
    except SyntaxError:
        raise SyntaxError(f"Syntax error in expression: '{expression}'")
    except Exception as e:
        raise TypeError(f"Error evaluating expression '{expression}': {e}")



def render_nested_value(nested_key, data):
    """
    Retrieves a value from a nested structure (dictionaries or objects with attributes) using a dot-separated key.

    This function is designed to navigate through a combination of dictionaries and objects. For an object to be
    compatible with this function, it must implement a `keys()` method that returns its attribute names.

    This function is intended as a subroutine of the more genera ``render_expression``

    Parameters
    ----------
    nested_key : str
        A dot-separated key string that specifies the path in the structure.
        For example, 'level1.level2.key' will retrieve data['level1']['level2']['key'] if data is a dictionary,
        or data.level1.level2.key if data is an object or a combination of dictionaries and objects.

    data : dict or object
        The starting dictionary or object from which to retrieve the value. This can be a nested structure
        of dictionaries and objects.

    Returns
    -------
    value
        The value retrieved from the nested structure using the specified key. 
        The type of the value depends on what is stored at the specified key in the structure.

    Raises
    ------
    KeyError
        If the specified nested key is not found in the data structure. The error message includes the part
        of the path that was successfully traversed and the available keys or attributes at the last valid level.
    """
    keys = nested_key.split('.')
    value = data
    traversed_path = []

    for key in keys:
        if isinstance(value, dict):
            # Handle dictionary-like objects
            if key in value:
                traversed_path.append(key)
                value = value[key]
            else:
                valid_keys = ', '.join(value.keys())
                traversed_path_str = '.'.join(traversed_path) if traversed_path else 'root'
                raise KeyError(f"Nested key '{key}' not found at '{traversed_path_str}'. Available keys: {valid_keys}")
        elif hasattr(value, key):
            # Handle objects with attributes
            traversed_path.append(key)
            value = getattr(value, key)
        else:
            traversed_path_str = '.'.join(traversed_path)
            available_keys = ', '.join(value.keys())
            raise KeyError(f"Key '{key}' not found in object at '{traversed_path_str}'. Available keys: {available_keys}")

    if not is_numeric(value):
        raise ValueError(f"The key '{nested_key}' is not numeric. Key value is: {value}")

    return value


def evaluate_constraints(data, constraints):
    """
    Evaluates the constraints based on the provided data and constraint definitions.

    Parameters
    ----------
    data : dict
        A dictionary containing performance data against which the constraints will be evaluated.
    constraints : list of dicts
        A list of constraint definitions, where each constraint is defined as a dictionary.
        Each dictionary must have 'variable' (str), 'type' (str, one of '=', '>', '<'), and 'value' (numeric).

    Returns
    -------
    tuple of numpy.ndarray
        Returns two numpy arrays: the first is an array of equality constraints, and the second is an array of 
        inequality constraints. These arrays are flattened and concatenated from the evaluated constraint values.

    Raises
    ------
    ValueError
        If an unknown constraint type is specified in the constraints list.
    """
    # Initialize constraint lists
    c_eq = []    # Equality constraints
    c_ineq = []  # Inequality constraints

    # Loop over all constraint from configuration file
    for constraint in constraints:
        name = constraint['variable']
        type = constraint['type']
        target = constraint['value']
        normalize = constraint.get("normalize", False)

        # Get the performance value for the given variable name
        current = render_nested_value(name, data)

        # Evaluate constraint
        mismatch = current - target

        # Normalize constraint according to specifications
        normalize_factor = normalize if is_numeric(normalize) else target
        if normalize is not False:
            if normalize_factor == 0:
                raise ValueError(f"Cannot normalize constraint '{name} {type} {target}' because the normalization factor is '{normalize_factor}' (division by zero).")
            mismatch /= normalize_factor

        # Add constraints to lists
        if type == '=':
            c_eq.append(mismatch)
        elif type == '>':
            c_ineq.append(mismatch)
        elif type == '<':
            # Change sign because optimizer handles c_ineq > 0
            c_ineq.append(-mismatch)
        else:
            raise ValueError(f"Unknown constraint type: {type}")

    # Flatten and concatenate constraints
    c_eq = np.hstack([np.atleast_1d(item) for item in c_eq]) if c_eq else np.array([])
    c_ineq = np.hstack([np.atleast_1d(item) for item in c_ineq]) if c_ineq else np.array([])

    return c_eq, c_ineq


def evaluate_objective_function(data, objective_function):
    """
    Evaluates the objective function based on the provided data and configuration.

    Parameters
    ----------
    data : dict
        A dictionary containing performance data against which the objective function will be evaluated.
    objective_function : dict
        A dictionary defining the objective function. It must have 'variable' (str) 
        and 'type' (str, either 'minimize' or 'maximize').

    Returns
    -------
    float
        The value of the objective function, adjusted for optimization. Positive for minimization and 
        negative for maximization.

    Raises
    ------
    ValueError
        If an unknown objective function type is specified in the configuration.
    """

    # Get the performance value for the given variable name
    name = objective_function['variable']
    type = objective_function['type']
    value = render_nested_value(name, data)

    if not np.isscalar(value):
        raise ValueError(f"The objective function '{name}' must be an scalar, but the value is: {value}")

    if type == 'minimize':
        return value
    elif type == 'maximize':
        return -value
    else:
        raise ValueError(f"Unknown objective function type: {type}")

