import os
import re
import yaml
import logging
from datetime import datetime

from collections.abc import Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from cycler import cycler


def read_configuration_file(filename):
    """
    Retrieve cascades data from a YAML configuration file and process the geometry data.

    This function reads a YAML configuration file to obtain cascades data, then converts
    the geometry values to numpy arrays. It also initializes `fixed_params` and `overall`
    dictionaries within the cascades data. String expressions in the YAML file representing
    numerical values are evaluated to actual numerical values when possible.

    Parameters
    ----------
    filename : str
        Path to the YAML configuration file to be read.

    Returns
    -------
    cascades_data : dictionary with fields:
        - "geometry": A dictionary where each key corresponds to a geometry parameter and its associated value is a numpy array.
        - "fixed_params": An empty dictionary, initialized for future usage.
        - "overall": An empty dictionary, initialized for future usage.

    Notes
    -----
    The function uses `postprocess_config` to evaluate string expressions found in the
    YAML data. For example, a YAML entry like "value: "np.pi/2"" will be converted to
    its numerical equivalent.
    """

    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    # Validate required and allowed sections
    validate_config_sections(config)

    # Convert configuration options
    config = postprocess_config(config)

    return config


def validate_config_sections(config):
    """
    Validate the presence of required configuration sections and check for any unexpected sections.

    This function ensures that all required sections are present in the configuration and
    that there are no sections other than the allowed ones. It raises a ConfigurationError if
    either required sections are missing or unexpected sections are found.

    Parameters
    ----------
    config : dict
        The configuration dictionary loaded from a YAML file. The keys of this dictionary
        represent the different configuration sections.

    """
    required_sections = {"operation_points", "solver_options", "model_options"}
    allowed_sections = required_sections.union({"performance_map", "geometry", "optimization"})
    validate_keys(config, required_sections, allowed_sections)


def postprocess_config(config):
    """
    Postprocesses the YAML configuration data by converting string values
    to numbers and lists to numpy arrays. Numerical expressions like "1+2" or
    "2*np.pi" are evaluated into the corresponding numerical values

    Parameters
    ----------
    config : dict
        The configuration data loaded from a YAML file.

    Returns
    -------
    config : dict
        The postprocessed configuration data.

    Raises
    ------
    ConfigurationError
        If a list contains elements of different types after conversion.
    """

    def convert_to_numbers(data):
        """Recursively convert string expressions in the configuration to numbers."""
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

    def convert_to_arrays(data, parent_key=""):
        """
        Convert lists to numpy arrays if all elements are numeric and of the same type.
        Raises ConfigurationError if a list contains elements of different types.
        """
        if isinstance(data, dict):
            return {k: convert_to_arrays(v, parent_key=k) for k, v in data.items()}
        elif isinstance(data, list):
            if not data:  # Empty list
                return data
            first_type = type(data[0])
            if not all(isinstance(item, first_type) for item in data):
                raise ConfigurationError(
                    "Option contains elements of different types.",
                    key=parent_key,
                    value=data,
                )
            return np.array(data)
        else:
            return data

    config = convert_to_numbers(config)
    config = convert_to_arrays(config)

    return config


def validate_keys(checked_dict, required_keys, allowed_keys=None):
    """
    Validate the presence of required keys and check for any unexpected keys in a dictionary.

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
        raise ConfigurationError("; ".join(error_messages))
    

class ConfigurationError(Exception):
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

    Args:
        input_dict (dict): The input dictionary.
        suffix (str): The string to add to each key.

    Returns:
        dict: A new dictionary with modified keys.

    Example:
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
        if solver and hasattr(solver, 'elapsed_time'):
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
        lines_to_output.append(f"Failed operation points: {', '.join(map(str, failed_points))}")

   # Add time statistics only if there are valid times
    if times.size > 0:
        lines_to_output.extend([
            f" Average calculation time per operation point: {np.mean(times):.3f} seconds",
            f" Minimum calculation time of all operation points: {np.min(times):.3f} seconds",
            f" Maximum calculation time of all operation points: {np.max(times):.3f} seconds",
            f" Total calculation time for all operation points: {np.sum(times):.3f} seconds",
        ])
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
