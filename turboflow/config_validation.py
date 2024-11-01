from pydantic import BaseModel, ValidationError
from .turbo_configurations import AxialTurbine, CentrifugalCompressor
from typing import Union
import yaml
import numbers
import numpy as np
from . import utilities

TURBOMACHINERIES = [
    "axial_turbine",
    "centrifugal_compressor"]


def convert_configuration_options(config):
    """
    Processes configuration data by evaluating string expressions as numerical values and converting lists to numpy arrays.

    This function iteratively goes through the configuration dictionary and converts string representations of numbers
    (e.g., "1+2", "2*np.pi") into actual numerical values using Python's `eval` function. It also ensures that all numerical
    values are represented as Numpy types for consistency across the application.

    Parameters
    ----------
    config : dict
        The configuration data loaded from a YAML file, typically containing a mix of strings, numbers, and lists.

    Returns
    -------
    dict
        The postprocessed configuration data where string expressions are evaluated as numbers, and all numerical values
        are cast to corresponding NumPy types.

    Raises
    ------
    ConfigurationError
        If a list contains elements of different types after conversion, indicating an inconsistency in the expected data types.
    """

    def convert_strings_to_numbers(data):
        """
        Recursively converts string expressions within the configuration data to numerical values.

        This function handles each element of the configuration: dictionaries are traversed recursively, lists are processed
        element-wise, and strings are evaluated as numerical expressions. Non-string and valid numerical expressions are
        returned as is. The conversion supports basic mathematical operations and is capable of recognizing Numpy functions
        and constants when used in the strings.

        Parameters
        ----------
        data : dict, list, str, or number
            A piece of the configuration data that may contain strings representing numerical expressions.

        Returns
        -------
        The converted data, with all string expressions evaluated as numbers.
        """
        if isinstance(data, dict):
            return {
                key: convert_strings_to_numbers(value) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [convert_strings_to_numbers(item) for item in data]
        elif isinstance(data, bool):
            return data
        elif isinstance(data, str):
            # Evaluate strings as numbers if possible
            try:
                data = eval(data)
                return convert_numbers_to_numpy(data)
            except (NameError, SyntaxError, TypeError):
                return data
        elif isinstance(data, numbers.Number):
            # Convert Python number types to corresponding NumPy number types
            return convert_numbers_to_numpy(data)
        else:
            return data

    def convert_numbers_to_numpy(data):
        """
        Casts Python native number types (int, float) to corresponding Numpy number types.

        This function ensures that all numeric values in the configuration are represented using Numpy types.
        It converts integers to `np.int64` and floats to `np.float64`.

        Parameters
        ----------
        data : int, float
            A numerical value that needs to be cast to a Numpy number type.

        Returns
        -------
        The same numerical value cast to the corresponding Numpy number type.

        """
        if isinstance(data, int):
            return np.int64(data)
        elif isinstance(data, float):
            return np.float64(data)
        else:
            return data

    def convert_to_arrays(data, parent_key=""):
        """
        Convert lists within the input data to Numpy arrays.

        Iterates through the input data recursively. If a list is encountered, the function checks if all elements are of the same type.
        If they are, the list is converted to a Numpy array. If the elements are of different types, a :obj:`ConfigurationError` is raised.

        Parameters
        ----------
        data : dict or list or any
            The input data which may contain lists that need to be converted to NumPy arrays. The data can be a dictionary (with recursive processing for each value), a list, or any other data type.
        parent_key : str, optional
            The key in the parent dictionary corresponding to `data`, used for error messaging. The default is an empty string.

        Returns
        -------
        dict or list or any
            The input data with lists converted to NumPy arrays. The type of return matches the type of `data`. Dictionaries and other types are returned unmodified.

        Raises
        ------
        ValueError
            If a list within `data` contains elements of different types. The error message includes the problematic list and the types of its elements.

        """
        if isinstance(data, dict):
            return {k: convert_to_arrays(v, parent_key=k) for k, v in data.items()}
        elif isinstance(data, list):
            if not data:  # Empty list
                return data
            first_type = type(data[0])
            if not all(isinstance(item, first_type) for item in data):
                element_types = [type(item) for item in data]
                raise ValueError(
                    f"Option '{parent_key}' contains elements of different types: {data}, "
                    f"types: {element_types}"
                )
            return np.array(data)
        else:
            return data

    # Convert the configuration options to Numpy arrays when relevant
    config = convert_strings_to_numbers(config)
    config = convert_to_arrays(config)

    return config


def object_to_dict(obj):
    """
    Recursively convert an object's attributes to a dictionary.

    This function takes an object and converts its attributes to a dictionary format.
    It handles nested dictionaries, lists, and objects with a `__dict__` attribute,
    recursively converting all attributes to dictionaries or arrays as appropriate.

    Parameters
    ----------
    obj : any
        The object to convert. This can be a dictionary, list, or an object
        with a `__dict__` attribute.

    Returns
    -------
    dict
        A dictionary representation of the object, where all nested attributes
        are recursively converted.
    """
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return np.array([object_to_dict(i) for i in obj])
    elif hasattr(obj, "__dict__"):
        return {k: object_to_dict(v) for k, v in vars(obj).items()}
    else:
        return obj


# Function to load and validate the configuration file
def load_config(config_file_path: str, print_summary=False):
    """
    Load and process a configuration file.

    This function reads a YAML configuration file, validates its contents,
    and converts it into a configuration object for turbomachinery analysis.
    The configuration can be printed as a summary if specified.

    Parameters
    ----------
    config_file_path : str
        Path to the YAML configuration file.
    print_summary : bool, optional
        Whether to print a summary of the loaded configuration (default is True).

    Returns
    -------
    dict
        A dictionary representation of the configuration object,
        with all nested attributes recursively converted.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    ValueError
        If required fields are missing or contain invalid values.
    yaml.YAMLError
        If there is an error parsing the YAML file.

    """

    try:
        with open(config_file_path, "r") as config_file:
            # Load config file
            config_data = yaml.safe_load(config_file)

            # Convert string representation of numbers to number and lists to arrays
            config_data = convert_configuration_options(config_data)
            if "turbomachinery" not in config_data:
                raise ValueError("Missing 'turbomachinery' field in configuration.")

            turbomachinery_type = config_data["turbomachinery"]

            if turbomachinery_type == "axial_turbine":
                config = AxialTurbine(**config_data)
            elif turbomachinery_type == 'centrifugal_compressor':
                config =  CentrifugalCompressor(**config_data)
            else:
                raise ValueError(
                    f"Unknown turbomachinery type: {turbomachinery_type}. Available turbomachineries are: {TURBOMACHINERIES}"
                )
            # Convert configuration object to a nested dictionary
            config = object_to_dict(config)

            if config is not None:
                # Print configuration summary
                if print_summary:
                    succsess_message = "Configuration loaded successfully: "
                    dashed_line = "-" * len(succsess_message)
                    print(dashed_line)
                    print(succsess_message)
                    print(dashed_line)
                    utilities.print_dict(config)
                    print(dashed_line)
                    print("\n")

            return config

    except FileNotFoundError:
        print("Configuration file not found.")
    except ValidationError as e:
        print("Validation error in configuration file:")
        print(e)
    except yaml.YAMLError as e:
        print("Error parsing YAML file:")
        print(e)


def read_configuration_file(filename):
    """
    Reads a YAML configuration file and converts options to NumPy types when possible.

    Parameters
    ----------
    filename : str
        The path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the configuration options, with values converted to NumPy types where applicable.

    Raises
    ------
    Exception
        If there is an error reading or parsing the configuration file.

    """
    # Read configuration file
    try:
        with open(filename, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        raise Exception(
            f"Error parsing configuration file: '{filename}'. Original error: {e}"
        )

    # Convert options to Numpy when possible
    config = convert_configuration_options(config)

    return config


# Example usage
if __name__ == "__main__":
    config_file_path = "config.yaml"  # Path to your YAML configuration file
    config = load_config(config_file_path, mode="optimization")
    if config:
        print("Configuration loaded successfully")
