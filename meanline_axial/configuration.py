import yaml
import numpy as np
from numbers import Number
from . import utilities as util
from .validation import configuration_schema

NUMERIC = "<numeric value>"



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

    # Convert configuration options
    config = convert_config_to_numpy(config)

    # Validate required and allowed sections
    validate_configuration_file(config, configuration_schema)

    return config

def convert_config_to_numpy(config):
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
            return {key: convert_strings_to_numbers(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [convert_strings_to_numbers(item) for item in data]
        elif isinstance(data, bool):
            return data
        elif isinstance(data, str):
            # Evaluate strings as numbers if possible
            try:
                data = eval(data)
                return cast_numbers_to_numpy(data)
            except (NameError, SyntaxError, TypeError):
                return data
        elif isinstance(data, Number):
            # Convert Python number types to corresponding NumPy number types
            return cast_numbers_to_numpy(data)
        else:
            return data


    def cast_numbers_to_numpy(data):
        """
        Casts Python native number types (int, float) to corresponding Numpy number types.

        This function ensures that all numeric values in the configuration are represented using Numpy types.
        It converts integers to `np.int64` and floats to `np.float64`.

        Parameters
        ----------
        data : int, float
            A numerical value that needs to be cast to a NumPy number type.

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

    config = convert_strings_to_numbers(config)
    config = convert_to_arrays(config)

    return config






def replace_types(obj):
    """
    Recursively replace type objects in a given data structure with their string names.

    This function is intended to process a nested data structure (like a dictionary or list)
    that contains Python type objects. It replaces these type objects with their corresponding
    string names (e.g., 'str' for `str`, 'float' for `float`). This is particularly useful for
    rendering the options of variable types in a human-readable format when exporting to YAML.

    Parameters
    ----------
    obj : dict or list
        The data structure (dictionary or list) containing the type objects to be replaced.
        This structure can be nested, and the function will process it recursively.

    Returns
    -------
    dict or list
        A new data structure with the same format as the input, where type objects have been
        replaced with their string names. The type of the return matches the type of the input
        (dict or list).

    Examples
    --------
    >>> example_schema = {
    ...     "parameter1": {
    ...         "type": int,
    ...         "default": 10
    ...     },
    ...     "parameter2": {
    ...         "type": str,
    ...         "default": "example"
    ...     }
    ... }
    >>> replace_types(example_schema)
    {
        'parameter1': {'type': 'int', 'default': 10},
        'parameter2': {'type': 'str', 'default': 'example'}
    }
    """
    if isinstance(obj, dict):
        return {k: replace_types(v) for k, v in obj.items()}
    elif isinstance(obj, type):
        return obj.__name__
    elif isinstance(obj, (list, tuple)):
        return [replace_types(e) for e in obj]
    elif obj is None:
        return "None"
    else:
        return obj


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration options."""

    def __init__(self, messages):
        self.messages = messages
        super().__init__(self._format_message())

    def _format_message(self):
        return "Configuration errors detected.\n" + ";\n".join(self.messages)


def validate_configuration_file(config, schema):
    """
    Validates thee configuration against the provided schema.
    This function performs several checks, including:

    - Presence of unexpected keys: Ensures there are no extra fields in the configuration
      that are not defined in the schema.
    - Mandatory fields: Checks that all fields marked as mandatory in the schema are present
      in the configuration.
    - Data type validation: Verifies that each field in the configuration matches the expected
      data type(s) defined in the schema. The expected type can be a single type or a combination of types.
    - Valid options: For fields with a specified list of valid values, it checks that the
      field's value in the configuration is one of these valid options. This includes support for numeric values
      where a string placeholder like '<numeric value>' is used in the schema.
    - Default values: For non-mandatory fields that are not present in the configuration,
      assigns a default value if one is specified in the schema.
    - Nested fields: Recursively validates nested dictionaries within the configuration, 
      regardless of whether they are presented as individual dictionaries or as lists/arrays 
      of dictionaries. The validation is conducted according to the nested schema defined 
      under the '_nested' key. This approach ensures consistency in handling nested 
      configurations, facilitating flexibility in configuration structure.

    The function raises a ConfigurationError if any discrepancies are found and also prints
    messages for fields where default values are used.

    The function allows for rapid development and prototyping by bypassing validation for any configuration options
    prefixed with an underscore ('_'). This feature is particularly useful for testing new or experimental settings
    without having to update the entire validation schema.

    Parameters
    ----------
    config : dict
        The 'model_options' section of the configuration to be validated.
    schema : dict
        The schema against which the configuration will be validated.

    Returns
    -------
    dict
        The updated configuration dictionary after applying default values.

    Raises
    ------
    ConfigurationError
        If there are any discrepancies between the configuration and the schema.
    """

    def validate_field(conf, schema, parent, errors, info):
        # Check for unexpected keys
        # Bypass validation for keys starting with '_' to skip options during development
        keys_to_validate = set(k for k in conf.keys() if not k.startswith("_"))
        unexpected_keys = keys_to_validate - set(schema.keys())
        if unexpected_keys:
            errors.append(f"Unexpected keys in '{parent}': {unexpected_keys}")

        # Loop through all the possible configuration options
        for key, specs in schema.items():
            # Define option path recursively
            current_path = f"{parent}/{key}" if parent else key

            # Ensure expected_type is a tuple
            # Check needed because some variables multiple types
            expected_types = specs["expected_type"]
            if not isinstance(expected_types, tuple):
                expected_types = (
                    tuple(expected_types)
                    if isinstance(expected_types, list)
                    else (expected_types,)
                )

            # Check if the key is present in the configuration
            if key in conf:
                # Validate the type
                if not isinstance(conf[key], expected_types):
                    
                    type_expected = ", ".join([t.__name__ for t in expected_types])
                    type_actual = type(conf[key]).__name__
                    msg = f"Incorrect type for field '{current_path}': '{type_actual}'. Expected {type_expected}"
                    errors.append(msg)

                # Validate option from list if there are valid options defined
                elif specs.get("valid_options") is not None:
                    conf_values = util.ensure_iterable(conf[key])
                    for item in conf_values:
                        # Check single value or each item of the list/array
                        if item not in specs["valid_options"] and not (
                            NUMERIC in specs["valid_options"]
                            and isinstance(item, (int, float))
                        ):
                            msg = f"Invalid value '{item}' for field '{current_path}'. Valid options are: {specs['valid_options']}"
                            errors.append(msg)
                    
            # If the key is not present and is mandatory, add error message
            elif specs["is_mandatory"]:
                errors.append(f"Missing required field: '{current_path}'")

            # If the key is not present and is non-mandatory, use the default value if provided
            elif not specs["is_mandatory"] and "default_value" in specs:
                conf[key] = specs["default_value"]
                msg = f"Field '{current_path}' not specified; using default value: {specs['default_value']}"
                info.append(msg)

            else:
                raise Exception("The configuration validation function encountered an unexpected case")

            # Recursively validate nested fields
            if "_nested" in specs:
                nested_conf = conf.get(key, {})

                # Ensure that nested configurations are iterable (operation_points can be a list of dictionaries)
                nested_conf = util.ensure_iterable(nested_conf) if specs["_nested"] else nested_conf

                # If it's a list or array of dictionaries, iterate through each item
                if isinstance(nested_conf, (list, np.ndarray)):
                    for item in nested_conf:
                        if isinstance(item, dict):
                            validate_field(item, specs["_nested"], current_path, errors, info)
                        else:
                            errors.append(f"Each item in '{current_path}' must be a dictionary.")
                elif isinstance(nested_conf, dict): # Single dictionary, validate directly
                    validate_field(nested_conf, specs["_nested"], current_path, errors, info)
                else:
                    errors.append(f"Invalid type for nested field '{current_path}'. Expected a dictionary or list of dictionaries.")

        return conf

    errors = []
    messages = []
    conf = validate_field(config, schema, None, errors, messages)

    if messages:
        messages.insert(0, "Some parameters were not defined. Using default options:")
        messages = "\n".join(messages)
        print(messages)
        print("")

    if errors:
        raise ConfigurationError(errors)

    return conf

