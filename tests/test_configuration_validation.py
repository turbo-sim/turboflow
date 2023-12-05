import os
import sys
import difflib
import copy
import datetime
import numpy as np
import pandas as pd
import pytest
import os
import sys
import yaml

desired_path = os.path.abspath("../")

if desired_path not in sys.path:
    sys.path.append(desired_path)


# from meanline_axial.configuration import _validate_configuration_file
from meanline_axial.utilities import validate_configuration_file
from meanline_axial.config_validation import CONFIGURATION_OPTIONS

# Define run settings
DATA_DIR = "regression_data"
CONFIG_DIR = "config_files"
CONFIG_FILES = [
    "configuration_valid.yaml",
    "configuration_invalid.yaml",
]

# Create a directory to save regression test data
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


@pytest.mark.parametrize("config_file", CONFIG_FILES)
def test_configuration_validation(config_file):

    # Combine current output into one string for comparison
    current_output = create_validation_message(config_file)

    # Read configuration regression data
    base_name = os.path.splitext(config_file)[0]
    regression_filepath = os.path.join(DATA_DIR, f"{base_name}_out.txt")
    with open(regression_filepath, "r") as file:
        regression_output = file.read()

    # Split the strings into lines and get the difference
    str1_lines = regression_output.splitlines(keepends=True)
    str2_lines = current_output.splitlines(keepends=True)
    diff = difflib.ndiff(str1_lines, str2_lines)

    # Filter out lines that don't start with '-' or '+'
    filtered_diff = [line for line in diff if line.startswith('-') or line.startswith('+')]

    if filtered_diff:
        print()
        print("Differences found between regression and current output:")
        print("Lines starting with '-' are present only in the first string (original).")
        print("Lines starting with '+' are present only in the second string (current).")
        print(''.join(filtered_diff))
    else:
        print()
        print("No differences found between regression and current output.")

    # Assert that the outputs match
    check = current_output == regression_output
    assert check, f"Output does not match for {config_file}"

    
def create_validation_message(config_file, out_dir=None):
    """
    Generates a validation message for a given configuration file.

    This function reads a specified configuration file, validates it, and then either returns 
    the validation messages as a string or saves them to a file, based on the parameters provided.

    When 'out_dir' is not provided, the function returns the validation messages directly. 
    This is particularly useful for testing purposes, where the output needs to be compared 
    against regression data.

    When 'out_dir' is provided, the function saves the validation messages to a file in the 
    specified directory. This is useful for creating and updating regression data files. 
    The file name is derived from the configuration file name with an added '_out.txt' suffix.
    If a file with the same name already exists, a datetime suffix is appended to avoid overwriting.

    Parameters
    ----------
    config_file : str
        The file name of the configuration file to be validated. This should include the path
        if the file is not in the current working directory.
    out_dir : str, optional
        The directory where the validation messages file will be saved. If None, the validation 
        messages are returned as a string.

    Returns
    -------
    str or None
        A string containing the validation messages if 'out_dir' is None. If 'out_dir' is provided,
        the function returns None after saving the messages to a file.
    """

    # Read configuration file
    filepath = os.path.join(CONFIG_DIR, config_file)
    config, info, error = validate_configuration_file(filepath, CONFIGURATION_OPTIONS)

    # Prepare output string
    output_string = [
        "Information messages:",
        info,
        "",
        "Error messages:",
        str(error),
    ]

    if out_dir is None:
        return "\n".join(output_string)

    else:
        # Construct base file name for output
        base_name = os.path.splitext(config_file)[0]
        filepath_out = os.path.join(out_dir, f"{base_name}_out.txt")

        # Safeguard mechanism: append datetime suffix if file exists
        if os.path.exists(filepath_out):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filepath_out = os.path.join(out_dir, f"{base_name}_out_{timestamp}.txt")

        # Write information and error messages to file
        with open(filepath_out, "w") as file:
            file.write("Information messages:\n")
            file.write(info)
            file.write("\n\n")
            file.write("Error messages:\n")
            file.write(str(error))

        print(f"Regression data saved: {filepath_out}")



if __name__ == "__main__":

    # # Create configuration validation regression files
    # for config_file in CONFIG_FILES:
    #     create_validation_message(config_file, DATA_DIR)

    # Running pytest from Python
    pytest.main([__file__, "-s", "-vv"])
