import os
import re
import yaml
import time
import logging
from datetime import datetime


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


def find_latest_results_file(results_path, prefix="performance_analysis_"):
    """
    Retrieve all files that match the given prefix and extension .xlsx
    """
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

