import re
import numbers
import numpy as np
import pandas as pd

from collections.abc import Iterable

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
    Fill an array of length `n` with values that sum to 1,
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

