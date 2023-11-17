import os
import sys
import yaml

package_path = os.path.abspath("..")

if package_path not in sys.path:
    sys.path.append(package_path)

import meanline_axial as ml

# Import the options dictionary
config_options = ml.CONFIGURATION_OPTIONS

# Define name map for nice visuals
FIELD_MAP = {
    "description": "Description",
    "is_mandatory": "Mandatory",
    "default_value": "Default value",
    "expected_type": "Valid types",
    "valid_options": "Valid options",
}


def format_value(value, key):
    """Format as strings and add highlighting"""
    if key == "expected_type":
        types =  [v if isinstance(v, str) else v.__name__ for v in ml.ensure_iterable(value)]
        return ", ".join([f"``{v}``" for v in types]  
        )
    elif key == "valid_options":
        return ", ".join(
            [f"``{v}``" if isinstance(v, str) else v for v in ml.ensure_iterable(value)]
        )
    return str(value)


def create_options_rst_tabs(config_dict):
    """
    Generate reStructuredText for tabs from a configuration dictionary.

    This function traverses a configuration dictionary to produce reStructuredText (RST)
    formatted content using sphinx-design tabs. It handles both root-level and 
    nested dictionary entries, formatting them into readable dropdown sections.

    The `dict_to_rst_tabs` nested function is used to recursively process
    the dictionary. It formats root-level keys with human-readable titles and
    maintains the original format for nested keys.

    Parameters
    ----------
    config_dict : dict
        The dictionary containing configuration options, where each key-value pair
        represents a tab or a nested structure within the tab.

    Returns
    -------
    str
        Formatted reStructuredText string representing the configuration options
        as interactive tab.

    """
    def dict_to_rst_tabs(d, indent=0, is_root=True, tab_key=None):
        """
        Recursively converts a nested dictionary into reStructuredText format for sphinx-design tabs.

        This function formats each dictionary key-value pair as a section in a tab,
        and handles nested structures by creating sub-tabs. The formatting of tab
        titles varies based on whether the processing is at the root level or a nested level.

        Parameters
        ----------
        d : dict
            The dictionary to convert into reStructuredText.
        indent : int, optional
            The indentation level for formatting, by default 0.
        parent_key : str, optional
            The parent key for nesting, by default ''.
        tab_key : str, optional
            The current dictionary key to use as the tab title, by default None.
        is_root : bool, optional
            Flag to indicate if the current level is the root, by default False.

        Returns
        -------
        str
            The formatted reStructuredText string for the given dictionary.
        """
        first_indent = "   " * indent
        tabs_indent = "   " * (indent + 1)
        nested_indent = "   " * (indent + 2)

        rst_output = []

        # Handling the root element
        if is_root and tab_key:
            rst_output.append(f"{first_indent}.. tab-item:: {tab_key}\n")

        # Process description
        description = d.get("description", "")
        if description:
            rst_output.append(f"{tabs_indent}{description}\n")

        # Create table for other keys
        keys = [k for k in d if k not in ["_nested", "description"]]
        if keys:
            table_header = (
                f"{tabs_indent}.. list-table::\n"
                f"{tabs_indent}   :widths: 20 80\n"
                f"{tabs_indent}   :header-rows: 0\n"
            )
            rst_output.append(table_header)

            table_rows = [
                f"{tabs_indent}   * - **{FIELD_MAP.get(key, key)}**\n"
                f"{tabs_indent}     - {format_value(d[key], key)}"
                for key in keys
                if not (key == "valid_options" and d[key] is None)
            ]
            table_rows.append("")
            rst_output.extend(table_rows)

        # Handle nested dictionaries
        nested_dict = d.get("_nested")
        if nested_dict:
            rst_output.append(f"\n{tabs_indent}.. tab-set::\n")
            nested_rst = [
                f"{nested_indent}.. tab-item:: {nested_key}\n\n"
                f"{dict_to_rst_tabs(nested_value, indent + 2, False, nested_key)}"
                for nested_key, nested_value in nested_dict.items()
            ]
            rst_output.extend(nested_rst)

        return "\n".join(rst_output)

    # Main function execution
    rst_content = ".. tab-set::\n\n"
    for tab_key, tab_value in config_dict.items():
        rst_content += dict_to_rst_tabs(tab_value, indent=1, is_root=True, tab_key=tab_key)
    rst_content += "\n"

    return rst_content


def create_options_rst_dropdowns(config_dict):
    """
    Generate reStructuredText for dropdowns from a configuration dictionary.

    This function traverses a configuration dictionary to produce reStructuredText (RST)
    formatted content using sphinx-design dropdowns. It handles both root-level and 
    nested dictionary entries, formatting them into readable dropdown sections.

    The `dict_to_rst_dropdowns` nested function is used to recursively process
    the dictionary. It formats root-level keys with human-readable titles and
    maintains the original format for nested keys.

    Parameters
    ----------
    config_dict : dict
        The dictionary containing configuration options, where each key-value pair
        represents a dropdown or a nested structure within the dropdown.

    Returns
    -------
    str
        Formatted reStructuredText string representing the configuration options
        as interactive dropdowns.

    """

    def dict_to_rst_dropdowns(d, indent=0, parent_key='', dropdown_key=None, is_root=False):
        """
        Recursively converts a nested dictionary into reStructuredText format for sphinx-design dropdowns.

        This function formats each dictionary key-value pair as a section in a dropdown,
        and handles nested structures by creating sub-dropdowns. The formatting of dropdown
        titles varies based on whether the processing is at the root level or a nested level.

        Parameters
        ----------
        d : dict
            The dictionary to convert into reStructuredText.
        indent : int, optional
            The indentation level for formatting, by default 0.
        parent_key : str, optional
            The parent key for nesting, by default ''.
        dropdown_key : str, optional
            The current dictionary key to use as the dropdown title, by default None.
        is_root : bool, optional
            Flag to indicate if the current level is the root, by default False.

        Returns
        -------
        str
            The formatted reStructuredText string for the given dictionary.
        """
        first_indent = "   " * indent
        content_indent = "   " * (indent + 1)
        nested_indent = "   " * (indent + 2)

        rst_output = []

        # Construct the full dropdown title including the parent key  
        dropdown_title = f"{parent_key}.{dropdown_key}" if parent_key else dropdown_key
        if is_root:
            formated_root_key = dropdown_key.replace('_', ' ').title()
            rst_output.append(f"{first_indent}.. dropdown:: {formated_root_key}")
            rst_output.append(f"{content_indent}:color: primary\n")
                
        else:
            rst_output.append(f"{first_indent}.. dropdown:: {dropdown_title}")
            rst_output.append(f"{content_indent}:color: secondary\n")

        # Add option description outside the table
        description = d.get('description', '')
        if description:
            rst_output.append(f"{content_indent}{description}\n\n")

        # Create table for all the other keys
        keys = [k for k in d if k not in ['_nested', 'description']]
        if keys:
            table_header = (
                f"{content_indent}.. list-table::\n"
                f"{content_indent}   :widths: 20 80\n"
                f"{content_indent}   :header-rows: 0\n"
            )
            rst_output.append(table_header)

            table_rows = [
                f"{content_indent}   * - **{FIELD_MAP.get(key, key)}**\n"
                f"{content_indent}     - {format_value(d[key], key)}"
                for key in keys if not (key == 'valid_options' and d[key] is None)
            ]
            table_rows.append("")
            rst_output.extend(table_rows)

        # Handle nested dictionaries
        nested_dict = d.get('_nested')
        if nested_dict:
            nested_rst = [
                f"{dict_to_rst_dropdowns(nested_value, indent + 1, dropdown_title, nested_key)}"
                for nested_key, nested_value in nested_dict.items()
            ]
            rst_output.extend(nested_rst)

        return '\n'.join(rst_output)

    # Main function execution
    rst_content = ""
    for tab_key, tab_value in config_dict.items():
        rst_content += dict_to_rst_dropdowns(tab_value, indent=0, dropdown_key=tab_key, is_root=True)
    rst_content += "\n"

    return rst_content



def convert_types_to_strings(obj):
    """
    Recursively replace type objects in a given data structure with their string names.

    This function is intended to process a nested data structure (like a dictionary or list)
    that contains Python type objects. It replaces these type objects with their corresponding
    string names (e.g., 'str' for `str`, 'float' for `float`). This is used to render types as
    strings when exporting a dictionary to YAML in a human-readable format.

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
        return {k: convert_types_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, type):
        return obj.__name__
    elif isinstance(obj, (list, tuple)):
        return [convert_types_to_strings(e) for e in obj]
    elif obj is None:
        return "None"
    else:
        return obj


if __name__ == "__main__":

    # Options as dropdowns
    rst_content = create_options_rst_dropdowns(config_options)
    with open("source/configuration_options_dropdowns.rst", "w") as file:
        file.write(rst_content)

    # Options as tabs
    rst_content = create_options_rst_tabs(config_options)
    with open("source/configuration_options_tabs.rst", "w") as file:
        file.write(rst_content)

    # Options as YAML file
    config_options = convert_types_to_strings(config_options)
    with open("source/configuration_options.yaml", "w") as file:
        yaml.dump(config_options, file, default_flow_style=False, sort_keys=False)
