import os
import sys
import yaml

package_path = os.path.abspath("..")

if package_path not in sys.path:
    sys.path.append(package_path)

import meanline_axial as ml


mapping = {
    "description": "Description",
    "is_mandatory": "Mandatory",
    "default_value": "Default value",
    "expected_type": "Valid types",
    "valid_options": "Valid options",
}


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


# def dict_to_rst(d, indent=0, is_root=True, tab_key=None):
#     rst_output = ""
#     tab_indent = "   " * indent
#     bullet_indent = "   " * (indent + +1)
#     tabs_indent = "   " * (indent + 1)
#     nested_indent = "   " * (indent + 2)

#     # Handling the root element
#     if is_root and tab_key:
#         rst_output += f"{tab_indent}.. tab-item:: {tab_key}\n\n"

#     # Process all keys
#     for key, value in d.items():
#         if key == "_nested":
#             continue
#         elif key == "description":
#             rst_output += f"{bullet_indent}{value}\n\n"
#         elif key == "valid_options":
#             if value is not None:
#                 rst_output += f"{bullet_indent}* **{key}**: {value}\n"
#         elif key == "expected_type":
#             value = ml.ensure_iterable(value)
#             type_strings = ", ".join(
#                 [v if isinstance(v, str) else v.__name__ for v in value]
#             )
#             rst_output += f"{bullet_indent}* **{key}**: {type_strings}\n"
#         else:
#             rst_output += f"{bullet_indent}* **{key}**: {value}\n"

#     # Handle nested dictionaries
#     nested_dict = d.get("_nested")
#     if nested_dict:
#         rst_output += f"\n{tabs_indent}.. tab-set::\n\n"
#         for nested_key, nested_value in nested_dict.items():
#             rst_output += f"{nested_indent}.. tab-item:: {nested_key}\n\n"
#             rst_output += dict_to_rst(nested_value, indent + 2, is_root=False)
#             rst_output += "\n"

#     return rst_output

def format_value(value, key):
    if key == "expected_type":
        return ", ".join(
            [v if isinstance(v, str) else v.__name__ for v in ml.ensure_iterable(value)]
        )
    return str(value)


def process_config_options(config_dict, type):
    
    if type == "tabs":
        rst_content = ".. tab-set::\n\n"
        for tab_key, tab_value in config_dict.items():
            rst_content += dict_to_rst_tabs(tab_value, indent=1, is_root=True, tab_key=tab_key)
        rst_content += "\n"
    elif type == "dropdowns":
        rst_content = ""
        for tab_key, tab_value in config_dict.items():
            rst_content += dict_to_rst_dropdowns(tab_value, indent=0, dropdown_key=tab_key)
        rst_content += "\n"
    else:
        raise Exception("Invalid options, choose tabs or dropdowns")



    return rst_content


def dict_to_rst_tabs(d, indent=0, is_root=True, tab_key=None):
    """
    Convert a dictionary to reStructuredText format using sphinx-design tabs.

    This function recursively processes a dictionary and its nested dictionaries
    to generate reStructuredText content, particularly focusing on creating tabs
    and tables to represent the dictionary structure.

    Parameters:
    d (dict): The dictionary to be converted to reStructuredText.
    indent (int, optional): The current indentation level for reStructuredText formatting. Defaults to 0.
    is_root (bool, optional): Flag to indicate if the current processing is at the root level. Defaults to True.
    tab_key (str, optional): The label for the tab if the current dictionary represents a tab. Defaults to None.

    Returns:
    str: The generated reStructuredText content as a string.
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
            f"{tabs_indent}   * - **{mapping.get(key, key)}**\n"
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







def dict_to_rst_dropdowns(d, indent=0, parent_key='', dropdown_key=None):
    """
    Convert a dictionary to reStructuredText format using sphinx-design dropdowns.

    This function recursively processes a dictionary and its nested dictionaries
    to generate reStructuredText content, particularly focusing on creating dropdowns
    and tables to represent the dictionary structure.

    Parameters:
    d (dict): The dictionary to be converted to reStructuredText.
    indent (int, optional): The current indentation level for reStructuredText formatting. Defaults to 0.
    parent_key (str, optional): The parent key to be included in the dropdown title. Defaults to ''.
    dropdown_key (str, optional): The label for the dropdown if the current dictionary represents a dropdown. Defaults to None.

    Returns:
    str: The generated reStructuredText content as a string.
    """
    first_indent = "   " * indent
    content_indent = "   " * (indent + 1)
    nested_indent = "   " * (indent + 2)

    rst_output = []

    # Construct the full dropdown title including the parent key
    full_dropdown_title = f"{parent_key}.{dropdown_key}" if parent_key else dropdown_key
    rst_output.append(f"{first_indent}.. dropdown:: {full_dropdown_title}\n\n")

    # Process description
    description = d.get('description', '')
    if description:
        rst_output.append(f"{content_indent}{description}\n\n")

    # Create table for other keys
    keys = [k for k in d if k not in ['_nested', 'description']]
    if keys:
        table_header = (
            f"{content_indent}.. list-table::\n"
            f"{content_indent}   :widths: 20 80\n"
            f"{content_indent}   :header-rows: 0\n"
        )
        rst_output.append(table_header)

        table_rows = [
            f"{content_indent}   * - **{mapping.get(key, key)}**\n"
            f"{content_indent}     - {format_value(d[key], key)}"
            for key in keys if not (key == 'valid_options' and d[key] is None)
        ]
        table_rows.append("")
        rst_output.extend(table_rows)

    # Handle nested dictionaries
    nested_dict = d.get('_nested')
    if nested_dict:
        nested_rst = [
            f"{dict_to_rst_dropdowns(nested_value, indent + 1, full_dropdown_title, nested_key)}"
            for nested_key, nested_value in nested_dict.items()
        ]
        rst_output.extend(nested_rst)

    return '\n'.join(rst_output)


if __name__ == "__main__":


    OPTIONS_MODEL = ml.CONFIGURATION_OPTIONS
    rst_content = process_config_options(OPTIONS_MODEL, "tabs")
    # Writing to a file
    with open("source/configuration_options_1.rst", "w") as file:
        file.write(rst_content)


    rst_content = process_config_options(OPTIONS_MODEL, "dropdowns")
    # Writing to a file
    with open("source/configuration_options_2.rst", "w") as file:
        file.write(rst_content)

    print("RST file generated successfully.")


    # config_options = convert_types_to_strings(ml.CONFIGURATION_OPTIONS)
    # with open("source/configuration_options.yaml", "w") as file:
    #     yaml.dump(config_options, file, default_flow_style=False, sort_keys=False)
