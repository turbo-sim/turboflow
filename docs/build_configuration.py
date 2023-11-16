import os
import sys
import yaml

package_path = os.path.abspath("..")

if package_path not in sys.path:
    sys.path.append(package_path)

import meanline_axial as ml




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




def dict_to_rst(d, indent=0, is_root=True, tab_key=None):
    rst_output = ""
    tab_indent = "   " * indent
    bullet_indent = "   " * (indent + +1)
    tabs_indent = "   " * (indent + 1)
    nested_indent = "   " * (indent + 2)

    # Handling the root element
    if is_root and tab_key:
        rst_output += f"{tab_indent}.. tab:: {tab_key}\n\n"

    # Process all keys
    for key, value in d.items():
        if key == "_nested":
            continue
        elif key == "description":
            rst_output += f"{bullet_indent}{value}\n\n"
        elif key == "valid_options":
            if value is not None:
                rst_output += f"{bullet_indent}* **{key}**: {value}\n"
        elif key == "expected_type":
            value = ml.ensure_iterable(value)
            type_strings = ", ".join(
                [v if isinstance(v, str) else v.__name__ for v in value]
            )
            rst_output += f"{bullet_indent}* **{key}**: {type_strings}\n"
        else:
            rst_output += f"{bullet_indent}* **{key}**: {value}\n"

    # Handle nested dictionaries
    nested_dict = d.get("_nested")
    if nested_dict:
        rst_output += f"\n{tabs_indent}.. tabs::\n\n"
        for nested_key, nested_value in nested_dict.items():
            rst_output += f"{nested_indent}.. tab:: {nested_key}\n\n"
            rst_output += dict_to_rst(nested_value, indent + 2, is_root=False)
            rst_output += "\n"

    return rst_output


def process_config_options(config_dict):
    # Root level tabs
    rst_content = ".. tabs::\n\n"
    for tab_key, tab_value in config_dict.items():
        rst_content += dict_to_rst(tab_value, indent=1, is_root=True, tab_key=tab_key)
        rst_content += "\n"

    return rst_content


OPTIONS_MODEL = ml.CONFIGURATION_OPTIONS
rst_content = process_config_options(OPTIONS_MODEL)

# Writing to a file
with open("source/configuration_options.rst", "w") as file:
    file.write(rst_content)

print("RST file generated successfully.")


if __name__ == "__main__":
    config_options = convert_types_to_strings(ml.CONFIGURATION_OPTIONS)
    with open("source/configuration_options.yaml", "w") as file:
        yaml.dump(config_options, file, default_flow_style=False, sort_keys=False)



