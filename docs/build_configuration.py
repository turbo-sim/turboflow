import os
import sys
import yaml

package_path = os.path.abspath("..")

if package_path not in sys.path:
    sys.path.append(package_path)

import meanline_axial as ml


if __name__ == "__main__":
    config_options = ml._convert_types_to_strings(ml.CONFIGURATION_OPTIONS)
    with open("source/configuration_options.yaml", "w") as file:
        yaml.dump(config_options, file, default_flow_style=False, sort_keys=False)


import yaml

# def parse_yaml_to_rst(yaml_file):
#     with open(yaml_file, 'r') as file:
#         config = yaml.safe_load(file)

#     rst_content = ''
#     for key, value in config.items():
#         rst_content += convert_to_rst(key, value)

#     return rst_content

# def convert_to_rst(key, value, indent=0):
#     indent_space = '   ' * indent
#     rst = f"{indent_space}* **{key}**:\n"
    
#     for subkey, subvalue in value.items():
#         if isinstance(subvalue, dict):
#             # Recursive call for nested dictionaries
#             rst += convert_to_rst(subkey, subvalue, indent + 1)
#         else:
#             rst += f"{indent_space}   - *{subkey}*: {subvalue}\n"
    
#     return rst

# if __name__ == "__main__":


#     # Example usage
#     rst_output = parse_yaml_to_rst('source/configuration_options.yaml')
#     print(rst_output)

def dict_to_rst_tabs(d, level=0):
    """
    Recursively convert a nested dictionary to rst format string for Sphinx Tabs.
    """
    rst = ""
    indent = "   " * (level + 2)  # Adjust indentation level for content

    # Start tabs for each level, but with different indentation
    if level == 0:
        rst += ".. tabs::\n\n"
    elif level > 0:
        rst += "   " * level + ".. tabs::\n\n"

    for key, value in d.items():
        tab_title = key.replace('_', ' ').title()
        rst += f"{indent[:-3]}.. tab:: {tab_title}\n\n"

        if isinstance(value, dict):
            # Nested dictionary, recurse
            rst += dict_to_rst_tabs(value, level + 1)
        else:
            # Base case, simple key-value
            rst += f"{indent}   - {value}\n\n"

    return rst

# Example usage
# Replace this with your actual configuration dictionary
config_dict = {
    "geometry": {
        "description": "Defines the turbine's geometric parameters.",
        "is_mandatory": True,
        "expected_type": "dict",
        "valid_options": None,
        "nested": {
            "cascade_type": "Specifies the types of cascade of each blade row.",
            "radius_hub": "Hub radius at the inlet and outlet of each cascade."
            # Add more nested items here...
        }
    },
    "model_options": {
        "description": "Specifies the options related to the physical modeling.",
        # Add more items here...
    }
    # Add more top-level items here...
}

# Convert the dictionary to rst format
rst_output = dict_to_rst_tabs(config_dict)

# Add a header to the rst output
rst_header = ".. _configuration_options:\n\nConfiguration options\n======================\n\n"
complete_rst = rst_header + rst_output

# Write the output to a file
with open('source/configuration_options_2.rst', 'w') as file:
    file.write(complete_rst)

print("rst file generated successfully.")