import yaml
import numpy as np
from .properties import FluidCoolProp_2Phase


def read_configuration_file(filename):
    with open(filename, 'r') as file:
        return postprocess_config(yaml.safe_load(file))


def postprocess_config(config):
    """
    Postprocesses the YAML configuration data by converting string values that represent
    numerical expressions to actual numerical values.

    This function helps ensure that configuration parameters, which may be specified as
    strings in the YAML file (e.g., "np.pi/180*45"), are correctly evaluated to numerical
    values (e.g., 1.0).

    Parameters:
        config (dict): The configuration data loaded from a YAML file.

    Returns:
        dict: The postprocessed configuration data with numerical values.

    Example:
        # Example YAML content:
        # fluid_name: R125
        # p0_in: 3650000
        # T0_in: 463.15
        # custom_value: "np.pi/180*45"
        yaml_content = '''
            fluid_name: R125
            p0_in: 3650000
            T0_in: 463.15
            custom_value: "np.pi/180*45"
        '''
        parsed_data = yaml.safe_load(yaml_content)
        postprocessed_data = postprocess_config(parsed_data)

    Note:
        The function uses `eval()` to evaluate string expressions. While this is safe for
        trusted input, exercise caution if the YAML file is sourced from untrusted or
        external input. Always validate and sanitize input if security is a concern.
    """
    def convert_to_numbers(data):
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

    return convert_to_numbers(config)


def print_dict(data, indent=0):
    for key, value in data.items():
        print('    ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print('')
            print_dict(value, indent+1)
        else:
            print(value)
            
def get_cascades_data(filename):
    cascades_data = read_configuration_file(filename)

    fluid_name = cascades_data["BC"]["fluid_name"]
    cascades_data["geometry"] = {key: np.asarray(value) for key, value in cascades_data["geometry"].items()}
    cascades_data["BC"]["fluid"] = FluidCoolProp_2Phase(fluid_name)
    cascades_data["fixed_params"] = {}
    cascades_data["overall"] = {}
    
    return cascades_data


