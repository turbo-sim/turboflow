import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp
from scipy.optimize._numdiff import approx_derivative

import os
import sys

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml



numeric_option = '<numeric value>'


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration options."""

    def __init__(self, messages):
        self.messages = messages
        super().__init__(self._format_message())

    def _format_message(self):
        return "Configuration errors detected.\n" + ";\n".join(self.messages)


def validate_model_options(config, schema):
    """
    Validates the 'model_options' section of the configuration against the provided schema.
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
    - Nested fields: Recursively validates nested dictionaries within the configuration according 
      to the nested schema defined under the '_nested' key.

    The function raises a ConfigurationError if any discrepancies are found and also prints 
    messages for fields where default values are used.

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
    def validate_field(conf, schema, path, errors, messages):

        # Check for unexpected keys
        unexpected_keys = set(conf.keys()) - set(schema.keys())
        if unexpected_keys:
            errors.append(f"Unexpected keys in '{path}': {unexpected_keys}")

        # Loop through all the possible configuration options
        for key, specs in schema.items():

            # Define option path recursively
            current_path = f"{path}/{key}"

            # Ensure expected_type is a tuple (needed because some variables accept more than 1 type)
            expected_types = specs["expected_type"]
            if not isinstance(expected_types, tuple):
                expected_types = tuple(expected_types) if isinstance(expected_types, list) else (expected_types,)

            # Check if the key is present in the configuration
            if key in conf:
                
                # Validate the type
                if not isinstance(conf[key], expected_types):
                    type_expected = ', '.join([t.__name__ for t in expected_types])
                    type_actual = type(conf[key]).__name__
                    errors.append(f"Incorrect type for field '{current_path}': '{type_actual}'. Expected {type_expected}")

                # Validate the value if there are specific valid options defined
                elif specs.get("valid_options") is not None:
                    if conf[key] not in specs["valid_options"] and numeric_option in specs["valid_options"] and not isinstance(conf[key], (int, float)):
                        errors.append(f"Invalid value for field '{current_path}'. Valid options are: {specs['valid_options']}")

            # If the key is not present and is mandatory, add error message
            elif specs["is_mandatory"]:
                    errors.append(f"Missing required field: '{current_path}'")

            # If the key is not present and is non-mandatory, use the default value if provided
            elif not specs["is_mandatory"] and "default_value" in specs:
                conf[key] = specs["default_value"]
                messages.append(f"Field '{current_path}' not specified; using default value: {specs['default_value']}")

            else:
                raise Exception("Something went wrong")

            # Recursively validate nested fields
            if "_nested" in specs and key in conf and isinstance(conf[key], dict):
                validate_field(conf[key], specs["_nested"], current_path, errors, messages)

        return conf
    

    errors = []
    messages = []   
    conf = validate_field(config, schema, "model_options", errors, messages)

    if messages:
        messages.insert(0, "Some parameters were not defined. Using default options:")
        messages = "\n   * ".join(messages)
        print(messages)
        print("")
    
    if errors:
        raise ConfigurationError(errors)

    return conf





model_options_schema = {
    "choking_condition": {
        "is_mandatory": True,
        "expected_type": str,
        "valid_options": ["deviation", "mach_critical", "mach_unity"],
    },
    "deviation_model": {
        "is_mandatory": True,
        "expected_type": str,
        "valid_options": ["aungier", "ainley_mathieson", "zero_deviation"],
    },
    "throat_blockage": {
        "is_mandatory": False,
        "default_value": 0.00,
        "expected_type": [float, str],  # Allowing float and specific string values
        "valid_options": [
            "flat_plate_turbulent", '<numeric value>'
        ],
    },
    "rel_step_fd": {
        "is_mandatory": False,
        "default_value": 1e-3,
        "expected_type": float,
        "valid_options": None,
    },
    "loss_model": {
        "is_mandatory": True,
        "expected_type": dict,
        "valid_options": None,
        "_nested": {
            "model": {
                "is_mandatory": True,
                "expected_type": str,
                "valid_options": [
                    "benner",
                    "moustapha",
                    "kacker_okapuu",
                    "benner_moustapha",
                    "isentropic",
                ],
            },
            "loss_coefficient": {
                "is_mandatory": True,
                "expected_type": str,
                "valid_options": ["stagnation_pressure"],  # Add valid options
            },
            "inlet_displacement_thickness_height_ratio": {
                "is_mandatory": False,
                "default_value": 0.011,
                "expected_type": float,
                "valid_options": None,
            },
            "tuning_factors": {
                "is_mandatory": False,
                "expected_type": dict,
                "valid_options": None,  # None since it's a nested structure
                "_nested": {
                    "profile": {
                        "is_mandatory": False,
                        "default_value": 1.00,
                        "expected_type": float,
                        "valid_options": None,
                    },
                    "incidence": {
                        "is_mandatory": False,
                        "default_value": 1.00,
                        "expected_type": float,
                        "valid_options": None,
                    },
                    "secondary": {
                        "is_mandatory": False,
                        "default_value": 1.00,
                        "expected_type": float,
                        "valid_options": None,
                    },
                    "trailing": {
                        "is_mandatory": False,
                        "default_value": 1.00,
                        "expected_type": float,
                        "valid_options": None,
                    },
                    "clearance": {
                        "is_mandatory": False,
                        "default_value": 1.00,
                        "expected_type": float,
                        "valid_options": None,
                    },
                },
            },
        },
    },
}


operation_points_schema = {
    "fluid_name": {"is_mandatory": True, "expected_type": str, "valid_options": None},
    "T0_in": {"is_mandatory": True, "expected_type": float, "valid_options": None},
    "p0_in": {"is_mandatory": True, "expected_type": float, "valid_options": None},
    "p_out": {"is_mandatory": True, "expected_type": list, "valid_options": None},
    "omega": {"is_mandatory": True, "expected_type": int, "valid_options": None},
    "alpha_in": {"is_mandatory": True, "expected_type": int, "valid_options": None},
}


performance_map_schema = {
    "fluid_name": {"is_mandatory": True, "expected_type": str, "valid_options": None},
    "T0_in": {"is_mandatory": True, "expected_type": float, "valid_options": None},
    "p0_in": {"is_mandatory": True, "expected_type": float, "valid_options": None},
    "p_out": {
        "is_mandatory": True,
        "expected_type": str,
        "valid_options": None,
    },
    "omega": {"is_mandatory": True, "expected_type": list, "valid_options": None},
    "alpha_in": {"is_mandatory": True, "expected_type": int, "valid_options": None},
}

solver_options_schema = {
    "method": {
        "is_mandatory": True,
        "expected_type": str,
        "valid_options": ["hybr", "other_methods"],
    },
    "tolerance": {"is_mandatory": True, "expected_type": float, "valid_options": None},
    "max_iterations": {
        "is_mandatory": True,
        "expected_type": int,
        "valid_options": None,
    },
    "derivative_method": {
        "is_mandatory": True,
        "expected_type": str,
        "valid_options": ["2-point", "3-point"],
    },
    "derivative_rel_step": {
        "is_mandatory": True,
        "expected_type": float,
        "valid_options": None,
    },
    "display_progress": {
        "is_mandatory": True,
        "expected_type": bool,
        "valid_options": None,
    },
}


# Load configuration file
CONFIG_FILE = "config_one_stage.yaml"
case_data = ml.read_configuration_file(CONFIG_FILE)


# ml.print_dict(case_data)

# ml.print_dict(model_options_schema)


ml.print_dict(case_data["model_options"])
config = validate_model_options(config=case_data["model_options"], schema=model_options_schema)


ml.print_dict(config)
