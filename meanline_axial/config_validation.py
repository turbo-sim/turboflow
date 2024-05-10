import numpy as np
from . import axial_turbine
from . import utilities


SETTINGS = {
    "description": "Defines general settings controlling the behavior of the program.",
    "is_mandatory": False,  # The code always sets a default value of {} if not provided
    "expected_type": dict,
    "valid_options": None,
    "_nested": {
        "skip_validation": {
            "description": "Whether to skip the configuration validation or not.",
            "is_mandatory": False,
            "default_value": False,
            "expected_type": bool,
            "valid_options": None,
        },
        # "logger": {
        #     "description": "To be completed",
        #     "is_mandatory": False,
        #     "default_value": False,
        #     "expected_type": bool,
        #     "valid_options": None,
        # }
    },
}

OPTIONS_GEOMETRY = {
    "description": "Defines the turbine's geometric parameters.",
    "is_mandatory": True,
    "expected_type": dict,
    "valid_options": None,
    "_nested": {
        "cascade_type": {
            "description": "Specifies the types of cascade of each blade row.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": ["stator", "rotor"],
        },
        "radius_hub": {
            "description": "Hub radius at the inlet and outlet of each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "radius_tip": {
            "description": "Tip radius at the inlet and outlet of each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "pitch": {
            "description": "Blade pitch (aslo known as spacing) for each cascade",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "chord": {
            "description": "Blade chord for each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "stagger_angle": {
            "description": "Blade stagger angle for each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "opening": {
            "description": "Blade opening for each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "diameter_le": {
            "description": "Leading-edge diameter for each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "wedge_angle_le": {
            "description": "Wedge angle at the leading edge of each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "metal_angle_le": {
            "description": "Metal angle at the leading edge of each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "metal_angle_te": {
            "description": "Metal angle at the trailing edge of each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "thickness_te": {
            "description": "Trailing edge thickness of the blades for each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "thickness_max": {
            "description": "Maximum thicknesses of the blades for each cascade.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "tip_clearance": {
            "description": "Tip clearance of the blades for each cascade (usually zero for stator blades).",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
        "throat_location_fraction": {
            "description": "Defines the position of the throat in the blade passages as a fraction of the cascade's axial length. This parameter is relevant when the annulus shape varies from the inlet to the outlet of the cascade, due to factors like flaring or non-constant radius. A value of 1 indicates that the throat is located exactly at the exit plane, aligning the throat's area and radius with the exit plane's dimensions. Adjusting this fraction allows for precise modeling of the throat location relative to the exit.",
            "is_mandatory": True,
            "expected_type": (list, np.ndarray),
            "valid_options": None,
        },
    },
}


OPTIONS_OPERATION_POINTS = {
    "description": "Defines operating conditions for turbine performance analysis. This can be provided in two formats. The first format is as a list of dictionaries, where each dictionary defines a single operation point. The second format is as a single dictionary where each key has a single value or an array of values. In this case, the function internally generates all possible combinations of operation points, similar to creating a performance map, by taking the Cartesian product of these ranges.",
    "is_mandatory": True,
    "expected_type": (dict, list, np.ndarray),
    "valid_options": None,
    "_nested": {
        "fluid_name": {
            "description": "Name of the working fluid.",
            "is_mandatory": True,
            "expected_type": str,
            "valid_options": None,
        },
        "T0_in": {
            "description": "Stagnation temperature at the inlet. Unit [K].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
        "p0_in": {
            "description": "Stagnation pressure at the inlet. Unit [Pa].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
        "p_out": {
            "description": "Static pressure at the exit. Unit [Pa].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
        "omega": {
            "description": "Angular speed. Unit [rad/s].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
        "alpha_in": {
            "description": "Flow angle at the inlet. Unit [deg].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
    },
}


OPTIONS_PERFORMANCE_MAP = {
    "description": "Specifies a range of operating conditions for creating the turbine's performance map. This option is expected to be a dictionary where each key corresponds to a parameter (like inlet pressure, angular speed, etc.) and its value is a scalar or an array of possible values for that parameter. The code generates the complete set of operation points internally by calculating all possible combinations of operating conditions (i.e., taking the cartesian product of the ranges).",
    "is_mandatory": False,
    "expected_type": dict,
    "valid_options": None,
    "_nested": {
        "fluid_name": {
            "description": "Name of the working fluid.",
            "is_mandatory": True,
            "expected_type": str,
            "valid_options": None,
        },
        "T0_in": {
            "description": "Stagnation temperature at the inlet. Unit [K].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
        "p0_in": {
            "description": "Stagnation pressure at the inlet. Unit [Pa].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
        "p_out": {
            "description": "Static pressure at the exit. Unit [Pa].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
        "omega": {
            "description": "Angular speed. Unit [rad/s].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
        "alpha_in": {
            "description": "Flow angle at the inlet. Unit [deg].",
            "is_mandatory": True,
            "expected_type": (np.number, np.ndarray, list),
            "valid_options": None,
        },
    },
}


OPTIONS_MODEL = {
    "description": "Specifies the options related to the physical modeling of the problem",
    "is_mandatory": True,
    "expected_type": dict,
    "valid_options": None,
    "_nested": {
        "deviation_model": {
            "description": "Deviation model used to predict the exit flow angle at subsonic conditions.",
            "is_mandatory": True,
            "expected_type": str,
            "valid_options": axial_turbine.DEVIATION_MODELS,
        },
        "blockage_model": {
            "description": "Model used to predict the blockage factor due to boundary layer displacement thickness.",
            "is_mandatory": True,
            "default_value": 0.00,
            "expected_type": [float, str],  # Allowing float and specific string values
            "valid_options": axial_turbine.BLOCKAGE_MODELS + [utilities.NUMERIC],
        },
        "rel_step_fd": {
            "description": "Relative step size of the finite differences used to approximate the critical condition Jacobian.",
            "is_mandatory": False,
            "default_value": 1e-3,
            "expected_type": float,
            "valid_options": None,
        },
        "loss_model": {
            "description": "Specifies the options of the methods to estimate losses.",
            "is_mandatory": True,
            "expected_type": dict,
            "valid_options": None,
            "_nested": {
                "model": {
                    "description": "Name of the model used to calculate the losses.",
                    "is_mandatory": True,
                    "expected_type": str,
                    "valid_options": axial_turbine.LOSS_MODELS,
                },
                "loss_coefficient": {
                    "description": "Definition of the loss coefficient used to characterize the losses.",
                    "is_mandatory": True,
                    "expected_type": str,
                    "valid_options": axial_turbine.LOSS_COEFFICIENTS,
                },
                "inlet_displacement_thickness_height_ratio": {
                    "description": "Ratio of the endwall boundary layer displacement thickness at the inlet of a cascade to the height of the blade. Used in the secondary loss calculations of the `benner` loss model.",
                    "is_mandatory": False,
                    "default_value": 0.011,
                    "expected_type": float,
                    "valid_options": None,
                },
                "tuning_factors": {
                    "description": "Specifies tuning factors to have control over the weight of the different loss components.",
                    "is_mandatory": False,
                    "expected_type": dict,
                    "valid_options": None,  # None for nested structures
                    "_nested": {
                        "profile": {
                            "description": "Multiplicative factor for the profile losses.",
                            "is_mandatory": False,
                            "default_value": 1.00,
                            "expected_type": float,
                            "valid_options": None,
                        },
                        "incidence": {
                            "description": "Multiplicative factor for the incidence losses.",
                            "is_mandatory": False,
                            "default_value": 1.00,
                            "expected_type": float,
                            "valid_options": None,
                        },
                        "secondary": {
                            "description": "Multiplicative factor for the secondary losses.",
                            "is_mandatory": False,
                            "default_value": 1.00,
                            "expected_type": float,
                            "valid_options": None,
                        },
                        "trailing": {
                            "description": "Multiplicative factor for the trailing edge losses.",
                            "is_mandatory": False,
                            "default_value": 1.00,
                            "expected_type": float,
                            "valid_options": None,
                        },
                        "clearance": {
                            "description": "Multiplicative factor for the tip clearance losses.",
                            "is_mandatory": False,
                            "default_value": 1.00,
                            "expected_type": float,
                            "valid_options": None,
                        },
                    },
                },
            },
        },
    },
}


OPTIONS_SOLVER = {
    "description": "Specifies options related to the numerical methods used to solve the problem",
    "is_mandatory": False,
    "default_value": {},
    "expected_type": dict,
    "valid_options": None,
    "_nested": {
        "method": {
            "description": "Name of the numerical method used to solve the problem. Different methods may offer various advantages in terms of accuracy, speed, or stability, depending on the problem being solved",
            "is_mandatory": False,
            "default_value": list(axial_turbine.SOLVER_MAP.keys())[0],
            "expected_type": str,
            "valid_options": list(axial_turbine.SOLVER_MAP.keys()),
        },
        "tolerance": {
            "description": "Termination tolerance for the solver. This value determines the precision of the solution. Lower tolerance values increase the precision but may require more computational time.",
            "is_mandatory": False,
            "default_value": 1e-8,
            "expected_type": (float, np.float64),
            "valid_options": None,
        },
        "max_iterations": {
            "description": "Maximum number of solver iterations. This sets an upper limit on the number of iterations to prevent endless computation in cases where convergence is slow or not achievable.",
            "is_mandatory": False,
            "default_value": 100,
            "expected_type": (int, np.int64),
            "valid_options": None,
        },
        "derivative_method": {
            "description": "Finite difference method used to calculate the problem Jacobian",
            "is_mandatory": False,
            "default_value": "2-point",
            "expected_type": str,
            "valid_options": ["2-point", "3-point"],
        },
        "derivative_rel_step": {
            "description": "Relative step size of the finite differences used to approximate the problem Jacobian. This step size is crucial in balancing the truncation error and round-off error. A larger step size may lead to higher truncation errors, whereas a very small step size can increase round-off errors due to the finite precision of floating point arithmetic. Choosing the appropriate step size is key to ensuring accuracy and stability in the derivative estimation process.",
            "is_mandatory": False,
            "default_value": 1e-4,
            "expected_type": float,
            "valid_options": None,
        },
        "display_progress": {
            "description": "Whether to print the convergence history to the console. Enabling this option helps in monitoring the solver's progress and diagnosing convergence issues during the solution process.",
            "is_mandatory": False,
            "default_value": True,
            "expected_type": bool,
            "valid_options": None,
        },
    },
}


CONFIGURATION_OPTIONS = {
    "geometry": OPTIONS_GEOMETRY,
    "operation_points": OPTIONS_OPERATION_POINTS,
    "performance_map": OPTIONS_PERFORMANCE_MAP,
    "model_options": OPTIONS_MODEL,
    "solver_options": OPTIONS_SOLVER,
    "general_settings": SETTINGS,
}


# Managing configuration validation for a more complex and diverse codebase, like one supporting various
# types of turbines and compressors, requires a balance between flexibility and maintainability.

# 1. Modular Schema Approach
# Divide your configuration schema into modular sections based on machine types.
# Each module (like axial_turbine, radial_turbine, centrifugal_compressor, etc.) would have its own schema.
# Keep a clear structure in your documentation by having separate sections or pages for each machine type.
# Easily extend or modify schemas for specific machine types without affecting others.
# Maintain a consistent naming convention across different types (e.g., geometry, loss_model) while allowing for type-specific validation.




def read_configuration_file(filename, validate=True):
    """
    Reads and validates the specified YAML configuration file.

    This function reads the specified YAML configuration file and performens 2 post-processing operations:

    1. Evaluates string expressions in the configuration file to actual numerical values and converts configuration options to Numpy types.
    2. Validates the configuration file against a predefined schema, checking that the configuration options and data types are compliant.

    Parameters
    ----------
    filename : str
        The file path of the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the validated configuration options.

    See Also
    --------
    :obj:`convert_configuration_options`
    :obj:`validate_configuration_options`
    """

    if validate:
        # Read and validate configuration file
        config, info, error = utilities.validate_configuration_file(filename, CONFIGURATION_OPTIONS)

        # Print info messages
        if info:
            print(info)

        # Raise errors from validation
        if error:
            raise error

    else:
        config = utilities.read_configuration_file(filename)

    return config

