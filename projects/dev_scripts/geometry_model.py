import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml


# Function to check if all items in the array are numeric (floats or ints)
def all_numeric(arr):
    return np.issubdtype(arr.dtype, np.number)


# Function to check if all items in the array are non-negative
def all_non_negative(arr):
    return np.all(arr >= 0)


def validate_axial_turbine_geometry(geom):
    """
    Performs validation of an axial turbine's geometry configuration.

    - Ensures all required geometry parameters are specified in 'geom'.
    - Verifies that no parameters beyond the required set are included.
    - Checks 'number_of_cascades' is an integer and correctly dictates sizes of other parameters.
    - Confirms 'radius_hub' and 'radius_tip' arrays have sizes twice that of 'number_of_cascades'.
    - Validates that all other parameter arrays match the size of 'number_of_cascades'.
    - Asserts that all geometry parameter values are numeric (integers or floats).
    - Ensures non-angle parameters contain only non-negative values.
    - Angle parameters ('metal_angle_leading', 'metal_angle_trailing') can be negative.

    This function is intended as a preliminary check before in-depth geometry analysis.

    Parameters:
    -----------
    geom : dict
        The geometry configuration parameters for the turbine as a dictionary.

    Returns:
    --------
    bool
        True if all validations pass, indicating a correctly structured geometry configuration.
    """

    required_keys = {
        "number_of_cascades",
        "radius_hub",
        "radius_tip",
        "pitch",
        "chord",
        "stagger_angle",
        "opening",
        "radius_leading",
        "radius_trailing",
        "metal_angle_leading",
        "metal_angle_trailing",
        "wedge_angle_leading",
        "tip_clearance",
        "thickness_max",
    }

    # Check for required fields
    missing_keys = required_keys - geom.keys()
    if missing_keys:
        raise KeyError(f"Missing geometry parameters: {missing_keys}")

    # Check for extra fields
    extra_keys = geom.keys() - required_keys
    if extra_keys:
        raise KeyError(f"Unexpected geometry parameters: {extra_keys}")

    # Check the number of cascades
    number_of_cascades = geom["number_of_cascades"]
    if not np.issubdtype(number_of_cascades.dtype, np.integer):
        raise ValueError("'number_of_cascades' must be an integer")

    # Check the size of radius_hub and radius_tip are twice the number of cascades
    for key in ["radius_hub", "radius_tip"]:
        if len(geom[key]) != 2 * number_of_cascades:
            raise ValueError(
                f"The size of '{key}' must be twice the number of cascades."
            )

    # Check that the size of all other arrays is equal to the number of cascades
    for key in required_keys - {"number_of_cascades", "radius_hub", "radius_tip"}:
        if len(geom[key]) != number_of_cascades:
            raise ValueError(
                f"The size of '{key}' must be equal to the number of cascades."
            )

    for key, value in geom.items():
        if key == "number_of_cascades":
            continue  # Skip the number_of_cascades in the loop

        if not isinstance(value, np.ndarray):
            raise TypeError(f"Parameter '{key}' must be a NumPy array.")

        if not all_numeric(value):
            raise ValueError(f"Parameter '{key}' must be an array of numeric types.")

        angles = ["metal_angle_leading", "metal_angle_trailing"]
        if key not in angles and not all_non_negative(value):
            raise ValueError(
                f"Parameter '{key}' must be an array of non-negative numbers."
            )

    return True


def calculate_throat_radius(radius_in, radius_out):
    """
    Calculate the throat radius using the given rule by Ainley and Mathieson,
    which is (1/6)*radius_in + (5/6)*radius_out.

    Parameters:
    -----------
    radius_in: np.array
        The radius values at the inlet sections.
    radius_out: np.array
        The radius values at the outlet sections.

    Returns:
    --------
    np.array
        The calculated throat radius values.
    """
    return (1 / 6) * radius_in + (5 / 6) * radius_out


def calculate_full_geometry(geometry):
    """
    Computes the complete geometry of an axial turbine based on input geometric parameters.

    Parameters:
    -----------
    geometry (dict): A dictionary containing the input geometry parameters

    Returns:
    --------
    dict: A new dictionary with both the original and newly computed geometry parameters.
    """

    # Create a copy of the input dictionary to avoid mutating the original
    geom = geometry.copy()

    # Compute axial chord
    axial_chord = geom["chord"]*np.cos(geom["stagger_angle"])

    # Extract initial radii and compute mean radius
    radius_hub = geom["radius_hub"]
    radius_tip = geom["radius_tip"]
    radius_mean = (radius_hub + radius_tip) / 2  # Average of hub and tip radius

    # Calculate shroud radius by adding clearance to tip radius
    radius_shroud = geom["radius_tip"] + np.repeat(geom["tip_clearance"], 2)

    # Calculate the inner and outer radii for the mean section
    radius_mean_in = radius_mean[::2]  # Even indices: in
    radius_mean_out = radius_mean[1::2]  # Odd indices: out
    radius_mean_throat = calculate_throat_radius(
        radius_mean_in, radius_mean_out
    )  # Calculate throat radii for mean section

    # Repeat calculations for hub
    radius_hub_in = radius_hub[::2]
    radius_hub_out = radius_hub[1::2]
    radius_hub_throat = calculate_throat_radius(radius_hub_in, radius_hub_out)

    # Repeat calculations for tip
    radius_tip_in = radius_tip[::2]
    radius_tip_out = radius_tip[1::2]
    radius_tip_throat = calculate_throat_radius(radius_tip_in, radius_tip_out)

    # Repeat calculations for shroud
    radius_shroud_in = radius_shroud[::2]
    radius_shroud_out = radius_shroud[1::2]
    radius_shroud_throat = calculate_throat_radius(radius_shroud_in, radius_shroud_out)

    # Compute hub to tip radius ratio
    hub_tip_ratio = radius_hub / radius_tip
    hub_tip_ratio_in = radius_hub_in / radius_tip_out
    hub_tip_ratio_out = radius_hub_out / radius_tip_out
    hub_tip_ratio_throat = radius_hub_throat / radius_tip_throat

    # Compute areas for the full, in, out, and throat sections
    A = np.pi * (radius_tip**2 - radius_hub**2)
    A_in = np.pi * (radius_tip_in**2 - radius_hub_in**2)
    A_out = np.pi * (radius_tip_out**2 - radius_hub_out**2)
    A_throat = np.pi * (radius_tip_throat**2 - radius_hub_throat**2)

    # Compute blade height at different sections
    height = geom["radius_tip"] - geom["radius_hub"]
    height_in = radius_tip_in - radius_hub_in
    height_out = radius_tip_out - radius_hub_out
    height_mean = (height_in + height_out) / 2

    # Compute geometric ratios
    aspect_ratio = height_mean / axial_chord
    pitch_to_chord_ratio = geom["pitch"] / geom["chord"]
    thickness_max_to_chord_ratio = geom["thickness_max"] / geom["chord"]
    trailing_edge_to_opening_ratio = geom["radius_trailing"] / geom["opening"]
    # throat_to_exit_opening_ratio = 

    # Create a dictionary with the newly computed parameters
    new_parameters = {
        "axial_chord": axial_chord,
        "radius_mean": radius_mean,
        "radius_shroud": radius_shroud,
        "radius_mean_in": radius_mean_in,
        "radius_mean_out": radius_mean_out,
        "radius_mean_throat": radius_mean_throat,
        "radius_hub_in": radius_hub_in,
        "radius_hub_out": radius_hub_out,
        "radius_hub_throat": radius_hub_throat,
        "radius_tip_in": radius_tip_in,
        "radius_tip_out": radius_tip_out,
        "radius_tip_throat": radius_tip_throat,
        "radius_shroud_in": radius_shroud_in,
        "radius_shroud_out": radius_shroud_out,
        "radius_shroud_throat": radius_shroud_throat,
        "hub_tip_ratio": hub_tip_ratio,
        "hub_tip_ratio_in": hub_tip_ratio_in,
        "hub_tip_ratio_out": hub_tip_ratio_out,
        "hub_tip_ratio_throat": hub_tip_ratio_throat,
        "A": A,
        "A_in": A_in,
        "A_out": A_out,
        "A_throat": A_throat,
        "height": height,
        "height_in": height_in,
        "height_out": height_out,
        "height_mean": height_mean,
        "aspect_ratio": aspect_ratio,
        "pitch_to_chord_ratio": pitch_to_chord_ratio,
        "thickness_max_to_chord_ratio": thickness_max_to_chord_ratio,
        "trailing_edge_to_opening_ratio": trailing_edge_to_opening_ratio,
    }

    return {**geometry, **new_parameters}




def check_axial_turbine_geometry(geometry):
    """
    Checks if the geometry parameters are within the predefined recommended ranges.

    Parameters:
    geometry (dict): A dictionary containing the computed geometry of the turbine.

    Returns:
    list: A list of messages with a summary of the checks.
    """

    # TODO: Check that this function behaves as intended

    msgs = []

    # Define good practice ranges for each parameter
    recommended_ranges = {
        "hub_tip_ratio": {"min": 0.3, "max": 1.0},
        "aspect_ratio": {"min": 0.5, "max": 4.0},
        "pitch_to_chord_ratio": {"min": 0.3, "max": 1.1},
        "metal_angle_leading": {"min": -45*np.pi/180, "max": +45*np.pi/180},
        "metal_angle_trailing": {"min": -80*np.pi/180, "max": +80*np.pi/180},
        "wedge_angle_leading": {"min": 10*np.pi/180, "max": 60*np.pi/180},
        "thickness_max_to_chord_ratio": {"min": 0.05, "max": 0.25},
        "trailing_edge_to_opening_ratio": {"min": 0.00, "max": 0.50},
        "thickness_max": {"min": 1e-3, "max": np.inf},
        "radius_trailing": {"min": 2e-4, "max": np.inf},
        "tip_clearance": {"min": 5e-4, "max": np.inf},
    }

    # Iterate over each good practice parameter and perform checks
    for parameter, limits in recommended_ranges.items():
        lb, ub = limits["min"], limits["max"]
        value_array = geometry[parameter]
        for index, value in enumerate(np.atleast_1d(value_array)):
            if not lb <= value <= ub:
                msgs.append(f"{parameter}[{index}]={value:0.4f} is out of recommended range ({lb}, {ub}).")

    if not msgs:
        msgs.append("All parameters are within recommended ranges")

    return msgs



# Load configuration file
CONFIG_FILE = "config_one_stage.yaml"
case_data = ml.read_configuration_file(CONFIG_FILE)


validate_axial_turbine_geometry(case_data["geometry_new"])
geom = calculate_full_geometry(case_data["geometry_new"])
geom_info = check_axial_turbine_geometry(geom)

for msg in geom_info:
    print(msg)


# TODO: improve logic for angle conventions of rotor/stator
# TODO: improve logic so tip clearance can be zero for stator