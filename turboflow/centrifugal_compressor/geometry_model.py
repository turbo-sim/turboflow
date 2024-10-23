
import numpy as np
from .. import math
from .. import utilities as utils

def calculate_full_geometry(geometry):

    r"""
    Computes the complete geometry of a centrifugal compressor based on input geometric parameters.

    Parameters
    ----------
    geometry : dict
        A dictionary containing the input geometry parameters

    Returns
    -------
    dict
        A dictionary with both the original and newly computed geometry parameters.
    """

    # Create a copy of the input dictionary to avoid mutating the original
    for key in geometry.keys():
        if key == "impeller":
            imp_geometry = calculate_impeller_geoemtry(geometry["impeller"])
            geometry["impeller"] = imp_geometry
        elif key == "vaneless_diffuser":
            vld_geometry = calculate_vaneless_diffuser_geometry(geometry["vaneless_diffuser"])
            geometry["vaneless_diffuser"] = vld_geometry
        elif key == "vaned_diffuser":
            vd_geometry = calculate_vaned_diffuser_geometry(geometry["vaned_diffuser"])
            geometry["vaned_diffuser"] = vd_geometry
        elif key == "volute":
            vol_geometry = calculate_volute_geometry(geometry["volute"])
            geometry["volute"] = vol_geometry

    return geometry

def calculate_impeller_geoemtry(imp_geometry):

    # Extract initial radii and compute throat values
    radius_hub_in = imp_geometry["radius_hub_in"]
    radius_tip_in = imp_geometry["radius_tip_in"]
    radius_out = imp_geometry["radius_out"]
    width_out = imp_geometry["width_out"]

    # Calculate geometry
    radius_mean_in = (radius_tip_in + radius_hub_in) / 2
    A_in = np.pi * (radius_tip_in**2 - radius_hub_in**2)
    A_out = 2*np.pi*radius_out*width_out

    new_parameters = {"radius_mean_in" : radius_mean_in,
                      "area_in" : A_in,
                      "area_out" : A_out}

    return {**imp_geometry, **new_parameters} 

def calculate_vaneless_diffuser_geometry(vld_geometry):

    # # Load geometry
    # r_out = vld_geometry["radius_out"]
    # b_out = vld_geometry["width_out"]

    # # Calculate geometry
    # A_out = 2*np.pi*r_out*b_out

    # new_parameters = {"area_out" : A_out}

    # return {**vld_geometry, **new_parameters} 
    return vld_geometry

def calculate_vaned_diffuser_geometry(vd_geometry):

    # Load geometry
    r_out = vd_geometry["radius_out"]
    b_out = vd_geometry["width_out"]
    theta_in = vd_geometry["leading_edge_angle"]
    theta_out = vd_geometry["trailing_edge_angle"]
    z = vd_geometry["number_of_vanes"]

    # Calculate geometry
    A_out = 2*np.pi*r_out*b_out
    theta_avg = (theta_in + theta_out)/2
    camber_angle = theta_out - theta_in
    solidity = z*(r_out -r_in)/(2*np.pi*r_in*np.sin(theta_avg))
    loc_camber_max = (2-(theta_avg-theta_in)/(theta_out-theta_in))/3

    new_parameters = {"area_out" : A_out,
                      "theta_mean" : theta_avg,
                      "camber_angle" : camber_angle,
                      "solidity" : solidity,
                      "loc_camber_max" : loc_camber_max,
                      }

    return {**vd_geometry, **new_parameters} 

def calculate_volute_geometry(vol_geometry):

    # Load geometry
    r_out = vol_geometry["radius_out"]

    # Calculate geometry
    A_out = np.pi*r_out**2

    new_parameters = {"area_out" : A_out}

    return {**vol_geometry, **new_parameters} 


def validate_compressor_geometry(geom, display=False):
    """
    Performs validation of a centrifugal compressor's geometry configuration.

    - Ensures all required geometry parameters are specified in 'geom'.
    - Verifies that no parameters beyond the required set are included.
    - Asserts that all geometry parameter values are numeric (integers or floats).
    - Ensures non-angle parameters contain only non-negative values.
    - Angle parameters can be negative.

    This function is intended as a preliminary check before in-depth geometry analysis.

    Parameters
    ----------
    geom : dict
        The geometry configuration parameters for the turbine as a dictionary.

    Returns
    -------
    bool
        True if all validations pass, indicating a correctly structured geometry configuration.
    """

    # Define the list of keys that must be defined in the configuration file
    required_keys = {
        "radius_hub_in",
        "radius_tip_out",
        "radius_out",
        "impeller_width_out",
    }

    # Angular variables need not be non-negative
    angle_keys = []

    # Check that there are no missing or extra variables
    utils.validate_keys(geom, required_keys, required_keys)

    # Check array size and values
    for key, value in geom.items():

        # Check that all variables are numpy arrays
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Parameter '{key}' must be a NumPy array.")

        # Check that values of parameters are numeric
        if not math.all_numeric(value):
            raise ValueError(
                f"Parameter '{key}' must be an array of numeric types. Current value: {value}"
            )

        # Check that non-angle variables are not negative
        if key not in angle_keys and not math.all_non_negative(value):
            raise ValueError(
                f"Parameter '{key}' must be an array of non-negative numbers. Current value: {value}"
            )

    msg = "The structure of the turbine geometry parameters is valid"
    if display:
        print(msg)

    return msg


