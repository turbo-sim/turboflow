import numpy as np
from .. import math


def get_subsonic_deviation(Ma_exit, Ma_crit, opening_to_pitch, model="aungier"):
    """
    Calculate subsonic relative exit flow angle based on the selected deviation model.

    Available deviation models:

    - "aungier": Calculate deviation using the method proposed by :cite:`aungier_turbine_2006`.
    - "ainley_mathieson": Calculate deviation using the model proposed by :cite:`ainley_method_1951`.
    - "metal_angle": Assume the exit flow angle is given by the gauge angle (zero deviation).

    Parameters
    ----------
    deviation_model : str
        The deviation model to use (e.g., 'aungier', 'ainley_mathieson', 'zero_deviation').
    Ma_exit : float or numpy.array
        The exit Mach number.
    Ma_crit : float
        The critical Mach number (possibly lower than one).
    opening_to_pitch : float
        The ratio of cascade opening to pitch.

    Returns
    -------
    float
        The relative exit flow angle including deviation in degrees (subsonic flow only).

    Raises
    ------
    ValueError
        If an invalid deviation model is provided.
    """

    # Function mappings for each deviation model
    deviation_model_functions = {
        "aungier": get_exit_flow_angle_aungier,
        "ainley_mathieson": get_exit_flow_angle_ainley_mathieson,
        "zero_deviation": get_exit_flow_angle_zero_deviation,
    }

    # Evaluate deviation model
    if model in deviation_model_functions:
        return deviation_model_functions[model](Ma_exit, Ma_crit, opening_to_pitch)
    else:
        options = ", ".join(f"'{k}'" for k in deviation_model_functions)
        raise ValueError(
            f"Invalid deviation model: '{model}'. Available options: {options}"
        )
    
def get_exit_flow_angle_aungier(Ma_exit, Ma_crit, opening_to_pitch):
    """
    Calculate deviation angle using the method proposed by :cite:`aungier_turbine_2006`.
    """
    # TODO add equations of Aungier model to docstring

    # Compute deviation for  Ma<0.5 (low-speed)
    o_to_s = opening_to_pitch
    beta_g = math.arcsind(o_to_s)
    delta_0 = math.arcsind(o_to_s * (1 + (1 - o_to_s) * (beta_g / 90) ** 2)) - beta_g

    # Initialize an empty array to store the results
    delta = np.empty_like(Ma_exit, dtype=float)

    # Compute deviation for Ma_exit < 0.50 (low-speed)
    low_speed_mask = Ma_exit < 0.50
    delta[low_speed_mask] = delta_0

    # Compute deviation for 0.50 <= Ma_exit < 1.00
    medium_speed_mask = (0.50 <= Ma_exit) & (Ma_exit < 1.00)
    X = (2 * Ma_exit[medium_speed_mask] - 1) / (2 * Ma_crit - 1)
    delta[medium_speed_mask] = delta_0 * (1 - 10 * X**3 + 15 * X**4 - 6 * X**5)

    # Extrapolate to zero deviation for supersonic flow
    supersonic_mask = Ma_exit >= 1.00
    delta[supersonic_mask] = 0.00

    # Compute flow angle from deviation
    beta = math.arccosd(o_to_s) - delta

    return beta


def get_exit_flow_angle_ainley_mathieson(Ma_exit, Ma_crit, opening_to_pitch):
    """
    Calculate deviation angle using the model proposed by :cite:`ainley_method_1951`.
    Equation digitized from Figure 5 of :cite:`ainley_method_1951`.
    """
    # TODO add equations of Ainley-Mathieson to docstring

    # Compute deviation for  Ma<0.5 (low-speed)
    beta_g = math.arccosd(opening_to_pitch)
    delta_0 = beta_g - (35.0 + (80.0 - 35.0) / (79.0 - 40.0) * (beta_g - 40.0))

    # Initialize an empty array to store the results
    delta = np.empty_like(Ma_exit, dtype=float)

    # Compute deviation for Ma_exit < 0.50 (low-speed)
    low_speed_mask = Ma_exit < 0.50
    delta[low_speed_mask] = delta_0

    # Compute deviation for 0.50 <= Ma_exit < 1.00
    medium_speed_mask = (0.50 <= Ma_exit) & (Ma_exit < 1.00)
    X = (2 * Ma_exit[medium_speed_mask] - 1) / (2 * Ma_crit - 1)
    delta[medium_speed_mask] = delta_0 * (1 - 10 * X**3 + 15 * X**4 - 6 * X**5)

    # Extrapolate to zero deviation for supersonic flow
    supersonic_mask = Ma_exit >= 1.00
    delta[supersonic_mask] = 0.00

    # Compute flow angle from deviation
    beta = beta_g - delta

    return beta


def get_exit_flow_angle_zero_deviation(Ma_exit, Ma_crit, opening_to_pitch):
    """
    The exit flow angle is given by the gauge angle at subsonic conditions
    """
    # TODO add equation of zero-deviation to docstring
    return np.full_like(Ma_exit, math.arccosd(opening_to_pitch))



