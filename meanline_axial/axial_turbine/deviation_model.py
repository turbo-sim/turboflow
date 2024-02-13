import numpy as np
from .. import math


DEVIATION_MODELS = ["aungier", "ainley_mathieson", "zero_deviation"]

def get_subsonic_deviation(Ma_exit, Ma_crit_throat, geometry, model):
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
        DEVIATION_MODELS[0]: get_exit_flow_angle_aungier,
        DEVIATION_MODELS[1]: get_exit_flow_angle_ainley_mathieson,
        DEVIATION_MODELS[2]: get_exit_flow_angle_zero_deviation,
        # DEVIATION_MODELS[3]: get_exit_flow_angle_borg_agromayor
    }

    # Evaluate deviation model
    if model in deviation_model_functions:
        Ma_exit = np.float64(Ma_exit)
        return deviation_model_functions[model](Ma_exit, Ma_crit_throat, geometry)
    else:
        options = ", ".join(f"'{k}'" for k in deviation_model_functions)
        raise ValueError(
            f"Invalid deviation model: '{model}'. Available options: {options}"
        )
    
def get_exit_flow_angle_aungier(Ma_exit, Ma_crit, geometry):
    """
    Calculate deviation angle using the method proposed by :cite:`aungier_turbine_2006`.
    """
    # TODO add equations of Aungier model to docstring
    # gauging_angle = geometry["metal_angle_te"]
    gauging_angle = math.arccosd(geometry["A_throat"]/geometry["A_out"])

    # Compute deviation for  Ma<0.5 (low-speed)
    beta_g = 90-abs(gauging_angle)
    delta_0 = math.arcsind(math.cosd(gauging_angle) * (1 + (1 - math.cosd(gauging_angle)) * (beta_g / 90) ** 2)) - beta_g

    # Initialize an empty array to store the results
    delta = np.empty_like(Ma_exit, dtype=float)

    # Compute deviation for Ma_exit < 0.50 (low-speed)
    low_speed_mask = Ma_exit < 0.50
    delta[low_speed_mask] = delta_0

    # Compute deviation for 0.50 <= Ma_exit < 1.00
    medium_speed_mask = (0.50 <= Ma_exit)  & (Ma_exit < Ma_crit)
    X = (2 * Ma_exit[medium_speed_mask] - 1) / (2 * Ma_crit - 1)
    delta[medium_speed_mask] = delta_0 * (1 - 10 * X**3 + 15 * X**4 - 6 * X**5)

    # Extrapolate to zero deviation for supersonic flow
    supersonic_mask = Ma_exit >= Ma_crit
    delta[supersonic_mask] = 0.00

    # Compute flow angle from deviation
    beta = abs(gauging_angle) - delta

    return beta


def get_exit_flow_angle_ainley_mathieson(Ma_exit, Ma_crit, geometry):
    """
    Calculate deviation angle using the model proposed by :cite:`ainley_method_1951`.
    Equation digitized from Figure 5 of :cite:`ainley_method_1951`.
    """
    # TODO add equations of Ainley-Mathieson to docstring
    gauging_angle = math.arccosd(geometry["A_throat"]/geometry["A_out"])
        
    # Compute deviation for  Ma<0.5 (low-speed)
    delta_0 = abs(gauging_angle) - (35.0 + (80.0 - 35.0) / (79.0 - 40.0) * (abs(gauging_angle) - 40.0))

    # Initialize an empty array to store the results
    delta = np.empty_like(Ma_exit, dtype=float)

    # Compute deviation for Ma_exit < 0.50 (low-speed)
    low_speed_mask = Ma_exit < 0.50
    delta[low_speed_mask] = delta_0

    # Compute deviation for 0.50 <= Ma_exit < 1.00
    medium_speed_mask = (0.50 <= Ma_exit) & (Ma_exit < Ma_crit)
    # delta[medium_speed_mask] = -delta_0/(Ma_crit-0.5)*Ma_exit[medium_speed_mask] + delta_0 + delta_0/(Ma_crit-0.5)*0.5
    delta[medium_speed_mask] = delta_0*(1+(0.5-Ma_exit[medium_speed_mask])/(Ma_crit-0.5))
    # FIXME: no linear interpolation now?

    # Extrapolate to zero deviation for supersonic flow
    supersonic_mask = Ma_exit >= 1.00
    delta[supersonic_mask] = 0.00

    # Compute flow angle from deviation
    beta = abs(gauging_angle) - delta
    
    return beta

# def get_exit_flow_angle_borg_agromayor(Ma_exit, beta_crit, Ma_crit_exit, geometry):
    
#     # Load geometry
#     area_throat = geometry["A_throat"]
#     area_exit = geometry["A_out"]
#     gauging_angle = math.arccosd(area_throat/area_exit)
    
#     # Compute deviation for  Ma<0.5 (low-speed)
#     beta_inc = (35.0 + (80.0 - 35.0) / (79.0 - 40.0) * (abs(gauging_angle) - 40.0))
    
#     # Define limit for incompressible flow angle
#     Ma_inc = 0.5
    
#     # Compute deviation angle
#     x = (Ma_exit - Ma_inc) / (Ma_crit_exit - Ma_inc)
#     # x2 = (Ma_crit_throat-Ma_inc)/(Ma_crit_exit-Ma_inc)
#     # y1 = 1
#     # x1 = 1
#     beta_max = math.arccosd(area_throat/area_exit)
#     # y2 = (beta_max-beta_inc)/(abs(gauging_angle)-beta_inc) 

#     # if y2-y1 < 1e-5:
#     #     k = 0
#     # else:
#     #     k = 2*(y1-y2)/(x1**2-2*x1*x2+x2**2 + 1e-6)*(x1-x2) # TODO: restrict k
#     k = 1
#     y = (3-k)*x**2+(k-2)*x**3
#     beta = beta_inc + (abs(beta_crit) - beta_inc) * y
    
#     return beta


def get_exit_flow_angle_zero_deviation(Ma_exit, Ma_crit_throat, geometry):
    """
    The exit flow angle is given by the gauge angle at subsonic conditions
    """
    # TODO add equation of zero-deviation to docstring
    # return np.full_like(Ma_exit, abs(geometry["metal_angle_te"]))
    return math.arccosd(geometry["A_throat"]/geometry["A_out"])



