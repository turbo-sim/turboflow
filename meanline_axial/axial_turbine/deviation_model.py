import numpy as np
from .. import math


DEVIATION_MODELS = ["aungier", "ainley_mathieson", "zero_deviation"]

def get_subsonic_deviation(Ma_exit, Ma_crit, geometry, model):
    r"""
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
    }

    # Evaluate deviation model
    if model in deviation_model_functions:
        Ma_exit = np.float64(Ma_exit)
        return deviation_model_functions[model](Ma_exit, Ma_crit, geometry)
    else:
        options = ", ".join(f"'{k}'" for k in deviation_model_functions)
        raise ValueError(
            f"Invalid deviation model: '{model}'. Available options: {options}"
        )
    
def get_exit_flow_angle_aungier(Ma_exit, Ma_crit, geometry):
    r"""
    Calculate the flow angle using the deviation model proposed by :cite:`aungier_turbine_2006`.

    This model defines the gauing angle with respect to tangential axis:

    .. math::

        \beta_g = 90 - \cos^{-1}\left(\frac{A_\mathrm{throat}}{A_\mathrm{out}}\right)
    
    The model involves a piecewise calculation depending on the mach number range:

    - For :math:`Ma_\mathrm{exit} < 0.50`, the deviation is a function of the gauging angle:

    .. math::

        \delta_0 = \sin^{-1}\left(\frac{A_\mathrm{throat}}{A_\mathrm{out}} \left(1+\left(1-\frac{A_\mathrm{throat}}{A_\mathrm{out}}\right)\cdot\left(\frac{\beta_g}{90}\right)^2\right)\right)

    - For :math:`0.50 \leq Ma_\mathrm{exit} < Ma_\mathrm{crit}`, the deviation is calculated by a fifth order interpolation between low and critical Mach numbers:

    .. math::
        \begin{align*}
        X &= \frac{2\cdot Ma_\mathrm{exit}-1}{2\cdot Ma_\mathrm{crit}-1} \\
        \delta &= \delta_0 \cdot (1-10X^3+15X^4-6X^5)
        \end{align*}

    - For :math:`Ma_\mathrm{exit} \geq Ma_\mathrm{crit}`, zero deviation is assumed:

    .. math:: 
        \delta = 0.00

    The flow angle is then computed based on the deviation and the gauging angle:

    .. math::
        \beta = 90 - \beta_g - \delta

    Parameters
    ----------
    Ma_exit : float
        Exit Mach number.
    Ma_crit : float
        Critical Mach number.
    method : str, optional
        Type of solver. Should be one of

            * 'hybr'       
            * 'lm'               
         
    Returns
    -------
    `\beta` : float

        Flow angle in degrees.
    """
    # Calculate gauging angle wrt axial axis
    gauging_angle = math.arccosd(geometry["A_throat"]/geometry["A_out"])

    # Compute deviation for  Ma<0.5 (low-speed)
    beta_g = 90-abs(gauging_angle)
    delta_0 = math.arcsind(geometry["A_throat"]/geometry["A_out"] * (1 + (1 - geometry["A_throat"]/geometry["A_out"]) * (beta_g / 90) ** 2)) - beta_g

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
    beta = gauging_angle - delta

    return beta


def get_exit_flow_angle_ainley_mathieson(Ma_exit, Ma_crit, geometry):
    r"""
    Calculate the flow angle using the deviation model proposed by :cite:`ainley_method_1951`.

    It involves a piecewise calculation
    depending on the Mach number range:
    - For Ma_exit < 0.50 (low-speed), the deviation is a function of the gauging angle:
        :math::
            \beta_g = cos^{-1}(A_\mathrm{throat} / A_\mathrm{out})
    The fucntion is obtained by digitizing the Figure 5 in :cite:`ainley_method_1951`:
        :math::
            delt_0 = \beta_g - (35.0 + \frac{80.0-35.0}{79.0-40.0}\cdot (\beta_g-40.0))
    - For 0.50 <= Ma_exit < Ma_crit (medium-speed), the deviation is calculated by a linear 
      interpolation between low and critical Mach numbers:
        :math::
            delta = delta_0\cdot \left(1+\frac{0.5-\mathrm{Ma_exit}}{\mathr{Ma_crit}-0.5}\right)
    - For Ma_exit >= 1.00 (supersonic), zero deviation is assumed:
        :math:: 
            \delta = 0.00

    The flow angle (beta) is then computed based on the deviation and the gauging angle:
    :math::
        \beta = \beta_g - \delta

    Parameters
    ----------
    Ma_exit : float 
        Exit Mach number.
    Ma_crit : float
        Critical Mach number.
    geometry : dict
        Dictionary containing geometric parameters.
        Should contain keys:
        - 'A_throat': float
            Cross-sectional area of the throat.
        - 'A_out': float
            Cross-sectional area of the exit.

    Returns
    -------
    float
        Flow angle in degrees.

    """
    # Compute gauging angle
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

def get_exit_flow_angle_zero_deviation(Ma_exit, Ma_crit, geometry):
    r"""
    Calculates the flow angle assuming zero deviation.
    This involves calculating the gauging angle, which is the angle of zero deviation.

    The gauing angle is calculated as:
    .. math::
        \beta_g = cos^{-1}(A_\mathrm{throat} / A_\mathrm{out})

    where :math:`A_\mathrm{throat}` is the cross-sectional area of the throat and
    :math:`A_\mathrm{out}` is the cross-sectional area of the exit.

    Parameters
    ----------
    Ma_exit : float
        Exit Mach number.
    Ma_crit : float
        Critical Mach number.
    geometry : dict
        Dictionary containing geometric parameters.
        Should contain keys:
        - 'A_throat': float
            Cross-sectional area of the throat.
        - 'A_out': float
            Cross-sectional area of the exit.

    Returns
    -------
    float
        Flow angle in degrees.

    """
    return math.arccosd(geometry["A_throat"]/geometry["A_out"])



