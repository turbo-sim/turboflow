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
    }

    # Evaluate deviation model
    if model in deviation_model_functions:
        Ma_exit = np.float64(Ma_exit)
        beta, delta = deviation_model_functions[model](Ma_exit, Ma_crit_throat, geometry)
        return beta
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

    The flow angle (:math:`\beta`) is then computed based on the deviation and the gauging angle:

    .. math::
        \beta = 90 - \beta_g - \delta

    Parameters
    ----------
    Ma_exit : float
        Exit Mach number.
    Ma_crit : float
        Critical Mach number.
    geometry : dict
        Dictionary containing geometric parameters. Must contain floats `A_throat` and `A_out`, representing the cascade throat and exit area respectively.          
         
    Returns
    -------
    float
        Flow angle in degrees.
    """

    def _get_aungier_interpolant(x):
        x = np.array(x)  # Ensure x is a NumPy array for vectorized operations
        p = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5  # Compute polynomial
        p = np.where(x < 0, 1, p)  # Trim to 1 where x < 0
        p = np.where(x > 1, 0, p)  # Trim to 0 where x > 1
        return p

    # TODO add equations of Aungier model to docstring
    gauging_angle = math.arccosd(geometry["A_throat"]/geometry["A_out"])

    # Compute deviation for Ma<0.5 (low-speed)
    Ma_0 = 0.5
    beta_g = 90-abs(gauging_angle)
    delta_0 = math.arcsind(math.cosd(gauging_angle) * (1 + (1 - math.cosd(gauging_angle)) * (beta_g / 90) ** 2)) - beta_g

    # Compute deviation
    X = (Ma_exit - Ma_0) / (Ma_crit - Ma_0)
    delta = delta_0 * _get_aungier_interpolant(X)

    # Compute flow angle from deviation
    beta = abs(gauging_angle) - delta

    return beta, delta


def get_exit_flow_angle_ainley_mathieson(Ma_exit, Ma_crit, geometry):
    r"""
    Calculate the flow angle using the deviation model proposed by :cite:`ainley_method_1951`.

    This model defines the gauing angle with respect to axial direction:

    .. math::
        \beta_g = \cos^{-1}(A_\mathrm{throat} / A_\mathrm{out})

    - For :math:`\mathrm{Ma_exit} < 0.50` (low-speed), the deviation is a function of the gauging angle:

    .. math::
        \delta_0 = \beta_g - (35.0 + \frac{80.0-35.0}{79.0-40.0}\cdot (\beta_g-40.0))

    - For :math:`0.50 \leq \mathrm{Ma_exit} < \mathrm{Ma_crit}` (medium-speed), the deviation is calculated by a linear
      interpolation between low and critical Mach numbers:

    .. math::
        \delta = \delta_0\cdot \left(1+\frac{0.5-\mathrm{Ma_exit}}{\mathrm{Ma_crit}-0.5}\right)

    - For :math:`\mathrm{Ma_exit} \geq \mathrm{Ma_crit}` (supersonic), zero deviation is assumed:

    .. math::
        \delta = 0.00

    The flow angle (:math:`\beta`) is then computed based on the deviation and the gauging angle:

    .. math::
        \beta = \beta_g - \delta

    Parameters
    ----------
    Ma_exit : float
        Exit Mach number.
    Ma_crit : float
        Critical Mach number.
    geometry : dict
        Dictionary containing geometric parameters. Must contain floats `A_throat` and `A_out`, representing the cascade throat and exit area respectively.

    Returns
    -------
    float
        Flow angle in degrees.

    """

    def _get_ainley_mathieson_interpolant(x):
        x = np.array(x)  # Ensure x is a NumPy array for vectorized operations
        p = 1 - x  # Compute polynomial
        p = np.where(x < 0, 1, p)  # Trim to 1 where x < 0
        p = np.where(x > 1, 0, p)  # Trim to 0 where x > 1
        return p

    # TODO add equations of Ainley-Mathieson to docstring
    # TODO Add warning that AM method is inaccurate if gauge_angle>70 and does not make sense if gauge_angle>72
    gauging_angle = math.arccosd(geometry["A_throat"]/geometry["A_out"])
        
    # Compute deviation for Ma < Ma_0 (low-speed)
    Ma_0 = 0.5
    delta_0 = abs(gauging_angle) - (35.0 + (80.0 - 35.0) / (79.0 - 40.0) * (abs(gauging_angle) - 40.0))

    # Compute deviation
    X = (Ma_exit - Ma_0) / (Ma_crit - Ma_0)
    delta = delta_0 * _get_ainley_mathieson_interpolant(X)

    # Compute flow angle from deviation
    beta = abs(gauging_angle) - delta
    
    return beta, delta


def get_exit_flow_angle_zero_deviation(Ma_exit, Ma_crit, geometry):
    r"""
    Calculates the flow angle assuming zero deviation.
    This involves calculating the gauging angle, which is the angle of zero deviation.

    The gauging angle is calculated as:

    .. math::
        \beta_g = \cos^{-1}(A_\mathrm{throat} / A_\mathrm{out})

    where :math:`A_\mathrm{throat}` is the cross-sectional area of the throat and
    :math:`A_\mathrm{out}` is the cross-sectional area of the exit.

    Parameters
    ----------
    Ma_exit : float
        Exit Mach number.
    Ma_crit : float
        Critical Mach number.
    geometry : dict
        Dictionary containing geometric parameters. Must contain floats `A_throat` and `A_out`, representing the cascade throat and exit area respectively.

    Returns
    -------
    float
        Flow angle in degrees.

    """
    return math.arccosd(geometry["A_throat"]/geometry["A_out"]), 0