# import numpy as np
from typing import Dict, Literal

import jax
import jax.numpy as jnp
from jax import lax

from turboflow import math

DEVIATION_MODELS = ["aungier", "ainley_mathieson", "zero_deviation"]
_MODEL_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(DEVIATION_MODELS)}


@jax.jit
def _get_subsonic_deviation_jit(
    Ma_exit: jnp.ndarray,
    Ma_crit_throat: jnp.ndarray,
    A_throat: jnp.ndarray,
    A_out: jnp.ndarray,
    model_id: int,
):
    """JIT core: numeric-only inputs; model dispatch via lax.switch."""

    def aungier_fn(args):
        Ma_exit, Ma_crit, A_throat, A_out = args
        gauging_angle = math.arccosd(A_throat / A_out)
        Ma_0 = 0.5
        beta_g = 90 - jnp.abs(gauging_angle)
        delta_0 = (
            math.arcsind(
                math.cosd(gauging_angle)
                * (1 + (1 - math.cosd(gauging_angle)) * (beta_g / 90) ** 2)
            )
            - beta_g
        )
        X = (Ma_exit - Ma_0) / (Ma_crit - Ma_0)
        p = 1 - 10 * X**3 + 15 * X**4 - 6 * X**5
        p = jnp.where(X < 0, 1.0, p)
        p = jnp.where(X > 1, 0.0, p)
        delta = delta_0 * p
        beta = jnp.abs(gauging_angle) - delta
        return beta, delta

    def ainley_mathieson_fn(args):
        Ma_exit, Ma_crit, A_throat, A_out = args
        gauging_angle = math.arccosd(A_throat / A_out)
        Ma_0 = 0.5
        delta_0 = jnp.abs(gauging_angle) - (
            35.0
            + (80.0 - 35.0) / (79.0 - 40.0) * (jnp.abs(gauging_angle) - 40.0)
        )
        X = (Ma_exit - Ma_0) / (Ma_crit - Ma_0)
        p = 1 - X
        p = jnp.where(X < 0, 1.0, p)
        p = jnp.where(X > 1, 0.0, p)
        delta = delta_0 * p
        beta = jnp.abs(gauging_angle) - delta
        return beta, delta

    def zero_dev_fn(args):
        _, _, A_throat, A_out = args
        beta = math.arccosd(A_throat / A_out)
        return beta, jnp.array(0.0)

    return lax.switch(
        model_id,
        (aungier_fn, ainley_mathieson_fn, zero_dev_fn),
        (Ma_exit, Ma_crit_throat, A_throat, A_out),
    )


def get_subsonic_deviation(
    Ma_exit,
    Ma_crit_throat,
    geometry,
    model: Literal["aungier", "ainley_mathieson", "zero_deviation"],
):
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

    if model not in _MODEL_TO_ID:
        options = ", ".join(f"'{k}'" for k in _MODEL_TO_ID)
        raise ValueError(f"Invalid deviation model: '{model}'. Available options: {options}")

    if "A_throat" not in geometry or "A_out" not in geometry:
        raise KeyError("geometry must contain 'A_throat' and 'A_out'")

    model_id = _MODEL_TO_ID[model]
    Ma_exit = jnp.asarray(Ma_exit)
    Ma_crit_throat = jnp.asarray(Ma_crit_throat)
    A_throat = jnp.asarray(geometry["A_throat"])
    A_out = jnp.asarray(geometry["A_out"])

    beta, _ = _get_subsonic_deviation_jit(
        Ma_exit=Ma_exit,
        Ma_crit_throat=Ma_crit_throat,
        A_throat=A_throat,
        A_out=A_out,
        model_id=model_id,
    )
    return beta


def get_exit_flow_angle_aungier(Ma_exit, Ma_crit, geometry):
    r"""
    Calculate the flow angle using the deviation model proposed by :cite:`aungier_turbine_2006`.

    This model defines the gauging angle with respect to tangential axis:

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
        x = jnp.array(x)  # Ensure x is a NumPy array for vectorized operations
        p = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5  # Compute polynomial
        p = jnp.where(x < 0, 1, p)  # Trim to 1 where x < 0
        p = jnp.where(x > 1, 0, p)  # Trim to 0 where x > 1
        return p

    # TODO add equations of Aungier model to docstring
    gauging_angle = math.arccosd(geometry["A_throat"] / geometry["A_out"])

    # Compute deviation for Ma<0.5 (low-speed)
    Ma_0 = 0.5
    beta_g = 90 - abs(gauging_angle)
    delta_0 = (
        math.arcsind(
            math.cosd(gauging_angle)
            * (1 + (1 - math.cosd(gauging_angle)) * (beta_g / 90) ** 2)
        )
        - beta_g
    )

    # Compute deviation
    X = (Ma_exit - Ma_0) / (Ma_crit - Ma_0)
    delta = delta_0 * _get_aungier_interpolant(X)

    # Compute flow angle from deviation
    beta = abs(gauging_angle) - delta

    return beta, delta


def get_exit_flow_angle_ainley_mathieson(Ma_exit, Ma_crit, geometry):
    r"""
    Calculate the flow angle using the deviation model proposed by :cite:`ainley_method_1951`.

    This model defines the gauging angle with respect to axial direction:

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
        x = jnp.array(x)  # Ensure x is a NumPy array for vectorized operations
        p = 1 - x  # Compute polynomial
        p = jnp.where(x < 0, 1, p)  # Trim to 1 where x < 0
        p = jnp.where(x > 1, 0, p)  # Trim to 0 where x > 1
        return p

    # TODO add equations of Ainley-Mathieson to docstring
    # TODO Add warning that AM method is inaccurate if gauge_angle>70 and does not make sense if gauge_angle>72
    gauging_angle = math.arccosd(geometry["A_throat"] / geometry["A_out"])

    # Compute deviation for Ma < Ma_0 (low-speed)
    Ma_0 = 0.5
    delta_0 = abs(gauging_angle) - (
        35.0 + (80.0 - 35.0) / (79.0 - 40.0) * (abs(gauging_angle) - 40.0)
    )

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
    return math.arccosd(geometry["A_throat"] / geometry["A_out"]), 0
