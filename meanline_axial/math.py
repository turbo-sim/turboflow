import numpy as np


def smooth_max(x, method="boltzmann", alpha=10, axis=None, keepdims=False):
    r"""
    Compute a smooth approximation to the maximum of an array using the specified method.

    The p-norm approximation to the maximum is given by :cite:p:`weisstein_vector_2023`:

    .. math::

       f_{\alpha}(x) = \left( \sum_i x_i^\alpha \right)^{\frac{1}{\alpha}}

    The Boltzmann approximation is given by :cite:p:`blanchard_accurately_2021`:

    .. math::

       f_{\alpha}(x) = \frac{1}{\alpha} \log \left( \sum_i e^{\alpha \,(x_i - \hat{x})} \right) + \hat{x}


    This LogSumExp approximation is given by :cite:p:`blanchard_accurately_2021`:

    .. math::

       f_{\alpha}(x) = \frac{1}{\alpha} \log \left( \sum_i e^{\alpha \,(x_i - \hat{x})} \right) + \hat{x}

    where the shift :math:`\hat{x}` is defined as:

    .. math::

       \hat{x} = \text{sign}(\alpha) \cdot \max(\text{sign}(\alpha) \cdot x)

    Shifting `x` prevents numerical overflow due to the finite precision of floating-point operations

    Parameters
    ----------
    x : array_like
        Input data.
    method : str, optional
        Method to be used for the approximation. Supported methods are:

        - ``boltzmann`` (default)
        - ``logsumexp``
        - ``p-norm``

    alpha : float, optional
        Sharpness parameter. Large values produce a tight approximation to the
        maximum function
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is
        used. If this is a tuple of ints, the maximum is selected over
        multiple axes, instead of a single axis or all the axes as before.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    Returns
    -------
    max_approx : ndarray or scalar
        Approximate maximum of `x` using the specified method.

    Raises
    ------
    ValueError
        If an unsupported method is specified.
    """

    if method == "logsumexp":
        return _smooth_max_logsumexp(x, np.abs(alpha), axis, keepdims)
    elif method == "boltzmann":
        return _smooth_max_boltzmann(x, np.abs(alpha), axis, keepdims)
    elif method == "p-norm":
        return _smooth_max_pnorm(x, np.abs(alpha), axis, keepdims)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'logsumexp'\n"
            "- 'boltzmann'\n"
            "- 'p-norm'"
        )


def smooth_min(x, method="boltzmann", alpha=10, axis=None, keepdims=False):
    r"""
    Compute a smooth approximation to the minimum of an array using the specified method.

    The smooth minimum approximation is equivalent to the smooth maximum with a
    negative value of the sharpness parameter :math:`\alpha`. See documentation
    of the ```smooth_max()`` function for more information about the approximation
    methods available.


    Parameters
    ----------
    x : array_like
        Input data.
    method : str, optional
        Method to be used for the approximation. Supported methods are:

        - ``boltzmann`` (default)
        - ``logsumexp``
        - ``p-norm``

        Default is 'boltzmann'.
    alpha : float, optional
        Sharpness parameter.Large values produce a tight approximation to the
        minimum function
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is
        used. If this is a tuple of ints, the minimum is selected over
        multiple axes, instead of a single axis or all the axes as before.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    Returns
    -------
    min_approx : ndarray or scalar
        Approximate minimum of `x` using the specified method.

    Raises
    ------
    ValueError
        If an unsupported method is specified.
    """

    if method == "logsumexp":
        return _smooth_max_logsumexp(x, -np.abs(alpha), axis, keepdims)
    elif method == "boltzmann":
        return _smooth_max_boltzmann(x, -np.abs(alpha), axis, keepdims)
    elif method == "p-norm":
        return _smooth_max_pnorm(x, -np.abs(alpha), axis, keepdims)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'logsumexp'\n"
            "- 'boltzmann'\n"
            "- 'p-norm'"
        )


def _smooth_max_logsumexp(x, alpha, axis=None, keepdims=False):
    """Smooth approximation to the maximum of an array using the log-sum-exp method"""

    # Determine the shift for numerical stability
    shift_value = np.sign(alpha) * np.max(np.sign(alpha) * x, axis=axis, keepdims=True)

    # Compute log-sum-exp with the shift and scale by alpha
    log_sum = np.log(
        np.sum(np.exp(alpha * (x - shift_value)), axis=axis, keepdims=True)
    )

    # Normalize the result by alpha and correct for the shift
    approx_max = (log_sum + alpha * shift_value) / alpha

    # Handle keepdims
    if not keepdims:
        if isinstance(axis, tuple):
            for ax in sorted(axis, reverse=True):
                approx_max = np.squeeze(approx_max, axis=ax)
        elif isinstance(axis, int):
            approx_max = np.squeeze(approx_max, axis=axis)

    return approx_max


def _smooth_max_boltzmann(x, alpha, axis=None, keepdims=False):
    """Smooth approximation to the maximum of an array using the Boltzmann weighted average"""

    # Compute the shift for numerical stability
    shift = np.sign(alpha) * np.max(np.sign(alpha) * x, axis=axis, keepdims=True)

    # Compute the weighted sum (numerator) of the elements of x
    weighted_sum = np.sum(x * np.exp(alpha * (x - shift)), axis=axis, keepdims=True)

    # Compute the sum of the smooth_max weights (denominator)
    weight_sum = np.sum(np.exp(alpha * (x - shift)), axis=axis, keepdims=True)

    # Compute the Boltzmann-weighted average avoiding division by zero
    max_approx = weighted_sum / (weight_sum + np.finfo(float).eps)

    if not keepdims:
        # Remove the dimensions of size one if keepdims is False
        max_approx = np.squeeze(max_approx, axis=axis)

    return max_approx


def _smooth_max_pnorm(x, alpha, axis=None, keepdims=False):
    """Smooth approximation to the maximum of an array using the p-norm method"""

    # Compute the p-norm approximation
    pnorm_approx = np.sum(np.power(x, alpha), axis=axis, keepdims=True) ** (1 / alpha)

    if not keepdims:
        # Remove the dimensions of size one if keepdims is False
        pnorm_approx = np.squeeze(pnorm_approx, axis=axis)

    return pnorm_approx


def smooth_abs(x, method="quadratic", epsilon=1e-5):
    r"""
    Compute a smooth approximation of the absolute value function according to the specified method.
 
    1. The quadratic approximation is given by :cite:p:`ramirez_x_2013`:

    .. math::
       f_{\epsilon}(x) = \sqrt{x^2 + \epsilon}

    2. The hyperbolic tangent approximation is given by :cite:p:`bagul_smooth_2017`:

    .. math::
       f_{\epsilon}(x) = \epsilon \tanh{x / \epsilon}

    3. The log-cosh approximation is given by :cite:p:`saleh_statistical_2022`:

    .. math::
       f_{\epsilon}(x) = \epsilon \log\left(\cosh\left(\frac{x}{\epsilon}\right)\right)

    The quadratic method is the most computationally efficient, but also requires a smaller value of :math:`\epsilon` to yield a good approximation.
    The transcendental methods give a better approximation to the absolute value function, at the expense of a higher computational time.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s) for which the approximation is computed.
    method : str, optional
        The method of approximation:

         - ``quadratic``
         - ``hyperbolic``
         - ``logcosh``

    epsilon : float, optional
        A small positive constant affecting the approximation. Default is 1e-5.

    Returns
    -------
    float or np.ndarray
        Smooth approximation value(s) of the absolute function.

    Raises
    ------
    ValueError
        If an unsupported approximation method is provided.
    """

    if method == "quadratic":
        return _smooth_abs_quadratic(x, epsilon)
    elif method == "hyperbolic":
        return _smooth_abs_tanh(x, epsilon)
    elif method == "logarithmic":
        return _smooth_abs_logcosh(x, epsilon)

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'quadratic'\n"
            "- 'hyperbolic'\n"
            "- 'logcosh'"
        )


def _smooth_abs_quadratic(x, epsilon):
    """Quadratic approximation of the absolute value function."""
    return np.sqrt(x**2 + epsilon)

def _smooth_abs_tanh(x, epsilon):
    """Hyperbolic tangent approximation of the absolute value function."""
    return x * np.tanh(x / epsilon)

def _smooth_abs_logcosh(x, epsilon):
    """Log-cosh approximation of the absolute value function."""
    return epsilon * np.log(np.cosh(x / epsilon))


def sind(x):
    """
    Compute the sine of an angle given in degrees.

    Parameters:
    x (float or array-like): Angle in degrees.

    Returns:
    float or ndarray: Sine of the input angle.
    """
    return np.sin(x * np.pi / 180)

def cosd(x):
    """
    Compute the cosine of an angle given in degrees.

    Parameters:
    x (float or array-like): Angle in degrees.

    Returns:
    float or ndarray: Cosine of the input angle.
    """
    return np.cos(x * np.pi / 180)

def tand(x):
    """
    Compute the tangent of an angle given in degrees.

    Parameters:
    x (float or array-like): Angle in degrees.

    Returns:
    float or ndarray: Tangent of the input angle.
    """
    return np.tan(x * np.pi / 180)

def arcsind(x):
    """
    Compute the arcsine of a value and return the result in degrees.

    Parameters:
    x (float or array-like): Value in the range [-1, 1].

    Returns:
    float or ndarray: Arcsine of the input value in degrees.
    """
    return np.arcsin(x) * 180 / np.pi

def arccosd(x):
    """
    Compute the arccosine of a value and return the result in degrees.

    Parameters:
    x (float or array-like): Value in the range [-1, 1].

    Returns:
    float or ndarray: Arccosine of the input value in degrees.
    """
    return np.arccos(x) * 180 / np.pi

def arctand(x):
    """
    Compute the arctangent of a value and return the result in degrees.

    Parameters:
    x (float or array-like): Value.

    Returns:
    float or ndarray: Arctangent of the input value in degrees.
    """
    return np.arctan(x) * 180 / np.pi

def is_odd(number):
    return number % 2 != 0

def is_even(number):
    return number % 2 == 0