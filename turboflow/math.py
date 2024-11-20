import numpy as np
import jax.numpy as jnp

def smooth_maximum(x1, x2, method="boltzmann", alpha=10):
    r"""
    Element-wise smoooth maximum approximation of array elements. Smoothed version of numpy.maximum().

    The :math:`p`-norm approximation to the maximum is given by :cite:p:`weisstein_vector_2023`:

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
    # Ensure x1 and x2 have the same shape by broadcasting
    x1, x2 = jnp.broadcast_arrays(x1, x2)
    
    # Stack the input arrays along a new axis to treat them as a single array
    x = jnp.stack([x1, x2], axis=0)

    # Compute smooth maximum approximation according to specified method
    if method == "logsumexp":
        return _smooth_max_logsumexp(x, jnp.abs(alpha), axis=0)
    elif method == "boltzmann":
        return _smooth_max_boltzmann(x, jnp.abs(alpha), axis=0)
    elif method == "p-norm":
        return _smooth_max_pnorm(x, jnp.abs(alpha), axis=0)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'logsumexp'\n"
            "- 'boltzmann'\n"
            "- 'p-norm'"
        )


def smooth_minimum(x1, x2, method="boltzmann", alpha=10):
    r"""
    Element-wise smoooth minimum approximation of array elements. Smoothed version of numpy.minimum().

    The smooth minimum approximation is equivalent to the smooth maximum with a
    negative value of the sharpness parameter :math:`\alpha`. See documentation
    of the ```smooth_max()`` function for more information about the smoothing
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
    # Ensure x1 and x2 have the same shape by broadcasting
    x1, x2 = jnp.broadcast_arrays(x1, x2)
    
    # Stack the input arrays along a new axis to treat them as a single array
    x = jnp.stack([x1, x2], axis=0)

    # Compute smooth maximum approximation according to specified method
    if method == "logsumexp":
        return _smooth_max_logsumexp(x, -jnp.abs(alpha), axis=0)
    elif method == "boltzmann":
        return _smooth_max_boltzmann(x, -jnp.abs(alpha), axis=0)
    elif method == "p-norm":
        return _smooth_max_pnorm(x, -jnp.abs(alpha), axis=0)
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
    shift_value = jnp.sign(alpha) * jnp.max(jnp.sign(alpha) * x, axis=axis, keepdims=True)

    # Compute log-sum-exp with the shift and scale by alpha
    log_sum = jnp.log(jnp.sum(jnp.exp(alpha * (x - shift_value)), axis=axis, keepdims=True))

    # Normalize the result by alpha and correct for the shift
    smooth_max = (log_sum + alpha * shift_value) / alpha

    # Remove the dimensions of size one if keepdims is False
    if not keepdims:
        smooth_max = jnp.squeeze(smooth_max, axis=axis)

    return smooth_max


def _smooth_max_boltzmann(x, alpha, axis=None, keepdims=False):
    """Smooth approximation to the maximum of an array using the Boltzmann weighted average"""

    # Compute the shift for numerical stability
    shift = jnp.sign(alpha) * jnp.max(jnp.sign(alpha) * x, axis=axis, keepdims=True)

    # Compute the weighted sum (numerator) of the elements of x
    weighted_sum = jnp.sum(x * jnp.exp(alpha * (x - shift)), axis=axis, keepdims=True)

    # Compute the sum of the smooth_max weights (denominator)
    weight_sum = jnp.sum(jnp.exp(alpha * (x - shift)), axis=axis, keepdims=True)

    # Compute the Boltzmann-weighted average avoiding division by zero
    smooth_max = weighted_sum / (weight_sum + jnp.finfo(float).eps)

    # Remove the dimensions of size one if keepdims is False
    if not keepdims:
        smooth_max = jnp.squeeze(smooth_max, axis=axis)

    return smooth_max


def _smooth_max_pnorm(x, alpha, axis=None, keepdims=False):
    """Smooth approximation to the maximum of an array using the p-norm method"""

    # Compute the p-norm approximation
    smooth_max = jnp.sum(jnp.power(x, alpha), axis=axis, keepdims=True) ** (1 / alpha)

    # Remove the dimensions of size one if keepdims is False
    if not keepdims:
        smooth_max = jnp.squeeze(smooth_max, axis=axis)

    return smooth_max


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

        - ``quadratic`` (default)
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
    return jnp.sqrt(x**2 + epsilon)


def _smooth_abs_tanh(x, epsilon):
    """Hyperbolic tangent approximation of the absolute value function."""
    return x * jnp.tanh(x / epsilon)


def _smooth_abs_logcosh(x, epsilon):
    """Log-cosh approximation of the absolute value function."""
    return epsilon * jnp.log(jnp.cosh(x / epsilon))



def sind(x):
    """Compute the sine of an angle given in degrees."""
    return jnp.sin(x * np.pi / 180.)


def cosd(x):
    """Compute the cosine of an angle given in degrees."""

    return jnp.cos(x * np.pi / 180.)


def tand(x):
    """Compute the tangent of an angle given in degrees."""
    return jnp.tan(x * np.pi / 180.)


def arcsind(x):
    """Compute the arcsine of a value and return the result in degrees."""
    return jnp.arcsin(x) * 180. / np.pi


def arccosd(x):
    """Compute the arccosine of a value and return the result in degrees."""
    return jnp.arccos(x) * 180 / np.pi


def arctand(x):
    """Compute the arctangent of a value and return the result in degrees."""
    return jnp.arctan(x) * 180. / np.pi


def is_odd(number):
    """Check if a number is odd. Returns True if the provided number is odd, and False otherwise."""
    return number % 2 != 0


def is_even(number):
    """Check if a number is even. Returns True if the provided number is even, and False otherwise."""
    return number % 2 == 0


def all_numeric(array):
    """Check if all items in Numpy array are numeric (floats or ints)"""
    return np.issubdtype(array.dtype, np.number)


def all_non_negative(array):
    "Check if all items in Numpy array are non-negative"
    return np.all(array >= 0)


def sigmoid_hyperbolic(x, x0=0.5, alpha=1):
    r"""
    Compute the sigmoid hyperbolic function.

    This function calculates a sigmoid function based on the hyperbolic tangent.
    The formula is given by:

    .. math::
        \sigma(x) = \frac{1 + \tanh\left(\frac{x - x_0}{\alpha}\right)}{2}

    Parameters
    ----------
    x : array_like
        Input data.
    x0 : float, optional
        Center of the sigmoid function. Default is 0.5.
    alpha : float, optional
        Scale parameter. Default is 0.1.

    Returns
    -------
    array_like
        Output of the sigmoid hyperbolic function.

    """
    x = np.array(x)  # Ensure x is a NumPy array for vectorized operations
    sigma = (1 + np.tanh((x - x0) / alpha)) / 2
    return sigma


def sigmoid_rational(x, n, m):
    r"""
    Compute the sigmoid rational function.

    This function calculates a sigmoid function using an algebraic approach based on a rational function.
    The formula is given by:

    .. math::
        \sigma(x) = \frac{x^n}{x^n + (1-x)^m}

    Parameters
    ----------
    x : array_like
        Input data.
    n : int
        Power of the numerator.
    m : int
        Power of the denominator.

    Returns
    -------
    array_like
        Output of the sigmoid algebraic function.

    """
    x = np.array(x)  # Ensure x is a NumPy array for vectorized operations
    x = np.where(x < 0, 0, x)  # Set x to 0 where x < 0
    x = np.where(x > 1, 1, x)  # Set x to 1 where x > 1
    sigma = x**n / (x**n + (1 - x) ** m)
    return sigma


def sigmoid_smoothstep(x):
    """
    Compute the smooth step function.

    This function calculates a smooth step function with zero first-order
    derivatives at the endpoints. More information available at:
    https://resources.wolframcloud.com/FunctionRepository/resources/SmoothStep/

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    array_like
        Output of the sigmoid smoothstep function.

    """
    x = np.array(x)  # Ensure x is a NumPy array for vectorized operations
    x = np.where(x < 0, 0, x)  # Set x to 0 where x < 0
    x = np.where(x > 1, 1, x)  # Set x to 1 where x > 1
    return x**2 * (3 - 2 * x)


def sigmoid_smootherstep(x):
    """
    Compute the smoother step function.

    This function calculates a smoother step function with zero second-order
    derivatives at the endpoints. It is a modification of the smoothstep
    function to provide smoother transitions.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    array_like
        Output of the sigmoid smootherstep function.

    """
    x = np.array(x)  # Ensure x is a NumPy array for vectorized operations
    x = np.where(x < 0, 0, x)  # Set x to 0 where x < 0
    x = np.where(x > 1, 1, x)  # Set x to 1 where x > 1
    return 6 * x**5 - 15 * x**4 + 10 * x**3
