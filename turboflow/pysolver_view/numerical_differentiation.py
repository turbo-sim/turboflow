import numpy as np
from scipy.optimize._numdiff import approx_derivative

from . import jax, jnp, JAX_AVAILABLE

DERIVATIVE_METHODS = [
    "2-point",
    "3-point",
    "Complex step method",
    "forward_finite_differences",
    "central_finite_differences",
    "complex_step",
]


def approx_gradient(
    function_handle, x, f0=None, abs_step=None, method="central_finite_differences"
):
    """
    Approximate the derivatives of a given function at a point using various differentiation methods.

    Parameters
    ----------
    function_handle : callable
        The function for which derivatives are to be approximated.
    x : array_like
        The point at which derivatives are to be approximated.
    abs_step : float or np.ndarray with the same size as x, optional
        Step size for finite difference methods. By default uses a suitable value for each method.
    method : str, optional
        The method to use for differentiation. Available options are:
        - 'forward_finite_differences' or '2-point': Forward finite differences.
        - 'central_finite_differences' or '3-point': Central finite differences.
        - 'complex_step' or 'cs': Complex step method.

        Defaults to 'central_finite_differences'.

    Returns
    -------
    numpy.ndarray
        The approximated derivatives of the function at the given point `x`.

    Raises
    ------
    ValueError
        If an invalid method is provided.
    """
    if method == "forward_finite_differences" or method == "2-point":
        return forward_finite_differences(function_handle, x, abs_step, f0)
    elif method == "central_finite_differences" or method == "3-point":
        return central_finite_differences(function_handle, x, abs_step)
    elif method == "complex_step" or method == "cs":
        return complex_step_derivative(function_handle, x, abs_step)
    else:
        raise ValueError(
            f"Invalid method '{method}' provided. Available methods: 'forward_finite_differences', "
            "'central_finite_differences', 'complex_step'."
        )


def forward_finite_differences(function_handle, x, abs_step=None, f0=None):
    """
    Gradient approximation by forward finite differences.

    Parameters
    ----------
    function_handle : callable
        The function for which the derivative is calculated.
    x : np.ndarray
        The point at which the derivative is calculated.
    abs_step : float or np.ndarray with the same size as x, optional
        The step size for finite differences. Default is square root of machine epsilon.

    Returns
    -------
    np.ndarray
        The gradient of the function at point x.
    """

    # Default step size
    if abs_step is None:
        # abs_step = np.finfo(float).eps ** (1 / 2)
        abs_step = np.finfo(float).eps ** (1 / 2)

    # Ensure abs_step is an array of the same shape as x
    x = np.atleast_1d(x)
    abs_step = np.broadcast_to(abs_step, x.shape)

    # Avoid one function evaluation if f0 is provided
    if f0 is None:
        f0 = np.atleast_1d(function_handle(x))
    else:
        f0 = np.atleast_1d(f0)

    # Compute gradient
    m = np.size(f0)
    n = np.size(x)
    df = np.zeros((m, n))
    for i in range(n):
        x_step = np.copy(x)
        x_step[i] += abs_step[i]
        df[:, i] = (np.atleast_1d(function_handle(x_step)) - f0) / abs_step[i]

    return df.squeeze()


def central_finite_differences(function_handle, x, abs_step=None):
    """
    Gradient approximation by central finite differences.

    Parameters
    ----------
    function_handle : callable
        The function for which the derivative is calculated.
    x : np.ndarray
        The point at which the derivative is calculated.
    abs_step : float or np.ndarray with the same size as x, optional
        The step size for finite differences. Default is cobic root of machine epsilon.

    Returns
    -------
    np.ndarray
        The gradient of the function at point x.
    """

    # Default step size
    if abs_step is None:
        abs_step = np.finfo(float).eps ** (1 / 3)

    # Ensure abs_step is an array of the same shape as x
    x = np.atleast_1d(x)
    abs_step = np.broadcast_to(abs_step, x.shape)

    # Compute gradient
    f0 = np.atleast_1d(function_handle(x))
    m = np.size(f0)
    n = np.size(x)
    df = np.zeros((m, n))
    for i in range(n):
        x_step_forward = np.copy(x)
        x_step_backward = np.copy(x)
        x_step_forward[i] += abs_step[i]
        x_step_backward[i] -= abs_step[i]
        df[:, i] = (
            np.atleast_1d(function_handle(x_step_forward))
            - np.atleast_1d(function_handle(x_step_backward))
        ) / (2 * abs_step[i])

    return df.squeeze()


def complex_step_derivative(function_handle, x, abs_step=None):
    """
    Gradient approximation using the complex step method.

    Parameters
    ----------
    function_handle : callable
        The function for which the derivative is calculated.
    x : np.ndarray
        The point at which the derivative is calculated.
    abs_step : float or np.ndarray with the same size as x, optional
        The step size for the complex step. Default is machine epsilon.

    Returns
    -------
    np.ndarray
        The gradient of the function at point x.
    """

    # Default step size
    if abs_step is None:
        abs_step = np.finfo(float).eps

    # Ensure abs_step is an array of the same shape as x
    abs_step = np.broadcast_to(abs_step, x.shape)

    # Compute gradient
    f0 = np.atleast_1d(function_handle(x))
    m = np.size(f0)
    n = np.size(x)
    df = np.zeros((m, n), dtype=np.complex128)
    for i in range(n):
        x_step = x.astype(np.complex128)
        x_step[i] += abs_step[i] * 1j
        df[:, i] = np.imag(np.atleast_1d(function_handle(x_step))) / abs_step[i]

    return df.squeeze().real


def approx_jacobian_hessians(f, x, abs_step=1e-5, lower_triangular=True):
    """
    Calculate the Hessian matrices for each component of a vector-valued function using finite differences.

    Parameters
    ----------
    f : callable
        The function for which to find the Hessian matrices. It must take a
        single argument which is a numpy array and can return a scalar or a numpy array.
    x : numpy array
        The point at which the Hessian matrices are calculated.
    abs_step : float, optional
        The step size for the finite differences, default is 1e-5.
    lower_triangular : bool, optional
        If True, the Hessians are returned in a lower triangular form suitable for Pygmo, default is True.

    Returns
    -------
    Hessians : numpy array
        A tensor where each slice along the first dimension corresponds to the Hessian matrix
        of each component of the function f at x. If f returns a scalar, the first dimension size is 1.
        If lower_triangular is True, the Hessians are returned in a lower triangular form.
    """
    x = np.asarray(x)
    f0 = np.atleast_1d(f(x))
    m = len(f0)
    n = len(x)
    Hessians = np.zeros((m, n, n))

    hh = np.eye(n) * abs_step

    for i in range(n):
        for j in range(i, n):
            f_ij = np.atleast_1d(f(x + hh[i] + hh[j]))
            f_i = np.atleast_1d(f(x + hh[i]))
            f_j = np.atleast_1d(f(x + hh[j]))

            Hessians[:, i, j] = (f_ij - f_i - f_j + f0) / abs_step**2
            if i != j:  # Symmetry
                Hessians[:, j, i] = Hessians[:, i, j]

    # Reformat to lower-triangular form suitable for Pygmo
    if lower_triangular:
        k = (n * (n + 1)) // 2
        Hessians_L = np.zeros((m, k))
        for i in range(m):
            Hessians_L[i] = Hessians[i][np.tril_indices(n)]
        Hessians = Hessians_L

    return Hessians


def compute_hessians_jax(fitness_function, x, lower_triangular=True):
    """
    Compute the Hessians of a vector-valued fitness function.

    Parameters
    ----------
    fitness_function : callable
        The vector-valued function for which to compute the Hessians.
    x : array-like
        The point at which to compute the Hessians.
    lower_triangular : bool, optional
        If True, return the Hessians in lower triangular format. Default is True.

    Returns
    -------
    Hessians : ndarray
        A tensor where each slice along the first dimension corresponds to the Hessian
        of a component of the fitness function. If lower_triangular is True, the Hessians
        are returned in a flattened lower triangular format.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not installed. Install with `pip install pysolver_view[jax]` to use this feature.")

    # Compute full Hessians with JAX
    H_full = jax.jacfwd(jax.jacrev(fitness_function))(x)

    # Convert to NumPy for post-processing
    H_full = np.array(H_full)

    if lower_triangular:
        # Extract the lower triangular part
        n = H_full.shape[1]
        k = (n * (n + 1)) // 2  # Number of elements in the lower triangular part
        H_lower_triangular = np.zeros((H_full.shape[0], k))

        # Loop over all components of the fitness
        for i in range(H_full.shape[0]):
            H_lower_triangular[i] = H_full[i][np.tril_indices(n)]

        return H_lower_triangular

    return H_full