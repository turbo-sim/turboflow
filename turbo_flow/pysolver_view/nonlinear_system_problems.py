import numpy as np
from scipy.optimize._numdiff import approx_derivative

from .nonlinear_system import NonlinearSystemProblem


class LorentzEquations(NonlinearSystemProblem):
    r"""
    Implementation of the Lorentz System of Nonlinear Equations.

    This class implements the following system of algebraic nonlinear equations:

    .. math::

        \begin{align}
        \dot{x} &= \sigma(y - x) = 0\\
        \dot{y} &= x(\rho - z) - y = 0\\
        \dot{z} &= xy - \beta z = 0
        \end{align}

    Where:

    - :math:`\sigma` is related to the Prandtl number
    - :math:`\rho` is related to the Rayleigh number
    - :math:`\beta` is a geometric factor

    References
    ----------
    - Edward N. Lorenz. "Deterministic Nonperiodic Flow". Journal of the Atmospheric Sciences, 20(2):130-141, 1963.
    
    Methods
    -------
    evaluate_problem(vars)`:
        Evaluate the Lorentz system at a given state.

    Attributes
    ----------
    sigma : float
        The Prandtl number.
    beta : float
        The geometric factor.
    rho : float
        The Rayleigh number.
    """

    def __init__(self, sigma=1.0, beta=2.0, rho=3.0):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho

    def residual(self, vars):
        x, y, z = vars
        eq1 = self.sigma * (y - x)
        eq2 = x * (self.rho - z) - y
        eq3 = x * y - self.beta * z
        return np.array([eq1, eq2, eq3])
    
    def gradient(self, x):
        return approx_derivative(self.residual, x, method="cs")
