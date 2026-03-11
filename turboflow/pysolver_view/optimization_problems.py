import numpy as np

from . import jax, jnp, JAX_AVAILABLE
from .optimization import OptimizationProblem, combine_objective_and_constraints
from .numerical_differentiation import approx_derivative, approx_jacobian_hessians


class RosenbrockProblem(OptimizationProblem):
    r"""
    Implementation of the Rosenbrock problem.

    The Rosenbrock problem, also known as Rosenbrock's valley or banana function,
    is defined as:

    .. math::
       
        \begin{align}
        \text{minimize} \quad  & f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right] \\
        \end{align}

    Methods
    -------
    evaluate_problem(x)
        Evaluates the Rosenbrock function and its constraints.
    get_bounds()
        Returns the bounds for the problem.
    get_n_eq()
        Returns the number of equality constraints.
    get_n_ineq()
        Returns the number of inequality constraints.
    """

    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        """Rosenbrock function value"""
        f = np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
        return combine_objective_and_constraints(f, None, None)

    def gradient(self, x):
        """Rosenbrock function gradient"""
        x = np.asarray(x)
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        grad = np.zeros_like(x)
        grad[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
        grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        grad[-1] = 200 * (x[-1] - x[-2] ** 2)
        return grad

    def hessians(self, x, lower_triangular=True):
        """Rosenbrock function gradient"""
        x = np.atleast_1d(x)
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        diagonal = np.zeros(len(x), dtype=x.dtype)
        diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
        H = H + np.diag(diagonal)
        if lower_triangular:
            H = H[np.tril_indices(len(x))]  # Lower triangular
            H = np.asarray([H])  # Correct Pygmo array shape
        # return H.squeeze()
        return H

    # def hessians(self, x, lower_triangular=True):
    #     return approx_jacobian_hessians(self.fitness, x, lower_triangular=lower_triangular)

    def get_bounds(self):
        return (-10 * np.ones(self.dim), 10 * np.ones(self.dim))

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0


class RosenbrockProblemConstrained(OptimizationProblem):
    r"""
    Implementation of the Chained Rosenbrock function with trigonometric-exponential constraints.

    This problem is also referred to as Example 5.1 in the report by Luksan and Vlcek. The optimization problem is described as:

    .. math::

        \begin{align}
        \text{minimize} \quad & \sum_{i=1}^{n-1}\left[100\left(x_i^2-x_{i+1}\right)^2 + \left(x_i-1\right)^2\right] \\
        \text{s.t.} \quad & 3x_{k+1}^3 + 2x_{k+2} - 5 + \sin(x_{k+1}-x_{k+2})\sin(x_{k+1}+x_{k+2}) + \\
                            & + 4x_{k+1} - x_k \exp(x_k-x_{k+1}) - 3 = 0, \; \forall k=1,...,n-2 \\
                            & -5 \le x_i \le 5, \forall i=1,...,n
        \end{align}

    References
    ----------
    - Luksan, L., and Jan Vlcek. “Sparse and partially separable test problems for unconstrained and equality constrained optimization.” (1999). `doi: link provided <http://hdl.handle.net/11104/0123965>`_.

    Methods
    -------
    evaluate_problem(x):
        Compute objective, equality, and inequality constraint.
    get_bounds():
        Return variable bounds.
    get_n_eq():
        Get number of equality constraints.
    get_n_ineq():  
        Get number of inequality constraints.
    """

    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        
        # Use jax.numpy if JAX is available, else use numpy
        xp = jnp if JAX_AVAILABLE else np
        # xp = np
        x = xp.asarray(x)

        # Objective function
        f = [xp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)]

        # Equality constraints
        c_eq = []
        for k in range(self.dim - 2):
            val = (
                3 * x[k + 1] ** 3
                + 2 * x[k + 2]
                - 5
                + xp.sin(x[k + 1] - x[k + 2]) * xp.sin(x[k + 1] + x[k + 2])
                + 4 * x[k + 1]
                - x[k] * xp.exp(x[k] - x[k + 1])
                - 3
            )
            c_eq.append(val)

        
        fitness = xp.concatenate((xp.asarray(f), xp.asarray(c_eq)))
        
        return fitness

        # c_ineq = None

        # return combine_objective_and_constraints(f, c_eq, c_ineq)


    def gradient(self, x, method="automatic_differentiation"):
        if method == "automatic_differentiation":
            if not JAX_AVAILABLE:
                print("JAX not available. Falling back to finite differences.")
                method = "finite_differences"
            else:
                try:
                    jac_fn = jax.jacobian(self.fitness)
                    return np.array(jac_fn(jnp.array(x)))
                except Exception as e:
                    print(f"Automatic differentiation failed ({e}). Falling back to finite differences.")
                    method = "finite_differences"

        if method == "finite_differences":
            return approx_derivative(
                self.fitness,
                x,
                method="2-point",
                abs_step=1e-8,
            )

        raise ValueError(f"Unsupported method '{method}'. Use 'finite_differences' or 'automatic_differentiation'.")


    # def gradient(self, x):
    #     gradient = approx_derivative(
    #         self.fitness,
    #         x,
    #         method="2-point",
    #         abs_step=1e-8,
    #     )
    #     return gradient

    # def hessians(self, x):
    #     H = approx_jacobian_hessians(
    #         self.fitness, x, abs_step=1e-4, lower_triangular=True
    #     )
    #     return H

    def get_bounds(self):
        return ([-10] * self.dim, [+10] * self.dim)

    def get_nec(self):
        return self.dim - 2

    def get_nic(self):
        return 0


class HS71Problem(OptimizationProblem):
    r"""
    Implementation of the Hock Schittkowski problem No.71.

    This class implements the following optimization problem:

    .. math::

        \begin{align}
        \text{minimize} \quad  & f(\mathbf{x}) = x_1x_4(x_1+x_2+x_3) + x_3 \\
        \text{s.t.} \quad      & x_1^2 + x_2^2 + x_3^2 + x_4^2 = 40 \\
                                & 25 - x_1 x_2 x_3 x_4 \le 0 \\
                                & 1 \le x_1, x_2, x_3, x_4 \le 5
        \end{align}

       
    References
    ----------
    - W. Hock and K. Schittkowski. Test examples for nonlinear programming codes. Lecture Notes in Economics and Mathematical Systems, 187, 1981. `doi: 10.1007/978-3-642-48320-2 <https://doi.org/10.1007/978-3-642-48320-2>`_.


    Methods
    -------
    evaluate_problem(x)`:
        Compute objective, equality, and inequality constraint.
    get_bounds()`:
        Return variable bounds.
    get_n_eq()`:
        Get number of equality constraints.
    get_n_ineq()`:  
        Get number of inequality constraints.

    """

    def fitness(self, x):
        # Objective function
        f = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

        # Equality constraints
        c_eq = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2) - 40.0

        # Inequality constraints
        c_ineq = 25 - x[0] * x[1] * x[2] * x[3]
        # c_ineq = None

        return combine_objective_and_constraints(f, c_eq, c_ineq)

    def get_bounds(self):
        return (4 * [1.0], 4 * [5])

    def get_nec(self):
        return 1

    def get_nic(self):
        return 1


class LorentzEquationsOpt(OptimizationProblem):
    r"""
    Implementation of the Lorentz System of Nonlinear Equations as an optimization problem

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

    def fitness(self, vars):
        x, y, z = vars
        eq1 = self.sigma * (y - x)
        eq2 = x * (self.rho - z) - y
        eq3 = x * y - self.beta * z
        return [0, eq1, eq2, eq3]

    def get_nec(self):
        return 3

    def get_nic(self):
        return 0

    def get_bounds(self):
        return (-10 * np.ones(3), 10 * np.ones(3))





# class SimoneProblem(OptimizationProblem):

#     def gradient(self, x):
#         return (4 * (x[0] - 2) * (x[1] - 4)**4, 8 * (x[0] - 2)**2 * (x[1] - 4)**3)
    
#     def fitness(self, x):
#         y=2*(x[0]-2)**2*(x[1]-4)**4
#         print('x = [%14f,%14f]'%(x[0],x[1]))
#         return np.atleast_1d(y)
    
#     def get_bounds(self):
#         return (2*[-1e6], 2*[1e6])

#     def get_nec(self):
#         return 0

#     def get_nic(self):
#         return 0