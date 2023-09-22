import numpy as np
from scipy.optimize._numdiff import approx_derivative

from .optimization import OptimizationProblem


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
    get_values(x)
        Evaluates the Rosenbrock function and its constraints.
    get_bounds()
        Returns the bounds for the problem.
    get_n_eq()
        Returns the number of equality constraints.
    get_n_ineq()
        Returns the number of inequality constraints.
    """

    
    def __init__(self):
        self.f = None
        self.c_eq = None
        self.c_ineq = None

    def get_values(self, x):

        # Objective function
        self.x = x
        self.f = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

        # Equality constraints
        self.c_eq = []
            
        # No inequality constraints given for this problem
        self.c_ineq = []

        # Combine objective function and constraints
        objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)

        return objective_and_constraints


    def get_bounds(self):
        return None

    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)

    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)


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
    get_values(x):
        Compute objective, equality, and inequality constraint.
    get_bounds():
        Return variable bounds.
    get_n_eq():
        Get number of equality constraints.
    get_n_ineq():  
        Get number of inequality constraints.
    """
    
    def __init__(self):
        self.f = None
        self.c_eq = None
        self.c_ineq = None

    def get_values(self, x):

        # Objective function
        self.x = x
        self.f = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

        # Equality constraints
        self.c_eq = []
        for k in range(len(x) - 2):
            val = (3 * x[k+1]**3 + 2 * x[k+2] - 5 +
                np.sin(x[k+1] - x[k+2]) * np.sin(x[k+1] + x[k+2]) +
                4 * x[k+1] - x[k] * np.exp(x[k] - x[k+1]) - 3)
            self.c_eq.append(val)

        # No inequality constraints given for this problem
        self.c_ineq = []

        # Combine objective function and constraints
        objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)

        return objective_and_constraints
    
    # def get_jacobian(self, x):
    #     jac = approx_derivative(self.get_values, x, method="2-point")
    #     return np.atleast_2d(jac)

    def get_bounds(self):
        bounds = [(-5, 5) for _ in range(len(self.x))]
        return bounds

    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)

    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)



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
    get_values(x)`:
        Compute objective, equality, and inequality constraint.
    get_bounds()`:
        Return variable bounds.
    get_n_eq()`:
        Get number of equality constraints.
    get_n_ineq()`:  
        Get number of inequality constraints.

    """

    def __init__(self):
        self.f = None
        self.c_eq = None
        self.c_ineq = None

    def get_values(self, x):

        # Objective function
        self.f = x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]
        
        # Equality constraints (as a list)
        self.c_eq = [(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) - 40.0]

        # Inequality constraints (as a list)
        self.c_ineq = x[0]*x[1]*x[2]*x[3] - 25.0

        # Combine objective function and constraints
        objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)

        return objective_and_constraints

    def get_bounds(self):
        return [(1, 5), (1, 5), (1, 5), (1, 5)]

    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)

    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)

