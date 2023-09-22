import os
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from abc import ABC, abstractmethod
from datetime import datetime

class OptimizationSolver:
    r"""

    Solver class for general nonlinear programming problems.

    The solver is designed to handle constrained optimization problems of the form:

    Minimize:

    .. math::
        f(\mathbf{x}) \; \mathrm{with} \; \mathbf{x} \in \mathbb{R}^n

    Subject to:

    .. math::
        c_{\mathrm{eq}}(\mathbf{x}) = 0
    .. math::
        c_{\mathrm{in}}(\mathbf{x}) \geq 0
    .. math::
        \mathbf{x}_l \leq \mathbf{x} \leq \mathbf{x}_u

    where:

    - :math:`\mathbf{x}` is the vector of decision variables (i.e., degree of freedom).
    - :math:`f(\mathbf{x})` is the objective function to be minimized. Maximization problems can be casted into minimization problems by changing the sign of the objective function.
    - :math:`c_{\mathrm{eq}}(\mathbf{x})` are the equality constraints of the problem.
    - :math:`c_{\mathrm{in}}(\mathbf{x})` are the inequality constraints of the problem. Constraints of type :math:`c_{\mathrm{in}}(\mathbf{x}) \leq 0` can be casted into :math:`c_{\mathrm{in}}(\mathbf{x}) \geq 0` type by changing the sign of the constraint functions.
    - :math:`\mathbf{x}_l` and :math:`\mathbf{x}_u` are the lower and upper bounds on the decision variables.

    The class interfaces with the `minimize` method from `scipy.optimize` to solve the problem and provides a structured framework for initialization, solution monitoring, and post-processing.

    This class employs a caching mechanism to avoid redundant evaluations. For a given set of independent variables, x, the optimizer requires the objective function, equality constraints, and inequality constraints to be provided separately. When working with complex models, these values are typically calculated all at once. If x hasn't changed from a previous evaluation, the caching system ensures that previously computed values are used, preventing unnecessary recalculations.

    Parameters
    ----------
    problem : OptimizationProblem
        An instance of the optimization problem to be solved.
    x0 : array_like
        Initial guess for the independent variables.
    display : bool, optional
        If True, displays the convergence progress. Defaults to True.
    plot : bool, optional
        If True, plots the convergence progress. Defaults to False.
    logger : logging.Logger, optional
        Logger object to which logging messages will be directed. Defaults to None.

    Methods
    -------
    solve(x0, method="slsqp", tol=1e-9, options=None):
        Solve the optimization problem using scipy's minimize.
    get_values(x):
        Evaluates the optimization problem values at a given point x.
    get_jacobian(x):
        Evaluates the Jacobians of the optimization problem at a given point x.
    print_convergence_history():
        Print the final result and convergence history of the optimization problem.
    plot_convergence_history():
        Plot the convergence history of the optimization problem.
    """

    def __init__(self, problem, x0, display=True, plot=False, logger=None):
        # Initialize class variables
        self.problem = problem
        self.display = display
        self.plot = plot
        self.logger = logger

        # Check for logger validity
        if self.logger is not None:
            if not isinstance(self.logger, logging.Logger):
                raise ValueError(
                    "The provided logger is not a valid logging.Logger instance."
                )

        # Initialize problem
        self.x0 = x0
        eval = self.problem.get_values(self.x0)
        if np.any([item is None for item in eval]):
            raise ValueError(
                "Problem evaluation contains None values. There may be an issue with the problem object provided."
            )

        # Number of optimization constraints
        self.N_eq = self.problem.get_n_eq()
        self.N_ineq = self.problem.get_n_ineq()

        # Initialize variables for convergence report
        self.last_x = None
        self.grad_count = 0
        self.func_count = 0 
        self.func_count_tot = 0
        self.solution = None
        self.solution_report = []
        self.convergence_history = {
            "grad_count": [],
            "func_count": [],
            "func_count_total": [],
            "objective_value": [],
            "constraint_violation": [],
            "norm_step": [],
        }

        # Initialize convergence plot
        # (convergence_history must be initialized before)
        if self.plot:
            self._plot_callback(initialize=True)

        # Initialize dictionaries for cached variables
        self.cache = {
            "x": None,
            "f": None,
            "c_eq": None,
            "c_ineq": None,
            "objective_and_constraints": None,
        }
        self.cache_jac = {
            "x_jac": None,
            "f_jac": None,
            "c_eq_jac": None,
            "c_ineq_jac": None,
            "objective_and_constraints_jac": None,
        }

        # Get handles to objective/constraint fucntions and their Jacobians
        self.f = lambda x: self.get_values(x, print_progress=True)[0]
        self.c_eq = lambda x: self.get_values(x, print_progress=True)[1]
        self.c_ineq = lambda x: self.get_values(x, print_progress=True)[2]
        self.f_jac = lambda x: self.get_jacobian(x, print_progress=True)[0]
        self.c_eq_jac = lambda x: self.get_jacobian(x, print_progress=True)[1]
        self.c_ineq_jac = lambda x: self.get_jacobian(x, print_progress=True)[2]

        # Define list of constraint dictionaries
        self.constraints = []
        if self.N_eq > 0:
            self.constraints.append(
                {"type": "eq", "fun": self.c_eq, "jac": self.c_eq_jac}
            )
        if self.N_ineq > 0:
            self.constraints.append(
                {"type": "ineq", "fun": self.c_ineq, "jac": self.c_ineq_jac}
            )

        # Initialize problem bounds
        self.bounds = self.problem.get_bounds()


    def solve(self, x0=None, method="slsqp", tol=1e-9, options=None):
        """
        Solve the optimization problem using scipy's minimize.

        Parameters
        ----------
        Parameters:
        x0: array-like, optional
            Initial guess for the optimization problem. If not provided, the initial guess set during instantiation is used.
        method : str, optional
            Optimization method to be used by scipy's minimize.
            Two available solvers for nonlinear constrained optimization problems:

            - :code:`trust-constr`: An interior-point method. This method might require more iterations to converge compared to :code:`slsqp`.
            - :code:`slsqp`: A Sequential Quadratic Programming (SQP) method. It might have a different convergence rate compared to :code:`trust-constr`.

            Defaults to 'slsqp'.
        tol : float, optional
            Tolerance for the optimization termination. Defaults to 1e-9.
        options : dict, optional
            Additional options to be passed to scipy's minimize.

        Returns
        -------
        OptimizeResult
            A result object from scipy's minimize detailing the outcome of the optimization.

        Notes
        -----
        The choice between :code:`trust-constr` and :code:`slsqp` can be influenced by the specific problem and the desired balance
        between potential computational speed and the method's nature. It might be beneficial to experiment with both
        methods to determine which one is more suitable for a particular problem.
        """

        # Print report header
        self._write_header()

        # Solve the optimization problem
        self.x0 = self.x0 if x0 is None else x0
        self.solution = minimize(
            self.f,
            self.x0,
            jac=self.f_jac,
            constraints=self.constraints,
            bounds=self.bounds,
            method=method,
            tol=tol,
            options=options,
        )

        # Print report footer
        self._write_footer()

        return self.solution

    def get_values(self, x, single_output=False, print_progress=False):
        """
        Evaluates the optimization problem values at a given point x.

        This method queries the `get_values` method of the OptimizationProblem class to 
        compute the objective function value and constraint values. It first checks the cache 
        to avoid redundant evaluations. If no matching cached result exists, it proceeds to 
        evaluate the objective function and constraints.

        Parameters
        ----------
        x : array-like
            Vector of independent variables (i.e., degrees of freedom).
        single_output : bool, optional
            If True, returns all values (objective and constraints) in a single array.
            If False, returns a tuple with separate arrays for the objective, equality 
            constraints, and inequality constraints. Default is False.

        Returns
        -------
        tuple or array-like
            If `single_output` is True:
                Array containing the objective function value followed by equality and inequality constraints values.
            If `single_output` is False:
                Tuple containing:
                - Objective function value
                - Array of equality constraints values
                - Array of inequality constraints values

        Notes
        -----
        Cached values are based on the last evaluated point. If `x` matches the cached point, 
        the cached results are returned to save computational effort.
        """

        # If x hasn't changed, use cached values
        if np.array_equal(x, self.cache["x"]):
            if single_output:
                return self.cache["objective_and_constraints"]
            else:
                return self.cache["f"], self.cache["c_eq"], self.cache["c_ineq"]

        # Increase total counter (includes finite differences)
        self.func_count_tot += 1

        # Evaluate objective function and constraints at once
        func_constr = self.problem.get_values(x)

        # Split the combined result based on number of constraints
        f = func_constr[0]
        c_eq = func_constr[1 : 1 + self.N_eq]
        c_ineq = func_constr[1 + self.N_eq :]

        # Update cached variabled
        self.cache.update(
            {
                "x": x.copy(),  # Needed for finite differences
                "f": f,
                "c_eq": c_eq,
                "c_ineq": c_ineq,
                "objective_and_constraints": func_constr,
            }
        )

        # Update progress report
        # Better to hide progress at function evaluations (line searches)
        # Show progress only at jacobian evaluations (true iterations)
        if print_progress:
            self.func_count += 1  # Does not include finite differences
            self._print_convergence_progress(x)

        # Return objective and constraints as array or as tuple
        if single_output:
            return self.cache["objective_and_constraints"]
        else:
            return self.cache["f"], self.cache["c_eq"], self.cache["c_ineq"]

    def get_jacobian(self, x, print_progress=False):
        """
        Evaluates the Jacobians of the optimization problem at the given point x.

        This method will use the `get_jacobian` method of the OptimizationProblem class if it exists. 
        If the `get_jacobian` method is not implemented the Jacobian is approximated using forward finite differences.

        To prevent redundant calculations, cached results are checked first. If a matching 
        cached result is found, it is returned; otherwise, a fresh calculation is performed.

        Parameters
        ----------
        x : array-like
            Vector of independent variables (i.e., degrees of freedom).

        Returns
        -------
        tuple
            Three arrays representing:
                - Jacobian of the objective function
                - Jacobian of equality constraints
                - Jacobian of inequality constraints

        Notes
        -----
        Cached values are based on the last evaluated point. If `x` matches the cached point, 
        the cached Jacobians are returned to save computational effort.
        """

        # If x hasn't changed, use cached values
        if np.array_equal(x, self.cache_jac["x_jac"]):
            return (
                self.cache_jac["f_jac"],
                self.cache_jac["c_eq_jac"],
                self.cache_jac["c_ineq_jac"],
            )

        # Evaluate the Jacobian of objective function and constraints at once
        self.grad_count += 1
        if hasattr(self.problem, "get_jacobian"):
            # If the problem has its own Jacobian method, use it
            jacobian = self.problem.get_jacobian(x)

        else: 
            # Fall back to finite differences
            funct_constr = lambda x: self.get_values(x, single_output=True)
            func_constr_0 = (
                self.cache["objective_and_constraints"]
                if np.array_equal(x, self.cache["x"])
                else funct_constr(x)
            )            
            jacobian = approx_derivative(funct_constr, x, method="2-point", f0=func_constr_0)
            jacobian = np.atleast_2d(jacobian)  # Reshape for unconstrained problems

        # Get the jacobian of the objective function and constraint separately
        f_jac = jacobian[0]
        c_eq_jac = jacobian[1 : 1 + self.N_eq]
        c_ineq_jac = jacobian[1 + self.N_eq :]

        # Update cache
        self.cache_jac.update(
            {
                "x_jac": x.copy(),
                "f_jac": f_jac,
                "c_eq_jac": c_eq_jac,
                "c_ineq_jac": c_ineq_jac,
                "objective_and_constraints_jac": jacobian,
            }
        )

        # # Report convergence progress
        # if print_progress:
        #     self._print_convergence_progress(x)

        return f_jac, c_eq_jac, c_ineq_jac

    def _write_header(self):
        """
        Print a formatted header for the optimization report.

        This internal method is used to display a consistent header format at the
        beginning of the optimization process. The header includes columns for function
        evaluations, gradient evaluations, objective function value, constraint violations,
        and norm of the steps.
        """

        # Define header text
        initial_message = (
            f" Starting optimization process for {type(self.problem).__name__}"
        )
        self.header = f" {'Grad-eval':>13}{'Func-eval':>13}{'Func-value':>16}{'Infeasibility':>18}{'Norm of step':>18} "
        separator = "-" * len(self.header)
        lines_to_output = [
            separator,
            initial_message,
            separator,
            self.header,
            separator,
        ]

        # Display to stdout
        if self.display:
            for line in lines_to_output:
                print(line)

        # Write to log
        if self.logger:
            for line in lines_to_output:
                self.logger.info(line)

        # Store text in memory
        self.solution_report.extend(lines_to_output)

    def _print_convergence_progress(self, x):
        """
        Print the current optimization status and update convergence history.

        This method captures and prints the following metrics:
        - Number of gradient evaluations
        - Number of function evaluations
        - Objective function value
        - Maximum constraint violation
        - Norm of the update step

        The method also updates the stored convergence history for potential future analysis.

        Parameters
        ----------
        x : array-like
            The current solution (i.e., vector of independent variable values)

        Notes
        -----
        The norm of the update step is calculated as the two-norm of the difference
        between the current and the last independent variables. Constraints violation is
        computed as the infinity norm of the active constraints.
        """
        # Compute the norm of the last step
        norm_step = np.linalg.norm(x - self.last_x) if self.last_x is not None else 0
        self.last_x = x.copy()

        # Compute the maximun constraint violation
        violation_eq = self.cache["c_eq"]
        violation_ineq = np.minimum(self.cache["c_ineq"], 0)  # 0 if c_ineq >= 0
        violation_all = np.concatenate((violation_eq, violation_ineq))
        violation_max = np.max(np.abs(violation_all)) if len(violation_all) > 0 else 0.0

        # Store convergence status
        self.convergence_history["grad_count"].append(self.grad_count)
        self.convergence_history["func_count"].append(self.func_count)
        self.convergence_history["func_count_total"].append(self.func_count_tot)
        self.convergence_history["objective_value"].append(self.cache["f"])
        self.convergence_history["constraint_violation"].append(violation_max)
        self.convergence_history["norm_step"].append(norm_step)

        # Current convergence message
        status = f" {self.grad_count:13d}{self.func_count:13d}{self.cache['f']:+16.3e}{violation_max:+18.3e}{norm_step:+18.3e} "

        # Display to stdout
        if self.display:
            print(status)

        # Write to log
        if self.logger:
            self.logger.info(status)

        # Store text in memory
        self.solution_report.append(status)

        # Refresh the plot with current values
        if self.plot:
            self._plot_callback()

    def _write_footer(self):
        """
        Print a formatted footer for the optimization report.

        This method displays the final optimization result, including the
        exit message, success status, objective function value, and decision variables.

        Notes
        -----
        The footer's structure is intended to match the header's style,
        providing a consistent look to the optimization report.
        """
        # Define footer text
        separator = "-" * len(self.header)
        exit_message = f"Exit message: {self.solution.message}"
        success_status = f"Sucess: {self.solution.success}"
        solution_header = "Solution:"
        solution_objective = f"   f  = {self.solution.fun:+6e}"
        solution_vars = [f"   x{i} = {x:+6e}" for i, x in enumerate(self.solution.x)]
        lines_to_output = (
            [
                separator,
                exit_message,
                success_status,
                solution_header,
                solution_objective,
            ]
            + solution_vars
            + [separator]
        )

        # Display to stdout
        if self.display:
            for line in lines_to_output:
                print(line)

        # Write to log
        if self.logger:
            for line in lines_to_output:
                self.logger.info(line)

        # Store text in memory
        self.solution_report.extend(lines_to_output)

    def _plot_callback(self, initialize=False):
        """
        Callback function to dynamically update the convergence progress plot.

        This method initializes a matplotlib plot on the first iteration and updates
        the data for each subsequent iteration. The plot showcases the evolution of
        the objective function and the constraint violation with respect to the
        number of iterations.

        The left y-axis depicts the objective function values, while the right y-axis
        showcases the constraint violation values. The x-axis represents the number
        of iterations. Both lines are updated and redrawn dynamically as iterations progress.

        Note:
            This is an internal method, meant to be called within the optimization process.
        """

        # Initialize figure before first iteration
        if initialize:
            self.fig, self.ax_1 = plt.subplots()
            (self.obj_line_1,) = self.ax_1.plot(
                [], [], color="#0072BD", marker="o", label="Objective function"
            )
            self.ax_1.set_xlabel("Number of iterations")
            self.ax_1.set_ylabel("Objective function")
            self.ax_1.xaxis.set_major_locator(MaxNLocator(integer=True)) # Interger ticks
            if self.N_eq > 0 or self.N_ineq > 0:
                self.ax_2 = self.ax_1.twinx()
                self.ax_2.set_ylabel("Constraint violation")
                (self.obj_line_2,) = self.ax_2.plot(
                    [], [], color="#D95319", marker="o", label="Constraint violation"
                )
                lines = [self.obj_line_1, self.obj_line_2]
                labels = [l.get_label() for l in lines]
                self.ax_1.legend(lines, labels, loc="upper right")
            else:
                self.ax_1.legend(loc="upper right")
            self.fig.tight_layout(pad=1)

        # Update plot data with current values
        iteration = self.convergence_history["func_count"]
        # iteration = self.convergence_history["grad_count"]
        objective_function = self.convergence_history["objective_value"]
        constraint_violation = self.convergence_history["constraint_violation"]
        self.obj_line_1.set_xdata(iteration)
        self.obj_line_1.set_ydata(objective_function)
        if self.N_eq > 0 or self.N_ineq > 0:
            self.obj_line_2.set_xdata(iteration)
            self.obj_line_2.set_ydata(constraint_violation)

        # Adjust the plot limits
        self.ax_1.relim()
        self.ax_1.autoscale_view()
        if self.N_eq > 0 or self.N_ineq > 0:
            self.ax_2.relim()
            self.ax_2.autoscale_view()

        # Redraw the plot
        plt.draw()
        plt.pause(0.01)  # small pause to allow for update

    def print_convergence_history(self):
        """
        Print the convergence history of the problem.

        The convergence history includes:
            - Current iteration
            - Number of function evaluations
            - Objective function value
            - Maximum constraint violation
            - Two-norm of the update step

        The method provides a comprehensive report on the final solution, including:
            - Exit message
            - Success status
            - Final ojective function value
            - Final independent variables values

        Note:
            This method should be called after the optimization process is completed.
        """
        if self.solution:
            for line in self.solution_report:
                print(line)
        else:
            warnings.warn(
                "This method should be used after invoking the 'solve()' method."
            )

    def plot_convergence_history(self, savefig=True, name=None, use_datetime=False, path=None):
        """
        Plot the convergence history of the problem.

        This method plots the optimization progress against the number of iterations:
            - Objective function value (left y-axis)
            - Maximum constraint violation (right y-axis)

        Note:
            - This method should be called after the optimization process is completed.
            - The constraint violation is only displayed if the problem has constraints


        """
        if self.solution:
            self._plot_callback(initialize=True)
        else:
            warnings.warn(
                "This method should be used after invoking the 'solve()' method."
            )

        if savefig:

            # Give a name to the figure if it is not specified
            if name is None:
                name = f"convergence_history_{type(self.problem).__name__}"

            # Define figures directory if it is not provided
            if path is None:
                path = os.path.join(os.getcwd(), "figures")

            # Create figures directory if it does not exist
            if not os.path.exists(path):
                os.makedirs(path)

            # Define file name and path
            if use_datetime:
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(path, f"{name}_{current_time}")
            else:
                filename = os.path.join(path, f"{name}")
            
            # Save plots
            self.fig.savefig(filename + ".png", bbox_inches="tight")
            self.fig.savefig(filename + ".svg", bbox_inches="tight")
 

        


class OptimizationProblem(ABC):
    """
    Abstract base class for optimization problems.

    Derived optimization problem objects must implement the following methods:

    - `get_values`: Evaluate the objective function and constraints for a given set of decision variables.
    - `get_bounds`: Get the bounds for each decision variable.
    - `get_n_eq`: Return the number of equality constraints associated with the problem.
    - `get_n_ineq`: Return the number of inequality constraints associated with the problem.

    Additionally, specific problem classes can define the `get_jacobian` method to compute the Jacobians. If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Methods
    -------
    get_values(x)
        Evaluate the objective function and constraints for a given set of decision variables.
    get_bounds()
        Get the bounds for each decision variable.
    get_n_eq()
        Return the number of equality constraints associated with the problem.
    get_n_ineq()
        Return the number of inequality constraints associated with the problem.

    Examples
    --------
    Here's an example of how to derive from `OptimizationProblem`::

        class MyOptimizationProblem(OptimizationProblem):
            def get_values(self, x):
                # Implement evaluation logic here
                pass

            def get_bounds(self):
                # Return bounds of decision variables
                pass

            def get_n_eq(self):
                # Return number of equality constraints
                pass

            def get_n_ineq(self):
                # Return number of inequality constraints
                pass

    """

    @abstractmethod
    def get_values(self, x):
        """
        Evaluate the objective function and constraints for given decision variables.

        Parameters
        ----------
        x : array-like
            Vector of independent variables (i.e., degrees of freedom).

        Returns
        -------
        array_like
            Vector containing the objective function, equality constraints, and inequality constraints.
        """
        pass

    @abstractmethod
    def get_bounds(self):
        """
        Get the bounds for each decision variable.

        Returns
        -------
        bounds : list of tuple
            A list of tuples where each tuple contains two elements: the lower and upper bounds
            for the corresponding decision variable. For example, [(-2, 2), (-1, 1)] indicates
            that the first decision variable has bounds between -2 and 2, and the second has
            bounds between -1 and 1.
        """
        pass

    @abstractmethod
    def get_n_eq(self):
        """
        Return the number of equality constraints associated with the problem.

        Returns
        -------
        neq : int
            Number of equality constraints.
        """
        pass

    @abstractmethod
    def get_n_ineq(self):
        """
        Return the number of inequality constraints associated with the problem.

        Returns
        -------
        nineq : int
            Number of inequality constraints.
        """
        pass

    @staticmethod
    def merge_objective_and_constraints(f, c_eq, c_ineq):
        """
        Combine an objective function with its associated equality and inequality constraints.

        This function takes in an objective function value, a set of equality constraints,
        and a set of inequality constraints. It then returns a combined numpy array of
        these values. If the constraints are not given as a list, tuple, or numpy array,
        they will be appended as individual values.

        Parameters
        ----------
        f : float
            The value of the objective function.
        c_eq : float, list, tuple, np.ndarray, or None
            The equality constraint(s). This can be a single value or a collection of values.
            If `None`, no equality constraints will be added.
        c_ineq : float, list, tuple, np.ndarray, or None
            The inequality constraint(s). This can be a single value or a collection of values.
            If `None`, no inequality constraints will be added.

        Returns
        -------
        np.ndarray
            A numpy array consisting of the objective function value followed by equality and
            inequality constraints.

        Examples
        --------
        >>> combine_objective_and_constraints(1.0, [0.5, 0.6], [0.7, 0.8])
        array([1. , 0.5, 0.6, 0.7, 0.8])

        >>> combine_objective_and_constraints(1.0, 0.5, 0.7)
        array([1. , 0.5, 0.7])
        """
        # Add objective function
        combined_list = [f]

        # Add equality constraints
        if c_eq is not None:
            if isinstance(c_eq, (list, tuple, np.ndarray)):
                combined_list.extend(c_eq)
            else:
                combined_list.append(c_eq)

        # Add inequality constraints
        if c_ineq is not None:
            if isinstance(c_ineq, (list, tuple, np.ndarray)):
                combined_list.extend(c_ineq)
            else:
                combined_list.append(c_ineq)

        # print(combined_list)
        return np.array(combined_list)

    @staticmethod
    def get_number_of_constraints(var):
        """
        Retrieve the number of constraints based on the provided input.

        This function returns the count of constraints based on the nature of the
        input:

        - `None` returns 0
        - Scalar values return 1
        - Array-like structures return their length

        Parameters
        ----------
        var : None, scalar, or array-like (list, tuple, np.ndarray)
            The input representing the constraint(s). This can be `None`, a scalar value,
            or an array-like structure containing multiple constraints.

        Returns
        -------
        int
            The number of constraints:

            - 0 for `None`
            - 1 for scalar values
            - Length of the array-like for array-like inputs

        Examples
        --------
        >>> get_number_of_constraints(None)
        0

        >>> get_number_of_constraints(5.0)
        1

        >>> get_number_of_constraints([1.0, 2.0, 3.0])
        3
        """
        # If constraint is None
        if var is None:
            return 0
        # If constraint is a scalar (assuming it's numeric)
        elif np.isscalar(var):
            return 1
        # If constraint is array-like
        else:
            return len(var)
