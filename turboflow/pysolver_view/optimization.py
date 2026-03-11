import os
import time
import copy
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from abc import ABC, abstractmethod
from matplotlib.ticker import MaxNLocator

from . import numerical_differentiation
from . import optimization_wrappers as _opt
from .pysolver_utilities import savefig_in_formats, validate_keys

# Define valid libraries and their corresponding methods
OPTIMIZATION_LIBRARIES = {
    "scipy": _opt.minimize_scipy,
    "pygmo": _opt.minimize_pygmo,
    "pygmo_nlopt": _opt.minimize_nlopt,
}

VALID_LIBRARIES_AND_METHODS = {
    "scipy": _opt.SCIPY_SOLVERS,
    "pygmo": _opt.PYGMO_SOLVERS,
    "pygmo_nlopt": _opt.NLOPT_SOLVERS,
}


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
        c_{\mathrm{in}}(\mathbf{x}) \leq 0
    .. math::
        \mathbf{x}_l \leq \mathbf{x} \leq \mathbf{x}_u

    where:

    - :math:`\mathbf{x}` is the vector of decision variables (i.e., degree of freedom).
    - :math:`f(\mathbf{x})` is the objective function to be minimized. Maximization problems can be casted into minimization problems by changing the sign of the objective function.
    - :math:`c_{\mathrm{eq}}(\mathbf{x})` are the equality constraints of the problem.
    - :math:`c_{\mathrm{in}}(\mathbf{x})` are the inequality constraints of the problem. Constraints of type :math:`c_{\mathrm{in}}(\mathbf{x}) \leq 0` can be casted into :math:`c_{\mathrm{in}}(\mathbf{x}) \geq 0` type by changing the sign of the constraint functions.
    - :math:`\mathbf{x}_l` and :math:`\mathbf{x}_u` are the lower and upper bounds on the decision variables.

    The class interfaces with various optimization methods provided by libraries such as `scipy` and `pygmo` to solve the problem and provides a structured framework for initialization, solution monitoring, and post-processing.

    This class employs a caching mechanism to avoid redundant evaluations. For a given set of independent variables, x, the optimizer requires the objective function, equality constraints, and inequality constraints to be provided separately. When working with complex models, these values are typically calculated all at once. If x hasn't changed from a previous evaluation, the caching system ensures that previously computed values are used, preventing unnecessary recalculations.

    Parameters
    ----------
    problem : OptimizationProblem
        An instance of the optimization problem to be solved. The problem should be defined
        in physical space, with its own bounds and (optionally) analytic derivatives.
    library : str, optional
        The library to use for solving the optimization problem (default is 'scipy').
    method : str, optional
        The optimization method to use from the specified library (default is 'slsqp').
    tolerance : float, optional
        Tolerance for termination. The minimization algorithm sets some solver-specific tolerances
        equal to tol. (default is 1e-6)
    max_iterations : int, optional
        Maximum number of iterations for the optimizer (default is 100).
    extra_options : dict, optional
        A dictionary of solver-specific options that prevails over 'tolerance' and 'max_iterations'
    derivative_method : str, optional
        Method to use for derivative calculation (default is '2-point').
    derivative_abs_step : float, optional
        Finite difference absolute step size to be used when the problem Jacobian is not provided. Default depends on calculation method.
    problem_scale : float or None, optional
        Scaling factor used to normalize the problem. This parameter controls the transformation
        of physical variables into a normalized domain. Specifically, for a physical variable x,
        with lower and upper bounds lb and ub, the normalized variable is computed as:

            x_norm = problem_scale * (x - lb) / (ub - lb)

        - If a numeric value is provided (e.g. 1.0, 10.0, etc.), the problem is scaled accordingly.
          Increasing problem_scale reduces the relative step sizes in the normalized space during
          line searches, which can improve convergence by making the optimization less aggresive.
          The rationale for the scaling is that the initial line search step size of many solvers is 1.0.
        - If set to None, no scaling is applied and the problem is solved in its original physical units.
          This might be preferred if the problem is already well-conditioned or if the user wishes to
          preserve the exact scale of the original formulation.

    print_convergence : bool, optional
        If True, displays the convergence progress (default is True).
    plot_convergence : bool, optional
        If True, plots the convergence progress (default is False).
    plot_scale_objective : str, optional
        Specifies the scale of the objective function axis in the convergence plot (default is 'linear').
    plot_scale_constraints : str, optional
        Specifies the scale of the constraint violation axis in the convergence plot (default is 'linear').
    logger : logging.Logger, optional
        Logger object to which logging messages will be directed. Logging is disabled if `logger` is None.
    update_on : str, optional
        Specifies if the convergence report should be updated based on new function evaluations or gradient evaluations (default is 'gradient', alternative is 'function').
    callback_functions : list of callable or callable, optional
        Optional list of callback functions to pass to the solver.
    plot_improvement_only : bool, optional
        If True, plots only display iterations that improve the objective function value (useful for gradient-free optimizers) (default is False).

    Methods
    -------
    solve(x0):
        Solve the optimization problem using the specified initial guess `x0`.
    fitness(x):
        Evaluates the optimization problem objective function and constraints at a given point `x`.
    gradient(x):
        Evaluates the Jacobians of the optimization problem at a given point `x`.
    print_convergence_history():
        Print the final result and convergence history of the optimization problem.
    plot_convergence_history():
        Plot the convergence history of the optimization problem.
    """

    def __init__(
        self,
        problem,
        library="scipy",
        method="slsqp",
        tolerance=1e-6,
        max_iterations=100,
        extra_options={},
        derivative_method="2-point",
        derivative_abs_step=None,
        problem_scale=None,
        print_convergence=True,
        plot_convergence=False,
        plot_scale_objective="linear",
        plot_scale_constraints="linear",
        logger=None,
        update_on="gradient",
        callback_functions=None,
        plot_improvement_only=False,
        tolerance_check_cache=None
    ):
        # Initialize class variables
        self.problem = problem
        self.display = print_convergence
        self.plot = plot_convergence
        self.plot_scale_objective = plot_scale_objective
        self.plot_scale_constraints = plot_scale_constraints
        self.logger = logger
        self.library = library
        self.method = method
        self.derivative_method = derivative_method
        self.derivative_abs_step = derivative_abs_step
        self.callback_functions = self._validate_callback(callback_functions)
        self.callback_function_call_count = 0
        self.plot_improvement_only = plot_improvement_only
        
        # Tolerance to check if design variables are the same as in previous fitness call
        if tolerance_check_cache is None:
            self.tolerance_cache = 10 * np.finfo(float).eps
        else:
            self.tolerance_cache = tolerance_check_cache

        # Validate library and method
        self._validate_library_and_method()

        # Define options dictionary
        self.options = {"tolerance": tolerance, "max_iterations": max_iterations}
        self.options = self.options | extra_options

        # Set scaling for the optimization problem (define at the solver, rather than problem level)
        self.problem.problem_scale = problem_scale

        # Auto-generate design variable names if not provided
        if not hasattr(self.problem, "variable_names"):
            n_vars = len(self.problem.get_bounds()[0])
            self.problem.variable_names = [
                f"design_variable_{i}" for i in range(n_vars)
            ]

        # Check for logger validity
        if self.logger is not None:
            if not isinstance(self.logger, logging.Logger):
                raise ValueError(
                    "The provided logger is not a valid logging.Logger instance."
                )

        # Check for valid display_on value
        self.update_on = update_on
        if update_on not in ["function", "gradient"]:
            raise ValueError(
                "Invalid value for 'update_on'. It should be either 'function' or 'gradient'."
            )

        # Rename number of constraints
        self.N_eq = self.problem.get_nec()
        self.N_ineq = self.problem.get_nic()

        # Initialize variables for convergence report
        self.f_final = None
        self.x_final = None
        self.x_last_norm = None
        self.grad_count = 0
        self.func_count = 0
        self.func_count_tot = 0
        self.success = None
        self.message = None
        self.convergence_report = []
        self.elapsed_time = None
        self.convergence_history = {
            "x": [],
            "grad_count": [],
            "func_count": [],
            "func_count_total": [],
            "objective_value": [],
            "constraint_violation": [],
            "norm_step": [],
        }

        # Initialize dictionary for cached variables
        self.cache = {
            "x": None,
            "f": None,
            "c_eq": None,
            "c_ineq": None,
            "x_jac": None,
            "f_jac": None,
            "c_eq_jac": None,
            "c_ineq_jac": None,
            "fitness": None,
            "gradient": None,
        }

    def _validate_library_and_method(self):
        # Check if the library is valid
        if self.library not in VALID_LIBRARIES_AND_METHODS:
            error_message = (
                f"Invalid optimization library '{self.library}'. \nAvailable libraries:\n   - "
                + "\n   - ".join(VALID_LIBRARIES_AND_METHODS.keys())
                + "."
            )
            raise ValueError(error_message)

        # Check if the method is valid for the selected library
        if self.method and self.method not in VALID_LIBRARIES_AND_METHODS[self.library]:
            error_message = (
                f"Invalid method '{self.method}' for library '{self.library}'. \nValid methods are:\n   - "
                + "\n   - ".join(VALID_LIBRARIES_AND_METHODS[self.library])
                + "."
            )
            raise ValueError(error_message)

    def _validate_callback(self, callback):
        """Validate the callback functions argument."""
        if callback is None:
            return []
        if callable(callback):
            return [callback]
        elif isinstance(callback, list):
            non_callable_items = [item for item in callback if not callable(item)]
            if not non_callable_items:
                return callback
            else:
                error_msg = f"All elements in the callback list must be callable functions. Non-callable items: {non_callable_items}"
                raise TypeError(error_msg)
        else:
            error_msg = f"callback_func must be a function or a list of functions. Received type: {type(callback)} ({callback})"
            raise TypeError(error_msg)

    def solve(self, x0):
        """
        Solve the optimization problem using the specified library and solver.

        This method initializes the optimization process, manages the flow of the optimization,
        and handles the results, utilizing the solver from a specified library such as scipy or pygmo.

        Parameters
        ----------
        x0 : array-like, optional
            Initial guess for the solution of the optimization problem.

        Returns
        -------
        x_final : array-like
            An array with the optimal vector of design variables

        """
        # Initialize convergence plot
        if self.plot:
            self._plot_convergence_callback([], [], initialize=True)

        # Get start datetime
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Start timing with high-resolution timer
        start_time = time.perf_counter()

        # Normalize the problem as the very first thing
        if self.problem.get_bounds() is not None:
            x0 = self.problem.clip_to_bounds(x0, logger=self.logger)
        x0 = self.problem.scale_physical_to_normalized(x0)

        # Print report header
        self._write_header()

        # Define new problem with anonymous methods (avoid problems when Pygmo creates a deep copy)
        problem = _PygmoProblem(self)

        # Print initial guess evaluation when using gradient
        if self.update_on == "gradient":
            self._print_convergence_progress(x0)

        # Solve the problem
        lib_wrapper = OPTIMIZATION_LIBRARIES[self.library]
        solution = lib_wrapper(problem, x0, self.method, self.options)

        # Retrieve last solution (also works for gradient-free solvers when updating on gradient)
        x_final_norm = self.x_last_norm if self.x_last_norm is not None else self.cache["x"]
        self.fitness(x_final_norm)
        self.gradient(x_final_norm)

        # Store final solution
        self.f_final = copy.deepcopy(self.cache["f"])
        self.success, self.message = solution

        # Calculate elapsed time
        self.elapsed_time = time.perf_counter() - start_time

        # Print report footer
        self._print_convergence_progress(x_final_norm)
        self._write_footer()

        # Save variable in physical scale
        self.x_final_norm = copy.deepcopy(x_final_norm)
        self.x_final = self.problem.scale_normalized_to_physical(self.x_final_norm)

        return self.x_final

    def fitness(self, x_norm, called_from_grad=False):
        """
        Evaluates the optimization problem values at a given point x.

        This method queries the `fitness` method of the OptimizationProblem class to
        compute the objective function value and constraint values. It first checks the cache
        to avoid redundant evaluations. If no matching cached result exists, it proceeds to
        evaluate the objective function and constraints.

        Parameters
        ----------
        x : array-like
            Vector of independent variables (i.e., degrees of freedom).
        called_from_grad : bool, optional
            Flag used to indicate if the method is called during gradient evaluation.
            This helps in preventing redundant increments in evaluation counts during
            finite-differences gradient calculations. Default is False.

        Returns
        -------
        fitness : numpy.ndarray
            A 1D array containing the objective function, equality constraints, and inequality constraints at `x`.

        """
        # If x hasn't changed, use cached values
        if self.cache["x"] is not None and np.allclose(x_norm, self.cache["x"], rtol=0, atol=self.tolerance_cache):
            return self.cache["fitness"]

        # Increase total counter (includes finite differences)
        self.func_count_tot += 1

        # Evaluate objective function and constraints at once
        fitness = self.problem.fitness_normalized_input(x_norm)

        # Does not include finite differences
        if not called_from_grad:
            # Update cached variabled
            self.cache.update(
                {
                    "x": x_norm.copy(),  # Needed for finite differences
                    "f": fitness[0],
                    "c_eq": fitness[1 : 1 + self.N_eq],
                    "c_ineq": fitness[1 + self.N_eq :],
                    "fitness": fitness,
                }
            )

            # Increase minor iteration counter (line search)
            self.func_count += 1

            # Update progress report
            if self.update_on == "function":
                self._print_convergence_progress(x_norm)

        return fitness

    def gradient(self, x_norm):
        """
        Evaluates the Jacobian matrix of the optimization problem at the given point x.

        This method utilizes the `gradient` method of the OptimizationProblem class if implemented.
        If the `gradient` method is not implemented, the Jacobian is approximated using forward finite differences.

        To prevent redundant calculations, cached results are checked first. If a matching
        cached result is found, it is returned; otherwise, a fresh calculation is performed.

        Parameters
        ----------
        x : array-like
            Vector of independent variables (i.e., degrees of freedom).

        Returns
        -------
        numpy.ndarray
            A 2D array representing the Jacobian matrix of the optimization problem at `x`.
            The Jacobian matrix includes:
            - Gradient of the objective function
            - Jacobian of equality constraints
            - Jacobian of inequality constraints
        """

        # If x hasn't changed, use cached values
        if self.cache["x_jac"] is not None and np.allclose(x_norm, self.cache["x_jac"], rtol=0, atol=self.tolerance_cache):
            return self.cache["gradient"]

        # Use problem gradient method if it exists
        if hasattr(self.problem, "gradient"):
            grad = self.problem.gradient_normalized_input(x_norm)
        else:
            # Fall back to finite differences
            fun = lambda x: self.fitness(x, called_from_grad=True)
            grad = numerical_differentiation.approx_gradient(
                fun,
                x_norm,
                f0=fun(x_norm),
                method=self.derivative_method,
                abs_step=self.derivative_abs_step,
            )

        # Reshape gradient for unconstrained problems
        grad = np.atleast_2d(grad)

        # Update cache
        self.cache.update(
            {
                "x_jac": x_norm.copy(),
                "f_jac": grad[0, :],
                "c_eq_jac": grad[1 : 1 + self.N_eq, :],
                "c_ineq_jac": grad[1 + self.N_eq :, :],
                "gradient": grad,
            }
        )

        # Update progress report
        self.grad_count += 1
        if self.update_on == "gradient":
            self._print_convergence_progress(x_norm)

        return grad

    def _handle_output(self, text: str, savefile: bool = False, filename: str = None, to_console=False):
        """
        Unified output handler: print, log, and optionally save to a file.

        Parameters
        ----------
        text : str
            The content to emit. Can be single or multiline.
        savefile : bool, optional
            If True, saves `text` to the given filename. Default is False.
        filename : str or None, optional
            Path to file where output should be saved. Required if savefile is True.
        """
        # Print to logger
        if self.logger:
            for line in text.splitlines():
                self.logger.info(line)

        # Print to screen if display is enabled and no logger used
        # if self.display and not self.logger:
        if self.display and to_console:
            print(text)

        # Save to file if requested
        if savefile:
            if not filename:
                raise ValueError("Filename must be provided if savefile is True.")
            with open(filename, "w") as f:
                f.write(text)

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
            f" Starting optimization process for {type(self.problem).__name__}\n"
            f" Optimization algorithm employed: {self.method}"
        )
        self.header = f" {'Grad-eval':>13}{'Func-eval':>13}{'Func-value':>16}{'Infeasibility':>18}{'Norm of step':>18}{'Optimality':>18} "
        separator = "-" * len(self.header)
        lines_to_output = [
            separator,
            initial_message,
            separator,
            self.header,
            separator,
        ]

        # Print or log content
        for line in lines_to_output:
            self._handle_output(line, to_console=True)

        # Store text in memory
        self.convergence_report.extend(lines_to_output)

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

        # Ensure fitness is computed at least once before printing
        if self.cache["fitness"] is None:
            self.fitness(x)

        # Compute the norm of the last step
        norm_step = (
            np.linalg.norm(x - self.x_last_norm) if self.x_last_norm is not None else 0
        )
        self.x_last_norm = x.copy()

        # Compute the maximun constraint violation
        c_eq = self.cache["c_eq"]
        c_ineq = self.cache["c_ineq"]
        violation_all = np.concatenate((c_eq, np.maximum(c_ineq, 0)))
        violation_max = np.max(np.abs(violation_all)) if len(violation_all) > 0 else 0.0
        kkt_data = self.evaluate_kkt_conditions(x, 50*self.options["tolerance"])
        optimality = kkt_data["first_order_optimality"]

        # Store convergence status
        self.convergence_history["x"].append(self.x_last_norm)
        self.convergence_history["grad_count"].append(self.grad_count)
        self.convergence_history["func_count"].append(self.func_count)
        self.convergence_history["func_count_total"].append(self.func_count_tot)
        self.convergence_history["objective_value"].append(self.cache["f"])
        self.convergence_history["constraint_violation"].append(violation_max)
        self.convergence_history["norm_step"].append(norm_step)

        # Current convergence message
        status = f" {self.grad_count:13d}{self.func_count:13d}{self.cache['f']:+16.3e}{violation_max:+18.3e}{norm_step:+18.3e}{optimality:+18.3e} "
        # status = f" {self.grad_count:13d}{self.func_count:13d}{self.cache['f']:+16.3e}{violation_max:+18.3e}{norm_step:+18.3e} "
        self._handle_output(status, to_console=True)

        # Store text in memory
        self.convergence_report.append(status)

        # Refresh the plot with current values
        # TODO for some reason convergence plot and thermodynamic cycle are not being plotten when options set to true
        if self.plot:
            self._plot_convergence_callback([], [])

        # Evaluate callback functions
        if self.callback_functions:
            self.callback_function_call_count += 1
            for func in self.callback_functions:
                func(x, self.callback_function_call_count)

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
        exit_message = f" Exit message: {self.message}"
        success_status = f" Success: {self.success}"
        time_message = f" Solution time: {self.elapsed_time:.3f} seconds"
        lines_to_output = [separator, success_status, exit_message, time_message]
        lines_to_output += [separator]

        # Print or log content
        for line in lines_to_output:
            self._handle_output(line, to_console=True)

        # Store text in memory
        self.convergence_report.extend(lines_to_output)

    def _plot_convergence_callback(self, x, iter, initialize=False, showfig=True):
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
            self.ax_1.set_yscale(self.plot_scale_objective)
            self.ax_1.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )  # Integer ticks
            self.ax_1.grid(False)
            if self.N_eq > 0 or self.N_ineq > 0:
                self.ax_2 = self.ax_1.twinx()
                self.ax_2.grid(False)
                self.ax_2.set_ylabel("Constraint violation")
                self.ax_2.set_yscale(self.plot_scale_constraints)
                (self.obj_line_2,) = self.ax_2.plot(
                    [], [], color="#D95319", marker="o", label="Constraint violation"
                )
                lines = [self.obj_line_1, self.obj_line_2]
                labels = [l.get_label() for l in lines]
                self.ax_2.legend(lines, labels, loc="upper right")
            else:
                self.ax_1.legend(loc="upper right")

            self.fig.tight_layout(pad=1)

        # Update plot data with current values
        iteration = (
            self.convergence_history["func_count"]
            if self.update_on == "function"
            else self.convergence_history["grad_count"]
        )
        objective_function = self.convergence_history["objective_value"]
        constraint_violation = self.convergence_history["constraint_violation"]

        # Iterate through the objective_function values to create the new series
        if self.plot_improvement_only:
            for i in range(1, len(objective_function)):
                if objective_function[i] > objective_function[i - 1]:
                    objective_function[i] = objective_function[i - 1]

        # Update graphic objects data
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
        if showfig:
            plt.draw()
            plt.pause(0.01)  # small pause to allow for update


    def plot_convergence_history(
        self,
        savefile=False,
        filename=None,
        output_dir="output",
        showfig=True,
    ):
        """
        Plot the convergence history of the problem.

        This method plots the optimization progress against the number of iterations:
            - Objective function value (left y-axis)
            - Maximum constraint violation (right y-axis)

        The constraint violation is only displayed if the problem has nonlinear constraints

        This method should be called only after the optimization problem has been solved, as it relies on data generated by the solving process.

        Parameters
        ----------
        savefile : bool, optional
            If True, the plot is saved to a file instead of being displayed. Default is False.
        filename : str, optional
            The name of the file to save the plot to. If not specified, the filename is automatically generated
            using the problem name and the start datetime. The file extension is not required.
        output_dir : str, optional
            The directory where the plot file will be saved if savefile is True. Default is "output".

        Returns
        -------
        matplotlib.figure.Figure
            The Matplotlib figure object for the plot. This can be used for further customization or display.

        Raises
        ------
        ValueError
            If this method is called before the problem has been solved.
        """
        if self.x_final is None:
            raise ValueError("This method can only be used after invoking the 'solve()' method.")

        # Initialize and optionally show plot
        self._plot_convergence_callback([], [], initialize=True, showfig=showfig)

        # Save if requested
        if savefile:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                basename = f"{type(self.problem).__name__}_{self.start_datetime}"
                filename = f"convergence_history_{basename}"
            savefig_in_formats(self.fig, os.path.join(output_dir, filename), formats=[".png", ".svg"])

        # Close the figure if it should not be displayed
        if not showfig:
            plt.close(self.fig)

        return self.fig
    

    def print_convergence_history(self, savefile=False, filename=None, output_dir="output", to_console=True):
        """
        Print or save the convergence history of the optimization process.

        This function prints (or saves) a the report of the optimization convergence progress.
        It includes information collected at each iteration, including:

        - Number of gradient evaluations
        - Number of function evaluations
        - Objective function value
        - Maximum constraint violation
        - Two-norm of the update step

        It also includes a summary at the end of the run with:

        - Exit message
        - Success flag
        - Total solution time (in seconds)

        .. note::
            This method must be called **after** `solve()` has been executed. Otherwise, the convergence
            report is unavailable and a `ValueError` is raised.

        Parameters
        ----------
        savefile : bool, optional
            If True, the report is saved to a file. Otherwise, it is printed to the screen. Default is False.

        filename : str or None, optional
            The name of the file to save the report to. If None, a default name is generated based on the
            problem class name and the optimization start datetime.

        output_dir : str, optional
            Directory where the report file is saved if `savefile=True`. Default is "output".

        Raises
        ------
        ValueError
            If the method is called before the optimization problem has been solved.
        """
        if self.x_final is None:
            raise ValueError("This method can only be used after invoking the 'solve()' method.")

        # Join convergence history lines
        full_text = "\n".join(self.convergence_report)

        # Decide filename if saving to disk
        if savefile:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                default = f"{type(self.problem).__name__}_{self.start_datetime}"
                filename = f"convergence_history_{default}.txt"
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = None

        # Output the report (either print or save to file)
        self._handle_output(full_text, savefile=savefile, filename=filepath, to_console=to_console)


    def print_optimization_report(
        self,
        x=None,
        tol=None,
        include_design_variables=True,
        include_constraints=True,
        include_kkt_conditions=False,
        include_multipliers=False,
        savefile=False,
        filename=None,
        output_dir="output",
        to_console=True,
    ):
        """
        Generate and print or save a complete optimization report with customizable content.

        This method assembles a detailed summary of the optimization process and results,
        allowing fine-grained control over which components to include. It supports
        outputting the report to the console or saving it to a file.

        The report may include the following sections:
        - Convergence history: number of function and gradient evaluations, objective function value, maximum constraint violation, and step norm. The report also includes The method provides a detailed report on an exit message, success status, and execution time
        - Number of function evaluations
        - Number of gradient evaluations
        - Objective function value
        - Maximum constraint violation
        - Two-norm of the update step

    
    Objective value, constraint violation, and step norm over iterations.
        - Design variables: Final values, shown in physical units and (if applicable) normalized space.
        - Constraints: Numerical values and satisfaction status of all constraints.
        - KKT conditions: Checks of the Karush-Kuhn-Tucker optimality conditions.
        - Lagrange multipliers: Values of multipliers for equality, inequality, and bound constraints.

        This method is intended to be called after the `solve()` method has been completed.


        Parameters
        ----------
        include_convergence_history : bool
        include_design_variables_normalized : bool
        include_design_variables_physical : bool
        include_constraints : bool
        include_kkt : bool
        include_multipliers : bool
        savefile : bool
        filename : str or None
        tol : float or None  (tolerance for constraints/KKT)
        output_dir : str      (directory for saving file)
        """

        # Define the vector of independent variables
        x_phys = x if x is not None else self.x_final
        x_norm = self.problem.scale_physical_to_normalized(x_phys)

        # Use a loose tolerance if not given as argument
        tol = 5 * self.options["tolerance"] if tol is None else tol

        # Collect each section’s text and soin into a single string
        r = []
        if include_kkt_conditions:
            r.append(self.make_kkt_optimality_report(x_norm, tol))
        if include_design_variables:
            if self.problem.problem_scale is None:
                r.append(self.make_variables_report(x_norm, normalized=False))
            else:
                r.append(self.make_variables_report(x_norm, normalized=True))
                r.append(self.make_variables_report(x_phys, normalized=False))    
        if include_constraints:
            r.append(self.make_constraint_report(x_norm, tol))
        if include_multipliers:
            r.append(self.make_lagrange_multipliers_report(x_norm, tol))
        full_text = "\n".join(r)

        # Decide on filename if saving
        if savefile:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                filename = (
                    f"optimization_report_{type(self.problem).__name__}"
                    f"_{self.start_datetime}.txt"
                )
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = None

        # 5) emit once to console, logger, or file
        self._handle_output(full_text, savefile=savefile, filename=filepath, to_console=to_console)



    # def make_optimization_report(self, x_phys, tol=1e-6):
    #     """
    #     Print or save a complete optimization report, including variables and constraints.
    #     """
    #     report = []
    #     report.append(self.make_variables_report(x_phys, normalized=True))
    #     report.append(self.make_variables_report(x_phys, normalized=False))
    #     report.append(self.make_constraint_report(x_phys, tol=tol))
    #     return "\n".join(report)

    def make_variables_report(self, x_norm, normalized=True):
        """
        Generate design variable report as a string.

        Parameters
        ----------
        x_norm : array-like
            Normalized design variables (input to the solver).
        normalized : bool, optional
            Whether to report in normalized or physical values. Default is True.

        Returns
        -------
        str
            Formatted string report.
        """
        x_norm = np.asarray(x_norm)
        if normalized:
            values = x_norm
            lb_raw, ub_raw = self.problem.get_bounds_normalized()
        else:
            values = x_norm
            lb_raw, ub_raw = self.problem.get_bounds()

        names = self.problem.variable_names or [
            f"design_variable_{i}" for i in range(len(values))
        ]
        scale_type = "normalized" if normalized else "physical"
        max_name_width = 38

        lines = []
        lines.append("")
        lines.append("-" * 80)
        header = f"{' Optimization variables report (' + scale_type + ' values)':<80}"
        lines.append(header)
        lines.append("-" * 80)
        lines.append(f" {'Variable name':<39}{'Lower':>13}{'Value':>13}{'Upper':>13}")
        lines.append("-" * 80)

        for key, lb, val, ub in zip(names, lb_raw, values, ub_raw):
            if len(key) > max_name_width:
                key = "..." + key[-(max_name_width - 3):]

            if normalized:
                lines.append(f" {key:<39}{lb:>13.4f}{val:>13.4f}{ub:>13.4f}")
            else:
                lines.append(f" {key:<39}{lb:>13.3e}{val:>13.3e}{ub:>13.3e}")

        lines.append("-" * 80)
        return "\n".join(lines)

    def get_constraint_data(self, x_norm, tol):
        """
        Return constraint evaluation data for the current optimization problem.

        This method returns a list of dictionaries describing the satisfaction
        status of all equality and inequality constraints at a given normalized
        design point.

        If `self.problem.constraint_data` exists and is valid, it is returned
        directly after key validation.

        Parameters
        ----------
        x_norm : array_like
            Normalized design variable vector.
        tol : float
            Tolerance for constraint satisfaction.

        Returns
        -------
        list of dict
            Each dictionary contains:

            - **name** : str  
              Constraint identifier.

            - **type** : str  
              Either "=" for equality or "<" for inequality constraints.

            - **target** : float  
              Desired target value (0.0 for all constraints).

            - **value** : float  
              Actual evaluated constraint value.

            - **satisfied** : bool  
              Whether the constraint is satisfied (based on `tol`).

            Optionally, each dictionary may also include:

            - **normalized_mismatch** : float  
              Mismatch scaled by the normalization factor.

            - **mismatch** : float  
              Raw constraint violation (value - target).

            - **normalize** : float  
              Normalization factor (if defined).

        Raises
        ------
        TypeError
            If `self.problem.constraint_data` exists but is not a list of dictionaries,
            or if required keys are missing from any entry.
        """
        # Use cached constraint data if available and valid
        if hasattr(self.problem, "constraint_data"):
            if not isinstance(self.problem.constraint_data, list) or not all(
                isinstance(d, dict) for d in self.problem.constraint_data
            ):
                raise TypeError("Expected self.constraint_data to be a list of dicts")
            required = {"name", "type", "target", "value", "satisfied"}
            allowed = required | {"normalized_mismatch", "mismatch", "normalize"}
            for entry in self.problem.constraint_data:
                validate_keys(entry, required, allowed)
            return self.problem.constraint_data

        # Otherwise evaluate constraints from scratch
        x_norm = np.asarray(x_norm)
        fitness = self.fitness(x_norm)  # Uses cache if available
        c_eq = fitness[1 : 1 + self.N_eq]
        c_ineq = fitness[1 + self.N_eq:]
        data = []

        # Equality constraints
        for i, val in enumerate(c_eq):
            data.append(
                {
                    "name": f"equality_constraint_{i}",
                    "type": "=",
                    "target": 0.0,
                    "value": val,
                    "satisfied": abs(val) <= tol,
                }
            )

        # Inequality constraints
        for i, val in enumerate(c_ineq):
            data.append(
                {
                    "name": f"inequality_constraint_{i}",
                    "type": "<",
                    "target": 0.0,
                    "value": val,
                    "satisfied": val <= tol,
                }
            )

        return data


    def evaluate_kkt_conditions(self, x_norm, tol):
        """
        Evaluate the raw quantities required for KKT condition checks.

        This method performs all necessary calculations to evaluate:

            - Lagrangian gradient
            - Constraint violations (equality and inequality)
            - Lagrange multipliers for active constraints
            - Complementary slackness products

        It does not apply any tolerance threshold; that is handled separately.

        Returns
        -------
        dict
            A dictionary containing raw values needed to assess the KKT conditions.
        """

        # Calculate problem values and gradients (possibily retrieving cached values)
        self.fitness(x_norm)
        self.gradient(x_norm)

        # Retrieve gradients and Jacobians from cache
        f_jac = self.cache["f_jac"]
        c_eq_jac = self.cache["c_eq_jac"]
        c_ineq_jac = self.cache["c_ineq_jac"]
        c_eq = self.cache["c_eq"]
        c_ineq = self.cache["c_ineq"]

        # Re-cast bounds as inequality constraints
        lb, ub = self.problem.get_bounds_normalized()
        lb, ub = np.asarray(lb), np.asarray(ub)
        c_lb, c_ub = lb - x_norm, x_norm - ub
        n = x_norm.size
        I = np.eye(n)
        jac_lb = -I  # ∇(lb_i - x_i) = -e_i
        jac_ub = +I  # ∇(x_i - ub_i) = +e_i
        c_ineq = np.hstack([c_ineq, c_lb, c_ub])
        c_ineq_jac = np.vstack([c_ineq_jac, jac_lb, jac_ub])

        # Identify active inequality constraints (c_i(x) ≈ 0)
        idx_active_ineq = [i for i, v in enumerate(c_ineq) if abs(v) <= tol]

        # Build full constraint Jacobian for active constraints
        c_jac_blocks = []
        if self.N_eq > 0:
            c_jac_blocks.append(c_eq_jac)
        if idx_active_ineq:
            c_jac_blocks.append(c_ineq_jac[idx_active_ineq])

        # Solve stationarity condition: grad(f) + J^T * mu = 0
        if c_jac_blocks:
            c_jac = np.vstack(c_jac_blocks)
            mu_vec, *_ = np.linalg.lstsq(c_jac.T, -f_jac, rcond=None)
        else:
            mu_vec = np.zeros(0)

        mu_eq = mu_vec[: self.N_eq] if self.N_eq > 0 else np.array([])
        mu_ineq = mu_vec[self.N_eq :] if idx_active_ineq else np.array([])

        # Compute gradient of Lagrangian
        lagrangian_gradient = f_jac.copy()
        if self.N_eq > 0:
            lagrangian_gradient += c_eq_jac.T @ mu_eq
        for idx, mu in zip(idx_active_ineq, mu_ineq):
            lagrangian_gradient += mu * c_ineq_jac[idx]

        # Compute slackness products mu_i * h_i(x)
        slack_products = [mu * c_ineq[i] for i, mu in zip(idx_active_ineq, mu_ineq)]

        # Save in dictionary
        kkt_data = {
            "lagrangian_gradient": lagrangian_gradient,
            "equality_violation": c_eq,
            "inequality_violation": c_ineq,
            "multipliers_eq": mu_eq,
            "multipliers_ineq": {i: mu for i, mu in zip(idx_active_ineq, mu_ineq)},
            "slack_products": slack_products,
        }

        # ------------------------------------------ #
        # ---------- Check KKT conditions ---------- #
        # ------------------------------------------ #

        # 1. First-order optimality (||∇L|| ≈ 0)
        grad_L = kkt_data["lagrangian_gradient"]
        first_order_val = np.linalg.norm(grad_L)
        first_order_ok = first_order_val <= tol

        # 2. Equality feasibility (|c_eq_i| <= tol)
        c_eq = kkt_data["equality_violation"]
        if self.N_eq > 0:
            feasibility_eq_val = np.max(np.abs(c_eq))
            feasibility_eq_ok = feasibility_eq_val <= tol
        else:
            feasibility_eq_val = 0.0
            feasibility_eq_ok = True

        # 3. Inequality feasibility (c_ineq_i <= 0)
        c_ineq = kkt_data["inequality_violation"]
        if self.N_ineq > 0:
            feasibility_ineq_val = np.max(np.maximum(c_ineq, 0.0))
            feasibility_ineq_ok = feasibility_ineq_val <= tol
        else:
            feasibility_ineq_val = 0.0
            feasibility_ineq_ok = True

        # 4. Dual feasibility (mu_ineq_i >= 0)
        mu_ineq_vals = list(kkt_data["multipliers_ineq"].values())
        if mu_ineq_vals:
            feasibility_dual_val = np.min(mu_ineq_vals)
            feasibility_dual_ok = feasibility_dual_val >= -tol
        else:
            feasibility_dual_val = 0.0
            feasibility_dual_ok = True

        # 5. Complementary slackness (mu_ineq_i * c_ineq_i ≈ 0)
        slack_products = kkt_data["slack_products"]
        if slack_products:
            complementary_slack_val = np.max(np.abs(slack_products))
            complementary_slack_ok = complementary_slack_val <= tol
        else:
            complementary_slack_val = 0.0
            complementary_slack_ok = True

        return {
            **kkt_data,
            "first_order_optimality": first_order_val,
            "feasibility_equality": feasibility_eq_val,
            "feasibility_inequality": feasibility_ineq_val,
            "feasibility_dual": feasibility_dual_val,
            "complementary_slackness": complementary_slack_val,
            "first_order_optimality_ok": first_order_ok,
            "feasibility_equality_ok": feasibility_eq_ok,
            "feasibility_inequality_ok": feasibility_ineq_ok,
            "feasibility_dual_ok": feasibility_dual_ok,
            "complementary_slackness_ok": complementary_slack_ok,
        }

    def make_kkt_optimality_report(self, x_norm, tol):
        r"""
        Generate a detailed KKT condition satisfaction report (80-character width).

        This report includes five key Karush-Kuhn-Tucker (KKT) checks:

        .. math::

            \|\nabla L(x, \lambda)\| \leq \mathrm{tol} \quad &\text{(First order optimality)} \\
            \max |c_{\text{eq}}(x)| \leq \mathrm{tol} \quad &\text{(Equality feasibility)} \\
            \max(0, c_{\text{ineq}}(x)) \leq \mathrm{tol} \quad &\text{(Inequality feasibility)} \\
            \min(\lambda_{\text{ineq}}) \geq 0 \quad &\text{(Dual feasibility)} \\
            \max |\lambda_i \cdot c_i| \leq \mathrm{tol} \quad &\text{(Complementary slackness)}

        For each condition, the report shows:

            - Actual computed value
            - Comparison direction and target (tolerance or 0)
            - Satisfaction status

        Parameters
        ----------
        x_norm : array-like
            Normalized decision variable vector to evaluate the KKT conditions at.
        tol : float
            Numerical tolerance used for comparisons in optimality and feasibility checks.

        Returns
        -------
        str
            A formatted report string summarizing KKT condition satisfaction.
        """
        # Compute KKT conditions
        kkt_data = self.evaluate_kkt_conditions(x_norm, tol)

        # Build report lines
        sep = "-" * 80
        lines = [
            "",
            sep,
            f" {'Karush-Kuhn-Tucker (KKT) conditions':<36}{'Value':>16}{'Target':>16}{'Ok?':>10}",
            sep,
        ]

        entries = [
            (
                "First order optimality",
                kkt_data["first_order_optimality"],
                f"< {tol:+.3e}",
                kkt_data["first_order_optimality_ok"],
            ),
            (
                "Equality feasibility",
                kkt_data["feasibility_equality"],
                f"< {tol:+.3e}",
                kkt_data["feasibility_equality_ok"],
            ),
            (
                "Inequality feasibility",
                kkt_data["feasibility_inequality"],
                f"< {tol:+.3e}",
                kkt_data["feasibility_inequality_ok"],
            ),
            (
                "Dual feasibility",
                kkt_data["feasibility_dual"],
                f"> {-tol:+.3e}",
                kkt_data["feasibility_dual_ok"],
            ),
            (
                "Complementary slackness",
                kkt_data["complementary_slackness"],
                f"< {tol:+.3e}",
                kkt_data["complementary_slackness_ok"],
            ),
        ]

        for name, val, target, ok in entries:
            status = "yes" if ok else " no"
            lines.append(f" {name:<36}{val:>+16.3e}{target:>16}{status:>10}")

        lines.append(sep)
        return "\n".join(lines)
    

    # def make_constraint_report(self, x_norm, tol):
    #     """
    #     Generate a compact constraint report that includes:
    #     - Constraint value and target
    #     - Satisfaction status
    #     - Lagrange multiplier (if active), otherwise 'inactive'

    #     This merges value and multiplier reports for better readability.

    #     Parameters
    #     ----------
    #     x_norm : array-like
    #         Normalized design variable vector.
    #     tol : float
    #         Tolerance used for satisfaction and KKT activeness checks.

    #     Returns
    #     -------
    #     str
    #         A formatted string with the constraint summary.
    #     """
    #     max_name_width = 35
    #     constraint_data = self.get_constraint_data(x_norm, tol)
    #     kkt_data = self.evaluate_kkt_conditions(x_norm, tol)
    #     active_multipliers = kkt_data["multipliers_ineq"]
    #     multipliers_eq = kkt_data["multipliers_eq"]

    #     sep = "-" * 80
    #     lines = [
    #         "",
    #         sep,
    #         " Constraint summary report",
    #         sep,
    #         f" {'Constraint name':<{max_name_width}}{'Value':>12}{'Target':>13}{'Ok?':>6}{'Multiplier':>12}",
    #         sep,
    #     ]

    #     eq_count = 0
    #     ineq_count = 0
    #     var_names = self.problem.variable_names
    #     n_vars = len(var_names)
    #     n_ineq = self.problem.get_nic()

    #     for i, entry in enumerate(constraint_data):
    #         name = entry.get("name", "")
    #         ctype = entry.get("type", "")
    #         value = entry.get("value", 0.0)
    #         target = entry.get("target", 0.0)
    #         satisfied = "yes" if entry.get("satisfied", False) else " no"

    #         if len(name) > max_name_width:
    #             name = "..." + name[-(max_name_width - 3):]

    #         # Determine the target symbol
    #         symbol = "=" if ctype == "=" else "<"
    #         target_str = f"{symbol} {target:+.3e}"
    #         value_str = f"{value:+.3e}"

    #         # Multiplier
    #         if ctype == "=":
    #             multiplier = f"{multipliers_eq[eq_count]:+.3e}"
    #             eq_count += 1
    #         elif ctype == "<":
    #             # Inequality constraint or bounds
    #             if ineq_count in active_multipliers:
    #                 multiplier = f"{active_multipliers[ineq_count]:+.3e}"
    #             else:
    #                 multiplier = "inactive"
    #             ineq_count += 1
    #         else:
    #             multiplier = "---"

    #         lines.append(f" {name:<{max_name_width}}{value_str:>12}{target_str:>13}{satisfied:>6}{multiplier:>12}")

    #     lines.append(sep)
    #     return "\n".join(lines)


    # def make_constraint_report(self, x_norm, tol):
    #     """
    #     Generate a compact constraint report that includes:
    #     - Constraint value and target
    #     - Satisfaction status
    #     - Lagrange multiplier (if active), otherwise 'inactive'

    #     Supports equality, less-than, and greater-than constraints, including bounds.

    #     Parameters
    #     ----------
    #     x_norm : array-like
    #         Normalized design variable vector.
    #     tol : float
    #         Tolerance used for constraint satisfaction and activeness checks.

    #     Returns
    #     -------
    #     str
    #         A formatted string with the constraint summary.
    #     """
    #     max_name_width = 35
    #     var_names = self.problem.variable_names
    #     n_vars = len(var_names)

    #     constraint_data = self.get_constraint_data(x_norm, tol)
    #     kkt_data = self.evaluate_kkt_conditions(x_norm, tol)
    #     active_multipliers = kkt_data["multipliers_ineq"]
    #     multipliers_eq = kkt_data["multipliers_eq"]

    #     # Append bound constraints with natural inequality direction
    #     lb, ub = self.problem.get_bounds_normalized()
    #     bounds_lower, bounds_upper = [], []
    #     for i, (name, xi, lb_i, ub_i) in enumerate(zip(var_names, x_norm, lb, ub)):
    #         if lb_i is not None:
    #             bounds_lower.append({
    #                 "name": name,
    #                 "type": ">",
    #                 "value": xi,
    #                 "target": lb_i,
    #                 "satisfied": (xi - lb_i >= -tol)  # xi ≥ lb_i
    #             })
    #         if ub_i is not None:
    #             bounds_upper.append({
    #                 "name": name,
    #                 "type": "<",
    #                 "value": xi,
    #                 "target": ub_i,
    #                 "satisfied": (ub_i - xi >= -tol)  # xi ≤ ub_i
    #             })

    #     # Then concatenate in order:
    #     constraint_data = bounds_lower + bounds_upper + constraint_data
    #     print(constraint_data)


    #     # Build the report
    #     sep = "-" * 80
    #     lines = [
    #         "",
    #         sep,
    #         f" {'Constraint summary report':<{max_name_width}}{'Value':>12}{'Target':>13}{'Ok?':>6}{'Multiplier':>12}",
    #         sep,
    #     ]

    #     eq_count = 0
    #     ineq_count = 0
    #     for entry in constraint_data:
    #         name = entry.get("name", "")
    #         ctype = entry.get("type", "")
    #         value = entry.get("value", 0.0)
    #         target = entry.get("target", 0.0)
    #         satisfied = "yes" if entry.get("satisfied", False) else " no"

    #         if len(name) > max_name_width:
    #             name = "..." + name[-(max_name_width - 3):]

    #         # Set the symbol for the target constraint
    #         if ctype == "=":
    #             symbol = "="
    #         elif ctype == "<":
    #             symbol = "<"
    #         elif ctype == ">":
    #             symbol = ">"
    #         else:
    #             symbol = "?"

    #         value_str = f"{value:+.3e}"
    #         target_str = f"{symbol} {target:+.3e}"

    #         # Handle multipliers
    #         if ctype == "=":
    #             multiplier = f"{multipliers_eq[eq_count]:+.3e}"
    #             eq_count += 1
    #         elif ctype in {"<", ">"}:
    #             if ineq_count in active_multipliers:
    #                 multiplier = f"{active_multipliers[ineq_count]:+.3e}"
    #             else:
    #                 multiplier = "inactive"
    #             ineq_count += 1
    #         else:
    #             multiplier = "---"

    #         lines.append(f" {name:<{max_name_width}}{value_str:>12}{target_str:>13}{satisfied:>6}{multiplier:>12}")

    #     lines.append(sep)
    #     return "\n".join(lines)


    def make_constraint_report(self, x_norm, tol):
        """
        generate a formatted constraint report at x_phys, using get_constraint_data to build/validate the entries.
        """
        max_name_width = 48
        constraint_data = self.get_constraint_data(x_norm, tol)

        # from .pysolver_utilities import print_dict
        # print("Data dictionary")
        # for item in constraint_data:
        #     print()
        #     print_dict(item)

        lines = []
        lines = []
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"{' Optimization constraints report':<80}")
        lines.append("-" * 80)
        lines.append(f"{' Constraint name':<49}{'Value':>11}{'Target':>13}{'Ok?':>6}")
        lines.append("-" * 80)

        for entry in constraint_data:
            name = entry.get("name", "")
            ctype = entry.get("type", "")
            target = entry.get("target", 0.0)
            value = entry.get("value", 0.0)
            satisfied = "yes" if entry.get("satisfied", False) else "no"

            if len(name) > max_name_width:
                name = "..." + name[-(max_name_width - 3) :]

            symbol_target = f"{ctype} {target:+.2e}"
            lines.append(f" {name:<48}{value:>+11.2e}{symbol_target:>13}{satisfied:>6}")

        lines.append("-" * 80)
        return "\n".join(lines)


    def make_lagrange_multipliers_report(self, x_norm, tol):
        """
        Generate a report of all Lagrange multipliers:
        - Equalities: always included
        - Inequalities and bounds: show value if active, 'inactive' otherwise

        Returns
        -------
        str
            The formatted multipliers report as a single string.
        """
        # Compute KKT data with given tolerance
        kkt_data = self.evaluate_kkt_conditions(x_norm, tol)
        constraint_data = self.get_constraint_data(x_norm, tol)
        eq_data = [c for c in constraint_data if c["type"] == "="]
        ineq_data = [c for c in constraint_data if c["type"] == "<"]
        # TODO: add cases of ctype ">"?

        # Print Lagrange multipliers in tabular format
        sep = "-" * 80
        max_name_width = 48
        lines = [
            "",
            sep,
            " Lagrange multipliers report",
            sep,
            f" {'Multiplier for constraint':<{max_name_width}}{'Type':>15}{'Value':>15}",
            sep,
        ]

        # 1. Equality constraints
        for i, entry in enumerate(eq_data):
            name = entry["name"]
            ctype = "equality"
            lam = kkt_data["multipliers_eq"][i]
            valstr = f"{lam:+.3e}"
            if len(name) > max_name_width:
                name = "..." + name[-(max_name_width - 3) :]
            lines.append(f" {name:<{max_name_width}}{ctype:>15}{valstr:>15}")

        # 2. Inequality + bound constraints
        var_names = self.problem.variable_names
        n_vars = len(var_names)
        n_ineq = len(ineq_data)
        total_ineq = n_ineq + 2 * n_vars
        active = set(kkt_data["multipliers_ineq"].keys())

        for j in range(total_ineq):
            if j < n_ineq:
                name = ineq_data[j]["name"]
                ctype = "inequality"
            elif j < n_ineq + n_vars:
                idx = j - n_ineq
                name = var_names[idx]
                ctype = "lower bound"
            else:
                idx = j - n_ineq - n_vars
                name = var_names[idx]
                ctype = "upper bound"

            if len(name) > max_name_width:
                name = "..." + name[-(max_name_width - 3) :]

            if j in active:
                valstr = f"{kkt_data['multipliers_ineq'][j]:+.3e}"
            else:
                valstr = "inactive"

            lines.append(f" {name:<{max_name_width}}{ctype:>15}{valstr:>15}")

        lines.append(sep)
        return "\n".join(lines)


class OptimizationProblem(ABC):
    """
    Abstract base class for optimization problems.

    Derived optimization problem objects must implement the following methods:

    - `fitness`: Evaluate the objective function and constraints for a given set of decision variables.
    - `get_bounds`: Get the bounds for each decision variable.
    - `get_neq`: Return the number of equality constraints associated with the problem.
    - `get_nineq`: Return the number of inequality constraints associated with the problem.

    Additionally, specific problem classes can define the `gradient` method to compute the Jacobians. If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Methods
    -------
    fitness(x)
        Evaluate the objective function and constraints for a given set of decision variables.
    get_bounds()
        Get the bounds for each decision variable.
    get_neq()
        Return the number of equality constraints associated with the problem.
    get_nineq()
        Return the number of inequality constraints associated with the problem.

    Parameters
    ----------
    problem_scale : float, optional
        Scaling factor for normalization. Default is 1.0.
    variable_names : list of str, optional
        Names of the decision variables. Used for reporting and debugging.

    """

    @abstractmethod
    def fitness(self, x):
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
        Get the bounds for each decision variable (Pygmo format)

        Returns
        -------
        bounds : tuple of lists
            A tuple of two items where the first item is the list of lower bounds and the second
            item of the list of upper bounds for the vector of decision variables. For example,
            ([-2 -1], [2, 1]) indicates that the first decision variable has bounds between
            -2 and 2, and the second has bounds between -1 and 1.
        """
        pass

    @abstractmethod
    def get_nec(self):
        """
        Return the number of equality constraints associated with the problem.

        Returns
        -------
        neq : int
            Number of equality constraints.
        """
        pass

    @abstractmethod
    def get_nic(self):
        """
        Return the number of inequality constraints associated with the problem.

        Returns
        -------
        nineq : int
            Number of inequality constraints.
        """
        pass

    def get_bounds_normalized(self):
        """
        Return normalized bounds in [0, problem_scale]. If no scaling is applied (i.e., problem_scale is None),
        the physical bounds are returned instead.

        Returns
        -------
        tuple of lists
            Tuple (lb, ub) in normalized space or the original physical bounds if problem_scale is None.
        """
        if self.problem_scale is None:
            return self.get_bounds()

        n = len(self.get_bounds()[0])
        return [0.0] * n, [self.problem_scale] * n

    def scale_physical_to_normalized(self, x_phys):
        """
        Convert physical design variable values to normalized values in the range [0, problem_scale].
        If self.problem_scale is None, no scaling is applied.

        The method uses the bounds returned by `self.get_bounds()` and the internal `self.problem_scale`.
        It automatically handles the case of fixed variables (i.e., upper bound = lower bound) by returning 0.0.

        Parameters
        ----------
        x_phys : array-like
            Physical values of the decision variables.

        Returns
        -------
        np.ndarray
            Normalized values in the range [0, problem_scale] if scaling is applied,
            otherwise the original physical values.
        """
        x_phys = np.asarray(x_phys)
        if self.problem_scale is None:
            return x_phys
        lb, ub = self.get_bounds()
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        x_scaled = []
        for xi, lbi, ubi in zip(x_phys, lb, ub):
            if ubi == lbi:
                x_scaled.append(0.0)
            else:
                x_scaled.append(self.problem_scale * (xi - lbi) / (ubi - lbi))

        return np.array(x_scaled)

    def scale_normalized_to_physical(self, x_norm):
        """
        Convert normalized values in the range [0, problem_scale] back to physical variable values.
        If self.problem_scale is None, no scaling is applied.

        The method uses the bounds returned by `self.get_bounds()` and the internal `self.problem_scale`.

        Parameters
        ----------
        x_norm : array-like
            Normalized values of the decision variables.

        Returns
        -------
        np.ndarray
            Physical values corresponding to the normalized input, or the original values if no scaling.
        """
        x_norm = np.asarray(x_norm)
        if self.problem_scale is None:
            return x_norm
        lb, ub = self.get_bounds()
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        return lb + (ub - lb) * (x_norm / self.problem_scale)

    def fitness_normalized_input(self, x_norm):
        """
        Evaluate fitness starting from normalized input.

        Parameters
        ----------
        x_norm : array-like
            Normalized input vector.

        Returns
        -------
        array_like
            Output of `fitness` evaluated on the corresponding physical vector.
        """
        x_phys = self.scale_normalized_to_physical(x_norm)
        return self.fitness(x_phys)

    def gradient_normalized_input(self, x_norm):
        """
        Compute the gradient of the objective and constraints with respect to normalized variables.

        This function applies the chain rule to convert the gradient computed in the physical space
        to the corresponding gradient in the normalized space. If `problem_scale` is set to `None`,
        the problem is considered unscaled and the gradient is returned directly.

        Parameters
        ----------
        x_norm : array-like
            Normalized input vector (typically in [0, problem_scale]).

        Returns
        -------
        np.ndarray
            Gradient with respect to the normalized variables. The shape is:
            - (n,) for scalar objective or flat constraint vectors.
            - (m, n) for vector-valued constraints with m rows and n design variables.

        Notes
        -----
        The chain rule is applied as follows:

            x_phys = lb + (ub - lb) * (x_norm / problem_scale)
            ∇f(x_norm) = ∇f(x_phys) * d(x_phys)/d(x_norm)

        where:

            d(x_phys)/d(x_norm) = (ub - lb) / problem_scale    [elementwise]

        This correction ensures that exact user-defined gradients are consistent
        with the scaling applied during optimization.
        """
        x_phys = self.scale_normalized_to_physical(x_norm)
        grad_phys = self.gradient(x_phys)

        if self.problem_scale is None:
            return grad_phys

        lb, ub = self.get_bounds()
        lb, ub = np.asarray(lb), np.asarray(ub)
        scaling_factor = (ub - lb) / self.problem_scale

        # Apply scaling using broadcasting to support both 1D and 2D gradients
        return grad_phys * scaling_factor
    

    def clip_to_bounds(self, x_physical, logger=None):
        """
        Clip physical variable values to lie within specified bounds.

        Parameters
        ----------
        x_physical : array-like
            Input vector in physical space.
        logger : logging.Logger or None, optional
            Logger for outputting warnings. If None, warnings are printed to standard output.

        Returns
        -------
        np.ndarray
            Clipped vector.
        """
        lb, ub = self.get_bounds()
        x_physical = np.atleast_1d(x_physical)
        x_clipped = np.clip(x_physical, lb, ub)

        for i, (orig, new, low, high) in enumerate(zip(x_physical, x_clipped, lb, ub)):
            if orig != new:
                msg = (
                    "-" * 80 + "\n"
                    f"Warning: optimization variable out of bounds\n"
                    f"  Name       : {self.variable_names[i]}\n"
                    f"  Original   : {orig:.3e}\n"
                    f"  Bounds     : [{low:.3e}, {high:.3e}]\n"
                    f"  Clipped to : {new:.3e}\n"
                    + "-" * 80
                )
                if logger:
                    for line in msg.splitlines():
                        logger.warning(line)
                else:
                    print(msg)

        return x_clipped

def count_constraints(var):
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
    >>> count_constraints(None)
    0

    >>> count_constraints(5.0)
    1

    >>> count_constraints([1.0, 2.0, 3.0])
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


def combine_objective_and_constraints(f, c_eq=None, c_ineq=None):
    """
    Combine an objective function with its associated equality and inequality constraints.

    This function takes in an objective function value, a set of equality constraints,
    and a set of inequality constraints. It then returns a combined Numpy array of
    these values. The constraints can be given as a list, tuple, numpy array, or as
    individual values.

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

    # Validate objective function value
    if isinstance(f, (list, tuple, np.ndarray)):
        if len(f) != 1:
            raise ValueError(
                "Objective function value 'f' must be a scalar or single-element array."
            )
        f = f[0]  # Unwrap the single element to ensure it's treated as a scalar

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

    return np.array(combined_list)


class _PygmoProblem:
    """
    A wrapper class for optimization problems to be compatible with Pygmo's need for deep-copiable problems.
    This class uses anonymous functions (lambda) to prevent issues with deep copying complex objects,
    (like Coolprop's AbstractState objects) which are not deep-copiable.
    """

    def __init__(self, wrapped_problem):
        # Pygmo requires a flattened Jacobian for gradients, unlike SciPy's two-dimensional array.
        self.fitness = lambda x: wrapped_problem.fitness(x)
        self.gradient = lambda x: wrapped_problem.gradient(x).flatten()

        # Directly link bounds and constraint counts from the original problem.
        self.get_bounds = lambda: wrapped_problem.problem.get_bounds_normalized()
        self.get_nec = lambda: wrapped_problem.problem.get_nec()
        self.get_nic = lambda: wrapped_problem.problem.get_nic()

        # If the original problem defines Hessians, provide them as well.
        if hasattr(wrapped_problem.problem, "hessians"):
            self.hessians = lambda x: wrapped_problem.problem.hessians(x)

        # Define anonymous functions for objective and constraints with their Jacobians.
        self.f = lambda x: wrapped_problem.fitness(x)[0]
        self.c_eq = lambda x: wrapped_problem.fitness(x)[1 : 1 + self.get_nec()]
        self.c_ineq = lambda x: wrapped_problem.fitness(x)[1 + self.get_nec() :]

        self.f_jac = lambda x: wrapped_problem.gradient(x)[0, :]
        self.c_eq_jac = lambda x: wrapped_problem.gradient(x)[1 : 1 + self.get_nec(), :]
        self.c_ineq_jac = lambda x: wrapped_problem.gradient(x)[1 + self.get_nec() :, :]
