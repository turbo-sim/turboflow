import os
import time
import copy
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative
from abc import ABC, abstractmethod
from datetime import datetime

SOLVER_OPTIONS = ["hybr", "lm"]


class NonlinearSystemSolver:
    r"""
    Solver class for nonlinear systems of equations.

    The solver is designed to handle system of nonlinear equations of the form:

    .. math::

        F(x) = 0

    where :math:`F: \mathbb{R}^n \rightarrow \mathbb{R}^n` is a vector-valued
    function of the vector :math:`x`.

    The class interfaces with the `root` method from `scipy.optimize` to solve the equations and provides a structured framework for initialization, solution monitoring, and post-processing.

    Parameters
    ----------
    problem : NonlinearSystemProblem
        An instance of a problem defining the system of equations to be solved.
    x0 : array_like
        Initial guess for the independent variables.
    method : str, optional
        Method to be used by scipy's root for solving the nonlinear system. Available solvers are:

        - :code:`hybr`: Refers to MINPACK's 'hybrd' method. This is a modification of Powell's hybrid method and can be viewed as a quasi-Newton method. It is suitable for smooth functions and known to be robust, but can sometimes be slower than other methods.
        - :code:`lm`: The Levenberg-Marquardt method. Often used for least-squares problems, it blends the steepest descent and the Gauss-Newton method. This method can perform well for functions that are reasonably well-behaved.

        The choice between :code:`hybr` and :code:`lm` largely depends on the specifics of the problem at hand and the nature of the function. Both methods have their strengths and can be applicable in various contexts. It is advisable to understand the characteristics of the system being solved and experiment with both methods to determine the most appropriate choice for a given problem.
        Defaults to 'hybr'.
    tol : float, optional
        Tolerance for the solver termination. Defaults to 1e-9.
    max_iter : integer, optional
        Maximum number of function evaluations for the solver termination. Defaults to 100.
    options : dict, optional
        Additional options to be passed to scipy's root.
    derivative_method : str, optional
        Finite difference method to be used when the problem Jacobian is not provided. Defaults to '2-point'
    derivative_rel_step : float, optional
        Finite difference relative step size to be used when the problem Jacobian is not provided. Defaults to 1e-6
    display : bool, optional
        If True, displays the convergence progress. Defaults to True.
    plot : bool, optional
        If True, plots the convergence progress. Defaults to False.
    logger : logging.Logger, optional
        Logger object to which logging messages will be directed. Defaults to None.
    update_on : str, optional
        Specifies if the convergence report should be updated on a new function evaluations ("function") or on gradient evaluations ("gradient"). Defaults to "function".


    Methods
    -------
    solve(x0=None)
        Solve the system of nonlinear equations.
    get_residual_values(x)
        Evaluate the given nonlinear system problem.
    get_residual_jacobian(x)
        Evaluate the Jacobian of the system.
    print_convergence_history()
        Print the convergence history of the problem.
    plot_convergence_history()
        Plot the convergence history.

    """

    def __init__(
        self,
        problem,
        x0,
        method="hybr",
        tol=1e-9,
        max_iter=100,
        options={},
        derivative_method="2-point",
        derivative_rel_step=1e-6,
        display=True,
        plot=False,
        logger=None,
        update_on="function",
        callback_func=None,
    ):
        # Initialize class variables
        self.problem = problem
        self.display = display
        self.plot = plot
        self.logger = logger
        self.method = method
        self.tol = tol
        self.maxiter = max_iter
        self.options = copy.deepcopy(options) if options else {}
        self.derivative_method = derivative_method
        self.derivative_rel_step = derivative_rel_step
        self.callback_func = callback_func
        self.callback_func_call_count = 0

        # Define the maximun number of iterations
        if method == "hybr":
            self.options["maxfev"] = self.maxiter
        elif method == "lm":
            self.options["maxiter"] = self.maxiter
        else:
            raise ValueError(
                f"Invalid solver. Available options: {', '.join(SOLVER_OPTIONS)}"
            )

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

        # Initialize problem
        self.x0 = x0
        eval = self.problem.get_residual_values(self.x0)
        if np.any([item is None for item in eval]):
            raise ValueError(
                "Problem evaluation contains None values. There may be an issue with the problem object provided."
            )

        # Initialize variables for convergence report
        self.last_x = None
        self.last_residuals = None
        self.grad_count = 0
        self.func_count = 0
        self.func_count_tot = 0
        self.solution = None
        self.solution_report = []
        self.elapsed_time = None
        self.include_solution_in_footer = False
        self.convergence_history = {
            "grad_count": [],
            "func_count": [],
            "func_count_total": [],
            "norm_residual": [],
            "norm_step": [],
        }

        # Initialize convergence plot
        # (convergence_history must be initialized before)
        if self.plot:
            self._plot_callback(initialize=True)

    def solve(self, x0=None):
        """
        Solve the nonlinear system using scipy's root.

        Parameters
        ----------
        x0 : array-like, optional
            Initial guess for the solution of the nonlinear system. If not provided, the initial guess set during instantiation is used.

        Returns
        -------
        RootResults
            A result object from scipy's root detailing the outcome of the solution process.

        """

        # Start timing with high-resolution timer
        start_time = time.perf_counter()

        # Print report header
        self._write_header()

        # Solve the root finding problem
        self.x0 = self.x0 if x0 is None else x0
        self.solution = root(
            self.get_residual_values,
            self.x0,
            jac=self.get_residual_jacobian,
            method=self.method,
            tol=self.tol,
            options=self.options,
        )

        # Evaluate performance again for the converged solution
        self.get_residual_values(self.solution.x)

        # Calculate elapsed time
        self.elapsed_time = time.perf_counter() - start_time

        # Print report footer
        self._write_footer()

        return self.solution

    def get_residual_values(self, x, called_from_jac=False):
        """
        Evaluate the nonlinear system residuals.

        Parameters
        ----------
        x : array_like
            Independent variable vector.

        Returns
        -------
        array_like
            Residuals of the problem.
        """

        # Increase total counter (includes finite differences)
        self.func_count_tot += 1

        # Compute problem residuals
        self.last_residuals = self.problem.get_residual_values(x)

        # Update progress report
        if not called_from_jac:
            self.func_count += 1  # Does not include finite differences
            if self.update_on == "function":
                self._print_convergence_progress(x)

        return self.last_residuals

    def get_residual_jacobian(self, x):
        """
        Evaluates the Jacobian of the nonlinear system of equations at the specified point x.

        This method will use the `get_residual_jacobian` method of the NonlinearSystemProblem class if it exists.
        If the `get_residual_jacobian` method is not implemented the Jacobian is appoximated using forward finite differences.

        Parameters
        ----------
        x : array-like
            Vector of independent variables.

        Returns
        -------
        array-like
            Jacobian matrix of the residual vector.
        """

        # Evaluate the Jacobian of the residual vector
        self.grad_count += 1
        if hasattr(self.problem, "get_residual_jacobian"):
            # If the problem has its own Jacobian method, use it
            jacobian = self.problem.get_residual_jacobian(x)

        else:
            # Fall back to finite differences
            fun = lambda x: self.get_residual_values(x, called_from_jac=True)
            jacobian = approx_derivative(
                fun,
                x,
                f0=self.last_residuals,
                method=self.derivative_method,
                abs_step=self.derivative_rel_step * np.abs(x),
            )

        # Update progress report
        if self.update_on == "gradient":
            self._print_convergence_progress(x)

        return jacobian

    def _write_header(self):
        """
        Print a formatted header for the root finding report.

        This internal method is used to display a consistent header format at the
        beginning of the root finding process. The header includes columns for residual evaluations,
        gradient evaluations, norm of the residuals, and norm of the steps.
        """
        # Define header text
        initial_message = (
            f" Solve system of equations for {type(self.problem).__name__}"
        )
        self.header = f" {'Func-eval':>15}{'Grad-eval':>15}{'Norm of residual':>24}{'Norm of step':>24} "
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
        Print the current solution status and update convergence history.

        This method captures and prints the following metrics:
        - Number of function evaluations
        - Number of gradient evaluations
        - Two-norm of the residual vector
        - Two-norm of the update step

        The method also updates the stored convergence history for potential future analysis.

        Parameters
        ----------
        x : array-like
            The current solution (i.e., vector of independent variable values)

        Notes
        -----
        The norm of the update step is calculated as the two-norm of the difference
        between the current and the last independent variables. The norm of the residual
        vector is computed using the two-norm.
        """
        # Compute norm of residual vector
        residual = self.last_residuals
        norm_residual = np.linalg.norm(residual)
        # norm_residual = np.max([np.linalg.norm(residual), np.finfo(float).eps])

        # Compute the norm of the last step
        norm_step = np.linalg.norm(x - self.last_x) if self.last_x is not None else 0
        self.last_x = x.copy()

        # Store convergence status
        self.convergence_history["grad_count"].append(self.grad_count)
        self.convergence_history["func_count"].append(self.func_count)
        self.convergence_history["func_count_total"].append(self.func_count_tot)
        self.convergence_history["norm_residual"].append(norm_residual)
        self.convergence_history["norm_step"].append(norm_step)

        # Current convergence message
        status = f" {self.func_count:15d}{self.grad_count:15d}{norm_residual:24.6e}{norm_step:24.6e} "

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

        # Evaluate callback function
        if self.callback_func:
            self.callback_func_call_count += 1
            self.callback_func(x, self.callback_func_call_count)

    def _write_footer(self):
        """
        Print a formatted footer for the root finding report.

        This method displays the final root finding result, including the
        exit message, success status, and decision variables.

        Notes
        -----
        The footer's structure is intended to match the header's style,
        providing a consistent look to the root finding report.
        """
        # Define footer text
        separator = "-" * len(self.header)
        exit_message = f"Exit message: {self.solution.message}"
        success_status = f"Success: {self.solution.success}"
        time_message = f"Solution time: {self.elapsed_time:.3f} seconds"
        solution_header = "Solution:"
        solution_vars = [f"   x{i} = {x:+6e}" for i, x in enumerate(self.solution.x)]
        lines_to_output = [separator, success_status, exit_message, time_message]
        if self.include_solution_in_footer:
            lines_to_output += [solution_header]
            lines_to_output += solution_vars
        lines_to_output += [separator, ""]

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
        the residual vector as a function of number of iterations.

        Note:
            This is an internal method, meant to be called within the optimization process.
        """

        # Initialize figure before first iteration
        if initialize:
            self.fig, self.ax = plt.subplots()
            (self.obj_line,) = self.ax.plot([], [], color="#D95319", marker="o")
            self.ax.set_xlabel("Number of iterations")
            self.ax.set_ylabel("Two-norm of the residual vector")
            self.ax.set_yscale("log")  # Set y-axis to logarithmic scale
            self.ax.xaxis.set_major_locator(MultipleLocator(1))
            # self.ax.legend(loc="upper right")
            self.fig.tight_layout(pad=1)

        # Update plot data with current values
        iteration = (
            self.convergence_history["func_count"]
            if self.update_on == "function"
            else self.convergence_history["grad_count"]
        )
        residual = self.convergence_history["norm_residual"]
        self.obj_line.set_xdata(iteration)
        self.obj_line.set_ydata(residual)

        # Adjust the plot limits
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw the plot
        plt.draw()
        plt.pause(0.01)  # small pause to allow for update

    def print_convergence_history(
        self, savefile=False, filename=None, output_dir="output"
    ):
        """
        Print the convergence history of the problem.

        The convergence history includes:
            - Number of function evaluations
            - Number of function evaluations (including finite differences)
            - Number of gradient evaluations
            - Two-norm of the residual vector
            - Two-norm of the update step

        The method provides a comprehensive report on the final solution, including:
            - Exit message
            - Success status
            - Final independent variables values


        """
        if self.solution:
            if savefile:
                # Create figures directory if it does not exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Give a name to the file if it is not specified
                if filename is None:
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = os.path.join(
                        output_dir, f"convergence_history_{current_time}.txt"
                    )
                else:
                    filename = os.path.join(output_dir, f"{filename}.txt")

                # Save plots
                with open(filename, "w") as f:
                    f.write("\n".join(self.solution_report))

            else:
                for line in self.solution_report:
                    print(line)

        else:
            warnings.warn(
                "This method should be used after invoking the 'solve()' method."
            )

    def plot_convergence_history(
        self, savefig=True, name=None, use_datetime=False, path=None
    ):
        """
        Plot the convergence history of the problem.

        This method plots the two norm of the residual vector versus the number of iterations

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


class NonlinearSystemProblem(ABC):
    """
    Abstract base class for root-finding problems.

    Derived root-finding problem objects must implement the following method:

    - `get_residual_values`: Evaluate the system of equations for a given set of decision variables.

    Additionally, specific problem classes can define the `get_residual_jacobian` method to compute the Jacobians. If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Methods
    -------
    get_residual_values(x)
        Evaluate the system of equations for a given set of decision variables.

    Examples
    --------
    Here's an example of how to derive from `RootFindingProblem`::

        class MyRootFindingProblem(RootFindingProblem):
            def get_residual_values(self, x):
                # Implement evaluation logic here
                pass
    """

    @abstractmethod
    def get_residual_values(self, x):
        """
        Evaluate the system of equations for given decision variables.

        Parameters
        ----------
        x : array-like
            Vector of decision variables.

        Returns
        -------
        array_like
            Vector containing the values of the system of equations for the given decision variables.
        """
        pass
