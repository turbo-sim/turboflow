import os
import time
import copy
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from abc import ABC, abstractmethod
from matplotlib.ticker import MultipleLocator
from scipy.optimize import root

from . import numerical_differentiation
from .pysolver_utilities import savefig_in_formats

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
    method : str, optional
        Method to be used by scipy's root for solving the nonlinear system. Available solvers are:

        - :code:`hybr`:  Uses MINPACK's 'hybrd' method, which is is a modification of Powell's hybrid method.
        - :code:`lm`: The Levenberg-Marquardt method, which blends the steepest descent and the Gauss-Newton methods.

        The choice between :code:`hybr` and :code:`lm` largely depends on the specifics of the problem at hand.
        The :code:`hybr` usually requires less gradient evaluations and it is often faster when analytic gradients are not available.
        It is advisable to experiment with both methods to determine the most appropriate choice for a given problem.

        Defaults to 'hybr'.
    tol : float, optional
        Tolerance for the solver termination. Defaults to 1e-9.
    max_iter : integer, optional
        Maximum number of function evaluations for the solver termination. Defaults to 100.
    options : dict, optional
        Additional options to be passed to scipy's root.
    derivative_method : str, optional
        Finite difference method to be used when the problem Jacobian is not provided. Defaults to '2-point'
    derivative_abs_step : float, optional
        Finite difference absolute step size to be used when the problem Jacobian is not provided. Defaults to 1e-6
    print_convergence : bool, optional
        If True, displays the convergence progress. Defaults to True.
    plot_convergence : bool, optional
        If True, plots the convergence progress. Defaults to False.
    logger : logging.Logger, optional
        Logger object to which logging messages will be directed. Defaults to None.
    update_on : str, optional
        Specifies if the convergence report should be updated on a new function evaluations ("function") or on gradient evaluations ("gradient"). Defaults to "function".


    Methods
    -------
    solve(x0)
        Solve the system of nonlinear equations using the specified initial guess `x0`.
    residual(x)
        Evaluate the vector of residuals of the at a given point `x`.
    gradient(x)
        Evaluate the Jacobian of the system at a given point `x`.
    print_convergence_history()
        Print the convergence history of the problem.
    plot_convergence_history()
        Plot the convergence history.

    """

    def __init__(
        self,
        problem,
        method="hybr",
        tolerance=1e-6,
        max_iterations=100,
        options={},
        derivative_method="2-point",
        derivative_abs_step=1e-6,
        print_convergence=True,
        plot_convergence=False,
        plot_scale="log",
        logger=None,
        update_on="function",
        callback_func=None,
    ):
        # Initialize class variables
        self.problem = problem
        self.display = print_convergence
        self.plot = plot_convergence
        self.plot_scale = plot_scale
        self.logger = logger
        self.method = method
        self.tol = tolerance
        self.maxiter = max_iterations
        self.options = copy.deepcopy(options) if options else {}
        self.derivative_method = derivative_method
        self.derivative_abs_step = derivative_abs_step
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

        # Initialize variables for convergence report
        self.f_final = None
        self.x_final = None
        self.x_last = None
        self.residuals_last = None
        self.grad_count = 0
        self.func_count = 0
        self.func_count_tot = 0
        self.success = None
        self.message = None
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
        # TODO ??? (convergence_history must be initialized before)
        if self.plot:
            self._plot_callback(initialize=True)

    def solve(self, x0):
        """
        Solve the nonlinear system using the specified solver.

        Parameters
        ----------
        x0 : array-like
            Initial guess for the solution of the nonlinear system.

        Returns
        -------
        RootResults
            A result object from scipy's root detailing the outcome of the solution process.

        """
        # Get start datetime
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Start timing with high-resolution timer
        start_time = time.perf_counter()

        # Print report header
        self._write_header()

        # Solve the root finding problem
        solution = root(
            self.residual,
            x0,
            jac=self.gradient,
            method=self.method,
            tol=self.tol,
            options=self.options,
        )

        # Save solution
        self.x_final = copy.deepcopy(self.x_last)
        self.f_final = self.residual(self.x_final)
        self.success = solution.success
        self.message = solution.message

        # Check if solver actually converged
        norm_residual = np.linalg.norm(self.f_final)
        if self.success and norm_residual > 10*self.tol:
            self.success = False
            self.message = f"The equation solver returned a success exit flag.\nHowever, the two-norm of the final residual is higher than tolerance ({norm_residual:0.2e}>{self.tol:0.2e})"

        # Calculate elapsed time
        self.elapsed_time = time.perf_counter() - start_time

        # Print report footer
        self._print_convergence_progress(self.x_final)
        self._write_footer()

        return self.x_final

    def residual(self, x, called_from_jac=False):
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
        self.residuals_last = self.problem.residual(x)

        # Update progress report
        if not called_from_jac:
            self.func_count += 1  # Does not include finite differences
            if self.update_on == "function":
                self._print_convergence_progress(x)

        return self.residuals_last

    def gradient(self, x):
        """
        Evaluates the Jacobian of the nonlinear system of equations at the specified point x.

        This method will use the `gradient` method of the NonlinearSystemProblem class if it exists.
        If the `gradient` method is not implemented the Jacobian is appoximated using forward finite differences.

        Parameters
        ----------
        x : array-like
            Vector of independent variables.

        Returns
        -------
        array-like
            Jacobian matrix of the residual vector formatted as a 2D array.
        """

        # Evaluate the Jacobian of the residual vector
        self.grad_count += 1
        if hasattr(self.problem, "gradient"):
            # If the problem has its own Jacobian method, use it
            jacobian = self.problem.gradient(x)

        else:
            # Fall back to finite differences
            fun = lambda x: self.residual(x, called_from_jac=True)
            jacobian = numerical_differentiation.approx_gradient(
                fun,
                x,
                f0=self.residuals_last,
                method=self.derivative_method,
                abs_step=self.derivative_abs_step,
            )

        # Update progress report
        if self.update_on == "gradient":
            self._print_convergence_progress(x)

        return np.atleast_1d(jacobian)

    def _handle_output(self, message):
        """
        Handles output by printing to the screen or logging it.

        Parameters
        ----------
        message : str
            The message to output.
        log_only : bool, optional
            If True, only logs the message (ignores `self.display`). Default is False.
        """
        if self.logger:
            for line in message.splitlines():
                self.logger.info(line)
                
        if self.display and not self.logger:
            print(message)

    def _write_header(self):
        """
        Print a formatted header for the root finding report.

        This internal method is used to display a consistent header format at the
        beginning of the root finding process. The header includes columns for residual evaluations,
        gradient evaluations, norm of the residuals, and norm of the steps.
        """
        # Define header text
        initial_message = (
            f" Solve system of equations for {type(self.problem).__name__}\n"
            f" Root-finding algorithm employed: {self.method}"
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

        # Print or log content
        for line in lines_to_output:
            self._handle_output(line)

        # Store text in memory
        self.solution_report.extend(lines_to_output)

    def _print_convergence_progress(self, x):
        """
        Print the current solution status and update convergence history.

        This method captures and prints the following metrics:
        - Number of function evaluations
        - Number of gradient evaluations
        - Two-norm of residual vector
        - Two-norm of solution step

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
        residual = self.residuals_last
        norm_residual = np.linalg.norm(residual)
        # norm_residual = np.max([np.linalg.norm(residual), np.finfo(float).eps])

        # Compute the norm of the last step
        norm_step = np.linalg.norm(x - self.x_last) if self.x_last is not None else 0
        self.x_last = x.copy()

        # Store convergence status
        self.convergence_history["grad_count"].append(self.grad_count)
        self.convergence_history["func_count"].append(self.func_count)
        self.convergence_history["func_count_total"].append(self.func_count_tot)
        self.convergence_history["norm_residual"].append(norm_residual)
        self.convergence_history["norm_step"].append(norm_step)

        # Current convergence message
        status = f" {self.func_count:15d}{self.grad_count:15d}{norm_residual:24.6e}{norm_step:24.6e} "
        self._handle_output(status)

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
        exit_message = f"Exit message: {self.message}"
        success_status = f"Success: {self.success}"
        time_message = f"Solution time: {self.elapsed_time:.3f} seconds"
        solution_header = "Solution:"
        solution_vars = [f"   x{i} = {x:+6e}" for i, x in enumerate(self.x_final)]
        lines_to_output = [separator, success_status, exit_message, time_message]
        if self.include_solution_in_footer:
            lines_to_output += [solution_header]
            lines_to_output += solution_vars
        lines_to_output += [separator, ""]

        # Print or log content
        for line in lines_to_output:
            self._handle_output(line)
            
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
            self.ax.set_ylabel("Two-norm of residual vector")
            self.ax.set_yscale(self.plot_scale)
            # self.ax.xaxis.set_major_locator(MultipleLocator(1))
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
            - Number of gradient evaluations
            - Two-norm of the residual vector
            - Two-norm of the solution step

        The method provides a detailed report on:
            - Exit message
            - Success status
            - Execution time

        This method should be called only after the optimization problem has been solved, as it relies on data generated by the solving process.

        Parameters
        ----------
        savefile : bool, optional
            If True, the convergence history will be saved to a file, otherwise printed to standard output. Default is False.
        filename : str, optional
            The name of the file to save the convergence history. If not specified, the filename is automatically generated
            using the problem name and the start datetime. The file extension is not required.
        output_dir : str, optional
            The directory where the plot file will be saved if savefile is True. Default is "output".

        Raises
        ------
        ValueError
            If this method is called before the problem has been solved.

        """
        if self.x_final is not None:
            if savefile:
                # Create output directory if it does not exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Give a name to the file if it is not specified
                if filename is None:
                    filename = f"convergence_{type(self.problem).__name__}_{self.start_datetime}.txt"

                # Write report to file
                fullfile = os.path.join(output_dir, filename)
                with open(fullfile, "w") as f:
                    f.write("\n".join(self.solution_report))

            else:
                for line in self.solution_report:
                    print(line)

        else:
            raise ValueError(
                "This method can only be used after invoking the 'solve()' method."
            )

    def plot_convergence_history(
        self, savefile=False, filename=None, output_dir="output"
    ):
        """
        Plot the convergence history of the problem as the two-norm of the residual vector versus the number of iterations.

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
        if self.x_final is not None:
            self._plot_callback(initialize=True)
        else:
            raise ValueError(
                "This method can only be used after invoking the 'solve()' method."
            )

        if savefile:
            # Create output directory if it does not exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Give a name to the file if it is not specified
            if filename is None:
                filename = (
                    f"convergence_{type(self.problem).__name__}_{self.start_datetime}"
                )

            # Save plots
            fullfile = os.path.join(output_dir, filename)
            savefig_in_formats(self.fig, fullfile, formats=[".png", ".svg"])

        return self.fig


class NonlinearSystemProblem(ABC):
    """
    Abstract base class for root-finding problems.

    Derived root-finding problem objects must implement the following method:

    - `residual`: Evaluate the system of equations for a given set of decision variables.

    Additionally, specific problem classes can define the `gradient` method to compute the Jacobians. If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Methods
    -------
    residual(x)
        Evaluate the system of equations for a given set of decision variables.

    Examples
    --------
    Here's an example of how to derive from `RootFindingProblem`::

        class MyRootFindingProblem(RootFindingProblem):
            def residual(self, x):
                # Implement evaluation logic here
                pass
    """

    @abstractmethod
    def residual(self, x):
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
