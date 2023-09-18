import os
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative

from turbo_functions import set_plot_options
set_plot_options(grid=True)


def log_message(message, display=True):
    """Write message to file and print to console if display=True"""
    # logger.info(message)
    if display:
        print(message)


def solve_nonlinear_system(fun, x0, args=(), method='hybr', tol=None, options=None, display=True):
    """
    Solve a system of nonlinear equations and display the convergence progress.

    Parameters:
    -----------
    fun : callable
        The function to solve. Should accept an array of variables as the first argument and return 
        an array of residuals. Additional arguments can be passed using the `args` parameter.
    
    x0 : array_like
        Initial guess for the variables.
    
    args : tuple, optional
        Additional arguments to pass to the `fun` function. Default is an empty tuple.
    
    method : str, optional
        The solution method to use. Default is 'hybr'. Refer to `scipy.optimize.root` for available methods.
    
    tol : float, optional
        Tolerance for the solver. Solution is considered converged when the residual is less than `tol`. 
        Default is None, in which case the solver's default tolerance is used.
    
    options : dict, optional
        A dictionary of solver options. Default is None.¨

    display : bool, optional
        Choose whether to print the convergence history to console or not 

    Returns:
    --------
    solution
        The solution object returned by scipy.optimize.root

    Notes:
    ------
    During the solving process, the function displays the iteration number, number of residuals evaluations, 
    norm of the residual, and norm of the step. The convergence progress is printed at each iteration.
    """


    last_vars = None  # Independent variable vector
    last_residuals = None  # Residual vector
    evaluations = 0  # Counter for the number of function evaluations
    iteration = 0 # Counter for the number of solver iterations
    convergence_history = {"iteration": [],
                           "evaluations": [],
                           "norm_residual": [],
                           "norm_step": []}

    # Compute residual vector and store value
    def get_residual(vars):
        nonlocal evaluations, last_residuals # nonlocal variables maintain state across multiple calls 
        evaluations += 1  # Increment function evaluation counter
        last_residuals = fun(vars, args)
        return last_residuals

    # Compute the Jacobian of the residual vector by finite differences
    # Use the stored residual vector to avoid an unnecesary calculation
    # Evaluate the callback function to monitor progress (1 iteration = 1 jacobian evaluation)
    def get_residual_jacobian(vars):
        epsilon = np.sqrt(np.finfo(float).eps)*np.maximum(1, abs(vars))*np.sign(vars)
        jacobian = approx_derivative(get_residual, vars, method="2-point", f0=last_residuals, abs_step=epsilon)
        callback_function(vars)
        return jacobian
    
    # Callback function
    def callback_function(vars):
        """Display convergence progress"""

        # Use nonlocal variables maintain state across multiple calls 
        nonlocal iteration, last_vars
        
        # Increment iteration counter
        iteration += 1

        # Compute the norm of the residuals without recomputing them
        residual = last_residuals
        norm_residual = np.linalg.norm(residual)

        # Compute the norm of the independent last iteration step
        norm_step = np.linalg.norm(vars - last_vars) if last_vars is not None else 0
        last_vars = vars.copy()

        # Store convergence status
        convergence_history["iteration"].append(iteration)
        convergence_history["evaluations"].append(evaluations)
        convergence_history["norm_residual"].append(norm_residual)
        convergence_history["norm_step"].append(norm_step)

        # Print current iteration convergence details
        log_message(f" {iteration:20}{spacing}{evaluations:20d}{spacing}{norm_residual:20.6e}{spacing}{norm_step:20.6e} ", display=display)

    # Print header
    spacing = "    "
    header = f" {'Iteration':>20}{spacing}{'Residual count':>20}{spacing}{'Norm of residual':>20}{spacing}{'Norm of step':>20} "
    log_message('-'*len(header), display=display)
    log_message(header, display=display)
    log_message('-'*len(header), display=display)

    # Solve the system of equations
    # options = {"maxiter": 50}
    solution = root(get_residual, x0, jac=get_residual_jacobian, method=method, tol=tol)#, options=options)

    # Print exit message and solution after last iteration
    callback_function(solution.x)  # Callback after last iteration
    log_message('-'*len(header), display=display)
    log_message(f"Exit message: {solution.message}", display=display)
    log_message(f"Sucess: {solution.success}", display=display)
    log_message(f"Solution:", display=display)
    for i, x in enumerate(solution.x):
        log_message(f"   x{i} = {x:6e}", display=display)
    log_message('-'*len(header), display=display)
    log_message("", display=display)

    return solution, convergence_history



if __name__ == "__main__":

    # Create logs directory if it does not exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Set up logger with unique date-time name
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"{logs_dir}/convergence_history_{current_time}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Define a system of nonlinear equations (Lorentz equations)
    # https://en.wikipedia.org/wiki/Lorenz_system
    def lorentz_equations(vars, args):
        "Evaluate the RHS of the Lorentz equations"
        x, y, z = vars
        sigma = 1.0
        beta = 2.0
        rho = 3.0
        eq1 = sigma * (y - x)
        eq2 = x * (rho - z) - y
        eq3 = x * y - beta * z
        return np.array([eq1, eq2, eq3])

    # Find stationary point of the Lorentz equations
    initial_guess = [1.0, 3.0, 5.0]
    solution, convergence_history = solve_nonlinear_system(lorentz_equations, initial_guess, method='lm', tol=1e-12, options={"maxiter": 200}, display=True)

    # Plot the convergence history
    fig, ax = plt.subplots()
    ax.plot(convergence_history["iteration"], convergence_history["norm_residual"], marker='o', color="black")
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Two-norm of the residual vector')
    ax.set_title('Convergence history')
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    plt.tight_layout(pad=1)
    plt.show()