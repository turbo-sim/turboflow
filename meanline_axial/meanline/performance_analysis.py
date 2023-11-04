# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:40:52 2023

@author: laboan
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import cascade_series as cs
from .design_optimization import CascadesOptimizationProblem
from ..solver import (
    NonlinearSystemSolver,
    OptimizationSolver,
    NonlinearSystemProblem,
    OptimizationProblem,
)
from ..utilities import (
    set_plot_options,
    print_dict,
    print_boundary_conditions,
    flatten_dataframe,
    numpy_to_python,
)
from datetime import datetime


set_plot_options()


name_map = {"lm": "Lavenberg-Marquardt", "hybr": "Powell's hybrid"}


def initialize_solver(cascades_data, method, initial_guess=None):
    """
    Initialize a nonlinear system solver for solving a given problem.

    Parameters:
    ----------
    cascades_data : dict
        A dictionary containing data related to the problem.

    method : str
        The solver method to use.

    initial_guess : array-like or dict, optional
        The initial guess for the solver. If None, a default initial guess is generated.
        If a dictionary is provided, it should contain the following keys:
        - 'R': Initial guess for parameter R (float)
        - 'eta_tt': Initial guess for parameter eta_tt (float)
        - 'eta_ts': Initial guess for parameter eta_ts (float)
        - 'Ma_crit': Initial guess for parameter Mach_crit (float)

    Returns:
    -------
    solver : NonlinearSystemSolver
        A solver object configured to solve the nonlinear system problem.

    Raises:
    ------
    ValueError
        If the provided initial guess is not an array or dictionary.

    Notes:
    -----
    - If an initial guess is provided as an array, it will be used as is.
    - If no initial guess is provided, default values for R, eta_tt, eta_ts, and Ma_crit
      are used to generate an initial guess.
    - If an initial guess is provided as a dictionary, it should include the necessary
      parameters for the problem.

    """

    # Initialize problem object
    problem = CascadesNonlinearSystemProblem(cascades_data)

    # Define initial guess
    if isinstance(initial_guess, np.ndarray):
        pass  # Keep the initial guess as it is
    else:
        if initial_guess is None:
            print("No initial guess provided.")
            print("Generating initial guess from default parameters...")
            R = 0.5
            eta_tt = 0.9
            eta_ts = 0.8
            Ma_crit = 0.95
        elif isinstance(initial_guess, dict):
            R = initial_guess["R"]
            eta_tt = initial_guess["eta_tt"]
            eta_ts = initial_guess["eta_ts"]
            Ma_crit = initial_guess["Ma_crit"]
        else:
            raise ValueError("Initial guess must be an array or dictionary.")

        initial_guess = cs.generate_initial_guess(
            cascades_data, R, eta_tt, eta_ts, Ma_crit
        )

    # Always normalize initial guess
    initial_guess = cs.scale_to_normalized_values(initial_guess, cascades_data)

    # Initialize solver object
    solver = NonlinearSystemSolver(
        problem,
        initial_guess,
        method=method,
        tol=cascades_data["solver"]["tolerance"],
        max_iter=cascades_data["solver"]["max_iterations"],
        derivative_method=cascades_data["solver"]["derivative_method"],
        derivative_rel_step=cascades_data["solver"]["derivative_rel_step"],
        display=cascades_data["solver"]["display_progress"],
    )

    return solver


def compute_operation_point(
    boundary_conditions,
    cascades_data,
    initial_guess=None,
):
    """
    Compute an operation point for a given set of boundary conditions using multiple solver methods and initial guesses.

    Parameters:
    ----------
    boundary_conditions : dict
        A dictionary containing boundary conditions for the operation point.

    cascades_data : dict
        A dictionary containing data related to the cascades problem.

    initial_guess : array-like or dict, optional
        The initial guess for the solver. If None, default initial guesses are used.
        If provided, the initial guess should not be scaled (it is scaled internally)

    Returns:
    -------
    solution : object
        The solution object containing the results of the operation point calculation.

    Notes:
    -----
    - This function attempts to compute an operation point for a given set of boundary
      conditions using various solver methods and initial guesses before giving up.
    - The boundary_conditions dictionary should include the necessary parameters for the
      problem.
    - The initial_guess can be provided as an array-like object or a dictionary. If None,
      default values are used for initial guesses.
    - The function iteratively tries different solver methods, including user-specified
      methods and the Levenberg-Marquardt method, to solve the problem.
    - It also attempts solving with a heuristic initial guess and explores different
      initial guesses from parameter arrays.
    - The function prints information about the solver method and any failures during the
      computation.
    - If successful convergence is achieved, the function returns the solution object.
    - If all attempts fail to converge, a warning message is printed.

    """

    # Print boundary conditions of current operation point
    print_boundary_conditions(boundary_conditions)

    # Calculate performance at given boundary conditions with given geometry
    cascades_data["BC"] = boundary_conditions

    # Attempt solving with the specified method
    method = cascades_data["solver"]["method"]
    print(f"Trying to solve the problem using {name_map[method]} method")
    solver = initialize_solver(cascades_data, method, initial_guess=initial_guess)
    try:
        solution = solver.solve()
        success = solution.success
    except Exception as e:
        print(f"Error during solving: {e}")
        success = False
    if not success:
        print(f"Solution failed with {name_map[method]} method")

    # Attempt solving with Lavenberg-Marquardt method
    if method != "lm" and not success:
        method = "lm"
        print(f"Trying to solve the problem using {name_map[method]} method")
        solver = initialize_solver(cascades_data, method, initial_guess=initial_guess)
        try:
            solution = solver.solve()
            success = solution.success
        except Exception as e:
            print(f"Error during solving: {e}")
            success = False
        if not success:
            print(f"Solution failed with {name_map[method]} method")

    # TODO: Attempt solving with optimization algorithms?

    # Attempt solving with a heuristic initial guess
    if isinstance(initial_guess, np.ndarray) and not success:
        method = "lm"
        print("Trying to solve the problem with a new initial guess")
        solver = initialize_solver(cascades_data, method, initial_guess=None)
        try:
            solution = solver.solve()
            success = solution.success
        except Exception as e:
            print(f"Error during solving: {e}")
            success = False

    # Attempt solving using different initial guesses
    if not success:
        N = 11
        x0_arrays = {
            "R": np.linspace(0.0, 0.95, N),
            "eta_ts": np.linspace(0.6, 0.9, N),
            "eta_tt": np.linspace(0.7, 1.0, N),
            "Ma_crit": np.linspace(0.9, 0.9, N),
        }

        for i in range(N):
            x0 = {key: values[i] for key, values in x0_arrays.items()}
            print(f"Trying to solve the problem with a new initial guess")
            print_dict(x0)
            solver = initialize_solver(cascades_data, method, initial_guess=x0)
            try:
                solution = solver.solve()
                success = solution.success
            except Exception as e:
                print(f"Error during solving: {e}")
                success = False
            if not success:
                print(f"Solution failed with {name_map[method]} method")

        if not success:
            print("WARNING: All attempts failed to converge")
            solution = False
            # TODO: Add messages to Log file

    return solver


def performance_map(
    operation_points,
    case_data,
    filename=None,
    output_dir="output",
    use_previous=True,
):
    """
    Evaluates the performance at the specified operation points and exports the results to an Excel file.
    This function computes the performance of each operation point, collects and compiles them into
    a single Excel file with separate sheets for each aspect of the data.

    Parameters:
    - operation_points: List of dicts representing operation points to evaluate.
    - case_data: Dict with necessary data structures for computing operation points.
    - filename: Optional; if provided, specifies the output Excel filename.
                If None, a default unique filename will be created with a timestamp.
    - output_dir: Optional; directory to save the Excel file. Defaults to 'output'.
    - use_previous: Optional; a boolean flag to determine whether to use the
                    solution from the previous operation point as an initial guess
                    for the current point.

    Returns:
    - The absolute path to the generated Excel file containing the results.
    """

    if not operation_points:
        raise ValueError("operation_points list is empty or not provided.")

    # Create a directory to save simulation results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define filename with unique date-time identifier
    if filename == None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"performance_analysis_{current_time}"

    # Filter out 'operation_points' and any empty entries
    config_data = {k: v for k, v in case_data.items() if v and k != "operation_points"}
    config_data = {**config_data, "operation_points": operation_points}
    config_data = numpy_to_python(config_data)

    print(config_data)

    # Export input arguments as YAML file
    config_file = os.path.join(output_dir, f"{filename}.yaml")
    with open(config_file, "w") as file:
        yaml.dump(config_data, file, default_flow_style=None)

    # Initialize lists to hold dataframes for each operation point
    operation_point_data = []
    overall_data = []
    plane_data = []
    cascade_data = []
    stage_data = []
    solver_data = []
    

    # Use default initial guess for the first operation point
    x0 = None

    for i, operation_point in enumerate(operation_points):
        print(
            f"Computing performance for operation point {i + 1} of {len(operation_points)}"
        )

        try:
            # Compute performance
            solver = compute_operation_point(
                operation_point, case_data, initial_guess=x0
            )

            # Use converged solution as initial guess for the next operation point
            if use_previous == True:
                x0 = cs.scale_to_real_values(solver.solution.x, case_data)

            # Retrieve solver data
            solver_status = {
                "completed": True,
                "success": solver.solution.success,
                "message": solver.solution.message,
                "grad_count": solver.convergence_history["grad_count"][-1],
                "func_count": solver.convergence_history["func_count"][-1],
                "func_count_total": solver.convergence_history["func_count_total"][-1],
                "norm_residual_last": solver.convergence_history["norm_residual"][-1],
                "norm_step_last": solver.convergence_history["norm_step"][-1],
            }

            # Collect data
            operation_point_data.append(pd.DataFrame([operation_point]))
            overall_data.append(pd.DataFrame([case_data["overall"]]))
            plane_data.append(flatten_dataframe(case_data["plane"]))
            cascade_data.append(flatten_dataframe(case_data["cascade"]))
            stage_data.append(flatten_dataframe(case_data["stage"]))
            solver_data.append(pd.DataFrame([solver_status]))

        except Exception as e:
            print(
                f"An error occurred while computing the operation point {i}/{len(operation_points)}:\n\t{e}"
            )

            # Retrieve solver data
            solver_status = {"completed": False}

            # Collect data
            operation_point_data.append(pd.DataFrame([operation_point]))
            overall_data.append(pd.DataFrame([{}]))
            plane_data.append(pd.DataFrame([{}]))
            cascade_data.append(pd.DataFrame([{}]))
            stage_data.append(pd.DataFrame([{}]))
            solver_data.append(pd.DataFrame([solver_status]))

    # Dictionary to hold concatenated dataframes
    dfs = {
        "operation point": pd.concat(operation_point_data, ignore_index=True),
        "overall": pd.concat(overall_data, ignore_index=True),
        "plane": pd.concat(plane_data, ignore_index=True),
        "cascade": pd.concat(cascade_data, ignore_index=True),
        "stage": pd.concat(stage_data, ignore_index=True),
        "solver": pd.concat(solver_data, ignore_index=True),
    }

    # Add 'operation_point' column to each dataframe
    for sheet_name, df in dfs.items():
        df.insert(0, "operation_point", range(1, 1 + len(df)))

    # Write dataframes to excel
    filepath = os.path.join(output_dir, f"{filename}.xlsx")
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Performance data successfully written to {filepath}")


class CascadesNonlinearSystemProblem(NonlinearSystemProblem):
    def __init__(self, cascades_data):
        cs.calculate_number_of_stages(cascades_data)
        cs.update_fixed_params(cascades_data)
        cs.check_geometry(cascades_data)
        self.cascades_data = cascades_data

    def get_values(self, vars):
        residuals = cs.evaluate_cascade_series(vars, self.cascades_data)

        return residuals


class CascadesOptimizationProblem(OptimizationProblem):
    def __init__(self, cascades_data, R, eta_tt, eta_ts, Ma_crit, x0=None):
        cs.calculate_number_of_stages(cascades_data)
        cs.update_fixed_params(cascades_data)
        cs.check_geometry(cascades_data)
        if x0 == None:
            x0 = cs.generate_initial_guess(cascades_data, R, eta_tt, eta_ts, Ma_crit)
        self.x0 = cs.scale_to_normalized_values(x0, cascades_data)
        self.cascades_data = cascades_data

    def get_values(self, vars):
        residuals = cs.evaluate_cascade_series(vars, self.cascades_data)
        self.f = 0
        self.c_eq = residuals
        self.c_ineq = None
        objective_and_constraints = self.merge_objective_and_constraints(
            self.f, self.c_eq, self.c_ineq
        )

        return objective_and_constraints

    def get_bounds(self):
        n_cascades = self.cascades_data["geometry"]["n_cascades"]
        lb, ub = cs.get_dof_bounds(n_cascades)
        bounds = [(lb[i], ub[i]) for i in range(len(lb))]
        return bounds

    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)

    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)
