# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:40:52 2023

@author: laboan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import cascade_series as cs
from .design_optimization import CascadesOptimizationProblem
from ..solver import NonlinearSystemSolver, OptimizationSolver, NonlinearSystemProblem, OptimizationProblem
from ..utilities import set_plot_options, print_dict, print_boundary_conditions
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
        
    # Always scale initial guess    
    initial_guess = cs.scale_x0(initial_guess, cascades_data)

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


def compute_operating_point(
    boundary_conditions,
    cascades_data,
    initial_guess=None,
):
    """
    Compute an operating point for a given set of boundary conditions using multiple solver methods and initial guesses.

    Parameters:
    ----------
    boundary_conditions : dict
        A dictionary containing boundary conditions for the operating point.

    cascades_data : dict
        A dictionary containing data related to the cascades problem.

    initial_guess : array-like or dict, optional
        The initial guess for the solver. If None, default initial guesses are used.

    Returns:
    -------
    solution : object
        The solution object containing the results of the operating point calculation.

    Notes:
    -----
    - This function attempts to compute an operating point for a given set of boundary
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

    # Print boundary conditions of current operating point
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


def performance_map(boundary_conditions, cascades_data, filename=None):
    # Evaluate the cascades series at all conditions given in boundary_conditions
    # Exports the performance to ...

    x0 = None
    use_previous = True
    
    for i in range(len(boundary_conditions["fluid_name"])):
        
        conditions = {key: val[i] for key, val in boundary_conditions.items()}
        solver = compute_operating_point(conditions, cascades_data, initial_guess=x0)
        solution = solver.solution
        success = solution.success
        
        if use_previous == True:
            x0 = cs.convert_scaled_x0(solution.x, cascades_data)

        # Save performance
        BC = {key: val for key, val in cascades_data["BC"].items() if key != "fluid"}
        plane = cascades_data["plane"]
        cascade = cascades_data["cascade"]
        stage = cascades_data["stage"]
        overall = cascades_data["overall"]
        
        convergence_history = {"success" : success}

        plane_stack = plane.stack()
        plane_stack.index = [f"{idx}_{col+1}" for col, idx in plane_stack.index]
        cascade_stack = cascade.stack()
        cascade_stack.index = [f"{idx}_{col+1}" for col, idx in cascade_stack.index]
        stage_stack = stage.stack()
        stage_stack.index = [f"{idx}_{col+1}" for col, idx in stage_stack.index]

        if i == 0:
            BC_data = pd.DataFrame({key: [val] for key, val in BC.items()})
            plane_data = pd.DataFrame(
                {key: [val] for key, val in zip(plane_stack.index, plane_stack.values)}
            )
            cascade_data = pd.DataFrame(
                {
                    key: [val]
                    for key, val in zip(cascade_stack.index, cascade_stack.values)
                }
            )
            stage_data = pd.DataFrame(
                {key: [val] for key, val in zip(stage_stack.index, stage_stack.values)}
            )
            overall_data = pd.DataFrame({key: [val] for key, val in overall.items()})
            convergence_data = pd.DataFrame({key: [val] for key, val in convergence_history.items()})

        else:
            BC_data.loc[len(BC_data)] = BC
            plane_data.loc[len(plane_data)] = plane_stack
            cascade_data.loc[len(cascade_data)] = cascade_stack
            stage_data.loc[len(stage_data)] = stage_stack
            overall_data.loc[len(overall_data)] = overall
            convergence_data.loc[len(convergence_data)] = convergence_history

    # Write performance dataframe to excel
    if filename == None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Performance_data_{current_time}.xlsx"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        BC_data.to_excel(writer, sheet_name="BC", index=False)
        plane_data.to_excel(writer, sheet_name="plane", index=False)
        cascade_data.to_excel(writer, sheet_name="cascade", index=False)
        stage_data.to_excel(writer, sheet_name="stage", index=False)
        overall_data.to_excel(writer, sheet_name="overall", index=False)
        convergence_data.to_excel(writer, sheet_name="convergence", index=False)


class CascadesNonlinearSystemProblem(NonlinearSystemProblem):
    def __init__(self, cascades_data):
        cs.calculate_number_of_stages(cascades_data)
        cs.update_fixed_params(cascades_data)
        cs.check_geometry(cascades_data)

        # if x0 is None:
        #     x0 = cs.generate_initial_guess(cascades_data, R, eta_tt, eta_ts, Ma_crit)

        # x_scaled = cs.scale_x0(x0, cascades_data)

        # self.x0 = x_scaled
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
        self.x0 = cs.scale_x0(x0, cascades_data)
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
