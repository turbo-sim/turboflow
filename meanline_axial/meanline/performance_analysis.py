import os
import yaml
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import cascade_series as cs
from . import geometry as geom
from .. import math
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
    convert_numpy_to_python,
    ensure_iterable,
    print_operation_points,
)
from datetime import datetime

from ..properties import FluidCoolProp_2Phase
import CoolProp as CP

set_plot_options()

name_map = {"lm": "Lavenberg-Marquardt", "hybr": "Powell's hybrid"}


def compute_performance(
    operation_points,
    config,
    out_filename=None,
    out_dir="output",
    stop_on_failure=True,
):
    """
    Compute and export the performance of each specified operation point to an Excel file.

    This function handles two types of input for operation points:

        1. An explicit list of dictionaries, each detailing a specific operation point.

        2. A dictionary where each key has a range of values, representing the cross-product
        of all possible operation points. It generates the Cartesian product of these ranges
        internally.

    For each operation point, it computes performance based on the provided case data and compiles
    the results into an Excel workbook with multiple sheets for various data sections.

    The function validates the input operation points, and if they are given as ranges, it generates
    all possible combinations. Performance is computed for each operation point, and the results are
    then stored in a structured Excel file with separate sheets for each aspect of the data (e.g.,
    overall, plane, cascade, stage, solver, and solution data).

    The initial guess for the first operation point is set to a default value. For subsequent operation
    points, the function employs a strategy to use the closest previously computed operation point's solution
    as the initial guess. This approach is based on the heuristic that similar operation points have similar
    performance characteristics, which can improve convergence speed and robustness of the solution process.

    Parameters
    ----------
    operation_points : list of dict or dict
        A list of operation points where each is a dictionary of parameters, or a dictionary of parameter
        ranges from which operation points will be generated.
    config : dict
        A dictionary containing necessary configuration options for computing performance at each operation point.
    out_file : str, optional
        The name for the output Excel file. If not provided, a default name with a timestamp is generated.
    out_dir : str, optional
        The directory where the Excel file will be saved. Defaults to 'output'.

    Returns
    -------
    str
        The absolute path to the created Excel file.

    See Also
    --------
    compute_operation_point : For computing the performance of a single operation point.
    generate_operation_points : For generation of operation points from ranges.
    validate_operation_point : For validation of individual operation points.
    """

    # Initialize problem object
    problem = CascadesNonlinearSystemProblem(config)

    # Check the type of operation_points argument
    if isinstance(operation_points, dict):
        # Convert ranges to a list of operation points
        operation_points = generate_operation_points(operation_points)
    elif not isinstance(operation_points, list):
        msg = "operation_points must be either list of dicts or a dict with ranges."
        raise TypeError(msg)

    # Validate all operation points
    for operation_point in operation_points:
        validate_operation_point(operation_point)

    # Create a directory to save simulation results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Define filename with unique date-time identifier
    if out_filename == None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_filename = f"performance_analysis_{current_time}"

    # Export simulation configuration as YAML file
    config_data = {k: v for k, v in config.items() if v}  # Filter empty entries
    config_data = convert_numpy_to_python(config_data, precision=12)
    config_file = os.path.join(out_dir, f"{out_filename}.yaml")
    with open(config_file, "w") as file:
        yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)

    # Initialize lists to hold dataframes for each operation point
    operation_point_data = []
    overall_data = []
    plane_data = []
    cascade_data = []
    stage_data = []
    solver_data = []
    solution_data = []
    solver_container = []

    # Loop through all operation points
    print_operation_points(operation_points)
    for i, operation_point in enumerate(operation_points):
        print()
        print(f"Computing operation point {i+1} of {len(operation_points)}")
        print_boundary_conditions(operation_point)

        try:
            # Define initial guess
            if i == 0:
                # Use default initial guess for the first operation point
                initial_guess = None
                print(f"Using default initial guess")
            else:
                closest_x, closest_index = find_closest_operation_point(
                    operation_point,
                    operation_points[:i],  # Use up to the previous point
                    solution_data[:i],  # Use solutions up to the previous point
                )
                print(f"Using solution from point {closest_index+1} as initial guess")
                initial_guess = closest_x

            # Compute performance
            solver, results = compute_single_operation_point(
                operation_point, problem, initial_guess, config["solver_options"]
            )

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

            # Collect results
            operation_point_data.append(pd.DataFrame([operation_point]))
            overall_data.append(pd.DataFrame([results["overall"]]))
            plane_data.append(flatten_dataframe(results["plane"]))
            cascade_data.append(flatten_dataframe(results["cascade"]))
            stage_data.append(flatten_dataframe(results["stage"]))
            solver_data.append(pd.DataFrame([solver_status]))
            solution_data.append(solver.problem.scale_to_real_values(solver.solution.x))
            solver_container.append(solver)

        except Exception as e:
            if stop_on_failure:
                raise Exception(e)
            else:
                print(f"Computation of point {i+1}/{len(operation_points)} failed")
                print(f"Error: {e}")

            # Retrieve solver data
            solver = None
            solver_status = {"completed": False}

            # Collect data
            operation_point_data.append(pd.DataFrame([operation_point]))
            overall_data.append(pd.DataFrame([{}]))
            plane_data.append(pd.DataFrame([{}]))
            cascade_data.append(pd.DataFrame([{}]))
            stage_data.append(pd.DataFrame([{}]))
            solver_data.append(pd.DataFrame([solver_status]))
            solution_data.append([])
            solver_container.append(solver)

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
    filepath = os.path.join(out_dir, f"{out_filename}.xlsx")
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Performance data successfully written to {filepath}")

    return solver_container


def compute_single_operation_point(
    operating_point,
    problem,
    initial_guess,
    solver_options,
):
    """
    Compute an operation point for a given set of boundary conditions using multiple solver methods and initial guesses.

    Parameters
    ----------
    boundary_conditions : dict
        A dictionary containing boundary conditions for the operation point.

    cascades_data : dict
        A dictionary containing data related to the cascades problem.

    initial_guess : array-like or dict, optional
        The initial guess for the solver. If None, default initial guesses are used.
        If provided, the initial guess should not be scaled (it is scaled internally)

    Returns
    -------
    solution : object
        The solution object containing the results of the operation point calculation.

    Notes
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

    # Initialize problem object
    problem.update_boundary_conditions(operating_point)
    initial_guess = problem.get_initial_guess(initial_guess)

    # Attempt solving with the specified method
    method = solver_options["method"]
    print(f"Trying to solve the problem using {name_map[method]} method")
    solver = initialize_solver(problem, initial_guess, solver_options)
    try:
        solution = solver.solve()
        success = solution.success
    except Exception as e:
        print(f"Error during solving: {e}")
        success = False
    if not success:
        print(f"Solution failed for method '{solver_options['method']}'")

    # Attempt solving with Lavenberg-Marquardt method
    if solver_options["method"] != "lm" and not success:
        solver_options["method"] = "lm"
        print(f"Trying to solve the problem using {name_map[method]} method")
        solver = initialize_solver(problem, initial_guess, solver_options)
        try:
            solution = solver.solve()
            success = solution.success
        except Exception as e:
            print(f"Error during solving: {e}")
            success = False
        if not success:
            print(f"Solution failed for method '{solver_options['method']}'")

    # TODO: Attempt solving with optimization algorithms?

    # Attempt solving with a heuristic initial guess
    if isinstance(initial_guess, np.ndarray) and not success:
        solver_options["method"] = "lm"
        print("Trying to solve the problem with a new initial guess")
        solver = initialize_solver(problem, initial_guess, solver_options)
        try:
            solution = solver.solve()
            success = solution.success
        except Exception as e:
            print(f"Error during solving: {e}")
            success = False

    # Attempt solving using different initial guesses
    # TODO: To be improved with random generation of initial guess within ranges
    # TODO: use sampling techniques like latin hypercube/ montecarlo sampling (fancy word for random sampling) / orthogonal sampling
    # if not success:
    #     N = 11
    #     x0_arrays = {
    #         "R": np.linspace(0.0, 0.95, N),
    #         "eta_ts": np.linspace(0.6, 0.9, N),
    #         "eta_tt": np.linspace(0.7, 1.0, N),
    #         "Ma_crit": np.linspace(0.9, 0.9, N),
    #     }

    #     for i in range(N):
    #         x0 = {key: values[i] for key, values in x0_arrays.items()}
    #         print(f"Trying to solve the problem with a new initial guess")
    #         print_dict(x0)
    #         initial_guess = problem.get_initial_guess(x0)
    #         solver = initialize_solver(problem, initial_guess, solver_options)
    #         try:
    #             solution = solver.solve()
    #             success = solution.success
    #         except Exception as e:
    #             print(f"Error during solving: {e}")
    #             success = False
    #         if not success:
    #             print(f"Solution failed for method '{solver_options['method']}'")

        if not success:
            print("WARNING: All attempts failed to converge")
            solution = False
            # TODO: Add messages to Log file

    return solver, problem.results


def initialize_solver(problem, initial_guess, solver_options):
    """
    Compute a solution for an operating point of a problem using various solver methods and initial guesses.

    This function aims to find a solution for a specified operating point. If the initial attempt fails, the function
    will try to solve the problem using different algorithms and initial guesses

    - The function first updates the problem's boundary conditions using the operating point.
    - It attempts to solve the problem with the specified solver method. If it fails, it tries the Levenberg-Marquardt method.
    - If the provided initial guess is an array, the function will attempt to solve the problem with a heuristic initial guess.
    - The function also explores different initial guesses based on parameter arrays for parameters like 'R', 'eta_ts', 'eta_tt', and 'Ma_crit'.
    - If all attempts fail, a warning message is printed.
    - The function returns the solver object and the computed results, which can be further analyzed or used in subsequent calculations.

    Parameters
    ----------
    operating_point : dict
        A dictionary containing the operating point's parameters and values.

    problem : object
        An object representing the performance analysis problem

    initial_guess : array-like or dict
        The initial guess for the solver. If provided, it should not be scaled (scaling is handled internally).

    solver_options : dict
        A dictionary containing options for the solver
        - If an initial guess is provided as an array, it will be used as is.
        - If no initial guess is provided, default values for R, eta_tt, eta_ts, and Ma_crit
            are used to generate an initial guess.
        - If an initial guess is provided as a dictionary, it should include the necessary
            parameters for the problem.

    Returns
    -------
    tuple
        A tuple containing the solver object and the problem's results after computation.

    """

    # Initialize solver object
    solver = NonlinearSystemSolver(
        problem,
        initial_guess,
        method=solver_options["method"],
        tol=solver_options["tolerance"],
        max_iter=solver_options["max_iterations"],
        derivative_method=solver_options["derivative_method"],
        derivative_rel_step=solver_options["derivative_rel_step"],
        display=solver_options["display_progress"],
    )

    return solver


def find_closest_operation_point(current_op_point, operation_points, solution_data):
    """
    Find the solution vector and index of the closest operation point in the historical data.

    Parameters
    ----------
    current_op_point : dict
        The current operation point we want to compare.
    operation_points : list of dict
        A list of historical operation points to search through.
    solution_data : list
        A list of solution vectors corresponding to each operation point.

    Returns
    -------
    tuple
        A tuple containing the closest solution vector and the one-based index of the closest operation point.

    """
    min_distance = float("inf")
    closest_point_x = None
    closest_index = None

    for i, op_point in enumerate(operation_points):
        distance = get_operation_point_distance(current_op_point, op_point)
        if distance < min_distance:
            min_distance = distance
            closest_point_x = solution_data[i]
            closest_index = i

    return closest_point_x, closest_index


def get_operation_point_distance(point_1, point_2, delta=1e-8):
    """
    Calculate the normalized distance between two operation points, with special consideration
    for angle measurements and prevention of division by zero for very small values.

    Parameters
    ----------
    point_1 : dict
        First operation point with numeric values.
    point_2 : dict
        Second operation point with numeric values.
    delta : float, optional
        A small constant to prevent division by zero. Default is 1e-8.

    Returns
    -------
    float
        The calculated normalized distance.
    """
    deviation_array = []
    for key in point_1:
        if isinstance(point_1[key], (int, float)) and key in point_2:
            value_1 = point_1[key]
            value_2 = point_2[key]

            if key == "alpha_in":
                # Handle angle measurements with absolute scale normalization
                deviation = np.abs(value_1 - value_2) / 90
            else:
                # Compute the relative difference with protection against division by zero
                max_val = max(abs(value_1), abs(value_2), delta)
                deviation = abs(value_1 - value_2) / max_val

            deviation_array.append(deviation)

    # Calculate the two-norm of the deviations
    return np.linalg.norm(deviation_array)


def generate_operation_points(performance_map):
    """
    Generates a list of dictionaries representing all possible combinations of
    operation points from a given performance map. The performance map is a
    dictionary where keys represent parameter names and values are the ranges
    of values for those parameters. The function ensures that the combinations
    are generated such that the parameters related to pressure ('p0_in' and
    'p_out') are the last ones to vary, effectively making them the first
    parameters to sweep through in the operation points.

    Parameters
    ----------
    - performance_map (dict): A dictionary with parameter names as keys and
      lists of parameter values as values.

    Returns
    -------
    - operation_points (list of dict): A list of dictionaries, each representing
      a unique combination of parameters from the performance_map.
    """
    # Make sure all values in the performance_map are iterables
    performance_map = {k: ensure_iterable(v) for k, v in performance_map.items()}

    # Reorder performance map keys so first sweep is always through pressure
    priority_keys = ["p0_in", "p_out"]
    other_keys = [k for k in performance_map.keys() if k not in priority_keys]
    keys_order = other_keys + priority_keys
    performance_map = {
        k: performance_map[k] for k in keys_order if k in performance_map
    }

    # Create all combinations of operation points
    keys, values = zip(*performance_map.items())
    operation_points = [
        dict(zip(keys, combination)) for combination in itertools.product(*values)
    ]

    return operation_points


def validate_operation_point(op_point):
    """
    Validates that an operation point has exactly the required fields:
    'fluid_name', 'p0_in', 'T0_in', 'p_out', 'alpha_in', 'omega'.

    Parameters
    ----------
    op_point: dict
        A dictionary representing an operation point.

    Returns
    -------
    ValueError: If the dictionary does not contain the required fields or contains extra fields.
    """
    REQUIRED_FIELDS = {"fluid_name", "p0_in", "T0_in", "p_out", "alpha_in", "omega"}
    fields = set(op_point.keys())
    if fields != REQUIRED_FIELDS:
        missing = REQUIRED_FIELDS - fields
        extra = fields - REQUIRED_FIELDS
        raise ValueError(
            f"Operation point validation error: "
            f"Missing fields: {missing}, Extra fields: {extra}"
        )


# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #


class CascadesNonlinearSystemProblem(NonlinearSystemProblem):
    """
    A class representing a nonlinear system problem for cascade analysis.

    This class is designed for solving nonlinear systems of equations related to cascade analysis.
    Derived classes must implement the `get_values` method to evaluate the system of equations for a given set of decision variables.

    Additionally, specific problem classes can define the `get_jacobian` method to compute Jacobians.
    If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Attributes
    ----------
    fluid : FluidCoolProp_2Phase
        An instance of the FluidCoolProp_2Phase class representing the fluid properties.
    results : dict
        A dictionary to store results.
    BC : dict
        A dictionary containing boundary condition data.
    geometry : dict
        A dictionary containing geometry-related data.
    model_options : dict
        A dictionary containing options related to the analysis model.
    reference_values : dict
        A dictionary containing reference values for calculations.

    Methods
    -------
    get_values(x)
        Evaluate the system of equations for a given set of decision variables.

    Examples
    --------
    Here's an example of how to derive from `CascadesNonlinearSystemProblem`::

        class MyCascadeProblem(CascadesNonlinearSystemProblem):
            def get_values(self, x):
                # Implement evaluation logic here
                pass
    """

    def __init__(self, config):
        """
        Initialize a CascadesNonlinearSystemProblem.

        Parameters
        ----------
        case_data : dict
            A dictionary containing case-specific data.
        """

        # Process turbine geometry
        geom.validate_axial_turbine_geometry(config["geometry"])
        self.geometry = geom.calculate_full_geometry(config["geometry"])
        self.geom_info = geom.check_axial_turbine_geometry(self.geometry, display=True)

        # Initialize other attributes
        self.model_options = config["model_options"]
        self.keys = []

    def get_values(self, x):
        """
        Evaluate the system of equations for a given set of decision variables.

        Parameters
        ----------
        x : array-like
            Vector of decision variables.

        Returns
        -------
        tuple
            A tuple containing residuals and a list of keys for the residuals.
        """
        residuals, self.results = cs.evaluate_cascade_series(
            x,
            self.boundary_conditions,
            self.geometry,
            self.fluid,
            self.model_options,
            self.reference_values,
        )

        return np.array(list(residuals.values()))

    def update_boundary_conditions(self, operation_point):
        """
        Update the boundary conditions of the problem with the provided operation point.

        This method updates the internal state of the object by setting new boundary conditions
        as defined in the 'operation_point' dictionary. It also initializes a Fluid object
        using the 'fluid_name' specified in the operation point.

        The method computes additional properties and reference values like stagnation properties at
        the inlet, exit static properties, spouting velocity, and reference mass flow rate.
        These are stored in the object's internal state for further use in calculations.

        Parameters
        ----------
        operation_point : dict
            A dictionary containing the boundary conditions defining the operation point.
            It must include the following keys:
            - fluid_name: str, the name of the fluid to be used in the Fluid object.
            - T0_in: float, the inlet temperature (in Kelvin).
            - p0_in: float, the inlet pressure (in Pascals).
            - p_out: float, the outlet pressure (in Pascals).
            - omega: float, the rotational speed (in rad/s).
            - alpha_in: float, the inlet flow angle (in degrees).

        Returns
        -------
        None
            This method does not return a value but updates the internal state of the object.

        """

        # Define current operating point
        validate_operation_point(operation_point)
        self.boundary_conditions = operation_point

        # Initialize fluid object
        self.fluid = FluidCoolProp_2Phase(operation_point["fluid_name"])

        # Rename variables
        p0_in = operation_point["p0_in"]
        T0_in = operation_point["T0_in"]
        p_out = operation_point["p_out"]

        # Compute stagnation properties at inlet
        state_in_stag = self.fluid.get_props(CP.PT_INPUTS, p0_in, T0_in)
        h0_in = state_in_stag["h"]
        s_in = state_in_stag["s"]

        # Store the inlet stagnation (h,s) for the first stage
        # TODO: Improve logic of implementation?
        self.boundary_conditions["h0_in"] = h0_in
        self.boundary_conditions["s_in"] = s_in

        # Calculate exit static properties for a isentropic expansion
        state_out_s = self.fluid.get_props(CP.PSmass_INPUTS, p_out, s_in)
        h_isentropic = state_out_s["h"]
        d_isenthalpic = state_out_s["d"]

        # Calculate exit static properties for a isenthalpic expansion
        state_out_h = self.fluid.get_props(CP.HmassP_INPUTS, h0_in, p_out)
        s_isenthalpic = state_out_h["s"]

        # Calculate spouting velocity
        v0 = np.sqrt(2 * (h0_in - h_isentropic))

        # Define a reference mass flow rate
        A_out = self.geometry["A_out"][-1]
        m_ref = A_out * v0 * d_isenthalpic

        # Define reference_values
        self.reference_values = {
            "s_range": s_isenthalpic - s_in,
            "s_min": s_in,
            "v0": v0,
            "h_out_s": h_isentropic,
            "d_out_s": d_isenthalpic,
            "m_ref": m_ref,
            "angle_range": 180,
            "angle_min": -90,
            "delta_ref": 0.011 / 3e5 ** (-1 / 7),
        }

        return

    def scale_to_real_values(self, x):
        """
        Convert a normalized solution vector back to real-world values.

        Parameters
        ----------
        x: The normalized solution vector from the solver.
        cascades_data: Dictionary containing reference values and number of cascades.

        Returns
        -------
        An array of values converted back to their real-world scale.
        """

        # TODO Lasse: improve logic using dictionary with keys instead of array
        # TODO Lasse/Roberto: Does it make sense to have a single function for scaling and unscaling if most of the code is the same?

        v0 = self.reference_values["v0"]
        s_range = self.reference_values["s_range"]
        s_min = self.reference_values["s_min"]
        angle_range = self.reference_values["angle_range"]
        angle_min = self.reference_values["angle_min"]
        number_of_cascades = self.geometry["number_of_cascades"]

        # Slice x into actual and critical values
        x_real = x.copy()[0 : 5 * number_of_cascades + 1]
        xcrit_real = x.copy()[5 * number_of_cascades + 1 :]

        # Convert x to real values
        x_real[0] *= v0
        xcrit_real *= v0  # Scale all critical values with v0
        for i in range(number_of_cascades):
            x_real[5 * i + 1] *= v0
            x_real[5 * i + 2] *= v0
            x_real[5 * i + 3] = x_real[5 * i + 3] * s_range + s_min
            x_real[5 * i + 4] = x_real[5 * i + 4] * s_range + s_min
            x_real[5 * i + 5] = x_real[5 * i + 5] * angle_range + angle_min
            xcrit_real[3 * i + 2] = (
                xcrit_real[3 * i + 2] / v0 * s_range + s_min
            )  # Scale the entropy values of the critical variables

        x_real = np.concatenate((x_real, xcrit_real))

        return x_real

    def scale_to_normalized_values(self, x):
        """
        Scale a solution vector from actual values to a normalized scale.

        Parameters
        ----------
        x: The solution vector with actual values to be normalized.
        cascades_data: Dictionary containing reference values and number of cascades.

        Returns
        -------
        An array of values scaled to a normalized range for solver computations.
        """
        # TODO Lasse: improve logic using dictionary with keys instead of array

        # Load parameters
        v0 = self.reference_values["v0"]
        s_range = self.reference_values["s_range"]
        s_min = self.reference_values["s_min"]
        angle_range = self.reference_values["angle_range"]
        angle_min = self.reference_values["angle_min"]
        number_of_cascades = self.geometry["number_of_cascades"]

        # Slice x into actual and critical values
        x_scaled = x.copy()[: 5 * number_of_cascades + 1]
        xcrit_scaled = x.copy()[5 * number_of_cascades + 1 :]

        # Scale x0
        x_scaled[0] /= v0
        xcrit_scaled /= v0  # Scale all critical values with v0

        for i in range(number_of_cascades):
            x_scaled[5 * i + 1] /= v0
            x_scaled[5 * i + 2] /= v0
            x_scaled[5 * i + 3] = (x_scaled[5 * i + 3] - s_min) / s_range
            x_scaled[5 * i + 4] = (x_scaled[5 * i + 4] - s_min) / s_range
            x_scaled[5 * i + 5] = (x_scaled[5 * i + 5] - angle_min) / angle_range

            xcrit_scaled[3 * i + 2] = (
                xcrit_scaled[3 * i + 2] * v0 - s_min
            ) / s_range  # Scale the entropy values of the critical variables

        x_scaled = np.concatenate((x_scaled, xcrit_scaled))

        return x_scaled

    def get_initial_guess(self, initial_guess=None):
        # TODO Lasse: Improve the logic of this function based on the new configuration file
        # TODO Lasse: It should be possible to provide a dictionary of unscaled independent variables, or a dictionary with enthalpy-split fractions and efficiencies
        # TODO Lasse/Roberto: Should this function be moved within the problem object.
        # TODO It's logical that the problem object has all the methods related to initial guess generation for the specific problem
        #
        # ATTENTION! Using an np.ndarray as initial guess will not generate an initial guess.
        # This is used when using a previous converged solution as initial guess
        # This behavior should be changed only after the code has been updated to store the solution vector as a dictionary
        #

        # Define initial guess
        if isinstance(initial_guess, np.ndarray):
            pass  # Keep the initial guess as it is
        else:
            if initial_guess is None:
                print("Generating heuristic initial guess with default parameters")
                R = 0.5
                eta_tt = 0.9
                eta_ts = 0.8
                Ma_crit = 0.95
            elif isinstance(initial_guess, dict):
                print(
                    "Generating heuristic initial guess with the specific parameters."
                )
                R = initial_guess["R"]
                eta_tt = initial_guess["eta_tt"]
                eta_ts = initial_guess["eta_ts"]
                Ma_crit = initial_guess["Ma_crit"]
            else:
                raise ValueError("Initial guess must be a dictionary.")

            # Compute initial guess using several approximations
            initial_guess = self.compute_heuristic_initial_guess(
                R, eta_tt, eta_ts, Ma_crit
            )

        # Always normalize initial guess
        initial_guess = self.scale_to_normalized_values(initial_guess)

        # Store keys as an attribute in problem
        # problem.keys = initial_guess.keys()

        return initial_guess
    
    def compute_heuristic_initial_guess(self, enthalpy_distribution, eta_tt, eta_ts, Ma_crit):
        
        # Load object attributes
        geometry = self.geometry
        operation_point = self.operation_point
        fluid = self.fluid
        
        # Rename variables
        number_of_cascades = geometry["number_of_cascades"]
        p0_in = operation_point["p0_in"]
        T0_in = operation_point["T0_in"]
        alpha_in = operation_point["alpha_in"]
        angular_speed = self.angular_speed
        p_out = operation_point["p_out"]
        h_out_s = self.reference_values["h_out_s"]
        
        # Calculate inlet stagnation state
        stagnation_properties_in = fluid.get_props(CP.PT_INPUTS, p0_in, T0_in)
        h0_in = stagnation_properties_in["h"]
        s_in = stagnation_properties_in["s"]
        
        # Calculate exit enthalpy
        h0_out = h0_in - eta_ts * (h0_in - h_out_s)
        v_out = np.sqrt(2 * (h0_in - h_out_s - (h0_in - h0_out) / eta_tt))
        h_out = h0_out - 0.5 * v_out**2

        # Calculate exit static state for expansion with guessed efficiency
        static_properties_exit = fluid.get_props(CP.HmassP_INPUTS, h_out, p_out)
        s_out = static_properties_exit["s"]

        # Define entropy distribution
        entropy_distribution= np.linspace(s_in, s_out, number_of_cascades + 1)
        
        # Define relative stagnation state
        h0_rel_in = h0_in
        
        # Define initial guess dictionary
        initial_guess = {}
        
        for i in range(number_of_cascades):
            
            geometry_cascade = {key: values[i] for key, values in geometry.items()
                if key not in ["number_of_cascades", "number_of_stages"]
            }
            
            # Load enthalpy from initial guess
            h_in = enthalpy_distribution[i]
            h_throat = enthalpy_distribution[i+1]
            h_out = enthalpy_distribution[i+2]
            
            # Load entropy from assumed entropy distribution
            s_in = enthalpy_distribution[i]
            s_throat = entropy_distribution[i+1]
            s_out = entropy_distribution[i+1]
            
            # Rename necessary geometry
            theta_out = geometry_cascade["metal_angle_te"]
            A_out = geometry_cascade["A_out"]
            A_in = geometry_cascade["A_in"]
            radius_mean_in = geometry_cascade["radius_mean_in"]
            radius_mean_throat = geometry_cascade["radius_mean_throat"]
            radius_mean_out = geometry_cascade["radius_mean_out"]
            
            # Calculate rothalpy stgnation enthalpy
            blade_speed_in = angular_speed*(i%2)*radius_mean_in
            rothalpy = h0_rel_in - 0.5 * blade_speed_in**2
            
            # Calculate throat velocity from rothalpy and enthalpy distirbution
            blade_speed_throat = angular_speed*(i%2)*radius_mean_throat
            h0_rel_throat = rothalpy+0.5*blade_speed_throat**2
            w_throat = np.sqrt(2*(h0_rel_throat-h_throat)) 
            
            # Calculate throat velocity from rothalpy and enthalpy distirbution
            blade_speed_out = angular_speed*(i%2)*radius_mean_out
            h0_rel_out = rothalpy+0.5*blade_speed_out**2
            w_out = np.sqrt(2*(h0_rel_out-h_out)) 
            
            # Calculate static state at cascade inlet
            static_state_in = fluid.get_props(CP.HmassSmass_INPUTS, h_in, s_in)
            rho_in = static_state_in["d"]
            
            # Calculate static state at cascade exit
            static_state_out = fluid.get_props(CP.HmassSmass_INPUTS, h_out, s_out)
            a_out = static_state_out["a"]
            rho_out = static_state_out["d"]
            m_crit = a_out*np.cosd(theta_out)*rho_out*A_out
            
            v_m_in_crit = m_crit/rho_in/A_in
            v_in_crit = v_m_in_crit/np.cosd(alpha_in)
            
            # Store initial guess
            initial_guess.update({"w_throat" : w_throat,
                                  "w_out" : w_out,
                                  "s_throat" : s_throat,
                                  "s_out" : s_out,
                                  "v*_in" : v_in_crit,
                                  "w*_out" : a_out*Ma_crit,
                                  "s*_out" : s_out})
            
            # update variables for next cascade

    def compute_heuristic_initial_guess(self, enthalpy_distribution, eta_tt, eta_ts, Ma_crit):
        """
        Generate an initial guess for the root-finder and design optimization.

        Parameters
        ----------
            cascades_data (dict): Data structure containing boundary conditions, geometry, etc.
            eta (float): Efficiency guess (default is 0.9).
            R (float): Degree of reaction guess (default is 0.4).
            Ma_crit (float): Critical Mach number guess (default is 0.92).

        Returns
        -------
            numpy.ndarray: Initial guess for the root-finder.
        """

        # TODO Lasse - Extend logic to multi-stage turbines
        # TODO Lasse - Use dictionary keys instead of array indices
        # TODO Lasse - Update docstring

        # Load necessary parameters
        fluid = self.fluid
        p0_in = self.boundary_conditions["p0_in"]
        T0_in = self.boundary_conditions["T0_in"]
        p_out = self.boundary_conditions["p_out"]
        angular_speed = self.boundary_conditions["omega"]
        number_of_cascades = self.geometry["number_of_cascades"]
        number_of_stages = self.geometry["number_of_stages"]
        h_out_s = self.reference_values["h_out_s"]
        geometry = self.geometry

        # Calculate inlet stagnation state
        stagnation_properties_in = fluid.get_props(CP.PT_INPUTS, p0_in, T0_in)
        h0_in = stagnation_properties_in["h"]
        d0_in = stagnation_properties_in["d"]
        s_in = stagnation_properties_in["s"]
        
        # Calculate exit enthalpy
        h0_out = h0_in - eta_ts * (h0_in - h_out_s)
        v_out = np.sqrt(2 * (h0_in - h_out_s - (h0_in - h0_out) / eta_tt))
        h_out = h0_out - 0.5 * v_out**2

        # Calculate exit static state for expansion with guessed efficiency
        static_properties_exit = fluid.get_props(CP.HmassP_INPUTS, h_out, p_out)
        s_out = static_properties_exit["s"]
        d_out = static_properties_exit["d"]

        # Assume same entropy production in each cascade
        entropy_distribution= np.linspace(s_in, s_out, number_of_cascades + 1)

        # Initialize x0
        x0 = np.array([])
        x0_crit = np.array([])
        labels = ["v_in"]

        # Assume d1 = d01 for first inlet
        d1 = d0_in

        if number_of_stages != 0:
            h_stages = np.linspace(
                h0_in, h_out, number_of_stages + 1
            )  # Divide enthalpy loss equally between stages
            h_in = h_stages[0:-1]  # Enthalpy at inlet of each stage
            h_out = h_stages[1:]  # Enthalpy at exit of each stage
            h_mid = h_out + R * (
                h_in - h_out
            )  # Enthalpy at stator exit for each stage (computed by degree of reaction)
            h01 = h0_in

            for i in range(number_of_stages):
                # Iterate through each stage to calculate guess for velocity
                index_stator = i * 2
                index_rotor = i * 2 + 1

                # Define cascade for labelling
                cascade = "_" + str(index_stator + 1)

                # 3 stations: 1: stator inlet 2: stator exit/rotor inlet, 3: rotor exit
                # Rename parameters
                A1 = geometry["A_in"][index_stator]
                A2 = geometry["A_out"][index_stator]
                A3 = geometry["A_out"][index_rotor]

                alpha1 = geometry["metal_angle_le"][index_stator]
                alpha2 = geometry["metal_angle_te"][index_stator]
                beta2 = geometry["metal_angle_le"][index_rotor]
                beta3 = geometry["metal_angle_te"][index_rotor]
                radius2 = geometry["radius_mean_in"][index_rotor]
                radius3 = geometry["radius_mean_out"][index_rotor]

                h1 = h_in[i]
                h2 = h_mid[i]
                h3 = h_out[i]

                s1 = s_cascades[i * 2]
                s2 = s_cascades[i * 2 + 1]
                s3 = s_cascades[i * 2 + 2]

                # Condition at stator exit
                h02 = h01
                v2 = np.sqrt(2 * (h02 - h2))
                u2 = radius2 * angular_speed
                vel2_data = cs.evaluate_velocity_triangle_in(u2, v2, alpha2)
                w2 = vel2_data["w"]
                h0rel2 = h2 + 0.5 * w2**2
                x0 = np.append(x0, np.array([v2, v2, s2, s2, alpha2]))
                labels += [
                    "w_throat" + cascade,
                    "w_out" + cascade,
                    "s_throat" + cascade,
                    "s_out" + cascade,
                    "beta_out" + cascade,
                ]

                # Critical condition at stator exit
                static_properties_2 = fluid.get_props(CP.HmassSmass_INPUTS, h2, s2)
                a2 = static_properties_2["a"]
                d2 = static_properties_2["d"]
                h2_crit = h01 - 0.5 * a2**2
                static_properties_2_crit = fluid.get_props(
                    CP.HmassSmass_INPUTS, h2_crit, s2
                )
                d2_crit = static_properties_2_crit["d"]
                v2_crit = Ma_crit * a2
                m2_crit = d2_crit * v2_crit * math.cosd(alpha2) * A2
                vm1_crit = m2_crit / (d1 * A1)
                v1_crit = vm1_crit / math.cosd(alpha1)
                x0_crit = np.append(x0_crit, np.array([v1_crit, v2_crit, s2]))
                # x0 = np.append(x0, np.array([v1_crit, v2_crit, s2]))
                labels += ["v*_in" + cascade, "w*_out" + cascade, "s*_out" + cascade]

                # Update casacde for labelling
                cascade = "_" + str(index_rotor + 1)

                # Condition at rotor exit
                w3 = np.sqrt(2 * (h0rel2 - h3))
                u3 = radius3 * angular_speed
                vel3_data = cs.evaluate_velocity_triangle_out(u3, w3, beta3)
                v3 = vel3_data["v"]
                vm3 = vel3_data["v_m"]
                h03 = h3 + 0.5 * v3**2
                x0 = np.append(x0, np.array([w3, w3, s3, s3, beta3]))
                labels += [
                    "w_throat" + cascade,
                    "w_out" + cascade,
                    "s_throat" + cascade,
                    "s_out" + cascade,
                    "beta_out" + cascade,
                ]

                # Critical condition at rotor exit
                static_properties_3 = fluid.get_props(CP.HmassSmass_INPUTS, h3, s3)
                a3 = static_properties_3["a"]
                h3_crit = h0rel2 - 0.5 * a3**2
                static_properties_3_crit = fluid.get_props(
                    CP.HmassSmass_INPUTS, h3_crit, s3
                )
                d3_crit = static_properties_3_crit["d"]
                v3_crit = Ma_crit * a3
                m3_crit = d3_crit * v3_crit * math.cosd(beta3) * A3
                vm2_crit = m3_crit / (d2 * A2)
                v2_crit = vm2_crit / math.cosd(alpha2)
                x0_crit = np.append(x0_crit, np.array([v2_crit, v3_crit, s3]))
                vm1_crit = m2_crit / (d1 * A1)
                v1_crit = vm1_crit / math.cosd(alpha1)
                # x0 = np.append(x0, np.array([v1_crit, v2_crit, s2]))
                labels += ["v*_in" + cascade, "w*_out" + cascade, "s*_out" + cascade]

                # Inlet stagnation state for next cascade equal stagnation state for current cascade
                h01 = h03
                d1 = static_properties_3["d"]

            # Inlet guess from mass convervation
            A_in = geometry["A_in"][0]
            A_out = geometry["A_out"][number_of_cascades - 1]
            m_out = d_out * vm3 * A_out
            v_in = m_out / (A_in * d0_in)
            x0 = np.insert(x0, 0, v_in)

        else:
            # For only the case of only one cascade we split into two station: 1: cascade inlet, 3: cascade exit
            h01 = h0_in
            h3 = h_out
            s3 = s_out
            d3 = d_out

            alpha1 = geometry["metal_angle_in"][0]
            alpha3 = geometry["metal_angle_out"][0]
            A1 = geometry["A_in"][0]
            A_in = A1
            A3 = geometry["A_out"][0]

            v3 = np.sqrt(2 * (h01 - h3))
            vm3 = v3 * math.cosd(alpha3)
            x0 = np.append(x0, np.array([v3, v3, s3, s3, alpha3]))

            # Critical conditions
            static_properties_3 = fluid.get_props(CP.PSmass_INPUTS, h3, s3)
            a3 = static_properties_3["a"]
            h3_crit = h01 - 0.5 * a3**2
            static_properties_3_crit = fluid.get_props(
                CP.HmassSmass_INPUTS, h3_crit, s3
            )
            d3_crit = static_properties_3_crit["d"]
            v3_crit = a3 * Ma_crit
            m3_crit = d3_crit * v3_crit * math.cosd(alpha3) * A3
            vm1_crit = m3_crit / (d1 * A1)
            v1_crit = vm1_crit / math.cosd(alpha1)
            x0_crit = np.array([v1_crit, v3_crit, s3])

            # Inlet guess from mass convervation
            m3 = d3 * vm3 * A3
            v1 = m3 / (A1 * d1)
            x0 = np.insert(x0, 0, v1)

        # Merge initial guess for critical state and actual point
        x0 = np.concatenate((x0, x0_crit))

        return x0


# class CascadesOptimizationProblem(OptimizationProblem):
#     def __init__(self, cascades_data, R, eta_tt, eta_ts, Ma_crit, x0=None):
#         cs.calculate_number_of_stages(cascades_data)
#         cs.update_fixed_params(cascades_data)
#         cs.check_geometry(cascades_data)

#         # Define reference mass flow rate
#         v0 = cascades_data["fixed_params"]["v0"]
#         d_in = cascades_data["fixed_params"]["d0_in"]
#         A_in = cascades_data["geometry"]["A_in"][0]
#         m_ref = A_in * v0 * d_in  # Reference mass flow rate
#         cascades_data["fixed_params"]["m_ref"] = m_ref

#         if x0 == None:
#             x0 = cs.generate_initial_guess(cascades_data, R, eta_tt, eta_ts, Ma_crit)
#         self.x0 = cs.scale_to_normalized_values(x0, cascades_data)
#         self.cascades_data = cascades_data

#     def get_values(self, vars):
#         residuals = cs.evaluate_cascade_series(vars, self.cascades_data)
#         self.f = 0
#         self.c_eq = residuals
#         self.c_ineq = None
#         objective_and_constraints = self.merge_objective_and_constraints(
#             self.f, self.c_eq, self.c_ineq
#         )

#         return objective_and_constraints

#     def get_bounds(self):
#         number_of_cascades = self.cascades_data["geometry"]["number_of_cascades"]
#         lb, ub = cs.get_dof_bounds(number_of_cascades)
#         bounds = [(lb[i], ub[i]) for i in range(len(lb))]
#         return bounds

#     def get_n_eq(self):
#         return self.get_number_of_constraints(self.c_eq)

#     def get_n_ineq(self):
#         return self.get_number_of_constraints(self.c_ineq)
