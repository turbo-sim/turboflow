import os
import yaml
import copy
import time
import datetime
import itertools
import numpy as np
import pandas as pd
import CoolProp as cp
import matplotlib.pyplot as plt

from .. import math
from .. import solver as psv
from .. import utilities as util
from .. import properties as props
from . import geometry_model as geom
from . import flow_model as flow


util.set_plot_options()

SOLVERS_AVAILABLE = [
    "lm",
    "hybr",
]

SOLVER_MAP = {"lm": "Lavenberg-Marquardt", "hybr": "Powell's hybrid"}

#INITIAL_GUESSES = [{"enthalpy_loss_fractions" : [0.5, 0.5], "eta_ts" : 0.8, "eta_tt" : 0.9, "Ma_crit" : 1},
#                    {"enthalpy_loss_fractions" : [0.3, 0.7], "eta_ts" : 0.7, "eta_tt" : 0.8, "Ma_crit" : 1}]

def get_heuristic_guess_input(n):
    
   eta_ts_vec = [0.8, 0.7, 0.6]
    
   array = util.fill_array_with_increment(n)
   enthalpy_distributions = []
   enthalpy_distributions.append(np.ones(n)*1/n)
   enthalpy_distributions.append(array)    
   enthalpy_distributions.append(np.flip(array))
    
   initial_guesses  = []
   for eta_ts in eta_ts_vec:
       for enthalpy_distribution in enthalpy_distributions:            
           initial_guesses.append({"enthalpy_loss_fractions" : enthalpy_distribution,
                                    "eta_ts" : eta_ts,
                                    "eta_tt" : eta_ts+0.1,
                                    "Ma_crit" : 0.95})
           
   return initial_guesses   

def compute_performance(
    operation_points,
    config,
    initial_guess=None,
    out_filename=None,
    out_dir="output",
    stop_on_failure=False,
    export_results=True,
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
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_filename = f"performance_analysis_{current_time}"

    # Export simulation configuration as YAML file
    config_data = {k: v for k, v in config.items() if v}  # Filter empty entries
    config_data = util.convert_numpy_to_python(config_data, precision=12)
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
    geometry_data = []
    solver_container = []

    # Loop through all operation points
    util.print_operation_points(operation_points)
    for i, operation_point in enumerate(operation_points):
        print()
        print(f" Computing operation point {i+1} of {len(operation_points)}")
        util.print_boundary_conditions(operation_point)

        try:
            # Define initial guess
            if i == 0:
                # Use default initial guess for the first operation point
                if initial_guess == None:
                    print("Using default initial guess")
                else:
                    print("Using user defined initial guess")
            else:
                closest_x, closest_index = find_closest_operation_point(
                    operation_point,
                    operation_points[:i],  # Use up to the previous point
                    solution_data[:i],  # Use solutions up to the previous point
                )
                print(f" Using solution from point {closest_index+1} as initial guess")
                initial_guess = closest_x

            # Compute performance
            solver, results = compute_single_operation_point(
                operation_point, initial_guess, config
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
            overall_data.append(results["overall"])
            plane_data.append(util.flatten_dataframe(results["plane"]))
            cascade_data.append(util.flatten_dataframe(results["cascade"]))
            stage_data.append(util.flatten_dataframe(results["stage"]))
            geometry_data.append(util.flatten_dataframe(results["geometry"]))
            solver_data.append(pd.DataFrame([solver_status]))
            solution_data.append(solver.problem.vars_real)
            solver_container.append(solver)

        except Exception as e:
            if stop_on_failure:
                raise Exception(e)
            else:
                print(f" Computation of point {i+1}/{len(operation_points)} failed")
                print(f" Error: {e}")

            # Retrieve solver data
            solver = None
            solver_status = {"completed": False}

            # Collect data
            operation_point_data.append(pd.DataFrame([operation_point]))
            overall_data.append(pd.DataFrame([{}]))
            plane_data.append(pd.DataFrame([{}]))
            cascade_data.append(pd.DataFrame([{}]))
            stage_data.append(pd.DataFrame([{}]))
            geometry_data.append(pd.DataFrame([{}]))
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
        "geometry": pd.concat(geometry_data, ignore_index=True),
        "solver": pd.concat(solver_data, ignore_index=True),
    }

    # Add 'operation_point' column to each dataframe
    for sheet_name, df in dfs.items():
        df.insert(0, "operation_point", range(1, 1 + len(df)))

    # Write dataframes to excel
    if export_results:
        filepath = os.path.join(out_dir, f"{out_filename}.xlsx")
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f" Performance data successfully written to {filepath}")

    # Print final report
    util.print_simulation_summary(solver_container)

    return solver_container


def compute_single_operation_point(
    operating_point,
    initial_guess,
    config,
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
    problem = CascadesNonlinearSystemProblem(config)
    # TODO: A limitation of defining a new problem for each operation point is that the geometry generated and checked once for every point
    # TODO: Performing the computations is not a big problem, but displaying the geometry report for every point can be very long.
    # TODO: Perhaps we could add options of verbosity and perhaps only display the full geometry report when it fails
    problem.update_boundary_conditions(operating_point)
    # initial_guess = problem.get_initial_guess(initial_guess)
    solver_options = copy.deepcopy(config["solver_options"])

    initial_guesses = [initial_guess] + get_heuristic_guess_input(problem.geometry["number_of_cascades"])
    methods_to_try = [solver_options["method"]] + [method for method in SOLVERS_AVAILABLE if method != solver_options["method"]]

    for initial_guess in initial_guesses:
        for method in methods_to_try:
            success = False
            x0 = problem.get_initial_guess(initial_guess)
            print(f" Trying to solve the problem using {SOLVER_MAP[method]} method")
            solver = initialize_solver(problem, problem.x0, solver_options)
    
            try: 
                solution = solver.solve()
                success = solution.success
                
            except Exception as e:
                print(f" Error during solving: {e}")
                success = False
                    
            if success:
                break
        if success:
            break
    
    
    # Attempt solving with the specified method
    # name = SOLVER_MAP[solver_options["method"]]
    # print(f" Trying to solve the problem using {name} method")
    # solver = initialize_solver(problem, problem.x0, solver_options)
    # try:
    #     solution = solver.solve()
    #     success = solution.success
    # except Exception as e:
    #     print(f" Error during solving: {e}")
    #     success = False
    # if not success:
    #     print(f" Solution failed for the {name} method")
      
    # # Attempt solving with Lavenberg-Marquardt method
    # if solver_options["method"] != "lm" and not success:
    #     solver_options["method"] = "lm"
    #     name = SOLVER_MAP[solver_options["method"]]
    #     print(f" Trying to solve the problem using {name} method")
    #     solver = initialize_solver(problem, problem.x0, solver_options)
    #     try:
    #         solution = solver.solve()
    #         success = solution.success
    #     except Exception as e:
    #         print(f" Error during solving: {e}")
    #         success = False
    #     if not success:
    #         print(f" Solution failed for the {name} method")
      
    # # TODO: Attempt solving with optimization algorithms?
      
    # # Attempt solving with a heuristic initial guess
    # # TODO: To be improved with random generation of initial guess within ranges
    # if isinstance(initial_guess, np.ndarray) and not success:
    #     solver_options["method"] = "lm"
    #     name = SOLVER_MAP[solver_options["method"]]
    #     print(f" Trying to solve the problem with a new initial guess")
    #     print(f" Using robust solver: {name}")
    #     initial_guess = problem.get_initial_guess(None)
    #     solver = initialize_solver(problem, problem.x0, solver_options)
    #     try:
    #         solution = solver.solve()
    #         success = solution.success
    #     except Exception as e:
    #         print(f" Error during solving: {e}")
    #         success = False
      
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
        #         print(f" Trying to solve the problem with a new initial guess")
        #         print_dict(x0)
        #         initial_guess = problem.get_initial_guess(x0)
        #         solver = initialize_solver(problem, initial_guess, solver_options)
        #         try:
        #             solution = solver.solve()
        #             success = solution.success
        #         except Exception as e:
        #             print(f" Error during solving: {e}")
        #             success = False
        #         if not success:
        #             print(f" Solution failed for method '{solver_options['method']}'")

    if not success:
        print(" WARNING: All attempts failed to converge")
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

    initial_guess : array-like
        The initial guess for the solver.

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
    solver = psv.NonlinearSystemSolver(
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
    performance_map = {k: util.ensure_iterable(v) for k, v in performance_map.items()}

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


class CascadesNonlinearSystemProblem(psv.NonlinearSystemProblem):
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
        geom.validate_turbine_geometry(config["geometry"])
        self.geometry = geom.calculate_full_geometry(config["geometry"])
        self.geom_info = geom.check_turbine_geometry(self.geometry, display=True)

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

        # Create dictionary of scaled variables
        self.vars_scaled = dict(zip(self.keys, x))

        # Create dictionary of real variables
        self.vars_real = self.scale_values(self.vars_scaled, to_normalized=False)

        # Evaluate cascade series
        self.results = flow.evaluate_axial_turbine(
            self.vars_scaled,
            self.boundary_conditions,
            self.geometry,
            self.fluid,
            self.model_options,
            self.reference_values,
        )

        return np.array(list(self.results["residuals"].values()))

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
        self.fluid = props.FluidCoolProp_2Phase(operation_point["fluid_name"])

        # Rename variables
        p0_in = operation_point["p0_in"]
        T0_in = operation_point["T0_in"]
        p_out = operation_point["p_out"]

        # Compute stagnation properties at inlet
        state_in_stag = self.fluid.get_props(cp.PT_INPUTS, p0_in, T0_in)
        h0_in = state_in_stag["h"]
        s_in = state_in_stag["s"]

        # Store the inlet stagnation (h,s) for the first stage
        # TODO: Improve logic of implementation?
        self.boundary_conditions["h0_in"] = h0_in
        self.boundary_conditions["s_in"] = s_in

        # Calculate exit static properties for a isentropic expansion
        state_out_s = self.fluid.get_props(cp.PSmass_INPUTS, p_out, s_in)
        h_isentropic = state_out_s["h"]
        d_isenthalpic = state_out_s["d"]

        # Calculate exit static properties for a isenthalpic expansion
        state_out_h = self.fluid.get_props(cp.HmassP_INPUTS, h0_in, p_out)
        s_isenthalpic = state_out_h["s"]

        # Calculate spouting velocity
        v0 = np.sqrt(2 * (h0_in - h_isentropic))

        # Define a reference mass flow rate
        A_out = self.geometry["A_out"][-1]
        mass_flow_ref = A_out * v0 * d_isenthalpic

        # Define reference_values
        self.reference_values = {
            "s_range": s_isenthalpic - s_in,
            "s_min": s_in,
            "v0": v0,
            "h_out_s": h_isentropic,
            "d_out_s": d_isenthalpic,
            "mass_flow_ref": mass_flow_ref,
            "angle_range": 180,
            "angle_min": -90,
        }

        return

    def scale_values(self, variables, to_normalized=True):
        """
        Convert values between normalized and real values.

        Parameters
        ----------
        variables: Dictionary containing values to be scaled.
        to_real: If True, scale to real values; if False, scale to normalized values.

        Returns
        -------
        An array of values converted between scales.
        """

        # Load parameters
        v0 = self.reference_values["v0"]
        s_range = self.reference_values["s_range"]
        s_min = self.reference_values["s_min"]
        angle_range = self.reference_values["angle_range"]
        angle_min = self.reference_values["angle_min"]

        # Define dictionary of scaled values
        scaled_variables = {}

        for key, val in variables.items():
            if key.startswith("v") or key.startswith("w"):
                scaled_variables[key] = val / v0 if to_normalized else val * v0
            elif key.startswith("s"):
                scaled_variables[key] = (
                    (val - s_min) / s_range if to_normalized else val * s_range + s_min
                )
            elif key.startswith("b"):
                scaled_variables[key] = (
                    (val - angle_min) / angle_range
                    if to_normalized
                    else val * angle_range + angle_min
                )

        return scaled_variables

    def get_initial_guess(self, initial_guess=None):

        number_of_cascades = self.geometry["number_of_cascades"]

        # Define initial guess
        if isinstance(initial_guess, dict):
            valid_keys_1 = ["enthalpy_loss_fractions", "eta_ts", "eta_tt", "Ma_crit"]
            valid_keys_2 = [
                "v_in",
                "w_throat",
                "w_out",
                "s_throat",
                "s_out",
                "beta_out",
                "v*_in",
                "w*_throat",
                "s*_throat",
                "w*_out",
                "s*_out",
            ]
            valid_keys_3 = ["v_in"]

            for i in range(number_of_cascades):
                index = f"_{i+1}"
                valid_keys_index = [
                    key + index for key in valid_keys_2 if key != "v_in"
                ]
                valid_keys_3 += valid_keys_index

            check = []
            check.append(
                util.check_lists_match(valid_keys_1, list(initial_guess.keys()))
            )
            check.append(
                util.check_lists_match(valid_keys_2, list(initial_guess.keys()))
            )
            check.append(
                util.check_lists_match(valid_keys_3, list(initial_guess.keys()))
            )
            
            if check[0]:
                enthalpy_loss_fractions = initial_guess["enthalpy_loss_fractions"]
                eta_tt = initial_guess["eta_tt"]
                eta_ts = initial_guess["eta_ts"]
                Ma_crit = initial_guess["Ma_crit"]

                # Check that eta_tt, eta_ts and Ma_crit is in reasonable range
                for label, variable in zip(
                    ["eta_tt", "eta_ts", "Ma_crit"], [eta_tt, eta_ts, Ma_crit]
                ):
                    if not 0 <= variable <= 1:
                        raise ValueError(f"{label} should be between {0} and {1}.")

                # Check if enthalpy_loss_fractions is a list or a NumPy array
                if not isinstance(enthalpy_loss_fractions, (list, np.ndarray)):
                    raise ValueError(
                        "enthalpy_loss_fractions must be a list or NumPy array"
                    )

                # Check that enthalpy_loss_fractions is of the same length as the number_of_cascades
                if len(enthalpy_loss_fractions) != number_of_cascades:
                    raise ValueError(
                        f"enthalpy_loss_fractions must be of length {number_of_cascades}"
                    )

                # Check that the sum of enthalpy loss fractions is 1
                sum_fractions = np.sum(enthalpy_loss_fractions)
                epsilon = 1e-8  # Small epsilon value for floating-point comparison
                if not np.isclose(sum_fractions, 1, atol=epsilon):
                    raise ValueError(
                        f"Sum of enthalpy_loss_fractions must be 1 (now: {sum_fractions})."
                    )

                print(" Generating heuristic initial guess with given parameters")
                initial_guess = self.compute_heuristic_initial_guess(
                    enthalpy_loss_fractions, eta_tt, eta_ts, Ma_crit
                )

            # Check if initial guess is given with arrays: {"v_throat" : [215, 260]}
            elif check[1]:
                # Check that all dict values are either a number or a list/array
                if not all(
                    isinstance(val, (int, float, list, np.ndarray))
                    for val in initial_guess.values()
                ):
                    raise ValueError(
                        "All dictionary elements must be either a number, list or NumPy array"
                    )

                # Check that array are of correct length
                if not all(
                    len(val) == number_of_cascades
                    for key, val in initial_guess.items()
                    if not key == "v_in"
                ):
                    raise ValueError(
                        f"All arrays must be of length {number_of_cascades}"
                    )

                # Create dictionary with index
                initial_guess_index = {}
                for key, val in initial_guess.items():
                    if isinstance(val, (list, np.ndarray)):
                        for i in range(len(val)):
                            initial_guess_index[f"{key}_{i+1}"] = val[i]
                    else:
                        initial_guess_index[key] = val

                initial_guess = initial_guess_index

            # Check if initial guess is given with indices: {"v_throat_1" : 215, "v_throat_2" : 260}
            elif check[2]:
                # Check that all values are a number
                if not all(
                    isinstance(val, (int, float)) for val in initial_guess.values()
                ):
                    raise ValueError("All dictionary values must be a float or int")

            else:
                raise ValueError(
                    f"Invalid keys provided for initial_guess. "
                    f"Valid keys include either:"
                    f"{valid_keys_1} \n"
                    f"{valid_keys_2} \n"
                    f"{valid_keys_3} \n"
                )

        elif initial_guess == None:
            enthalpy_loss_fractions = np.full(
                number_of_cascades, 1 / number_of_cascades
            )
            eta_tt = 0.9
            eta_ts = 0.8
            Ma_crit = 0.95

            # Compute initial guess using several approximations
            initial_guess = self.compute_heuristic_initial_guess(
                enthalpy_loss_fractions, eta_tt, eta_ts, Ma_crit
            )
        else:
            raise ValueError("Initial guess must be either None or a dictionary.")


        # Always normalize initial guess
        initial_guess_scaled = self.scale_values(initial_guess)

        # Store labels
        self.keys = initial_guess_scaled.keys()
        self.x0 = np.array(list(initial_guess_scaled.values()))

        return initial_guess_scaled

    def compute_heuristic_initial_guess(
        self, enthalpy_loss_fractions, eta_tt, eta_ts, Ma_crit
    ):
        # Load object attributes
        geometry = self.geometry
        boundary_conditions = self.boundary_conditions
        fluid = self.fluid

        # Rename variables
        number_of_cascades = geometry["number_of_cascades"]
        p0_in = boundary_conditions["p0_in"]
        T0_in = boundary_conditions["T0_in"]
        alpha_in = boundary_conditions["alpha_in"]
        angular_speed = boundary_conditions["omega"]
        p_out = boundary_conditions["p_out"]
        h_out_s = self.reference_values["h_out_s"]

        # Calculate inlet stagnation state
        stagnation_properties_in = fluid.get_props(cp.PT_INPUTS, p0_in, T0_in)
        h0_in = stagnation_properties_in["h"]
        s_in = stagnation_properties_in["s"]
        rho0_in = stagnation_properties_in["d"]

        # Calculate exit enthalpy
        h0_out = h0_in - eta_ts * (h0_in - h_out_s)
        v_out = np.sqrt(2 * (h0_in - h_out_s - (h0_in - h0_out) / eta_tt))
        h_out = h0_out - 0.5 * v_out**2

        # Calculate exit static state for expansion with guessed efficiency
        static_properties_exit = fluid.get_props(cp.HmassP_INPUTS, h_out, p_out)
        s_out = static_properties_exit["s"]

        # Define entropy distribution
        entropy_distribution = np.linspace(s_in, s_out, number_of_cascades + 1)[1:]

        # Define enthalpy distribution
        total_enthalpy_loss = h0_in - h_out
        enthalpy_loss_per_cascade = [
            fraction * total_enthalpy_loss for fraction in enthalpy_loss_fractions
        ]
        enthalpy_distribution = [
            h0_in - sum(enthalpy_loss_per_cascade[: i + 1])
            for i in range(number_of_cascades)
        ]

        # Assums h0_in approx h_in for first inlet
        h_in = h0_in

        # Define initial guess dictionary
        initial_guess = {}

        for i in range(number_of_cascades):
            geometry_cascade = {
                key: values[i]
                for key, values in geometry.items()
                if key not in ["number_of_cascades", "number_of_stages"]
            }

            # Load enthalpy from initial guess
            h_out = enthalpy_distribution[i]

            # Load entropy from assumed entropy distribution
            s_out = entropy_distribution[i]

            # Rename necessary geometry
            theta_in = geometry_cascade["metal_angle_le"]
            theta_out = geometry_cascade["metal_angle_te"]
            A_out = geometry_cascade["A_out"]
            A_throat = geometry_cascade["A_throat"]
            A_in = geometry_cascade["A_in"]
            radius_mean_in = geometry_cascade["radius_mean_in"]
            radius_mean_throat = geometry_cascade["radius_mean_throat"]
            radius_mean_out = geometry_cascade["radius_mean_out"]

            # Calculate rothalpy at inlet of cascade
            blade_speed_in = angular_speed * (i % 2) * radius_mean_in
            if i == 0:
                h0_rel_in = h_in
                m_temp = rho0_in * A_in * math.cosd(alpha_in)
            else:
                v_in = np.sqrt(2 * (h0_in - h_in))
                velocity_triangle_in = flow.evaluate_velocity_triangle_in(
                    blade_speed_in, v_in, alpha_in
                )
                w_in = velocity_triangle_in["w"]
                h0_rel_in = h_in + 0.5 * w_in**2

            rothalpy = h0_rel_in - 0.5 * blade_speed_in**2

            # Calculate static state at cascade inlet
            static_state_in = fluid.get_props(cp.HmassSmass_INPUTS, h_in, s_in)
            rho_in = static_state_in["d"]

            # Calculate exit velocity from rothalpy and enthalpy distirbution
            blade_speed_out = angular_speed * (i % 2) * radius_mean_out
            h0_rel_out = rothalpy + 0.5 * blade_speed_out**2
            w_out = np.sqrt(2 * (h0_rel_out - h_out))
            velocity_triangle_out = flow.evaluate_velocity_triangle_out(
                blade_speed_out, w_out, theta_out
            )
            v_t_out = velocity_triangle_out["v_t"]
            v_m_out = velocity_triangle_out["v_m"]
            v_out = velocity_triangle_out["v"]
            h0_out = h_out + 0.5 * v_out**2

            # Calculate static state at cascade exit
            static_state_out = fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)
            a_out = static_state_out["a"]
            rho_out = static_state_out["d"]

            # Calculate mass flow rate
            mass_flow = rho_out * v_m_out * A_out

            # Calculate throat velocity depending on subsonic or supersonic conditions
            if w_out < a_out * Ma_crit:
                w_throat = w_out
                s_throat = s_out
            else:
                w_throat = a_out * Ma_crit
                blade_speed_throat = angular_speed * (i % 2) * radius_mean_throat
                h0_rel_throat = rothalpy + 0.5 * blade_speed_throat**2
                h_throat = h0_rel_throat - 0.5 * w_throat**2
                rho_throat = rho_out
                static_state_throat = fluid.get_props(
                    cp.DmassHmass_INPUTS, rho_throat, h_throat
                )
                s_throat = static_state_throat["s"]

            # Calculate critical state
            w_throat_crit = a_out * Ma_crit
            h_throat_crit = h0_rel_in - 0.5 * w_throat_crit**2 # FIXME: h0_rel_in works less good?
            static_state_throat_crit = fluid.get_props(
                cp.HmassSmass_INPUTS, h_throat_crit, s_throat
            )
            rho_throat_crit = static_state_throat_crit["d"]
            m_crit = w_throat_crit * math.cosd(theta_out) * rho_throat_crit * A_throat
            w_m_in_crit = m_crit / rho_in / A_in
            w_in_crit = w_m_in_crit / math.cosd(theta_in)  # XXX Works better with metal angle than inlet flow angle?
            velocity_triangle_crit_in = flow.evaluate_velocity_triangle_out(
                blade_speed_in, w_in_crit, theta_in)
            v_in_crit = velocity_triangle_crit_in["v"]
            
            rho_out_crit = rho_throat_crit
            w_out_crit = m_crit/(rho_out_crit*A_out*math.cosd(theta_out))

            # Store initial guess
            index = f"_{i+1}"
            initial_guess.update(
                {
                    "w_throat" + index: w_throat,
                    "w_out" + index: w_out,
                    "s_throat" + index: s_throat,
                    "s_out" + index: s_out,
                    "beta_out" + index: theta_out,
                    "v*_in" + index: v_in_crit,
                    "w*_throat" + index: w_throat_crit,
                    "s*_throat" + index: s_throat,
                    "w*_out" + index: w_out_crit,
                    "s*_out" + index: s_out,
                }
            )

            # Update variables for next cascade
            if i != (number_of_cascades - 1):
                A_next = geometry["A_in"][i + 1]
                radius_mean_next = geometry["radius_mean_in"][i + 1]
                v_m_in = v_m_out * A_out / A_next
                v_t_in = v_t_out * radius_mean_out / radius_mean_next
                v_in = np.sqrt(v_m_in**2 + v_t_in**2)
                alpha_in = math.arctand(v_t_in / v_m_in)
                h0_in = h0_out
                h_in = h0_in - 0.5 * v_in**2
                s_in = s_out

        # Calculate inlet velocity from
        initial_guess["v_in"] = mass_flow / m_temp

        return initial_guess


# class CascadesOptimizationProblem(OptimizationProblem):
#     def __init__(self, cascades_data, R, eta_tt, eta_ts, Ma_crit, x0=None):
#         flow.calculate_number_of_stages(cascades_data)
#         flow.update_fixed_params(cascades_data)
#         flow.check_geometry(cascades_data)

#         # Define reference mass flow rate
#         v0 = cascades_data["fixed_params"]["v0"]
#         d_in = cascades_data["fixed_params"]["d0_in"]
#         A_in = cascades_data["geometry"]["A_in"][0]
#         m_ref = A_in * v0 * d_in  # Reference mass flow rate
#         cascades_data["fixed_params"]["m_ref"] = m_ref

#         if x0 == None:
#             x0 = flow.generate_initial_guess(cascades_data, R, eta_tt, eta_ts, Ma_crit)
#         self.x0 = flow.scale_to_normalized_values(x0, cascades_data)
#         self.cascades_data = cascades_data

#     def get_values(self, vars):
#         residuals = flow.evaluate_cascade_series(vars, self.cascades_data)
#         self.f = 0
#         self.c_eq = residuals
#         self.c_ineq = None
#         objective_and_constraints = self.merge_objective_and_constraints(
#             self.f, self.c_eq, self.c_ineq
#         )

#         return objective_and_constraints

#     def get_bounds(self):
#         number_of_cascades = self.cascades_data["geometry"]["number_of_cascades"]
#         lb, ub = flow.get_dof_bounds(number_of_cascades)
#         bounds = [(lb[i], ub[i]) for i in range(len(lb))]
#         return bounds

#     def get_n_eq(self):
#         return self.get_number_of_constraints(self.c_eq)

#     def get_n_ineq(self):
#         return self.get_number_of_constraints(self.c_ineq)
