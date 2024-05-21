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
from .. import pysolver_view as psv
from .. import utilities as utils
from .. import properties as props
from . import geometry_model as geom
from . import flow_model as flow

SOLVER_MAP = {"lm": "Lavenberg-Marquardt", "hybr": "Powell's hybrid"}
"""
Available solvers for performance analysis.
"""


def get_heuristic_guess_input(n):
   
   r"""
   
   Generate a list of `n` number of different sets of dictionaries used to generate initial guesses.

   Total-to-total efficiency, total-to-static efficiency, enthalpy loss fractions for each cascade and critical mach can be used to generate
   an initial guess for performance analysis. This function generate a list of such different sets used to generate a full initial guess through the function `compute_heuristic_initial_guess`.
   
   The total-to-static efficiency, varies between  0.6, 0.7, or 0.8, while the total-to-total efficiency can be 0.7, 0.8 or 0.9. The enthalpy loss fractions can either split equally between the cascade,
   or with equal increments for neighbouring values (see `utils.fill_array_with_increment` for more information). The critical mach is assumed to be 0.95.

   Parameters
    ----------
    n : int
        Number of cascades.

    Returns
    -------
    list
        List of sets of variables used to generate an initial guess.

   """
    
   eta_ts_vec = [0.8, 0.7, 0.6]
    
   array = utils.fill_array_with_increment(n)
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
    r"""
    Compute and export the performance of each specified operation point to an Excel file.

    This function handles two types of input for operation points:

        1. An explicit list of dictionaries, each detailing a specific operation point.
        2. A dictionary where each key has a range of values, representing the cross-product of all possible operation points. It generates the Cartesian product of these ranges internally.

    For each operation point, it computes performance based on the provided case data and compiles
    the results into an Excel workbook with multiple sheets for various data sections.

    The function validates the input operation points, and if they are given as ranges, it generates
    all possible combinations. Performance is computed for each operation point, and the results are
    then stored in a structured Excel file with separate sheets for each aspect of the data (e.g.,
    overall, plane, cascade, stage, solver, and solution data).

    The initial guess variable is used for the first operation point. If given, it must be a dictionary with the following keys:

        - `enthalpy_loss_fractions`, which is a list containing the assumed fractions of enthalpy loss that occurs for each cascade.
        - `eta_ts`, which is the assumed total-to-static efficiency.
        - `eta_tt`, which is the assumed total-to-total efficiency.
        - `Ma_crit`, which is the assumed critical mash number.

    It can also be a dictionary containing the full set of initial guess that is provided directly to the solver. This 
    require care as the user must have a complete knowledge of the different variables, and setup, of the initial guess that must be given that 
    corresponds with the rest of the configuration file. If the initial guess is not given, it is set to a default value. 
    For subsequent operation points, the function employs a strategy to use the closest previously computed operation point's solution
    as the initial guess. This approach is based on the heuristic that similar operation points have similar
    performance characteristics, which can improve convergence speed and robustness of the solution process.
    If the solution fails to converge, a set of initial guesses is provided to try other guesses (see `get_heuristic_guess_input`).

    The function returns a list of solver object for each operation point. This contain information on both solver related performance (see psv.NonlinearSystemSolver) 
    and the object of the performance analysis problem (see CascadesNonlinearSystemProblem). 

    Parameters
    ----------
    operation_points : list of dict or dict
        A list of operation points where each is a dictionary of parameters, or a dictionary of parameter
        ranges from which operation points will be generated.
    config : dict
        A dictionary containing necessary configuration options for computing performance at each operation point.
    initial_guess : optional
        A dictionary with the required elements to generate an initial guess (see description above).
    out_file : str, optional
        The name for the output Excel file. If not provided, a default name with a timestamp is generated.
    out_dir : str, optional
        The directory where the Excel file will be saved. Defaults to 'output'.
    stop_on_failure: bool, optional
        If true, the analysis stops if the solution fails to converge for an operating point.
    export_result : bool, optional
        If true, the result is exported to an excel file. 

    Returns
    -------
    list
        A List of solver object for each operation point.

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
    print_operation_points(operation_points)
    for i, operation_point in enumerate(operation_points):
        print()
        print(f" Computing operation point {i+1} of {len(operation_points)}")
        print_boundary_conditions(operation_point)

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
                "success": solver.success,
                "message": solver.message,
                "grad_count": solver.convergence_history["grad_count"][-1],
                "func_count": solver.convergence_history["func_count"][-1],
                "func_count_total": solver.convergence_history["func_count_total"][-1],
                "norm_residual": solver.convergence_history["norm_residual"][-1],
                "norm_step": solver.convergence_history["norm_step"][-1],
            }

            # Collect results
            operation_point_data.append(pd.DataFrame([operation_point]))
            overall_data.append(results["overall"])
            plane_data.append(utils.flatten_dataframe(results["plane"]))
            cascade_data.append(utils.flatten_dataframe(results["cascade"]))
            # stage_data.append(utils.flatten_dataframe(results["stage"]))
            geometry_data.append(utils.flatten_dataframe(results["geometry"]))
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
            # stage_data.append(pd.DataFrame([{}]))
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
        # "stage": pd.concat(stage_data, ignore_index=True),
        "geometry": pd.concat(geometry_data, ignore_index=True),
        "solver": pd.concat(solver_data, ignore_index=True),
    }

    # Add 'operation_point' column to each dataframe
    for sheet_name, df in dfs.items():
        df.insert(0, "operation_point", range(1, 1 + len(df)))

    # Write dataframes to excel
    if export_results:
        # Create a directory to save simulation results
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Define filename with unique date-time identifier
        if out_filename == None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_filename = f"performance_analysis_{current_time}"

        # Export simulation configuration as YAML file
        config_data = {k: v for k, v in config.items() if v}  # Filter empty entries
        config_data = utils.convert_numpy_to_python(config_data, precision=12)
        config_file = os.path.join(out_dir, f"{out_filename}.yaml")
        with open(config_file, "w") as file:
            yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)

        # Export performance results in excel file
        filepath = os.path.join(out_dir, f"{out_filename}.xlsx")
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f" Performance data successfully written to {filepath}")

    # Print final report
    print_simulation_summary(solver_container)

    return solver_container


def compute_single_operation_point(
    operating_point,
    initial_guess,
    config,
):
    """
    Compute an operation point for a given set of boundary conditions using multiple solver methods and initial guesses.

    The initial guess make take three forms:

        - None, which generate a default initial guess
        - A dictionary which generates an initial guess through `compute_heuristic_initial_guess`. Required elements are:

            - `enthalpy_loss_fractions`, which is a list containing the assumed fractions of enthalpy loss that occurs for each cascade.
            - `eta_ts`, which is the assumed total-to-static efficiency.
            - `eta_tt`, which is the assumed total-to-total efficiency.
            - `Ma_crit`, which is the assumed critical mash number.

        - A dictionary containing the full set of variables needed to evaluate turbine performance. This option require that the user has complete knowledge of what are the required variables, and the setup of the inital guess dictionary for the given configuration. 

    Parameters
    ----------
    boundary_conditions : dict
        A dictionary containing boundary conditions for the operation point.
    initial_guess : dict, optional
        A dictionary with the required elements to generate an initial guess (see description above).
    config : dict
        A dictionary containing necessary configuration options for computing performance at the operational point.

    Returns
    -------
    psv.NonlinearSystemSolver
        The solution object containing the results of the operation point calculation.

    """

    # Initialize problem object
    problem = CascadesNonlinearSystemProblem(config)
    # TODO: A limitation of defining a new problem for each operation point is that the geometry generated and checked once for every point
    # TODO: Performing the computations is not a big problem, but displaying the geometry report for every point can be very long.
    # TODO: Perhaps we could add options of verbosity and perhaps only display the full geometry report when it fails
    problem.update_boundary_conditions(operating_point)
    solver_options = copy.deepcopy(config["solver_options"])

    initial_guesses = [initial_guess] + get_heuristic_guess_input(problem.geometry["number_of_cascades"])
    methods_to_try = [solver_options["method"]] + [method for method in SOLVER_MAP.keys() if method != solver_options["method"]]

    for initial_guess in initial_guesses:
        for method in methods_to_try:
            success = False
            x0 = problem.get_initial_guess(initial_guess)  # TODO: Roberto_17.05.2023: It seems we are not using this value. Why take it as output then?
            print(f" Trying to solve the problem using {SOLVER_MAP[method]} method") 
            solver_options["method"] = method

            solver = psv.NonlinearSystemSolver(problem, **solver_options)
            # TODO: Roberto: add the option to use optimizers as solver depending on the method specified?
            # TODO: Roberto: at some point in the past we tried to solve the system of equations with SLSQP, right?

            try: 
                solver.solve(problem.x0)                
                
            except Exception as e:
                print(f" Error during solving: {e}")
                solver.success = False
                    
            if solver.success:
                break
        if solver.success:
            break
    
    if not solver.success:
        print(" WARNING: All attempts failed to converge")
        # TODO: Add messages to Log file

    return solver, problem.results


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
    performance_map = {k: utils.ensure_iterable(v) for k, v in performance_map.items()}

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
    Derived classes must implement the `residual` method to evaluate the system of equations for a given set of decision variables.

    Additionally, specific problem classes can define the `get_jacobian` method to compute Jacobians.
    If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Attributes
    ----------
    fluid : FluidCoolProp_2Phase
        An instance of the FluidCoolProp_2Phase class representing the fluid properties.
    results : dict
        A dictionary to store results.
    boundary_conditions : dict
        A dictionary containing boundary condition data.
    geometry : dict
        A dictionary containing geometry-related data.
    model_options : dict
        A dictionary containing options related to the analysis model.
    reference_values : dict
        A dictionary containing reference values for calculations.
    vars_scaled
        A dicionary of scaled variables used to evaluate turbine performance.
    vars_real
        A dicionary of real variables used to evaluate turbine performance. 

    Methods
    -------
    get_values(x)
        Evaluate the system of equations for a given set of decision variables.
    update_boundary_conditions(operation_point)
        Update the boundary conditions of the problem with the provided operation point.
    scale_values(variables, to_normalized = False)
        Convert values between normalized and real values.
    get_initial_guess(initial_guess = None)
        Determine the initial guess for the performance analysis based on the given parameters or default values.
    compute_heuristic_initial_guess(enthalpy_loss_fractions, eta_tt, eta_ts, Ma_crit)
        Compute the heuristic initial guess for the performance analysis based on the given parameters.
    print_simulation_summary(solvers)
        Print a formatted footer summarizing the performance of all operation points.
    print_boundary_conditions(BC)
        Print the boundary conditions.
    print_operation_points(operation_points)
        Prints a summary table of operation points scheduled for simulation.

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
        config : dict
            A dictionary containing case-specific data.
        """

        # Process turbine geometry
        geom.validate_turbine_geometry(config["geometry"])
        self.geometry = geom.calculate_full_geometry(config["geometry"])
        # self.geom_info = geom.check_turbine_geometry(self.geometry, display=True)

        # Initialize other attributes
        self.model_options = config["model_options"]
        self.keys = []

    def residual(self, x):
        """
        Evaluate the system of equations for a given set of decision variables.

        Parameters
        ----------
        x : array-like
            Vector of decision variables.

        Returns
        -------
        numpy nd.array
            An array containing residual values.
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

        This method updates the boundary conditions attributes used to evaluate the turbine performance. 
        It also initializes a Fluid object using the 'fluid_name' specified in the operation point.
        The method computes additional properties and reference values like stagnation properties at
        the inlet, exit static properties, spouting velocity, and reference mass flow rate.
        These are stored in the object's internal state for further use in calculations.

        Parameters
        ----------
        operation_point : dict
            A dictionary containing the boundary conditions defining the operation point. It must include the following keys:

            - `fluid_name` (str) : The name of the fluid to be used in the Fluid object.
            - `T0_in` (float): The inlet temperature (in Kelvin).
            - `p0_in` (float): The inlet pressure (in Pascals).
            - `p_out` (float): The outlet pressure (in Pascals).
            - `omega` (float): The rotational speed (in rad/s).
            - `alpha_in` (float): The inlet flow angle (in degrees).

        Returns
        -------
        None
            This method does not return a value but updates the internal state of the object.

        """

        # Define current operating point
        self.boundary_conditions = operation_point

        # Initialize fluid object
        self.fluid = props.Fluid(operation_point["fluid_name"], exceptions=True)

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
        d_isentropic = state_out_s["d"]

        # Calculate exit static properties for a isenthalpic expansion
        state_out_h = self.fluid.get_props(cp.HmassP_INPUTS, h0_in, p_out)
        s_isenthalpic = state_out_h["s"]

        # Calculate spouting velocity
        v0 = np.sqrt(2 * (h0_in - h_isentropic))

        # Define a reference mass flow rate
        A_out = self.geometry["A_out"][-1]
        mass_flow_ref = A_out * v0 * d_isentropic

        # Define reference_values
        self.reference_values = {
            "s_range": s_isenthalpic - s_in,
            "s_min": s_in,
            "v0": v0,
            "h_out_s": h_isentropic,
            "d_out_s": d_isentropic,
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
        variables: dict
            A dictionary containing values to be scaled.
        to_real: bool
            If True, scale to real values; if False, scale to normalized values.

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

        """
        Determine the initial guess for the performance analysis based on the given parameters or default values.

        The given `initial_guess` make take three forms:

            - None, which generate a default initial guess
            - A dictionary which generates an initial guess through `compute_heuristic_initial_guess`. Required elements are:

                - `enthalpy_loss_fractions`, which is a list containing the assumed fractions of enthalpy loss that occurs for each cascade.
                - `eta_ts`, which is the assumed total-to-static efficiency.
                - `eta_tt`, which is the assumed total-to-total efficiency.
                - `Ma_crit`, which is the assumed critical mash number.
                
            - A dictionary containing the full set of variables needed to evaluate turbine performance. This option require that the user has complete knowledge of what are the required variables, and the setup of the inital guess dictionary for the given configuration. 

        The initial guess is scaled in this function. 
            
        Parameters
        ----------
        initial_guess : dict, optional
            A dictionary containing the initial guess parameters.

        Returns
        -------
        dict
            Initial guess for the performance analysis.

        Raises
        ------
        ValueError
            If the provided initial_guess is invalid or incompatible with the model options.

        """

        number_of_cascades = self.geometry["number_of_cascades"]

        # Define initial guess
        if isinstance(initial_guess, dict):
            valid_keys_1 = ["enthalpy_loss_fractions", "eta_ts", "eta_tt", "Ma_crit"]
            valid_keys_2 = [
                "w_out",
                "s_out",
                "beta_out",
                "v*_in",
                "w*_throat",
                "s*_throat",
            ]
            valid_keys_2 = ["v_in"] + [f"{key}_{i+1}" for i in range(number_of_cascades) for key in valid_keys_2]
            valid_keys_3 = [key for key in valid_keys_2 if not key.startswith("v*_in")] + [f"beta*_throat_{i+1}" for i in range(number_of_cascades)]
            valid_keys_4 = [key for key in valid_keys_2 if not (key.startswith("v*_in") or key.startswith("s*_throat") or key.startswith("beta*_throat"))]

            check = []
            check.append(set(valid_keys_1) == set(list(initial_guess.keys())))
            check.append(set(valid_keys_2) == set(list(initial_guess.keys())))
            check.append(set(valid_keys_3) == set(list(initial_guess.keys())))
            check.append(set(valid_keys_4) == set(list(initial_guess.keys())))
            
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

            elif check[1]:
                # Check that set of input correspond with model option
                if not self.model_options["choking_model"] == "evaluate_cascade_critical":
                    raise ValueError("Set of input corresponds with different choking_model (evaluate_cascade_critical)")

                # Check that all values are a number
                if not all(
                    isinstance(val, (int, float)) for val in initial_guess.values()
                ):
                    raise ValueError("All dictionary values must be a float or int")
                
            elif check[2]:
                # Check that set of input correspond with model option
                if not self.model_options["choking_model"] == "evaluate_cascade_throat":
                    raise ValueError("Set of input corresponds with different choking_model (evaluate_cascade_throat)")
                
                # Check that all values are a number
                if not all(
                    isinstance(val, (int, float)) for val in initial_guess.values()
                ):
                    raise ValueError("All dictionary values must be a float or int")
                
            elif check[3]:
                # Check that set of input correspond with model option
                if not self.model_options["choking_model"] == "evaluate_cascade_isentropic_throat":
                    raise ValueError("Set of input corresponds with different choking_model (evaluate_cascade_isentropic_throat)")
                
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
        
        if self.model_options["choking_model"] == "evaluate_cascade_throat":
            initial_guess = {key : val for key, val in initial_guess.items() if not key.startswith("v*_in")}
        elif self.model_options["choking_model"] == "evaluate_cascade_critical":
            initial_guess = {key : val for key, val in initial_guess.items() if not key.startswith("beta*_throat")}
        elif self.model_options["choking_model"] == "evaluate_cascade_isentropic_throat":
            initial_guess = {key : val for key, val in initial_guess.items() if not (key.startswith("v*_in") or key.startswith("s*_throat") or key.startswith("beta*_throat"))}

        # Always normalize initial guess
        initial_guess_scaled = self.scale_values(initial_guess)

        # Store labels
        self.keys = initial_guess_scaled.keys()
        self.x0 = np.array(list(initial_guess_scaled.values()))

        return initial_guess_scaled

    def compute_heuristic_initial_guess(
        self, enthalpy_loss_fractions, eta_tt, eta_ts, Ma_crit
    ):
        
        """
        Compute the heuristic initial guess for the performance analysis based on the given parameters.

        This function calculates the heuristic initial guess based on the provided enthalpy loss fractions for each cascade,
        total-to-static and total-to-total efficiencies, and critical Mach number. 

        Parameters
        ----------
        enthalpy_loss_fractions : array-like
            Enthalpy loss fractions for each cascade.
        eta_tt : float
            Total-to-total efficiency.
        eta_ts : float
            Total-to-static efficiency.
        Ma_crit : float
            Critical Mach number.

        Returns
        -------
        dict
            Heuristic initial guess for the performance analysis.

        """
        
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
            theta_in = geometry_cascade["leading_edge_angle"]
            theta_out = geometry_cascade["gauging_angle"]
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
                    "w_out" + index: w_out,
                    "s_out" + index: s_out,
                    "beta_out" + index: np.sign(theta_out)*math.arccosd(A_throat/A_out),
                    "v*_in" + index: v_in_crit,
                    "beta*_throat" + index : np.sign(theta_out)*math.arccosd(A_throat/A_out),
                    "w*_throat" + index: w_throat_crit,
                    "s*_throat" + index: s_throat,
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


def print_simulation_summary(solvers):
    """
    Print a formatted footer summarizing the performance of all operation points.

    This function processes a list of solver objects to provide a summary of the performance
    analysis calculations. It calculates and displays the number of successful points and a summary of
    simulation tme statistics. Additionally, it lists the indices of failed operation points, if any.

    The function is robust against solvers that failed and lack certain attributes like 'elapsed_time'.
    In such cases, these solvers are included in the count of failed operation points, but not in the
    calculation time statistics.

    Parameters
    ----------
    solvers : list
        A list of solver objects. Each solver object should contain attributes related to the
        calculation of an operation point, such as 'elapsed_time' and the 'solution' status.

    """

    # Initialize times list and track failed points
    times = []
    failed_points = []

    for i, solver in enumerate(solvers):
        # Check if the solver is not None and has the required attribute
        if solver and hasattr(solver, "elapsed_time"):
            times.append(solver.elapsed_time)
            if not solver.success:
                failed_points.append(i)
        else:
            # Handle failed solver or missing attributes
            failed_points.append(i)

    # Convert times to a numpy array for calculations
    times = np.asarray(times)
    total_points = len(solvers)

    # Define footer content
    width = 80
    separator = "-" * width
    lines_to_output = [
        "",
        separator,
        "Final summary of performance analysis calculations".center(width),
        separator,
        f" Simulation successful for {total_points - len(failed_points)} out of {total_points} points",
    ]

    # Add failed points message only if there are failed points
    if failed_points:
        lines_to_output.append(
            f"Failed operation points: {', '.join(map(str, failed_points))}"
        )

    # Add time statistics only if there are valid times
    if times.size > 0:
        lines_to_output.extend(
            [
                f" Average calculation time per operation point: {np.mean(times):.3f} seconds",
                f" Minimum calculation time of all operation points: {np.min(times):.3f} seconds",
                f" Maximum calculation time of all operation points: {np.max(times):.3f} seconds",
                f" Total calculation time for all operation points: {np.sum(times):.3f} seconds",
            ]
        )
    else:
        lines_to_output.append(" No valid calculation times available.")

    lines_to_output.append(separator)
    lines_to_output.append("")

    # Display to stdout
    for line in lines_to_output:
        print(line)



def print_boundary_conditions(BC):
    """
    Print the boundary conditions.

    This function prints the boundary conditions in a formatted manner. It takes a dictionary `BC`
    containing the following keys:

        - `fluid_name` (str): Name of the fluid.
        - `alpha_in` (float): Flow angle at inlet in degrees.
        - `T0_in` (float): Total temperature at inlet in Kelvin.
        - `p0_in` (float): Total pressure at inlet in Pascal.
        - `p_out` (float): Static pressure at outlet in Pascal.
        - `omega` (float): Angular speed in radians per second.

    Parameters
    ----------
    BC : dict
        A dictionary containing the boundary conditions.    

    """

    column_width = 25  # Adjust this to your desired width
    print("-" * 80)
    print(" Operating point: ")
    print("-" * 80)
    print(f" {'Fluid: ':<{column_width}} {BC['fluid_name']:<}")
    print(f" {'Flow angle in: ':<{column_width}} {BC['alpha_in']:<.2f} deg")
    print(f" {'Total temperature in: ':<{column_width}} {BC['T0_in']-273.15:<.2f} degC")
    print(f" {'Total pressure in: ':<{column_width}} {BC['p0_in']/1e5:<.3f} bar")
    print(f" {'Static pressure out: ':<{column_width}} {BC['p_out']/1e5:<.3f} bar")
    print(f" {'Angular speed: ':<{column_width}} {BC['omega']*60/2/np.pi:<.1f} RPM")
    print("-" * 80)
    print()


def print_operation_points(operation_points):
    """
    Prints a summary table of operation points scheduled for simulation.

    This function takes a list of operation point dictionaries, formats them
    according to predefined specifications, applies unit conversions where
    necessary, and prints them in a neatly aligned table with headers and units.

    Parameters
    ----------
    - operation_points (list of dict): A list where each dictionary contains
      key-value pairs representing operation parameters and their corresponding
      values.

    Notes
    -----
    - This function assumes that all necessary keys exist within each operation
      point dictionary.
    - The function directly prints the output; it does not return any value.
    - Unit conversions are hardcoded and specific to known parameters.
    - If the units of the parameters change or if different parameters are added,
      the unit conversion logic and `field_specs` need to be updated accordingly.
    """
    length = 80
    index_width = 8
    output_lines = [
        "-" * length,
        " Summary of operation points scheduled for simulation",
        "-" * length,
    ]

    # Configuration for each field with specified width and decimal places
    field_specs = {
        "fluid_name": {"name": "Fluid", "unit": "", "width": 8},
        "alpha_in": {"name": "angle_in", "unit": "[deg]", "width": 10, "decimals": 1},
        "T0_in": {"name": "T0_in", "unit": "[degC]", "width": 12, "decimals": 2},
        "p0_in": {"name": "p0_in", "unit": "[kPa]", "width": 12, "decimals": 2},
        "p_out": {"name": "p_out", "unit": "[kPa]", "width": 12, "decimals": 2},
        "omega": {"name": "omega", "unit": "[RPM]", "width": 12, "decimals": 0},
    }

    # Create formatted header and unit strings using f-strings and field widths
    header_str = f"{'Index':>{index_width}}"  # Start with "Index" header
    unit_str = f"{'':>{index_width}}"  # Start with empty string for unit alignment

    for spec in field_specs.values():
        header_str += f" {spec['name']:>{spec['width']}}"
        unit_str += f" {spec['unit']:>{spec['width']}}"

    # Append formatted strings to the output lines
    output_lines.append(header_str)
    output_lines.append(unit_str)

    # Unit conversion functions
    def convert_units(key, value):
        if key == "T0_in":  # Convert Kelvin to Celsius
            return value - 273.15
        elif key == "omega":  # Convert rad/s to RPM
            return (value * 60) / (2 * np.pi)
        elif key == "alpha_in":  # Convert radians to degrees
            return np.degrees(value)
        elif key in ["p0_in", "p_out"]:  # Pa to kPa
            return value / 1e3
        return value

    # Process and format each operation point
    for index, op_point in enumerate(operation_points, start=1):
        row = [f"{index:>{index_width}}"]
        for key, spec in field_specs.items():
            value = convert_units(key, op_point[key])
            if isinstance(value, float):
                # Format floats with the specified width and number of decimal places
                row.append(f"{value:>{spec['width']}.{spec['decimals']}f}")
            else:
                # Format strings to the specified width without decimals
                row.append(f"{value:>{spec['width']}}")
        output_lines.append(" ".join(row))  # Ensure spaces between columns

    output_lines.append("-" * length)  # Add a closing separator line

    # Join the lines and print the output
    formatted_output = "\n".join(output_lines)

    for line in output_lines:
        print(line)
    return formatted_output
