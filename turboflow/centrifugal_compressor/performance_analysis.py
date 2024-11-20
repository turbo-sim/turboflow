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
from scipy.stats import qmc
from scipy import optimize


SOLVER_MAP = {"lm": "Lavenberg-Marquardt", "hybr": "Powell's hybrid"}
"""
Available solvers for performance analysis.
"""

def compute_performance(
    config,
    operation_points,
    stop_on_failure=False,
    export_results = False,
    out_dir = "output",
    out_filename = None,
):
    # Get list of operating points
    operation_points = generate_operation_points(operation_points)

    # Initialize initial guess
    initial_guess = config["performance_analysis"]["initial_guess"]
    
    # Initialize lists to hold each solution and solver 
    solution_data = []
    solver_container = []

    # Calculate performance at all operation points
    print_operation_points(operation_points)
    for i, operation_point in enumerate(operation_points):
        print(f"\n Computing operation point {i+1} of {len(operation_points)}")
        print_boundary_conditions(operation_point)

        # Update initial guess
        if i > 0:
            closest_x, closest_index = find_closest_operation_point(
                operation_point,
                operation_points[:i],  # Use up to the previous point
                solution_data[:i],  # Use solutions up to the previous point
                solver_container[:i],
            )
            print(f" Using solution from point {closest_index+1} as initial guess")
            initial_guess = closest_x
        # try:
        # Compute performance
        solver = compute_single_operation_point(
                            operation_point,
                            initial_guess,
                            config["geometry"],
                            config["simulation_options"],
                            config["performance_analysis"]["solver_options"],
                            )

        # Collect solution and solver
        solution_data.append(solver.problem.vars_real)
        solver_container.append(solver)

        # except Exception as e:
        #     handle_failure(i, e, stop_on_failure)
        #     solver_container.append(None)
        #     solution_data.append([])

    # Print final report
    print_simulation_summary(solver_container)

    # Export results
    if export_results:
        export_results_excel(solver_container, out_dir, out_filename)

    return solver_container

def handle_failure(point_index, error, stop_on_failure):
    print(f" Computation of point {point_index} failed")
    print(f" Error: {error}")
    if stop_on_failure:
        raise Exception(error)

def export_results_excel(solver_container, out_dir, out_filename):

    """
    Export results to excel file
    Each component has a distinct sheet in the excel file
    """

    RESULTS_KEYS = ["overall", "boundary_conditions", "impeller", "vaneless_diffuser", "vaned_diffuser", "volute"]

    # Initialize results structures
    results = {key : [] for key in solver_container[0].problem.results.keys() if key in RESULTS_KEYS}
    solver_data = []
    
    # Loop through each solver result in the container
    for i, solver in enumerate(solver_container):
        if solver is None:
            # If solver failed, skip or add a placeholder
            for key in results.keys():
                results[key].append({})
            solver_data.append({})
            continue
        
        else:
            # Performance data
            for key in results.keys():
                val = solver.problem.results[key]
                # For components (impeller etc.)
                if isinstance(val, dict) and set(val.keys()).issuperset({"inlet_plane", "exit_plane"}):
                    results[key].append({
                        **utils.add_string_to_keys(val["inlet_plane"], "_in"), 
                        **utils.add_string_to_keys(val["exit_plane"], "_out"),
                        **(utils.add_string_to_keys(val["throat_plane"], "_throat") if "throat_plane" in val else {})
                    })
                
                # For entries with 
                elif isinstance(val, dict):
                    results[key].append(val)

            # Solver data
            solver_data.append({
                "completed": True,
                "success": solver.success,
                "message": solver.message,
                "grad_count": solver.convergence_history["grad_count"][-1],
                "func_count": solver.convergence_history["func_count"][-1],
                "func_count_total": solver.convergence_history["func_count_total"][-1],
                "norm_residual": solver.convergence_history["norm_residual"][-1],
                "norm_step": solver.convergence_history["norm_step"][-1],
            })



    # Convert lists to a DataFrame
    for key, val in results.items():
        results[key] = pd.DataFrame(results[key])
    results["solver"] = pd.DataFrame(solver_data)

    # Create a directory to save simulation results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Define filename with unique date-time identifier
    if out_filename == None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_filename = f"performance_analysis_{current_time}"

    filepath = os.path.join(out_dir, f"{out_filename}.xlsx")
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name, df in results.items():
            # Write each DataFrame to a specific sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Results successfully exported to {filepath}")

    return

def compute_single_operation_point(
    operating_point,
    initial_guess,
    geometry,
    simulation_options,
    solver_options,
):

    # Initialize problem object
    problem = CentrifugalCompressorProblem(geometry, simulation_options)
    problem.update_boundary_conditions(operating_point)
    solver_options = copy.deepcopy(solver_options)

    # Get solver method array
    solver_methods = [solver_options["method"]] + [method for method in SOLVER_MAP.keys() if method != solver_options["method"]]

    # Solve operation point
    initial_guesses = problem.get_initial_guess(initial_guess)
    for initial_guess in initial_guesses:
        initial_guess_scaled = problem.scale_values(initial_guess)
        x0 = np.array(list(initial_guess_scaled.values()))
        problem.keys = list(initial_guess_scaled.keys())
        for method in solver_methods:
            solver_options["method"] = method
            solver = psv.NonlinearSystemSolver(problem, **solver_options)
            try:
                solver.solve(x0)
            except Exception as e:
                if solver.func_count == 0:
                    raise e 
                print(f" Error during solving: {e}")
                solver.success = False
            if solver.success:
                    break
            
        if solver.success:
                    break
    
    if not solver.success:
        print("WARNING: All attempts failed to converge")
        # TODO: Add messages to Log file

    return solver


def find_closest_operation_point(current_op_point, operation_points, solution_data, solver_container):
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
    solver_container : list
        A list of solver objects corresponding to each operation point.

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
        if distance < min_distance and solver_container[i].success:
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


def generate_operation_points(operation_points):
    """
    Generates a list of dictionaries representing all possible combinations of
    operation points from a given input. If the input is a dictionary with ranges,
    it generates all combinations; if it's a list or array, it directly returns the
    operation points.

    Parameters
    ----------
    - operation_points (dict or list/np.ndarray): A dictionary with parameter names as
      keys and lists of parameter values as values, or a precomputed list of
      operation points.

    Returns
    -------
    - operation_points (list of dict): A list of dictionaries, each representing
      a unique combination of parameters from the performance map.
    """
    # If input is already a list or array, return it as operation points
    if isinstance(operation_points, (list, np.ndarray)):
        return operation_points

    # If input is a dictionary, generate operation points by creating all combinations
    if isinstance(operation_points, dict):
        # Ensure all values in the performance_map are iterables
        performance_map = {k: utils.ensure_iterable(v) for k, v in operation_points.items()}

        # Reorder performance map keys so 'p0_in' and 'mass_flow_rate' are prioritized last
        priority_keys = ["p0_in", "mass_flow_rate"]
        other_keys = [k for k in performance_map.keys() if k not in priority_keys]
        keys_order = other_keys + priority_keys
        performance_map = {k: performance_map[k] for k in keys_order if k in performance_map}

        # Generate all combinations of operation points
        keys, values = zip(*performance_map.items())
        operation_points = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        
        return operation_points

    # Raise an error if input is neither a dict nor a list/array
    msg = "operation_points must be either a list of dicts or a dict with ranges."
    raise TypeError(msg)



def validate_operation_point(op_point):
    """
    Validates that an operation point has exactly the required fields:
    'fluid_name', 'p0_in', 'T0_in', 'mass_flow_rate', 'alpha_in', 'omega'.

    Parameters
    ----------
    op_point: dict
        A dictionary representing an operation point.

    Returns
    -------
    ValueError: If the dictionary does not contain the required fields or contains extra fields.
    """
    REQUIRED_FIELDS = {"fluid_name", "p0_in", "T0_in", "mass_flow_rate", "alpha_in", "omega"}
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

class CentrifugalCompressorProblem(psv.NonlinearSystemProblem):

    def __init__(self, geometry, simulation_options):
        
        # Process turbine geometry
        self.geometry = geom.calculate_full_geometry(geometry)

        # Initialize simulation options
        # Specify loss model for each component if not given
        self.model_options = simulation_options
        if set(simulation_options["loss_model"].keys()) == {"model", "loss_coefficient"}:
            loss_model = simulation_options["loss_model"]
            self.model_options["loss_model"] = {key : loss_model for key in self.geometry.keys()}

        # Initialize list of keys for the independent variables
        self.keys = []

    def residual(self, x):
        
        # Create dictionary of scaled variables
        self.vars_scaled = dict(zip(self.keys, x))

        # Create dictionary of real variables
        self.vars_real = self.scale_values(self.vars_scaled, to_normalized=False)
        
        # Evaluate centrifugal compressor
        results = flow.evaluate_centrifugal_compressor(
            self.vars_scaled,
            self.boundary_conditions,
            self.geometry,
            self.fluid,
            self.model_options,
            self.reference_values,
        )

        self.results = results

        return np.array(list(results["residuals"].values()))

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
            A dictionary containing the boundary conditions defining the operation point. 

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
        omega = operation_point["omega"]

        # Compute stagnation properties at inlet
        state_in_stag = self.fluid.get_props(cp.PT_INPUTS, p0_in, T0_in)
        h0_in = state_in_stag["h"]
        s_in = state_in_stag["s"]

        # Store the inlet stagnation (h,s) for the first plane
        self.boundary_conditions["h0_in"] = h0_in
        self.boundary_conditions["s_in"] = s_in

        # Calculate reference velocity
        u_out = omega*self.geometry["impeller"]["radius_out"]
        h0_max = h0_in + 0.5*u_out**2
        ref_props_min = self.fluid.get_props(cp.PSmass_INPUTS, p0_in/2, s_in)
        h_ref_min = ref_props_min["h"]
        v0 = np.sqrt(2*(h0_max - h_ref_min))

        # Calculate max entropy
        s_max = s_in + u_out**2/T0_in

        # Define reference_values
        self.reference_values = {
            "s_range": s_max - s_in,
            "s_min": s_in,
            "v_max": v0,
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
        v0 = self.reference_values["v_max"]
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
            elif key.startswith("b") or key.startswith("a"):
                scaled_variables[key] = (
                    (val - angle_min) / angle_range
                    if to_normalized
                    else val * angle_range + angle_min
                )

        return scaled_variables
    
    def __getstate__(self):

        """
        This function is called when dumping object using pickle.
        This function ensures that attribute types that are not supported by pickle is reset.
        Every action in this function should correspond to an action in __setstate__ 
        """

        # Create a copy of the object's state dictionary
        state = self.__dict__.copy()
        # Remove the unpickleable 'fluid' entry
        state['fluid'] = None
        return state

    def __setstate__(self, state):

        """
        This function is called when loading a pickle file.
        This function ensures that attribute types that are not supported by pickle are restored.
        Every action in this function should correspond to an action in __getstate__ 
        """

        # Restore the attributes
        self.__dict__.update(state)

        # Recreate the 'fluid' attribute
        self.fluid = props.Fluid(self.boundary_conditions["fluid_name"])

    def get_initial_guess(self, initial_guess):

        """
        3 options:
            1. Give guess of efficiency_tt and phi_out. Initial guess calculated heuristically
            2. Give bounds of efficiency_tt and phi_out, and a sample size. Initial guess calculated by the best 
            3. Give complete set (real values)
        """

        valid_keys_1 = ["efficiency_impeller", "phi_impeller", "Ma_vaned_diffuser"]
        valid_keys_2 = ["efficiency_impeller", "phi_impeller", "Ma_vaned_diffuser", "n_samples"]
        check = []
        check.append(set(valid_keys_1) == set(list(initial_guess.keys())))
        check.append(set(valid_keys_2) == set(list(initial_guess.keys())))

        if check[0]:
            # Initial guess determined from heurustic guess, with given parameters
            if isinstance(initial_guess["efficiency_impeller"], (list, np.ndarray)):
                # Several initial guesses
                initial_guesses = []
                for i in range(len(initial_guess["efficiency_impeller"])):
                    guess = {key : val[i] for key, val in initial_guess.items()}
                    heuristic_guess = self.get_heuristic_guess(guess)
                    initial_guesses.append(heuristic_guess)
            else:
                # Single initial guess
                heuristic_guess = self.get_heuristic_guess(initial_guess)
                initial_guesses = [heuristic_guess]

        elif check[1]:
            # Generate initial guess using latin hypercube sampling
            bounds = [initial_guess["efficiency_impeller"], initial_guess["phi_impeller"], initial_guess["Ma_vaned_diffuser"]]

            n_samples = initial_guess["n_samples"]
            heuristic_inputs = latin_hypercube_sampling(bounds, n_samples)
            norm_residuals = np.array([])
            failures = 0
            for heuristic_input in heuristic_inputs:
                try:
                    guess = dict(zip(valid_keys_1, heuristic_input))
                    heuristic_guess = self.get_heuristic_guess(guess)
                    x = self.scale_values(heuristic_guess) # TODO: Add scaling
                    self.keys = x.keys() 
                    x0 = np.array(list(x.values()))
                    residual = self.residual(x0)
                    norm_residuals = np.append(norm_residuals, np.linalg.norm(residual))
                except:
                    failures += 1 
                    norm_residuals = np.append(norm_residuals, np.nan)

            print(f"Generating heuristic inital guesses from latin hypercube sampling")
            print(f"Number of failures: {failures} out of {n_samples} samples")
            print(f"Least norm of residuals: {np.nanmin(norm_residuals)}")
            heuristic_input = heuristic_inputs[np.nanargmin(norm_residuals)]
            initial_guess = dict(zip(valid_keys_1, heuristic_input))
            initial_guess = self.get_heuristic_guess(initial_guess)
            initial_guesses = [initial_guess]
        elif isinstance(initial_guess, dict):
            # Simply return the initial guess given
            initial_guesses = [initial_guess]
        else:
            raise ValueError("Provided initial guess is not valid.")
    
        return initial_guesses
    
    def get_heuristic_guess(self, guess):

        # Check vaneless diffuser model
        vaneless_model = self.model_options["vaneless_diffuser_model"]

        # initialize initial guess dictionary
        heuristic_guess = {}
        input = {}

        # Get initial guess
        for key in self.geometry.keys():
            guess_component = {var : val for var, val in guess.items() if var.endswith(key)}

            if key == "impeller":
                heuristic_guess_component, exit_state = self.get_impeller_guess(guess_component, input)
            elif key == "vaneless_diffuser" and vaneless_model == "algebraic":
                heuristic_guess_component, exit_state = self.get_vaneless_diffuser_guess(guess_component, input)
            elif key == "vaned_diffuser":
                heuristic_guess_component, exit_state = self.get_vaned_diffuser_guess(guess_component, input)
            elif key == "volute":
                heuristic_guess_component, exit_state = self.get_volute_guess(guess_component, input)

            # Store guess
            heuristic_guess = {**heuristic_guess, **heuristic_guess_component}

            # Prepare calculation of next component
            input = exit_state
        
        # Define keys to be deleted based on choking criterion
        choking_criterion = self.model_options["choking_criterion"]
        keys_to_delete = {
        "no_throat": {"w_throat_vaned_diffuser", "w_throat_impeller"},  
        "critical_isentropic_throat": {},  
        }
        # Remove specified keys for the selected choking criterion
        if choking_criterion in keys_to_delete:
            for key in keys_to_delete[choking_criterion]:
                heuristic_guess.pop(key, None)  # Remove key if it exists

        return heuristic_guess
    
    def get_impeller_guess(self, guess, input):

        # Load boundary conditions
        p0_in = self.boundary_conditions["p0_in"]
        T0_in = self.boundary_conditions["T0_in"]
        omega = self.boundary_conditions["omega"]
        mass_flow_rate = self.boundary_conditions["mass_flow_rate"]
        alpha_in = self.boundary_conditions["alpha_in"]

        # Load guess
        eta_tt = guess["efficiency_impeller"]
        phi_out = guess["phi_impeller"]
        
        # Load geometry
        z = self.geometry["impeller"]["number_of_blades"]
        theta_out = self.geometry["impeller"]["trailing_edge_angle"]
        r_out = self.geometry["impeller"]["radius_out"]
        r_in = self.geometry["impeller"]["radius_mean_in"]
        A_out = self.geometry["impeller"]["area_out"]
        A_in = self.geometry["impeller"]["area_in"]

        # Load velocity scale
        v0 = self.reference_values["v_max"]

        # Evaluate inlet thermpdynamic state
        stagnation_props_in = self.fluid.get_props(cp.PT_INPUTS, p0_in, T0_in)
        gamma = stagnation_props_in["cp"]/stagnation_props_in["cv"]
        h0_in = stagnation_props_in["h"]
        a0_in = stagnation_props_in["a"]
        s_in = stagnation_props_in["s"]

        # Approximate impeller exit total pressure
        u_out = r_out*omega
        Ma_out = u_out/a0_in
        slip_factor = 1-np.sqrt(math.cosd(theta_out))/(z**(0.7)*(1+phi_out*math.tand(theta_out)))
        p0_out = (1+(gamma-1)*eta_tt*slip_factor*(1+phi_out*math.tand(theta_out))*Ma_out**2)**(gamma/(gamma-1))*p0_in

        # Get impeller exit total enthalpy and entropy
        isentropic_props_out = self.fluid.get_props(cp.PSmass_INPUTS, p0_out, s_in)
        h0_out_is = isentropic_props_out["h"]
        h0_out = (h0_out_is-h0_in)/eta_tt + h0_in
        stagnation_props_out = self.fluid.get_props(cp.HmassP_INPUTS, h0_out, p0_out)
        s_out = stagnation_props_out["s"]

        # Get inlet velocity
        def calculate_inlet_residual(v_scaled):

            # Calculate velocity triangle
            u_in = r_in*omega
            v_in = v_scaled*v0
            v_m_in = v_in*math.cosd(alpha_in)
            v_t_in = v_in*math.sind(alpha_in)
            w_m_in = v_m_in
            w_t_in = v_t_in - u_in
            w_in = np.sqrt(w_m_in**2 + w_t_in**2)   
            beta_in = math.arctand(w_t_in/w_m_in)

            velocity_triangle_in = {"v_t" : v_t_in,
                    "v_m" : v_m_in,
                    "v" : v_in,
                    "alpha" : alpha_in,
                    "w_t" : w_t_in,
                    "w_m" : w_m_in,
                    "w" : w_in,
                    "beta" : beta_in,
                    "blade_speed" : u_in,
                    }

            # Calculate thermodynamic state
            h_in = h0_in - 0.5*v_in**2
            h0_rel_in = h_in + 0.5*w_in**2
            static_props_in = self.fluid.get_props(cp.HmassSmass_INPUTS, h_in, s_in)
            relative_props_in = self.fluid.get_props(cp.HmassSmass_INPUTS, h0_rel_in, s_in)

            # Calculate mass flow rate nad rothalpy
            m_in = static_props_in["d"]*v_m_in*A_in
            rothalpy_in = h0_rel_in - 0.5*u_in**2

            # Evaluate mass flow rate residual
            res = m_in - mass_flow_rate 

            check_in = {"mass_flow_rate" : m_in,
                        "rothalpy" : rothalpy_in,
                        "s_in" : s_in}

            return res, velocity_triangle_in, check_in
        
        def calculate_inlet_velocity():

            sol = optimize.root_scalar(lambda x: calculate_inlet_residual(x)[0], method = "secant", x0 = 0.2)
            delta_h, velocity_triangle_in, check_in  = calculate_inlet_residual(sol.root)
            return velocity_triangle_in, check_in

        velocity_triangle_in, check_in = calculate_inlet_velocity()

        # Calculate exit state
        v_m_out = phi_out*u_out
        v_t_out = slip_factor*(1+phi_out*math.tand(theta_out))*u_out
        v_out = np.sqrt(v_t_out**2 + v_m_out**2)
        alpha_out = math.arctand(v_t_out/v_m_out)
        w_m_out = v_m_out
        w_t_out = v_t_out - u_out
        w_out = np.sqrt(w_t_out**2 + w_m_out**2)
        beta_out = math.arctand(w_t_out/w_m_out)

        velocity_triangle_out = {"v_t" : v_t_out,
                    "v_m" : v_m_out,
                    "v" : v_out,
                    "alpha" : alpha_out,
                    "w_t" : w_t_out,
                    "w_m" : w_m_out,
                    "w" : w_out,
                    "beta" : beta_out,
                    "blade_speed" : u_out,
                    }
        
        # Get thermophysical properties
        h_out = h0_out -0.5*v_out**2
        h0_rel_out = h_out + 0.5*w_out**2
        static_props_out = self.fluid.get_props(cp.HmassSmass_INPUTS, h_out, s_out)

        # Calculate mass flow rate and rothalpy
        m_out = static_props_out["d"]*v_m_out*A_out
        rothalpy_out = h0_rel_out - 0.5*u_out**2

        exit_plane = {
            **velocity_triangle_out,
            **static_props_out,
        }

        # Store initial guess
        initial_guess = {
                    "v_in_impeller" : velocity_triangle_in["v"],
                    "w_throat_impeller" : velocity_triangle_in["w"]*0.5, # factor to ensure correct solution of thorat velocity
                    "w_out_impeller" : w_out,
                    "beta_out_impeller" : beta_out,
                    "s_out_impeller" : s_out,
                    }
        
        return initial_guess, exit_plane

    def get_vaneless_diffuser_guess(self, guess, input):
        """
        Guess constant alpha
        Simple correlation for static enthalpy loss coefficient
        """

        # Load input
        alpha_in  = input["alpha"]
        v_t_in = input["v_t"]
        v_in = input["v"]
        h_in = input["h"]
        d_in = input["d"]
        s_in = input["s"]

        # Load boundary conditions
        mass_flow_rate = self.boundary_conditions["mass_flow_rate"]

        # Load geometry
        r_out = self.geometry["vaneless_diffuser"]["radius_out"]
        r_in = self.geometry["vaneless_diffuser"]["radius_in"]
        b_in = self.geometry["vaneless_diffuser"]["width_in"]
        A_out = self.geometry["vaneless_diffuser"]["area_out"]

        # Load model options
        Cf = self.model_options["factors"]["skin_friction"]

        # Calculate exit velocity
        alpha_out = alpha_in
        # delta_M = np.exp(-Cf*(r_out-r_in)/(b_in*math.cosd(alpha_in)))
        delta_M = 1
        v_t_out = v_t_in*r_in/r_out*delta_M
        v_out = v_t_out/math.sind(alpha_out)
        v_m_out = v_out*math.cosd(alpha_out)
        velocity_triangle_out = {
            "v" : v_out,
            "v_t" : v_t_out,
            "v_m" : v_m_out,
            "alpha" : alpha_out
        }
        
        # Calculate exit entropy
        d_out = mass_flow_rate/(v_out*math.cosd(alpha_out)*A_out)
        h0_out = h_in + 0.5*v_in**2
        h_out = h0_out - 0.5*v_out**2
        static_props_out = self.fluid.get_props(cp.DmassHmass_INPUTS, d_out, h_out)
        s_out = static_props_out["s"]

        initial_guess = {
                    "v_out_vaneless_diffuser" : v_out,
                    "s_out_vaneless_diffuser" : s_out,
                    "alpha_out_vaneless_diffuser" : alpha_out,
                    }
        
        exit_state = {
            **velocity_triangle_out,
            **static_props_out,
        }

        return initial_guess, exit_state

    def get_vaned_diffuser_guess(self, guess, input):

        """
        Similar as vaneless diffuser
        Assume zero deviation at exit 
        """

        # Load guess
        Ma_out = guess["Ma_vaned_diffuser"]
        
        # Load input
        alpha_in  = input["alpha"]
        v_t_in = input["v_t"]
        v_in = input["v"]
        h_in = input["h"]
        a_in = input["a"]

        # Load boundary conditions
        mass_flow_rate = self.boundary_conditions["mass_flow_rate"]

        # Load geometry
        r_out = self.geometry["vaned_diffuser"]["radius_out"]
        r_in = self.geometry["vaned_diffuser"]["radius_in"]
        b_in = self.geometry["vaned_diffuser"]["width_in"]
        A_out = self.geometry["vaned_diffuser"]["area_out"]
        theta_out = self.geometry["vaned_diffuser"]["trailing_edge_angle"]

        # Load model options
        Cf = self.model_options["factors"]["skin_friction"]

        # Calculate exit velocity
        alpha_out = theta_out*0.9
        delta_M = np.exp(-Cf*(r_out-r_in)/(b_in*math.cosd(alpha_in)))
        # delta_M = 1
        v_t_out = v_t_in*r_in/r_out*delta_M
        v_out = v_t_out/math.sind(alpha_out)
        v_m_out = v_out*math.cosd(alpha_out)
        v_out = 0.2*a_in

        velocity_triangle_out = {
            "v" : v_out,
            "v_t" : v_t_out,
            "v_m" : v_m_out,
            "alpha" : alpha_out
        }
        
        # Calculate exit entropy
        d_out = mass_flow_rate/(v_out*math.cosd(alpha_out)*A_out)
        h0_out = h_in + 0.5*v_in**2
        h_out = h0_out - 0.5*v_out**2
        static_props_out = self.fluid.get_props(cp.DmassHmass_INPUTS, d_out, h_out)
        s_out = static_props_out["s"]

        initial_guess = {
                    "w_throat_vaned_diffuser" : input["v"]*0.5,
                    "v_out_vaned_diffuser" : v_out,
                    "s_out_vaned_diffuser" : s_out,
                    }
        
        exit_state = {
            **velocity_triangle_out,
            **static_props_out,
        }

        return initial_guess, exit_state

    def get_volute_guess(self, input):
        """
        Guess constant density
        Guess single velocity component at the exit
        
        """
        
        # Load input
        d_in = input["d"]
        h_in = input["h"]
        v_in = input["v"]

        # Load boundary conditions
        mass_flow_rate = self.boundary_conditions["mass_flow_rate"]

        # Load geometry
        A_out = self.geometry["volute"]["area_out"]

        # Calculate exit velocity
        d_out = d_in
        v_out = mass_flow_rate/(d_out*A_out)
        velocity_triangle_out = {
            "v" : v_out,
        }


        # Calculate exit entropy
        h0_out = h_in + 0.5*v_in**2
        h_out = h0_out - 0.5*v_out**2
        static_props_out = self.fluid.get_props(cp.DmassHmass_INPUTS, d_out, h_out)
        s_out = static_props_out["s"]

        initial_guess = {
                    "v_out_volute" : v_out,
                    "s_out_volute" : s_out,
                    }

        exit_plane = {
            **velocity_triangle_out,
            **static_props_out
        }

        return initial_guess, exit_plane


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
        if solver is not None and solver.success:
            times.append(solver.elapsed_time)
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
        - `mass_flow_rate` (float): Static pressure at outlet in Pascal.
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
    print(f" {'Mass flow rate: ':<{column_width}} {BC['mass_flow_rate']:<.3f} kg/s")
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
        "mass_flow_rate": {"name": "mass_flow_rate", "unit": "[kg/s]", "width": 12, "decimals": 2},
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
        elif key in ["p0_in"]:  # Pa to kPa
            return value / 1e3
        # elif key in ["mass_flow_rate"]:  # No change
        #     return value
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


def latin_hypercube_sampling(bounds, n_samples):
    """
    Generates samples using Latin Hypercube Sampling.

    Parameters:
    bounds (list of tuples): A list of (min, max) bounds for each variable.
    n_samples (int): The number of samples to generate.

    Returns:
    np.ndarray: An array of shape (n_samples, n_variables) containing the samples.
    """
    n_variables = len(bounds)
    # Create a Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=n_variables, seed = 1)
    
    # Generate samples in the unit hypercube
    unit_hypercube_samples = sampler.random(n=n_samples)
    
    # Scale the samples to the provided bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    scaled_samples = qmc.scale(unit_hypercube_samples, lower_bounds, upper_bounds)

    return scaled_samples

