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

def compute_single_operation_point(
    operating_point,
    initial_guess,
    geometry,
    simulation_options,
    solver_options,
    out_dir = "output",
    out_filename = None,
    export_results = False,
):

    # Initialize problem object
    problem = CentrifugalCompressorProblem(geometry, simulation_options)
    problem.update_boundary_conditions(operating_point)
    solver_options = copy.deepcopy(solver_options)

    # Get solver method array
    solver_methods = [solver_options["method"]] + [method for method in SOLVER_MAP.keys() if method != solver_options["method"]]

    x0 = np.array(list(initial_guess.values()))
    problem.keys = list(initial_guess.keys())
    solver = psv.NonlinearSystemSolver(problem, **solver_options)
    solver.solve(x0)
    
    if not solver.success:
        print("WARNING: All attempts failed to converge")
        # TODO: Add messages to Log file

    # Export results
    overall = solver.problem.results["overall"]
    overall = {key : [val] for key, val in overall.items()}
    overall = pd.DataFrame(overall)
    planes = solver.problem.results["planes"]

    if export_results:
        if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        # Define filename with unique date-time identifier
        if out_filename == None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_filename = f"cc_{current_time}"

        filepath = os.path.join(out_dir, f"{out_filename}.xlsx")
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Save the dictionary DataFrame to the first sheet
            overall.to_excel(writer, sheet_name='Overall', index=True)
            
            # Save the original DataFrame to the second sheet
            planes.to_excel(writer, sheet_name='Planes', index=True)

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
    are generated such that 'p0_in' and
    'mass_flow_rate' are the last ones to vary, effectively making them the first
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
    priority_keys = ["p0_in", "mass_flow_rate"]
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

        # Initialize other attributes
        self.model_options = simulation_options
        self.keys = []

    def residual(self, x):
        
        # Create dictionary of scaled variables
        self.vars_scaled = dict(zip(self.keys, x))
        
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

        # Compute stagnation properties at inlet
        state_in_stag = self.fluid.get_props(cp.PT_INPUTS, p0_in, T0_in)
        h0_in = state_in_stag["h"]
        s_in = state_in_stag["s"]

        # Store the inlet stagnation (h,s) for the first stage
        self.boundary_conditions["h0_in"] = h0_in
        self.boundary_conditions["s_in"] = s_in

        # Calculate max velocity for a compression to the critical pressure
        # p_crit = self.fluid.critical_point["p"]
        critical_properties = self.fluid._compute_critical_point()
        p_crit = critical_properties["p"]
        properties = self.fluid.get_props(cp.PSmass_INPUTS, p_crit, s_in)
        h_max = properties["h"]
        v_max = np.sqrt(2*(h_max - h0_in))

        # Calculate entropy for zero compression with maximum enthalpy
        properties = self.fluid.get_props(cp.HmassP_INPUTS, h_max, p0_in)
        s_max = properties["s"]

        # Define reference_values
        self.reference_values = {
            "s_range": s_max - s_in,
            "s_min": s_in,
            "v_max": v_max,
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
    print(f" {'Mass flow rate: ':<{column_width}} {BC['mass_flow_rate']/1e5:<.3f} kg/s")
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
        elif key in ["mass_flow_rate"]:  # No change
            return value
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

