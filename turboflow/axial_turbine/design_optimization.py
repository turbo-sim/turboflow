import numpy as np
import pandas as pd
import CoolProp as cp
import datetime
import os
import copy
import yaml
import dill
import warnings

from .. import pysolver_view as psv
from .. import utilities as utils
from . import geometry_model as geom
from . import flow_model as flow
from .. import properties as props
from . import performance_analysis as pa

from .. properties import perfect_gas_props
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # By default jax uses 32 bit, for scientific computing we need 64 bit precision


from functools import reduce

RADIUS_TYPE = ["constant_mean", "constant_hub", "constant_tip"]
ANGLE_KEYS = ["leading_edge_angle", "gauging_angle"]
INDEPENDENT_VARIABLES = [
    "v_in",
    "w_out",
    "s_out",
    "beta_out",
    "v_crit_in",
    "beta_crit_throat",
    "w_crit_throat",
    "s_crit_throat",
]
INDEXED_VARIABLES = [
    "hub_tip_ratio_in",
    "hub_tip_ratio_out",
    "aspect_ratio",
    "pitch_chord_ratio",
    "trailing_edge_thickness_opening_ratio",
    "leading_edge_angle",
    "gauging_angle",
    "throat_location_fraction",
    "leading_edge_diameter",
    "leading_edge_wedge_angle",
    "tip_clearance",
    "cascade_type",
]

GENETIC_ALGORITHMS = psv.GENETIC_SOLVERS
GRADIENT_ALGORITHMS = psv.GRADIENT_SOLVERS

def fitness_gradient(config,step_size):
    problem = CascadesOptimizationProblem(config)

    x = problem.initial_guess
    x_keys = problem.design_variables_keys
    
    grad_jax = jax.jacfwd(problem.fitness, argnums=0)(x)
    grad_FD = psv.approx_gradient(
                problem.fitness,
                x,
                f0=problem.fitness(x),
                method="2-point",
                # abs_step=config["design_optimization"]["solver_options"]["derivative_abs_step"],  ## TODO make sure it works when design variable takes value 0 * np.abs(x),
                abs_step= step_size
            )
    output_dict = problem.output_dict

    # Merge all dictionaries in the list into dict1
    for d in config["design_optimization"]["constraints"]:
        variable_name = d['variable']
        type_value = d['type']

        # To make the key unique, append or prepend the type to the variable name
        unique_key = f"{variable_name}_{type_value}"

        output_dict[unique_key] = d['value']

    output_dict["geometry.flaring_angle_<1"] = 1.0
    output_dict["geometry.flaring_angle_>1"] = 1.0
    # output_keys = list(output_dict.keys())
    fitness_func = problem.fitness(x)
    output = problem.output

    return x_keys, output_dict, output, grad_jax, grad_FD


def evaluate_jax_array(val):
    """
    Helper function to ensure that JAX arrays and tracers are converted to concrete values.
    """
    if isinstance(val, jax.core.Tracer):
        # If val is a JAX Tracer, extract the primal value (concrete array being traced)
        if hasattr(val, "primal"):
            return jax.device_get(val.primal)
        return jax.device_get(val)

    # If the value is a JAX array, convert it
    elif isinstance(val, jax.Array):
        return jax.device_get(val)

    # Handle NumPy scalars
    elif isinstance(val, np.generic):
        return np.asscalar(val)  # Convert to Python scalar

    # If the value is already a NumPy or Python type, return as is
    return val


def compute_optimal_turbine(
    config,
    out_filename=None,
    out_dir="output",
    export_results=True,
    logger=None,
):
    r"""
    Calculate the optimal turbine configuration based on the specified optimization problem.

    The function checks the configuration file before performing design optimization.

    Parameters
    ----------
    config : dict
        A dictionary containing necessary configuration options for computing optimal turbine.
    initial_guess : None or dict
        The initial guess of the design optimization. If None, a defualt initial guess is provided. If given, he initial guess must correspond
        with the given set of design variables given in the configuration dictionary.

    Returns
    -------
    solution : object
        The solution object containing the results of the design optimization.
    """

    # Initialize problem object
    problem = CascadesOptimizationProblem(config)

    # Check that the initial guess is within bounds and clip if necessary
    problem.initial_guess = check_and_clip_initial_guess(
        problem.initial_guess,
        problem.bounds,
        problem.design_variables_keys
    )

    # Perform initial function call to initialize problem
    # This populates the arrays of equality and inequality constraints
    # TODO: it might be more intuitive to create a new method called initialize_problem() that generates the initial guess and evaluates the fitness() function with it
    problem.fitness(problem.initial_guess)

    # Load solver configuration
    solver_config = config["design_optimization"]["solver_options"]

    # Initialize solver object using keyword-argument dictionary unpacking
    solver = psv.OptimizationSolver(problem, **solver_config, logger=logger)

    # Solve optimization problem for initial guess x0
    solver.solve(problem.initial_guess)


    dfs = {
        "operation point": pd.DataFrame(
            {key: [evaluate_jax_array(val)] for key, val in problem.boundary_conditions.items()}
        ),
        "overall": pd.DataFrame(
            {key: evaluate_jax_array(val) for key, val in problem.results["overall"].items()},
            index=[0],
        ),
        "planes": pd.DataFrame(
            {key: evaluate_jax_array(val) for key, val in problem.results["planes"].items()}
        ),
        "cascades": pd.DataFrame(
            {key: evaluate_jax_array(val) for key, val in problem.results["cascades"].items()}
        ),
        "stage": pd.DataFrame(
            {key: evaluate_jax_array(val) for key, val in problem.results["stage"].items()}
        ),
        "geometry": pd.DataFrame(
            {key: pd.Series(evaluate_jax_array(val)) for key, val in problem.geometry.items()}
        ),
        "solver": pd.DataFrame(
            {
                "completed": pd.Series(True, index=[0]),
                "success": pd.Series(solver.success, index=[0]),
                "message": pd.Series(solver.message, index=[0]),
                "elapsed_time": pd.Series(solver.elapsed_time, index=[0]),
                "grad_count": evaluate_jax_array(solver.convergence_history["grad_count"]),
                "func_count": evaluate_jax_array(solver.convergence_history["func_count"]),
                "func_count_total": evaluate_jax_array(solver.convergence_history["func_count_total"]),
                "objective_value": evaluate_jax_array(solver.convergence_history["objective_value"]),
                "constraint_violation": evaluate_jax_array(solver.convergence_history["constraint_violation"]),
                "norm_step": evaluate_jax_array(solver.convergence_history["norm_step"]),
            },
            index=range(len(solver.convergence_history["grad_count"])),
        ),
    }

    if export_results:
        # Create a directory to save simulation results
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Define filename with unique date-time identifier
        if out_filename == None:
            out_filename = "optimization"

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_filenames = [f"{out_filename}_{current_time}", f"{out_filename}_latest"]
        for out_filename in out_filenames:

            # Export simulation configuration as YAML file
            config_data = {k: v for k, v in config.items() if v}  # Filter empty entries
            config_data = utils.convert_numpy_to_python(config_data, precision=12)
            config_file = os.path.join(out_dir, f"{out_filename}.yaml")
            with open(config_file, "w") as file:
                yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)

            # Export optimal turbine in excel file
            filepath = os.path.join(out_dir, f"{out_filename}.xlsx")
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                for sheet_name, df in dfs.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)

            # Export optimal turbine as dill object
            filepath = os.path.join(out_dir, f"{out_filename}.pkl")
            solver.problem = None
            with open(filepath, 'wb') as file:
                # Serialize the object and write it to the file
                dill.dump(solver, file)

    return solver        



class CascadesOptimizationProblem(psv.OptimizationProblem):
    """
    A class representing a turbine design optimization problem.

    This class is designed for solving design optimization problems of axial turbines. It is initialized by providing a
    dictionary containing information on design variables, objective function, constraints and bounds.

    Parameters
    ----------
    config : dictionary
        A dictionary containing necessary configuration options for computing optimal turbine.

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
    design_variables_keys : list
        A list of strings defining the selected design variables.
    objective_function : str
        A string defining the selected objective function for the design optimization problem.
    eq_constraints : list
        A list of strings containing the selected equaility constraints for the design optimization problem.
    ineq_constraints : list
        A list of strings containing the selected inequaility constraints for the design optimization problem.
    bounds : list
        A list of tuples on the form (lower bound, upper bound) defining the bounds for the design optimization problem.
    radius_type : str
        A string deciding what type of turbine geometry should be condidered (constant mean, hub or tip radius).
    vars_scaled : dict
        A dict containing the scaled flow design variables used to evaluate turbine performance.


    Methods
    -------
    fitness(x)
        Evaluate the objective function and constraints at a given point `x`.
    get_bounds()
        Provide the bounds for the design optimization problem.
    get_n_eq()
        Get the number of equality constraints.
    get_n_ineq()
        Get the number of oinequality constraints.
    get_omega(specific_speed, mass_flow, h0_in, d_is, h_is)
        Convert specific speed to actual angular speed
    index_variables(variables)
        Convert a dict containing lists or arrays to a dictionaries of only one element.
    get_given_bounds(design_variable)
        Retrieves the lower and upper bounds for the given design variables.
    get_constraints(self, constraints):
        Converts a list of constraints to equality and inequality constraints.
    update_boundary_conditions(design_point)
        Update the boundary conditions of the turbine analysis problem with the design point of the design optimization problem.
    convert_performance_analysis_results(performance_problem)
        Generate a feasable set of initial guess from a solved performance analysis problem object.
    evaluate_constraints:
        Evaluate constraints
    get_objective_function:
        Change scale for the objective function depending on its type.
    get_nested_value:
        Get values from a dictionary of Dataframes
    """


    def __init__(self, config):
        r"""
        Initialize a CascadesOptimizationProblem.

        Parameters
        ----------
        config : dict
            A dictionary containing case-specific data.

        """

        # Get list of design variables
        config = copy.deepcopy(config)
        self.obj_func = self.get_objective_function(config["design_optimization"]["objective_function"])
        self.radius_type = config["design_optimization"]["radius_type"]
        self.eq_constraints, self.ineq_constraints = self.get_constraints(
            config["design_optimization"]["constraints"]
        )

        # Update design point 
        if isinstance(config["operation_points"], (list, jnp.ndarray)):
            self.update_boundary_conditions(config["operation_points"][0])
        else:
            self.update_boundary_conditions(config["operation_points"])

        # Initialize model options
        self.model_options = config["simulation_options"]

        # Initialize solver options to determine derivative calculation method
        self.solver_options = config["design_optimization"]["solver_options"]

        # Adjust given variables according to choking model (remove excess variables)
        variables = config["design_optimization"]["variables"]
        if self.model_options["choking_criterion"] == "critical_mach_number":
            variables = {
                key: var
                for key, var in variables.items()
                if not key.startswith("v_crit_in")
            }
        elif self.model_options["choking_criterion"] == "critical_mass_flow_rate":
            variables = {
                key: var
                for key, var in variables.items()
                if not key.startswith("beta_crit_throat")
            }
        elif (
            self.model_options["choking_criterion"] == "critical_isentropic_throat"
        ):
            variables = {
                key: var
                for key, var in variables.items()
                if not (
                    key.startswith("v_crit_in")
                    or key.startswith("s_crit_throat")
                    or key.startswith("beta_crit_throat")
                )
            }
        else:
            raise ValueError("STOP")

        # Separate fixed variables and design variables
        fixed_params = {}
        design_variables = {}
        for key, value in variables.items():
            if value["lower_bound"] is not None:
                design_variables[key] = value
            else:
                fixed_params[key] = value

        # Initialize initial guess
        initial_guess = self.index_variables(design_variables)
        self.design_variables_keys = list(initial_guess.keys())
        self.initial_guess = jnp.array(list(initial_guess.values()))

        # Index fixed variables
        self.fixed_params = self.index_variables(fixed_params)

        # Initialize bounds
        lb, ub = self.get_given_bounds(design_variables)
        self.bounds = (lb, ub)

        # Adjust set of independent variables according to choking model
        self.independent_variables = [
            key
            for key in initial_guess.keys()
            if any(key.startswith(var) for var in INDEPENDENT_VARIABLES)
        ]
        if self.model_options["choking_criterion"] == "critical_mach_number":
            self.independent_variables = [
                var for var in self.independent_variables if not var.startswith("v_crit_in")
            ]
        elif self.model_options["choking_criterion"] == "critical_mass_flow_rate":
            self.independent_variables = [
                var
                for var in self.independent_variables
                if not var.startswith("beta_crit_throat")
            ]
        elif (
            self.model_options["choking_criterion"] == "critical_isentropic_throat"
        ):
            self.independent_variables = [
                var
                for var in self.independent_variables
                if not (
                    var.startswith("v_crit_in")
                    or var.startswith("s_crit_throat")
                )
            ]
        else:
            raise ValueError("STOP")


    def fitness(self, x):
        r"""
        Evaluate the objective function and constraints at a given pint `x`.

        Parameters
        ----------
        x : array-like
            Vector of design variables.

        Returns
        -------
        numpy.ndarray
            An array containg the value of the objective function and constraints.
        """

        # Rename reference values
        h0_in = self.boundary_conditions["h0_in"]
        mass_flow = self.reference_values["mass_flow_ref"]
        h_is = self.reference_values["h_out_s"]
        d_is = self.reference_values["d_out_s"]
        v0 = self.reference_values["v0"]
        angle_range = self.reference_values["angle_range"]
        angle_min = self.reference_values["angle_min"]

        # Structure design variables to dictionary (Assume set of design variables)
        design_variables = dict(zip(self.design_variables_keys, x))

        # Construct array with independent variables
        self.vars_scaled = {
            key: design_variables[key] for key in self.independent_variables
        }

        # Contruct variables
        variables = {**design_variables, **self.fixed_params}

        # Calculate angular speed
        specific_speed = variables["specific_speed"]
        self.boundary_conditions["omega"] = self.get_omega(
            specific_speed, mass_flow, h0_in, d_is, h_is
        )

        # Calculate mean radius
        blade_jet_ratio = variables["blade_jet_ratio"]
        blade_speed = blade_jet_ratio * v0
        self.geometry = {}
        self.geometry["radius"] = blade_speed / self.boundary_conditions["omega"]

        # Assign geometry design variables to geometry attribute
        for key in INDEXED_VARIABLES:

            # Extract the values for this key
            values = [v for k, v in variables.items() if k.startswith(key)]
  
            # If the key corresponds to non-numeric data (like 'cascade_type'), store it as a regular list.
            if all(isinstance(v, str) for v in values):  # check if all values are strings
                self.geometry[key] = values  # store as a regular list
            else:
                # Otherwise, convert to JAX array (for numeric values)
                self.geometry[key] = jnp.array(values)
            
            if key in ANGLE_KEYS:
                self.geometry[key] = self.geometry[key] * angle_range + angle_min

        self.geometry = geom.prepare_geometry(self.geometry, self.radius_type)
        self.geometry = geom.calculate_full_geometry(self.geometry)

        # Evaluate turbine model
        self.results = flow.evaluate_axial_turbine(
            self.vars_scaled,
            self.boundary_conditions,
            self.geometry,
            self.fluid,
            self.model_options,
            self.reference_values,
        )

        # Evaluate objective function
        self.f = jnp.atleast_1d(self.get_nested_value(self.results, self.obj_func["variable"])/self.obj_func["scale"]) # self.obj.func on the form "key.column"

        # Evaluate additional constraints
        self.results["additional_constraints"] = {"interspace_area_ratio": self.geometry["A_in"][1:] / self.geometry["A_out"][0:-1]}
        self.output_dict = {}
        self.output_dict.update({"efficiency": self.obj_func})
        self.output_dict.update(self.results["residuals"])

        # Evaluate constraints
        self.c_eq = jnp.array(list(self.results["residuals"].values()))
        self.c_eq = jnp.append(self.c_eq, self.evaluate_constraints(self.eq_constraints))
        self.c_ineq = self.evaluate_constraints(self.ineq_constraints)
        objective_and_constraints = jnp.concatenate([self.f, self.c_eq, self.c_ineq])  
        self.output = objective_and_constraints

        return objective_and_constraints
    

    def gradient(self, x):

        # Use JAX for automatic differentiation
        method = self.solver_options["derivative_method"]
        if method == "jax":   
            grad = jax.jacfwd(self.fitness, argnums=0)(x)

        # Approximate gradient with finite differences
        else:  
            fun = lambda x: self.fitness(x)
            grad = psv.numerical_differentiation.approx_gradient(
                fun,
                x,
                f0=fun(x),
                method=method,
                abs_step=self.solver_options["derivative_abs_step"],
            )

        return grad


    def get_objective_function(self, objective):
        """
        Change scale for the objective function depending on its type.
        If objective function should be maximized, the sign of the scale is changed. 

        Parameters
        ----------
        objective : dict
            dictionary containing variable name, type and scale of the objective function.

        Returns 
        -------
        dict
            dictionary containing modified scale of the objective function
        """

        if objective["type"] == "maximize":
            objective["scale"] *= -1

        return objective

    def index_variables(self, variables):
        """
        Index the design variables for each cascade for the default initial guess.

        The design variables are given without indexing the cascades. For example, the aspect ratio is simply provided by aspect_ratio.
        This function extend the design_variable dictionary such that for each cascade specific design variable, one item is given for each cascade.
        for example `aspect_ratio_1` for first cascade and etc.

        Parameters
        ----------
        variables : dict
            A dictionary where each key maps to a dictionary containing at least a
            "value" key. The "value" can be a list, numpy array, or float.

        Returns
        -------
        dict
            A dictionary where each list or numpy array in the original dictionary
            is expanded into individual elements with keys formatted as "key_index".
            Float values are kept as is.

        """

        variables_extended = {}
        for key in variables.keys():
            if isinstance(variables[key]["value"], (list, jnp.ndarray, np.ndarray)):
                for i in range(len(variables[key]["value"])):
                    variables_extended[f"{key}_{i+1}"] = variables[key]["value"][i]
            elif isinstance(variables[key]["value"], float):
                variables_extended[key] = variables[key]["value"]

        return variables_extended

    def get_given_bounds(self, design_variables):
        """
        Retrieves the lower and upper bounds for the given design variables.

        Parameters
        ----------
        design_variables : dict
            A dictionary where each key maps to a dictionary containing a
            "value" key, a "lower_bound" key, and an "upper_bound" key. The values
            can be a list, numpy array, or float.

        Returns
        -------
        lb : list
            A list of lower bounds for the design variables.
        ub : list
            A list of upper bounds for the design variables.

        """

        lb = []
        ub = []
        for key in design_variables.keys():
            if isinstance(design_variables[key]["value"], list):
                lb += design_variables[key]["lower_bound"]
                ub += design_variables[key]["upper_bound"]
            elif isinstance(design_variables[key]["value"], (np.ndarray, jnp.ndarray)):
                lb += list(design_variables[key]["lower_bound"])
                ub += list(design_variables[key]["upper_bound"])
            elif isinstance(design_variables[key]["value"], float):
                lb += [design_variables[key]["lower_bound"]]
                ub += [design_variables[key]["upper_bound"]]

        return lb, ub
    
    def get_nested_value(self, d, path):
        """
        Get values from a dictionary of Dataframes. 
        Path is on the form `dataframe.column`, and returns `d[dataframe][column]`

        Parameters
        ----------
        d : dict
            Dictionary of DataFrames
        path : str
            String giving the path of the DataFrame values

        Returns
        -------
        numpy.ndarray  
           Array of specified values 
        """

        keys = path.split('.')

        return jnp.array(d[keys[0]][keys[1]])
    
    def evaluate_constraints(self, constraints_list):
        r"""
        Evaluate constraints. 

        This function evaluates the constraints from the information in `constraints_list`.
        Constraints are defined to be less than 0. 

        `constraints_list` is a list of dictionaries, where each dictionary have a `variable`, `scale` and `value` key. 

        Parameters
        ----------
        constraints_list : list
            List of dictionaries with constraint information

        Returns
        -------
        numpy.ndarray
            Array with constraint values
        """
        constraints = jnp.array([])
        for constraint in constraints_list:
            constraints = jnp.append(
                constraints,
                (self.get_nested_value(self.results, constraint["variable"]) - constraint["value"])
                / constraint["scale"],
            )
        return constraints


    def get_bounds(self):
        r"""
        Provide the bounds for the design optimization problem.

        Returns
        -------
        list
            List of toutples containg the lower and upper bound for each design variable.
        """

        return self.bounds

    def get_nec(self):
        r"""
        Get the number of equality constraints.

        Returns
        -------
        int
            Number of equality constraints.

        """

        return psv.count_constraints(self.c_eq)

    def get_nic(self):
        r"""
        Get the number of inequality constraints.

        Returns
        -------
        int
            Number of inequality constraints.

        """
        return psv.count_constraints(self.c_ineq)

    def get_omega(self, specific_speed, mass_flow, h0_in, d_is, h_is):
        r"""
        Convert specific speed to actual angular speed

        Parameters
        ----------
        specific_speed : float
            Given specific speed.
        mass_flow : float
            Given mass flow rate.
        h0_in : float
            Turbine inlet stagnation enthalpy.
        d_is : float
            Turbine exit density for an isentropic expansion.
        h_is : float
            Turbine exit enthalpy for an isentropic expansion.

        Returns
        -------
        float
            Actual angular speed.

        """

        return specific_speed * (h0_in - h_is) ** (3 / 4) / ((mass_flow / d_is) ** 0.5)

    def update_boundary_conditions(self, design_point):
        """
        Update the boundary conditions of the turbine analysis problem with the design point of the design optimization problem.

        This method updates the boundary conditions attributes used to evaluate the turbine performance.
        It also initializes a Fluid object using the 'fluid_name' specified in the operation point.
        The method computes additional properties and reference values like stagnation properties at
        the inlet, exit static properties, spouting velocity, and reference mass flow rate.
        These are stored in the object's internal state for further use in calculations.

        Parameters
        ----------
        design_point : dict
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
            This method does not return a value but updates the attributes of the object.

        """

        # Define current operating point
        self.boundary_conditions = design_point.copy()

        # Initialize fluid object
        self.fluid = props.Fluid(design_point["fluid_name"])

        # Rename variables
        p0_in = design_point["p0_in"]
        T0_in = design_point["T0_in"]
        p_out = design_point["p_out"]

        # Compute stagnation properties at inlet
        # state_in_stag  = self.fluid.get_props(cp.PT_INPUTS, p0_in, T0_in)
        state_in_stag = perfect_gas_props("PT_INPUTS", p0_in, T0_in)
        h0_in = state_in_stag["h"]
        s_in = state_in_stag["s"]

        # Store the inlet stagnation (h,s) for the first stage
        # TODO: Improve logic of implementation?
        self.boundary_conditions["h0_in"] = h0_in
        self.boundary_conditions["s_in"] = s_in

        # Calculate exit static properties for a isentropic expansion
        # state_out_s  = self.fluid.get_props(cp.PSmass_INPUTS, p_out, state_in_stag.s)
        state_out_s = perfect_gas_props("PSmass_INPUTS", p_out, s_in)
        
        h_isentropic = state_out_s["h"]
        d_isentropic = state_out_s["d"]

        # Calculate exit static properties for a isenthalpic expansion
        # state_out_h = self.fluid.get_props(cp.HmassP_INPUTS, state_in_stag.h, p_out)
        state_out_h = perfect_gas_props("HmassP_INPUTS", h0_in, p_out)
        s_isenthalpic = state_out_h["s"]

        # Calculate spouting velocity
        v0 = np.sqrt(2 * (h0_in - h_isentropic))

        mass_flow_rate = None  # Initialize mass_flow_rate to None
        # Try to find mass_flow_rate from eq_constraints
        for constraint in self.eq_constraints:
            if constraint.get("variable") == "overall.mass_flow_rate":
                mass_flow_rate = constraint.get("value")
                break

        # If mass_flow_rate not found, calculate using default formula
        if mass_flow_rate is None:
            mass_flow_rate = v0 * d_isentropic

        # Define reference_values
        self.reference_values = {
            "s_range": s_isenthalpic - s_in,
            "s_min": s_in,
            "v0": v0,
            "h_out_s": h_isentropic,
            "d_out_s": d_isentropic,
            "mass_flow_ref": mass_flow_rate,
            "angle_range": 180,
            "angle_min": -90,
        }

        return

    def convert_performance_analysis_results(self, performance_problem):
        r"""
        Generate a feasable set of initial guess from a solved performance analysis problem object.

        The function adopt the vars_scaled attribute from the solved performance analysis problem, the geometry and
        angular speed to generate an initial guess of correct form to be given to the design optimization problem.

        Returns
        -------
        dict
            Feasable initial guess.
        """

        design_variables = self.design_variables_keys
        radius_type = self.radius_type
        mass_flow = self.reference_values["mass_flow_ref"]

        geometry = performance_problem.geometry
        overall = performance_problem.results["overall"]
        vars_scaled = performance_problem.vars_scaled

        angle_min = self.reference_values["angle_min"]
        angle_range = self.reference_values["angle_range"]
        d_out_s = self.reference_values["d_out_s"]
        h0_in = self.boundary_conditions["h0_in"]
        h_out_s = self.reference_values["h_out_s"]

        initial_guess = {}
        for design_variable in set(design_variables) - {
            "specific_speed",
            "blade_jet_ratio",
        }:
            if "angle" in design_variable:
                added_dict = {
                    design_variable
                    + f"_{i+1}": (geometry[design_variable][i] - angle_min)
                    / angle_range
                    for i in range(len(geometry[design_variable]))
                }
            else:
                added_dict = {
                    design_variable + f"_{i+1}": geometry[design_variable][i]
                    for i in range(len(geometry[design_variable]))
                }
            initial_guess = {**initial_guess, **added_dict}

        if "blade_jet_ratio" in design_variables:
            if radius_type == "constant_mean":
                initial_guess["blade_jet_ratio"] = overall[
                    "blade_jet_ratio_mean"
                ].values[0]
            elif radius_type == "constant_hub":
                initial_guess["blade_jet_ratio"] = overall[
                    "blade_jet_ratio_hub"
                ].values[0]
            elif radius_type == "constant_mean":
                initial_guess["blade_jet_ratio"] = overall[
                    "blade_jet_ratio_tip"
                ].values[0]

        if "specific_speed" in design_variables:
            angular_speed = overall["angular_speed"].values[0]
            initial_guess["specific_speed"] = (
                angular_speed
                * (mass_flow / d_out_s) ** 0.5
                / ((h0_in - h_out_s) ** 0.75)
            )

        return {**vars_scaled, **initial_guess}

    def get_constraints(self, constraints):
        """
        Converts a list of constraints to equality and inequality constraints.

        Parameters
        ----------
        constraints : list of dict
            List of constraints where each constraint is represented as a dictionary
            with keys "type", "variable", "value", and "normalize". "type" represents
            the type of constraint ('=', '<', or '>'). "variable" is the variable
            involved in the constraint. "value" is the value of the constraint. "normalize"
            specifies whether to normalize the constraint value.

        Returns
        -------
        list of dict
            List of equality constraints, each represented as a dictionary with keys
            "variable", "value", and "scale". "variable" is the variable involved
            in the constraint. "value" is the value of the constraint. "scale" is
            the scaling factor applied to the value.
        list of dict
            List of inequality constraints, each represented as a dictionary with keys
            "variable", "value", and "scale". "variable" is the variable involved
            in the constraint. "value" is the value of the constraint. "scale" is
            the scaling factor applied to the value.
        """

        eq_constraints = []
        ineq_constraints = []
        for constraint in constraints:
            if constraint["type"] == "=":
                if constraint["normalize"]:
                    eq_constraints += [
                        {
                            "variable": constraint["variable"],
                            "value": constraint["value"],
                            "scale": abs(constraint["value"]),
                        }
                    ]
                else:
                    eq_constraints += [
                        {
                            "variable": constraint["variable"],
                            "value": constraint["value"],
                            "scale": 1,
                        }
                    ]

            elif constraint["type"] == "<":
                if constraint["normalize"]:
                    ineq_constraints += [
                        {
                            "variable": constraint["variable"],
                            "value": constraint["value"],
                            "scale": abs(constraint["value"]),
                        }
                    ]
                else:
                    ineq_constraints += [
                        {
                            "variable": constraint["variable"],
                            "value": constraint["value"],
                            "scale": 1,
                        }
                    ]

            elif constraint["type"] == ">":
                if constraint["normalize"]:
                    ineq_constraints += [
                        {
                            "variable": constraint["variable"],
                            "value": constraint["value"],
                            "scale": -1 * abs(constraint["value"]),
                        }
                    ]
                else:
                    ineq_constraints += [
                        {
                            "variable": constraint["variable"],
                            "value": constraint["value"],
                            "scale": -1,
                        }
                    ]
        return eq_constraints, ineq_constraints
    
    def __getstate__(self):
        # Create a copy of the object's state dictionary
        state = self.__dict__.copy()
        # Remove the unpickleable 'fluid' entry
        state['fluid'] = None
        return state

    def __setstate__(self, state):
        # Restore the attributes
        self.__dict__.update(state)
        # Recreate the 'fluid' attribute
        self.fluid = props.Fluid(self.boundary_conditions["fluid_name"])

class BlackBoxOptimization:

    def __init__(self):
        self.turbine_results = []
        self.iterations_convergence = np.array([])
        self.iterations_objective_function = np.array([])
        self.iterations_efficiency = np.array([])
        self.iterations_mass_flow_rate = np.array([])
        self.iterations_interspace_flaring = np.array([])
        self.iterations_flaring_1 = np.array([])
        self.iterations_flaring_2 = np.array([])
        self.constraint_violation = np.array([])
        self.root_finder_solutions = np.array([])
        self.failed_iterations = 0
        self.failed_iterations_percentage = 0
    
    def update_optimization_process(self, results, converged = True, f = None, violation = None, vars_scaled = None):

        if converged:
            self.iterations_objective_function = np.append(self.iterations_objective_function, f)
            self.iterations_efficiency = np.append(self.iterations_efficiency, results["overall"]["efficiency_ts"])
            self.iterations_convergence = np.append(self.iterations_convergence, converged)
            self.iterations_mass_flow_rate = np.append(self.iterations_mass_flow_rate, results["overall"]["mass_flow_rate"])
            self.iterations_interspace_flaring = np.append(self.iterations_interspace_flaring, results["additional_constraints"]["interspace_area_ratio"])
            self.iterations_flaring_1 = np.append(self.iterations_flaring_1, results["geometry"]["flaring_angle"].values[0])
            self.iterations_flaring_2 = np.append(self.iterations_flaring_2, results["geometry"]["flaring_angle"].values[1])
            self.constraint_violation = np.append(self.constraint_violation, violation)
            self.root_finder_solutions = np.append(self.root_finder_solutions, vars_scaled)
        else:
            self.iterations_objective_function = np.append(self.iterations_objective_function, np.nan)
            self.iterations_efficiency = np.append(self.iterations_efficiency, np.nan)
            self.iterations_convergence = np.append(self.iterations_convergence, converged)
            self.iterations_mass_flow_rate = np.append(self.iterations_mass_flow_rate, np.nan)
            self.iterations_interspace_flaring = np.append(self.iterations_interspace_flaring, np.nan)
            self.iterations_flaring_1 = np.append(self.iterations_flaring_1, np.nan)
            self.iterations_flaring_2 = np.append(self.iterations_flaring_2, np.nan)
            self.constraint_violation = np.append(self.constraint_violation, np.nan)
            self.root_finder_solutions = np.append(self.root_finder_solutions, np.nan)
            self.failed_iterations += 1

        self.turbine_results.append(results)
        self.failed_iterations_percentage = self.failed_iterations/len(self.iterations_objective_function)

    def print_optimization_process(self):

        print(f'Iteration: {len(self.iterations_objective_function)}')
        print(f'Objective function: {self.iterations_objective_function[-1]}')
        print(f'Mass flow rate: {self.iterations_mass_flow_rate[-1]}')
        print(f'Interspace flaring: {self.iterations_interspace_flaring[-1]}')
        print(f'Stator flaring: {self.iterations_flaring_1[-1]}')
        print(f'Rotor flaring: {self.iterations_flaring_2[-1]}')
        print('\n')

    def find_champion(self, solver):

        champion_i = np.nanargmin(solver.convergence_history["objective_value"])
        champion_x = solver.convergence_history["x"][champion_i]
        solver.problem.fitness(champion_x)

    def export_optimization_process(self, config):
        overall_data = []
        plane_data = []
        cascade_data = []
        stage_data = []
        solver_data = []
        geometry_data = []

        for i in range(len(self.turbine_results)):
            solver_status = {"Success" : self.iterations_convergence[i],
                        "Objective" : self.iterations_objective_function[i],
                        "Efficiency" : self.iterations_efficiency[i],
                        "Mass flow rate" : self.iterations_mass_flow_rate[i],
                        "Interspace flaring" : self.iterations_interspace_flaring[i],
                        "Flaring Stator" : self.iterations_flaring_1[i],
                        "Flaring Rotor" : self.iterations_flaring_2[i],
            }
            results = self.turbine_results[i]
            overall_data.append(results["overall"])
            plane_data.append(utils.flatten_dataframe(results["planes"]))
            cascade_data.append(utils.flatten_dataframe(results["cascades"]))
            stage_data.append(utils.flatten_dataframe(results["stage"]))
            geometry_data.append(utils.flatten_dataframe(results["geometry"]))
            solver_data.append(pd.DataFrame([solver_status]))

        all_results = {
            "overall": pd.concat(overall_data, ignore_index=True),
            "planes": pd.concat(plane_data, ignore_index=True),
            "cascades": pd.concat(cascade_data, ignore_index=True),
            "stage": pd.concat(stage_data, ignore_index=True),
            "geometry": pd.concat(geometry_data, ignore_index=True),
            "solver": pd.concat(solver_data, ignore_index=True),
        }
        
        # Create a directory to save simulation results
        out_dir = "output"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_filename = f"black_box_optimization_{timestamp}"

        # Export simulation configuration as YAML file
        config_data = {k: v for k, v in config.items() if v}  # Filter empty entries
        config_data = utils.convert_numpy_to_python(config_data, precision=12)
        config_file = os.path.join(out_dir, f"{out_filename}.yaml")
        with open(config_file, "w") as file:
            yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)

        # Export performance results in excel file
        filepath = os.path.join(out_dir, f"{out_filename}.xlsx")
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            for sheet_name, df in all_results.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def build_config(filename, performance_map, solver_options, initial_guess):
    
    """
    Build configuration from CascadesOptimizationProblem object
    """

    obj = utils.load_from_pickle(filename)
    bc_keys = ["fluid_name", "T0_in", "p0_in", "p_out", "omega" , "alpha_in"]
    geometry_keys = ["cascade_type", "radius_hub_in", "radius_hub_out", "radius_tip_in", "radius_tip_out", "pitch",
                     "chord", "stagger_angle", "opening", "leading_edge_angle", "leading_edge_wedge_angle",
                     "leading_edge_diameter", "trailing_edge_thickness", "maximum_thickness",
                     "tip_clearance", "throat_location_fraction"]
    boundary_conditions = {key :  val for key, val in obj.problem.boundary_conditions.items() if key in bc_keys}
    geometry = {key : val for key, val in obj.problem.geometry.items() if key in geometry_keys}
    config = {"geometry" : geometry,
              "simulation_options" : obj.problem.model_options,
              "operation_points" : boundary_conditions,
              "performance_analysis" : {"perfromance_map" : performance_map,
                                        "solver_options" : solver_options,
                                        "initial_guess" : initial_guess,}}
    
    return config


def check_and_clip_initial_guess(initial_guess, bounds, variable_names):
    """
    Checks if the initial guess is within the given bounds and clips it if necessary.
    
    Parameters
    ----------
    initial_guess : numpy.ndarray
        Initial guess for the optimization variables.
    bounds : tuple of numpy.ndarray
        A tuple containing two arrays: lower bounds and upper bounds.
    variable_names : list of str
        List of variable names corresponding to the design variables.
        
    Returns
    -------
    numpy.ndarray
        Adjusted initial guess.
    """
    lb, ub = bounds
    for i, (x, l, u) in enumerate(zip(initial_guess, lb, ub)):
        if not l <= x <= u:
            # Use JAX's .at method to modify the array
            initial_guess = initial_guess.at[i].set(jnp.clip(x, l, u))
            warnings.warn(
                f"Variable '{variable_names[i]}' was out of bounds ({x} not in [{l}, {u}]). "
                f"Clipped to {initial_guess[i]}.",
                UserWarning
            )

    return initial_guess
