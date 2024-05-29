import numpy as np
import pandas as pd
import CoolProp as cp
import datetime
import os
import yaml
from .. import pysolver_view as psv
from .. import utilities as utils
from . import geometry_model as geom
from . import flow_model as flow
from .. import properties as props
from . import performance_analysis as pa


CONSTRAINTS = ["mass_flow_rate", "interstage_flaring"]
AVAILABLE_INEQ_CONSTRAINTS = ["mass_flow_rate", "interstage_flaring"]
OBJECTIVE_FUNCTIONS = ["none", "efficiency_ts"]
RADIUS_TYPE = ["constant_mean",
                        "constant_hub",
                        "constant_tip"]
ANGLE_KEYS = ["leading_edge_angle", "gauging_angle"]
INDEPENDENT_VARIABLES = ["v_in",
                         "w_out",
                         "s_out",
                         "beta_out",
                         "v*_in",
                         "beta*_throat",
                         "w*_throat",
                         "s*_throat"]
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
VARIABLES = INDEXED_VARIABLES + ["specific_speed", "blade_jet_ratio"] + INDEPENDENT_VARIABLES


def compute_optimal_turbine(
    config, 
    out_filename=None,
    out_dir="output",
    export_results=True,
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

    # Perform initial function call to initialize problem
    # This populates the arrays of equality and inequality constraints
    # TODO: it might be more intuitive to create a new method called initialize_problem() that generates the initial guess and evaluates the fitness() function with it
    problem.fitness(problem.initial_guess)

    # Load solver configuration
    solver_config = config["design_optimization"]["solver_options"]

    # Initialize solver object using keyword-argument dictionary unpacking
    solver = psv.OptimizationSolver(problem, **solver_config)

    # Solve optimization problem for initial guess x0
    solver.solve(problem.initial_guess)

    dfs = {
        "operation point": pd.DataFrame({key : pd.Series(val) for key, val in problem.boundary_conditions.items()}),
        "overall": pd.DataFrame(problem.results["overall"], index = [0]),
        "plane": problem.results["plane"],
        "cascade": problem.results["cascade"],
        "stage": problem.results["stage"],
        "geometry": pd.DataFrame({key : pd.Series(val) for key, val in problem.geometry.items()}),
        "solver": pd.DataFrame({
                "completed": pd.Series(True, index = [0]),
                "success": pd.Series(solver.success, index = [0]),
                "message": pd.Series(solver.message, index = [0]),
                "grad_count": solver.convergence_history["grad_count"],
                "func_count": solver.convergence_history["func_count"],
                "func_count_total": solver.convergence_history["func_count_total"],
                "objective_value" : solver.convergence_history["objective_value"],
                "constraint_violation": solver.convergence_history["constraint_violation"],
                "norm_step": solver.convergence_history["norm_step"],
            }, index = range(len(solver.convergence_history["grad_count"]))),
    }

    if export_results:
        # Create a directory to save simulation results
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

	# Define filename with unique date-time identifier
        if out_filename == None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_filename = f"design_optimization_{current_time}"

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
    extend_variables(variables)
        Convert a dict containing lists or arrays to a dictionaries of only one element.
    get_given_bounds(design_variable)
        Retrieves the lower and upper bounds for the given design variables.
    get_constraints(self, constraints):
        Converts a list of constraints to equality and inequality constraints.
    update_boundary_conditions(design_point)
        Update the boundary conditions of the turbine analysis problem with the design point of the design optimization problem.
    convert_performance_analysis_results(performance_problem)
        Generate a feasable set of initial guess from a solved performance analysis problem object.

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
        self.obj_func = config["design_optimization"]["objective_function"]
        self.radius_type = config["design_optimization"]["radius_type"]
        self.eq_constraints, self.ineq_constraints = self.get_constraints(config["design_optimization"]["constraints"])

        # Update design point
        self.update_boundary_conditions(config["operation_points"])

        # Initialize model options
        self.model_options = config["simulation_options"]

        # Adjust given variables according to choking model (remove excess variables)
        variables = config["design_optimization"]["variables"]
        if self.model_options["choking_model"] == "evaluate_cascade_throat":
            variables = {key : var for key, var in variables.items() if not key.startswith("v*_in")}
        elif self.model_options["choking_model"] == "evaluate_cascade_critical":
            variables = {key : var for key, var in variables.items() if not key.startswith("beta*_throat")}
        elif self.model_options["choking_model"] == "evaluate_cascade_isentropic_throat":
            variables = {key : var for key, var in variables.items() if not (key.startswith("v*_in") or key.startswith("s*_throat") or key.startswith("beta*_throat"))}
        
        # Separate fixed variables and design variables
        fixed_params = {}
        design_variables = {}
        for key, value in variables.items():
            if value["lower_bound"] is not None:
                design_variables[key] = value
            else:
                fixed_params[key] = value

        # Initialize initial guess
        initial_guess = self.extend_variables(design_variables)
        self.design_variables_keys = initial_guess.keys()
        self.initial_guess = np.array(list(initial_guess.values()))

        # Extend fixed variables
        self.fixed_params = self.extend_variables(fixed_params)

        # Initialize bounds
        lb, ub = self.get_given_bounds(design_variables)
        self.bounds = (lb, ub)

        # Adjust set of independent variables according to choking model
        self.independent_variables = [key for key in initial_guess.keys() if any(key.startswith(var) for var in INDEPENDENT_VARIABLES)]
        if self.model_options["choking_model"] == "evaluate_cascade_throat":
            self.independent_variables = [var for var in self.independent_variables if not var.startswith("v*_in")]
        elif self.model_options["choking_model"] == "evaluate_cascade_critical":
            self.independent_variables = [var for var in self.independent_variables if not var.startswith("beta*_throat")]
        elif self.model_options["choking_model"] == "evaluate_cascade_isentropic_throat":
            self.independent_variables = [var for var in self.independent_variables if not (var.startswith("v*_in") or var.startswith("s*_throat") or var.startswith("beta*_throat"))]
    
    def extend_variables(self, variables):
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
            if isinstance(variables[key]["value"], (list, np.ndarray)):
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
            elif isinstance(design_variables[key]["value"], np.ndarray):
                lb += list(design_variables[key]["lower_bound"])
                ub += list(design_variables[key]["upper_bound"])
            elif isinstance(design_variables[key]["value"], float):
                lb += [design_variables[key]["lower_bound"]]
                ub += [design_variables[key]["upper_bound"]]
                    
        return lb, ub

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
        self.vars_scaled = {key : design_variables[key] for key in self.independent_variables}

        # Contruct variables
        variables = {**design_variables, **self.fixed_params}

        # Calculate angular speed
        specific_speed = variables["specific_speed"]
        self.boundary_conditions["omega"] = self.get_omega(specific_speed, mass_flow, h0_in, d_is, h_is)

        # Calculate mean radius
        blade_jet_ratio = variables["blade_jet_ratio"]
        blade_speed = blade_jet_ratio*v0
        self.geometry = {}
        self.geometry["radius"] = blade_speed/self.boundary_conditions["omega"]
        
        # Assign geometry design variables to geometry attribute
        for key in INDEXED_VARIABLES:
            self.geometry[key] = np.array([v for k, v in variables.items() if k.startswith(key)])
            if key in ANGLE_KEYS:
                self.geometry[key] = self.geometry[key]*angle_range + angle_min
    
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

        # Evaluate available objecitve functions
        objective_functions = {"none" : 0,
                              "efficiency_ts" : -self.results["overall"]["efficiency_ts"].values[0]/10}                       
        self.f = objective_functions[self.obj_func]

        # Evaluate available equality constraints    
        available_constraints = {"mass_flow_rate" : self.results["overall"]["mass_flow_rate"].values[0],
                                "interstage_flaring" :  self.geometry["A_in"][1:]/self.geometry["A_out"][0:-1],
                                "flaring" : self.geometry["flaring_angle"]}
        self.c_eq = np.array(list(self.results["residuals"].values()))
        self.c_ineq = np.array([])
        for constraint in self.eq_constraints:
            self.c_eq = np.append(self.c_eq, (available_constraints[constraint["variable"]] - constraint["value"])/constraint["scale"])
            
        for constraint in self.ineq_constraints:
            self.c_ineq = np.append(self.c_ineq, (constraint["value"] - available_constraints[constraint["variable"]])/constraint["scale"])

        objective_and_constraints = psv.combine_objective_and_constraints(self.f, self.c_eq, self.c_ineq)
        
        return objective_and_constraints
    
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

        return specific_speed*(h0_in-h_is)**(3/4)/((mass_flow/d_is)**0.5)
    
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
        self.boundary_conditions = design_point

        # Initialize fluid object
        self.fluid = props.Fluid(design_point["fluid_name"])

        # Rename variables
        p0_in = design_point["p0_in"]
        T0_in = design_point["T0_in"]
        p_out = design_point["p_out"]

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

        mass_flow_rate = None  # Initialize mass_flow_rate to None
        # Try to find mass_flow_rate from eq_constraints
        for constraint in self.eq_constraints:
            if constraint.get("variable") == "mass_flow_rate":
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
        for design_variable in set(design_variables) - {"specific_speed", "blade_jet_ratio"}:
            if "angle" in design_variable:
                added_dict = {design_variable + f"_{i+1}" : (geometry[design_variable][i]-angle_min)/angle_range for i in range(len(geometry[design_variable]))}
            else:
                added_dict = {design_variable + f"_{i+1}" : geometry[design_variable][i] for i in range(len(geometry[design_variable]))}
            initial_guess = {**initial_guess, **added_dict}


        if "blade_jet_ratio" in design_variables:
            if radius_type == "constant_mean":
                initial_guess["blade_jet_ratio"] = overall["blade_jet_ratio_mean"].values[0]
            elif radius_type == "constant_hub":
                initial_guess["blade_jet_ratio"] = overall["blade_jet_ratio_hub"].values[0]
            elif radius_type == "constant_mean":
                initial_guess["blade_jet_ratio"] = overall["blade_jet_ratio_tip"].values[0]
                
        if "specific_speed" in design_variables:
            angular_speed = overall["angular_speed"].values[0]
            initial_guess["specific_speed"] = angular_speed*(mass_flow/d_out_s)**0.5/((h0_in - h_out_s)**0.75)

        return {**vars_scaled, **initial_guess}
    
    def get_constraints(self, constraints):

        """
        Converts a list of constraints to equality and inequality constraints.

        Parameters:
        -----------
        constraints : list of dict
            List of constraints where each constraint is represented as a dictionary
            with keys "type", "variable", "value", and "normalize". "type" represents
            the type of constraint ('=', '<', or '>'). "variable" is the variable
            involved in the constraint. "value" is the value of the constraint. "normalize"
            specifies whether to normalize the constraint value.

        Returns:
        --------
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
        for constraint in constraints.keys():
            if constraints[constraint]["type"] == '=':
                if constraints[constraint]["normalize"]:
                    eq_constraints += [{"variable" : constraint,
                                    "value" : constraints[constraint]["value"],
                                    "scale" : constraints[constraint]["value"]}]
                else:
                    eq_constraints += [{"variable" : constraint,
                                    "value" : constraints[constraint]["value"],
                                    "scale" : 1}]
                    
            elif constraints[constraint]["type"] == "<": 
                if constraints[constraint]["normalize"]:
                    ineq_constraints += [{"variable" : constraint,
                                        "value": constraints[constraint]["value"],
                                        "scale" : abs(constraints[constraint]["value"])}]
                else:
                    ineq_constraints += [{"variable" : constraint,
                                        "value": constraints[constraint]["value"],
                                        "scale" : 1}]
                    
            elif constraints[constraint]["type"] == ">":
                if constraints[constraint]["normalize"]:
                    ineq_constraints += [{"variable" : constraint,
                                        "value": constraints[constraint]["value"],
                                        "scale" : -1*abs(constraints[constraint]["value"])}]
                else:
                    ineq_constraints += [{"variable" : constraint,
                                        "value": constraints[constraint]["value"],
                                        "scale" : -1}]
        return eq_constraints, ineq_constraints


