import numpy as np

from .. import solver as psv
from .. import utilities as utils
from . import geometry_model as geom
from . import flow_model as flow
from .. import properties as props
from . import performance_analysis as pa
import CoolProp as cp

AVAILABLE_EQ_CONSTRAINTS = ["mass_flow_rate", "interstage_flaring"]
AVAILABLE_INEQ_CONSTRAINTS = ["mass_flow_rate", "interstage_flaring"]
AVAILABLE_OBJECTIVE_FUNCTIONS = ["none", "efficiency_ts"]
AVAILABLE_DESIGN_VARIABLES = ["specific_speed", 
                              "blade_jet_ratio",
                                "hub_tip_ratio_in",
                                "hub_tip_ratio_out",
                                "aspect_ratio",
                                "pitch_chord_ratio",
                                "trailing_edge_thickness_opening_ratio",
                                "leading_edge_angle",
                                "gauging_angle"
                            ]
AVAILABLE_GEOMETRIES = ["constant_mean",
                        "constant_hub",
                        "constant_tip"]

def compute_optimal_turbine(config, initial_guess = None):

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


    # Check configuration
    config = check_optimization_config(config)

    # Initialize problem object
    problem = CascadesOptimizationProblem(config)

    # Get initial guess
    x0 = problem.get_initial_guess(initial_guess)
        
    # Initialize solver object  
    max_iter = config["optimization"]["solver_options"]["max_iter"]  
    solver = psv.OptimizationSolver(problem, x0, max_iter= max_iter, display = True)

    sol = solver.solve()
    solver.plot_convergence_history(savefig = False)
    
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
    get_optimization_values(x)
        Evaluate the objective function and constraints at a given point `x`.
    get_bounds()
        Provide the bounds for the design optimization problem.
    get_n_eq()
        Get the number of equality constraints.
    get_n_ineq()
        Get the number of oinequality constraints.
    get_omega(specific_speed, mass_flow, h0_in, d_is, h_is)
        Convert specific speed to actual angular speed
    get_initial_guess(initial_guess)
        Get the initial guess for the design optimization problem.
    extend_design_variables()
        Index the design variables for each cascade for the default initial guess.
    get_default_initial_guess()
        Generate a default initial guess for the design optimization problem.
    get_default_bounds()
        Generate a set of default bounds for each design variable.
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
        self.design_variables_keys = config["optimization"]["design_variables"]
        self.design_variables_geometry = set(self.design_variables_keys) - set(set(self.design_variables_keys) - set(AVAILABLE_DESIGN_VARIABLES)) - {"specific_speed", "blade_jet_ratio"}
        self.obj_func = config["optimization"]["objective_function"]
        self.eq_constraints = config["optimization"]["eq_constraints"]
        self.ineq_constraints = config["optimization"]["ineq_constraints"]
        self.bounds = config["optimization"]["bounds"]
        self.radius_type = config["optimization"]["radius_type"]

        # Update design point
        self.update_boundary_conditions(config["operation_points"])

        # Define geometry (fixed parameters)
        self.geometry = config["geometry"]

        # Initialize other attributes
        self.model_options = config["model_options"]
        
    def get_optimization_values(self, x):

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

        # Structure design variables to dictionary (Assume set of design variables) # TODO: Make flexible set of design variables 
        design_variables = dict(zip(self.design_variables_keys, x))

        # Construct array with independent variables
        self.vars_scaled = dict(zip(self.keys, x)) # TODO: Ensure independent variables are in the start of x! 

        # Calculate angular speed
        specific_speed = design_variables["specific_speed"]
        self.boundary_conditions["omega"] = self.get_omega(specific_speed, mass_flow, h0_in, d_is, h_is)
        
        # Calculate mean radius
        blade_jet_ratio = design_variables["blade_jet_ratio"]
        blade_speed = blade_jet_ratio*v0
        self.geometry["radius"] = blade_speed/self.boundary_conditions["omega"]
        
        # Assign geometry design variables to geometry attribute
        for des_key in self.design_variables_geometry:
            self.geometry[des_key] = np.array([value for key, value in design_variables.items() if (key.startswith(des_key) and key not in ["specific_speed", "blade_jet_ratio"])])
        self.geometry["leading_edge_angle"] = self.geometry["leading_edge_angle"]*angle_range + angle_min # TODO!
        self.geometry["gauging_angle"] = self.geometry["gauging_angle"]*angle_range + angle_min
    
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
                              "efficiency_ts" : -self.results["overall"]["efficiency_ts"].values[0]/100}                       
        self.f = objective_functions[self.obj_func]

        # Evaluate available equality constraints    
        if self.eq_constraints is not None:
            available_cons_eq = {"mass_flow_rate" : self.results["overall"]["mass_flow_rate"].values[0],
                                "interstage_flaring" :  self.geometry["A_in"][1:]/self.geometry["A_out"][0:-1]}
            cons_eq = np.array([])
            for key in self.eq_constraints.keys():
                cons_eq = np.append(cons_eq, (available_cons_eq[key] - self.eq_constraints[key]["value"])/self.eq_constraints[key]["scale"])

            residuals = np.array(list(self.results["residuals"].values()))
            self.c_eq = np.concatenate((residuals, cons_eq))
        else:
            self.c_eq = np.array(list(self.results["residuals"].values()))

        # Evaluate available inequality constraints
        if self.ineq_constraints is not None:
            available_cons_ineq = {"interstage_flaring" : self.geometry["A_in"][1:]/self.geometry["A_out"][0:-1],
                                #    "flaring" : self.geometry["flaring"],
                                   }
            cons_ineq = np.array([])
            for key in self.ineq_constraints.keys():
                cons_ineq = np.append(cons_ineq, (available_cons_ineq[key] - self.ineq_constraints[key]["lower_bound"])/self.ineq_constraints[key]["scale"])
                cons_ineq = np.append(cons_ineq, (self.ineq_constraints[key]["upper_bound"] - available_cons_ineq[key])/self.ineq_constraints[key]["scale"])
            self.c_ineq = cons_ineq
        else:
            self.c_ineq = None

        objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)
        
        return objective_and_constraints
    
    def get_bounds(self):

        r"""
        Provide the bounds for the design optimization problem. 

        Returns
        -------
        list
            List of toutples containg the lower and upper bound for each design variable. 
        """

        if self.bounds == None:
            self.bounds = self.get_default_bounds()

        return self.bounds
            
    def get_n_eq(self):
        
        r"""
        Get the number of equality constraints.

        Returns
        -------
        int
            Number of equality constraints.

        """

        return self.get_number_of_constraints(self.c_eq)
    
    def get_n_ineq(self):
        r"""
        Get the number of inequality constraints.

        Returns
        -------
        int
            Number of inequality constraints.

        """
        return self.get_number_of_constraints(self.c_ineq)
    
    def get_initial_guess(self, initial_guess):
        r"""
        
        Get the initial guess for the design optimization problem. 

        The initial guess can be given from a defualt set of values, generated from a solved performance analysis problem object or provided by the user. 
        If given by the user, the initial guess must be given as a dictionary where the key of each cascade specific value is indexed according 
        to the order at which the cascade occur in the turbine. For example, `aspect_ratio_1`, `aspect_ratio_2` etc. 
        In addition, the flow variables must occur first (velocities, entropies and flow angles).
        If a solved performance analysis problem object (CascadesNonlinearSystemProblem) is given, the code generate a feasable initial guess through the
        `convert_performance_analysis_results` method. 

        Parameters
        ----------
        inital_guess : None, dict or solved 
            The initial guess. Defualt guess if None.

        Returns
        -------
        numpy.ndarray
            Array of the values of the initial guess.

        Notes
        -----
        The keys are assigned as an attribute and is zipped together with the array in get_optimization_values(x).

        """
        if initial_guess == None:
            self.extend_design_variables()
            initial_guess = self.get_default_initial_guess()
        elif isinstance(initial_guess, pa.CascadesNonlinearSystemProblem):
            initial_guess = self.convert_performance_analysis_results(initial_guess)

        # Get keys with independent variables
        number_of_cascades = len(self.geometry["cascade_type"])
        if self.model_options["choking_model"] == "evaluate_cascade_throat":
            number_of_dof = number_of_cascades*6 + 1
        elif self.model_options["choking_model"] == "evaluate_cascade_critical":
            number_of_dof = number_of_cascades*6 + 1
        elif self.model_options["choking_model"] == "evaluate_cascade_isentropic_throat":
            number_of_dof = number_of_cascades*4 + 1
        self.keys = list(initial_guess.keys())[0:number_of_dof]
    
        self.design_variables_keys = initial_guess.keys()

        return np.array(list(initial_guess.values()))
    
    def extend_design_variables(self):
        r"""
        Index the design variables for each cascade for the default initial guess.

        The design variables are given without indexing the cascades. For example, the aspect ratio is simply provided by aspect_ratio. 
        This function extend the design_variable dictionary such that for each cascade specific design variable, one item is given for each cascade.
        for example `aspect_ratio_1` for first cascade and etc. 

        Returns
        -------
        None
            This method does not return a value but updates the `design_variables_keys` attribute of the object.       
        
        """

        number_of_cascades = len(self.geometry["cascade_type"])
        new_keys = []
        for key in self.design_variables_keys:
            if key == "specific_speed":
                new_keys += ["specific_speed"]
            elif key == "blade_jet_ratio":
                new_keys += ["blade_jet_ratio"]
            elif key == "hub_tip_ratio_in":
                new_keys += [f"hub_tip_ratio_in_{i+1}" for i in range(number_of_cascades)]
            elif key == "hub_tip_ratio_out":
                new_keys += [f"hub_tip_ratio_out_{i+1}" for i in range(number_of_cascades)]
            elif key == "aspect_ratio":
                new_keys += [f"aspect_ratio_{i+1}" for i in range(number_of_cascades)]
            elif key == "pitch_chord_ratio":
                new_keys += [f"pitch_chord_ratio_{i+1}" for i in range(number_of_cascades)]
            elif key == "trailing_edge_thickness_opening_ratio":
                new_keys += [f"trailing_edge_thickness_opening_ratio_{i+1}" for i in range(number_of_cascades)]
            elif key == "leading_edge_angle":
                new_keys += [f"leading_edge_angle_{i+1}" for i in range(number_of_cascades)]
            elif key == "gauging_angle":
                new_keys += [f"gauging_angle_{i+1}" for i in range(number_of_cascades)]

        self.design_variables_keys = new_keys

        return                      

    def get_default_initial_guess(self):
        """
        Generate a default initial guess for the design optimization problem. 

        Each available design variable have an associated defualt initial guess value. This function take the
        `design_variables_keys` attribute and assign a value for each key. 

        Returns
        -------
        dict
            Initial guess for the design optimization problem.

        """

        # Define dictionary with given design variables
        initial_guess = np.array([]) 
        for key in self.design_variables_keys:
            if key == "specific_speed":
                initial_guess = np.append(initial_guess, 1.2)
            elif key == "blade_jet_ratio":
                initial_guess = np.append(initial_guess, 0.5)
            elif key.startswith("hub_tip"):
                initial_guess = np.append(initial_guess, 0.6)
            elif key.startswith("aspect_ratio"):
                initial_guess = np.append(initial_guess, 1.5)
            elif key.startswith("pitch_chord_ratio"):
                initial_guess = np.append(initial_guess, 0.9)
            elif key.startswith("trailing_edge_thickness_opening_ratio"):
                initial_guess = np.append(initial_guess, 0.1)
            elif key.startswith("thickness_to_chord"):
                initial_guess = np.append(initial_guess, 0.2)
            elif key.startswith("leading_edge_angle"):
                if int(key[-1]) % 2 == 0:
                    initial_guess = np.append(initial_guess, 0.41) # -15 degrees for rotor
                else:
                    initial_guess = np.append(initial_guess, 0.5) # 0 degrees for stator 
            elif key.startswith("gauging_angle"):
                if int(key[-1]) % 2 == 0:
                    initial_guess = np.append(initial_guess, 0.17) # -60 degrees for rotor
                else:
                    initial_guess = np.append(initial_guess, 0.94) # 80 degrees for stator

        initial_guess = dict(zip(self.design_variables_keys, initial_guess))

        # Add vector of independent variables to the initial guess
        number_of_cascades = len(self.geometry["cascade_type"])
        initial_guess_variables = {"v_in" : 0.1}
        for i in range(number_of_cascades):
            index = f"_{i+1}"
            initial_guess_variables.update(
                {
                    "w_out" + index: 0.65,
                    "s_out" + index: 0.15,
                    "beta_out" + index: ((1-2*((i+1) % 2 == 0))*60+90)/180, # Trick to get different guess for stator/rotor
                    "v*_in" + index: 0.4,
                    "beta*_throat" + index: ((1-2*((i+1) % 2 == 0))*60+90)/180,
                    "w*_throat" + index: 0.65,
                    "s*_throat" + index: 0.15
                })
            
        # Adjust initial guess according to model options
        if self.model_options["choking_model"] == "evaluate_cascade_throat":
            initial_guess_variables = {key : val for key, val in initial_guess_variables.items() if not key.startswith("v*_in")}
        elif self.model_options["choking_model"] == "evaluate_cascade_critical":
            initial_guess_variables = {key : val for key, val in initial_guess_variables.items() if not key.startswith("beta*_throat")}
        elif self.model_options["choking_model"] == "evaluate_cascade_isentropic_throat":
            initial_guess_variables = {key : val for key, val in initial_guess_variables.items() if not (key.startswith("v*_in") or key.startswith("s*_throat") or key.startswith("beta*_throat"))}

        return {**initial_guess_variables, **initial_guess} 
          
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

        # Define a reference mass flow rate
        try:
            mass_flow_rate = self.eq_constraints["mass_flow_rate"]["value"]
        except:
            mass_flow_rate = v0*d_isentropic

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
    
    def get_default_bounds(self):
        r"""
        Generate a set of default bounds for each design variable. 

        The function iterates through each design variable in `design_variables_keys` and assignes a tuple with associated 
        defualt set of upper and lower bounds.

        Returns
        -------
        tuple
            A tuple of tuples with the lower and upper bound for each design variable.
        """

        bounds = []     

        for key in self.design_variables_keys:
            if key.startswith("w*_"):
                bounds += [(0.1, 1.0)]
            elif key.startswith("w_out"):
                bounds += [(0.1, 1.0)]
            elif key.startswith("v*_in_1"):
                bounds += [(0.01, 0.8)]
            elif key.startswith("v*_in"):
                bounds += [(0.1, 1)]
            elif key.startswith("v_in"):
                bounds += [(0.001, 0.5)]
            elif key.startswith("beta"): 
                if int(key[-1]) % 2 == 0: 
                    bounds += [(0.06, 0.28)] # Rotor (-40, -80) [degree]
                else:
                    bounds += [(0.72, 0.94)] # Stator (40, 80) [degree]
            elif key.startswith("s*"):
                bounds += [(0.0, 0.32)] 
            elif key.startswith("s_"):
                bounds += [(0.0, 0.32)]
            elif key == "specific_speed":
                bounds += [(0.01, 10)]
            elif key == "blade_jet_ratio":
                bounds += [(0.1, 0.9)]
            elif key.startswith("hub_tip"):
                bounds += [(0.6, 0.9)]
            elif key.startswith("aspect_ratio"):
                bounds += [(1.0, 2.0)]
            elif key.startswith("pitch_chord_ratio"):
                bounds += [(0.75, 1.10)]
            elif key.startswith("trailing_edge_thickness_opening_ratio"):
                bounds += [(0.05, 0.4)]
            elif key.startswith("leading_edge_angle"):
                if int(key[-1]) % 2 == 0: 
                    bounds += [(0.41, 0.92)] # Rotor (-15, 75) [degree]
                else:
                    bounds += [(0.08, 0.58)] # Stator (-75, 15) [degree]
            elif key.startswith("gauging_angle"):
                if int(key[-1]) % 2 == 0: 
                    bounds += [(0.06, 0.28)] # Rotor (-40, -80) [degree]
                else:
                    bounds += [(0.72, 0.94)] # Stator (40, 80) [degree]

        return tuple(bounds)
    
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

def check_optimization_config(config):

    r"""
    
    Checks the given configuration file for the desing optimization problem.

    The following checks are performed:

        - checks that `config['optimization']` contain `objective_function`, and that this is a string which is in `AVAILABLE_OBJECTIVE_FUNCTIONS`.
        - checks if `config['optimization']` contain `eq_constraints`. If so, it checks that each element is a dict, where the key is contained in `AVAILABLE_EQ_CONSTRAINTS` and that it has a scale and value item. If not specified in configuration file, `config['optimization']['eq_constraints]` is set to None.
        - checks if `config['optimization']` contain `ineq_constraints`. If so, it checks that each element is a dict, where the key is contained in `AVAILABLE_EQ_CONSTRAINTS` and that it has a scale and value item. If not specified in configuration file, `config['optimization']['ineq_constraints]` is set to None.
        - checks if `config['optimization']` contain `design_variables`. This should be a list of strings that is contained in `AVAILABLE_DESIGN_VARIABLES`.
        - checks if `config['optimization']` contain `bounds`. If given, it should be a iterable of tuples on the form (lower bound, upper bound), corresponding to the design variables. Set to None if not given. 
        - checks if `config['optimization']` contain `radius_type`, and that it is contained in `AVAILABLE_GEOMETRIES`. Set to constant_mean if not given.

    Parameters
    ----------
    config : dict
        A dictionary containing necessary configuration options for computing optimal turbine.

    Returns
    -------
    dict
        The configuration dictionary. It is the same as the input if all required elements are given correctly. 
    
    """

     # Check objective function
    if "objective_function" not in config["optimization"].keys():
        raise ValueError(f"Error: Objective function is not specified in the configuration file. Available objective functions are: {', '.join(AVAILABLE_OBJECTIVE_FUNCTIONS)}")
    if not config["optimization"]["objective_function"] in AVAILABLE_OBJECTIVE_FUNCTIONS:
        raise ValueError(f"Error: Objective function is not supported. Available objective functions are: {', '.join(AVAILABLE_OBJECTIVE_FUNCTIONS)}")
    
    # Check equality constraints
    if "eq_constraints" in config["optimization"].keys():
        constraint_keys = config["optimization"]["eq_constraints"].keys() 
        required_keys = {"scale", "value"}
        for key in config["optimization"]["eq_constraints"].keys():
            if key not in AVAILABLE_EQ_CONSTRAINTS:
                raise ValueError(f"Error: Equality constraint {key} is not supported. Available equality constraints are: {', '.join(AVAILABLE_EQ_CONSTRAINTS)}")
            if not set(list(config["optimization"]["eq_constraints"][key].keys())) == required_keys:
                raise ValueError(f"Error: Missing keys for eq constraint {key}: {required_keys - set(config['optimization']['eq_constraints'][key].keys())}")
    else:
        config["optimization"]["eq_constraints"] = None

    # Check inequality constraints
    if "ineq_constraints" in config["optimization"].keys():
        constraint_keys = config["optimization"]["ineq_constraints"].keys() 
        required_keys = {"scale", "lower_bound", "upper_bound"}
        for key in constraint_keys:
            if key not in AVAILABLE_INEQ_CONSTRAINTS:
                raise ValueError(f"Error: Inequality constraint {key} is not supported. Available inequality constraints are: {', '.join(AVAILABLE_INEQ_CONSTRAINTS)}")
            if not set(list(config["optimization"]["ineq_constraints"][key].keys())) == required_keys:
                raise ValueError(f"Error: Missing keys for ineq constraint {key}: {required_keys - set(config['optimization']['ineq_constraints'][key].keys())}")
    else:
        config["optimization"]["ineq_constraints"] = None

    # Check design variables: Must be a list of strings 
    if "design_variables" not in config["optimization"].keys():
        raise ValueError(f"Error: design_variables is not specified in the configuration file")
    else:
        design_variables = config["optimization"]["design_variables"]
        if not len(set(design_variables)-set(AVAILABLE_DESIGN_VARIABLES)) == 0:
            raise ValueError(f"Error: Design variables are not supported: {set(design_variables)-set(AVAILABLE_DESIGN_VARIABLES)}")

    # Check bounds: Must be a list of tuples corresponding to the design variables
    if "bounds" not in config["optimization"].keys():
        config["optimization"]["bounds"] = None
    else:
        if not len(config["optimization"]["bounds"]) == len(config["optimization"]["design_variables"]):
            raise ValueError(f"Error: Bounds not aligned with design variables. Number of bounds: {len(config['optimization']['bounds'])}. Number of design variables: {len(config['optimization']['design_variables'])}")

    # Check radius type
    if "radius_type" not in config["optimization"]:
        config["optimization"]["radius_type"] = "constant_mean"
    else:
        if not config["optimization"]["radius_type"] in AVAILABLE_GEOMETRIES:
            raise ValueError(f"Error: Radius type is not supported. Available radius types are: {', '.join(AVAILABLE_GEOMETRIES)}")
        
    return config


