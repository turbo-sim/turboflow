import numpy as np

from .. import solver as psv
from .. import utilities as utils
from . import geometry_model as geom
from . import flow_model as flow
from .. import properties as props
import CoolProp as cp

def compute_optimal_turbine(config, initial_guess = None):
    
    # Initialize problem object
    problem = CascadesOptimizationProblem(config)

    # Get initial guess
    x0 = problem.get_initial_guess(initial_guess = initial_guess)
        
    # Initialize solver object  
    max_iter = config["optimization"]["solver_options"]["max_iter"]  
    solver = psv.OptimizationSolver(problem, x0, max_iter= max_iter, display = True)

    sol = solver.solve()
    solver.plot_convergence_history(savefig = False)
    
    return solver

class CascadesOptimizationProblem(psv.OptimizationProblem):
    """
    A class representing a turbine design optimization problem

    Methods
    -------
    get_values(x)
        Evaluate the system of equations for a given set of decision variables.

    """

    def __init__(self, config):

        # Get list of design variables
        self.design_variables_keys = list(config["optimization"]["design_variables"])
        self.obj_func = config["optimization"]["objective_function"]
        self.bounds = []

        # Update design point
        self.update_boundary_conditions(config["operation_points"])

        # Define geometry (fixed parameters)
        self.geometry = config["geometry"]

        # Initialize other attributes
        self.model_options = config["model_options"]
        
    def get_optimization_values(self, x):

        # Rename reference values
        h0_in = self.boundary_conditions["h0_in"]
        mass_flow = self.reference_values["mass_flow_ref"]
        h_is = self.reference_values["h_out_s"]    
        d_is = self.reference_values["d_out_s"]
        angle_range = self.reference_values["angle_range"]
        angle_min = self.reference_values["angle_min"]

        # Structure design variables to dictionary (Assume set of design variables) # TODO: Make flexible set of design variables 
        design_variables = dict(zip(self.design_variables_keys, x))

        # Construct array with independent variables
        self.vars_scaled = dict(zip(self.keys, x)) # TODO: Ensure independent variables are in the start of x! 

        # Update boundary conditions
        omega_spec = design_variables["omega_spec"]
        self.boundary_conditions["omega"] = self.get_omega(omega_spec, mass_flow, h0_in, d_is, h_is)
        
        # Scale and compute full geometry
        diameter_spec = design_variables["diameter_spec"]
        self.geometry["radius_mean"] = self.get_radius(diameter_spec, mass_flow, h0_in, d_is, h_is) # Constant for all cascade with this formulation?
        self.geometry["hub_to_tip"] = np.array([value for key, value in design_variables.items() if key.startswith('hub_to_tip')]) # TODO: Fix such that it is ensured that the values are placed accroding to index
        self.geometry["aspect_ratio"] = np.array([value for key, value in design_variables.items() if key.startswith('aspect_ratio')])
        self.geometry["pitch_to_chord"] = np.array([value for key, value in design_variables.items() if key.startswith('pitch_to_chord')])
        self.geometry["trailing_edge_to_opening"] = np.array([value for key, value in design_variables.items() if key.startswith('trailing_edge_to_opening')])
        # self.geometry["thickness_to_chord"] = np.array([value for key, value in design_variables.items() if key.startswith('thickness_to_chord')])
        self.geometry["metal_angle_le"] = np.array([value*angle_range + angle_min for key, value in design_variables.items() if key.startswith('metal_angle_le')])
        self.geometry["metal_angle_te"] = np.array([value*angle_range + angle_min for key, value in design_variables.items() if key.startswith('metal_angle_te')])
        self.geometry = geom.calculate_geometry(self.geometry)
        # self.geometry = geom.calculate_full_geometry(self.geometry)

        # Evaluate turbine model
        self.results = flow.evaluate_axial_turbine(
            self.vars_scaled,
            self.boundary_conditions,
            self.geometry,
            self.fluid,
            self.model_options,
            self.reference_values,
        )
        if self.obj_func == 0:
            self.f = 0 
        elif self.obj_func == 'efficiency_ts':
            self.f = -self.results["overall"]["efficiency_ts"].values[0]/100

        cons = np.array((self.results["overall"]["mass_flow_rate"].values-mass_flow)/mass_flow)
        areas = np.array(self.geometry["A_out"][0:-1] - self.geometry["A_in"][1:])

        residuals = np.array(list(self.results["residuals"].values()))
        self.c_eq = np.concatenate((residuals, cons, areas))
        self.c_ineq = None

        objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)
        
        return objective_and_constraints
    
    def get_bounds(self):
        # TODO: improve to do checks in case bounds are given 
        if self.bounds == []:
            self.bounds = self.get_default_bounds()

        return self.bounds
            
    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)
    
    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)
    
    def get_initial_guess(self, initial_guess):
        """
        Structure the initial guess for the design optimization. 
        The intial guess can either be provided by the user or given from a defualt value

        """
        if initial_guess == None:
            self.extend_design_variables()
            initial_guess = self.get_default_initial_guess()
        
        # TODO: Do checks on given initial guess
        # x0 must be first
        # geometry with indices in key must be sortes from the smallest to largest
        self.design_variables_keys = initial_guess.keys()

        return np.array(list(initial_guess.values()))
    
    def extend_design_variables(self):
        """
        Extend list of design variables such that each cascade/plane specific variables have one value for each cascade/plane
        """

        number_of_cascades = len(self.geometry["cascade_type"])
        new_keys = []
        for key in self.design_variables_keys:
            if key == "omega_spec":
                new_keys += ["omega_spec"]
            elif key == "diameter_spec":
                new_keys += ["diameter_spec"]
            elif key == "hub_to_tip":
                new_keys += [f"hub_to_tip_{i+1}" for i in range(2*number_of_cascades)]
            elif key == "aspect_ratio":
                new_keys += [f"aspect_ratio_{i+1}" for i in range(number_of_cascades)]
            elif key == "pitch_to_chord":
                new_keys += [f"pitch_to_chord_{i+1}" for i in range(number_of_cascades)]
            elif key == "trailing_edge_to_opening":
                new_keys += [f"trailing_edge_to_opening_{i+1}" for i in range(number_of_cascades)]
            elif key == "thickness_to_chord":
                new_keys += [f"thickness_to_chord_{i+1}" for i in range(number_of_cascades)]
            elif key == "metal_angle_le":
                new_keys += [f"metal_angle_le_{i+1}" for i in range(number_of_cascades)]
            elif key == "metal_angle_te":
                new_keys += [f"metal_angle_te_{i+1}" for i in range(number_of_cascades)]

        self.design_variables_keys = new_keys

        return                      

    def get_default_initial_guess(self):
        """
        Generate a default initial guess for design optimization
        """

        # Define dictionary with given design variables
        initial_guess = np.array([]) 
        for key in self.design_variables_keys:
            if key == "omega_spec":
                initial_guess = np.append(initial_guess, 1.2)
            elif key == "diameter_spec":
                initial_guess = np.append(initial_guess, 1.2)
            elif key.startswith("hub_to_tip"):
                initial_guess = np.append(initial_guess, 0.6)
            elif key.startswith("aspect_ratio"):
                initial_guess = np.append(initial_guess, 1.5)
            elif key.startswith("pitch_to_chord"):
                initial_guess = np.append(initial_guess, 0.9)
            elif key.startswith("trailing_edge_to_opening"):
                initial_guess = np.append(initial_guess, 0.1)
            elif key.startswith("thickness_to_chord"):
                initial_guess = np.append(initial_guess, 0.2)
            elif key.startswith("metal_angle_le"):
                if int(key[-1]) % 2 == 0:
                    initial_guess = np.append(initial_guess, 0.41) # -15 degrees for rotor
                else:
                    initial_guess = np.append(initial_guess, 0.5) # 0 degrees for stator 
            elif key.startswith("metal_angle_te"):
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
            
        if self.model_options["choking_model"] == "evaluate_cascade_throat":
            initial_guess_variables = {key : val for key, val in initial_guess_variables.items() if not key.startswith("v*_in")}
        elif self.model_options["choking_model"] == "evaluate_cascade_critical":
            initial_guess_variables = {key : val for key, val in initial_guess_variables.items() if not key.startswith("beta*_throat")}
        elif self.model_options["choking_model"] == "evaluate_cascade_isentropic_throat":
            initial_guess_variables = {key : val for key, val in initial_guess_variables.items() if not (key.startswith("v*_in") or key.startswith("s*_throat") or key.startswith("beta*_throat"))}

        self.keys = initial_guess_variables.keys() # TODO: Move this
        # self.keys used to merge with independent variables in get_optimization_values
        # This implementation only works with default initial guess
        return {**initial_guess_variables, **initial_guess} 
                
    def get_omega(self, omega_spec, mass_flow, h0_in, d_is, h_is):
        return omega_spec*(h0_in-h_is)**(3/4)/((mass_flow/d_is)**0.5)
    
    def get_radius(self, diameter_spec, mass_flow, h0_in, d_is, h_is):
        return diameter_spec*(mass_flow/d_is)**0.5/((h0_in-h_is)**(1/4))/2
    
    def update_boundary_conditions(self, design_point):
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
        mass_flow_rate = design_point["mass_flow_rate"]
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
        """
        Gives default bounds to each design variable.
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
            elif key == "omega_spec":
                bounds += [(0.01, 10)]
            elif key == "diameter_spec":
                bounds += [(0.01, 10)]
            elif key.startswith("hub_to_tip"):
                bounds += [(0.6, 0.9)]
            elif key.startswith("aspect_ratio"):
                bounds += [(1.0, 2.0)]
            elif key.startswith("pitch_to_chord"):
                bounds += [(0.75, 1.10)]
            elif key.startswith("trailing_edge_to_opening"):
                bounds += [(0.05, 0.4)]
            elif key.startswith("thickness_to_chord"):
                bounds += [(0.15, 0.25)]
            elif key.startswith("metal_angle_le"):
                if int(key[-1]) % 2 == 0: 
                    bounds += [(0.41, 0.92)] # Rotor (-15, 75) [degree]
                else:
                    bounds += [(0.08, 0.58)] # Stator (-75, 15) [degree]
            elif key.startswith("metal_angle_te"):
                if int(key[-1]) % 2 == 0: 
                    bounds += [(0.06, 0.28)] # Rotor (-40, -80) [degree]
                else:
                    bounds += [(0.72, 0.94)] # Stator (40, 80) [degree]

        return tuple(bounds)

def find_variable(cascades_data, variable):
    
    # Function to find variable in cascades_data
    
    for key in cascades_data.keys():
        
        if any(element == variable for element in cascades_data[key]):
            
            return cascades_data[key][variable]

    raise Exception(f"Could not find column {variable} in cascades_data")