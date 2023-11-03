# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:32:24 2023

@author: laboan
"""

import numpy as np
from ..solver import OptimizationProblem, OptimizationSolver
from . import cascade_series as cs
from ..utilities import print_boundary_conditions


def compute_optimal_turbine(design_point, cascades_data, x0):
    
    # Calculate performance at given boundary conditions with given geometry
    cascades_data["BC"] = design_point    
    
    # Get optimization specific options
    method = cascades_data["optimization"].get("method", 'slsqp')
    design_variables = cascades_data["optimization"].get("design_variables", {})
    obj_func = cascades_data["optimization"].get("obj_func", 0)
    constraints_eq = cascades_data["optimization"].get("constraints_eq", {})
    constraints_ineq = cascades_data["optimization"].get("constraints_ineq",{})
    bounds = cascades_data["optimization"].get("bounds",[])
    
    # Initialize problem object
    problem = CascadesOptimizationProblem(cascades_data, 
                                          design_variables,
                                          obj_func,
                                          constraints_eq,
                                          constraints_ineq,
                                          bounds)
        
    # Construct initial guess array
    design_variables_values = []
    for key, val in design_variables.items():
        if isinstance(val, list):
            design_variables_values.extend(val)
        else:
            design_variables_values.append(val)
    x = np.concatenate((design_variables_values, x0))
        
    # Initialize solver object    
    solver = OptimizationSolver(problem, x, display = True)#, update_on="function")

    sol = solver.solve(method = method, options = {"maxiter" : 200})
    solver.plot_convergence_history(savefig = False)
    
    return solver

class CascadesOptimizationProblem(OptimizationProblem):

    def __init__(self, cascades_data, design_variables = {}, obj_func = 0, constraints_eq = {}, constraints_ineq = {}, bounds = []):
        
        cs.calculate_number_of_stages(cascades_data)
        cs.update_fixed_params(cascades_data)
        
        # Define reference mass flow rate
        v0 = cascades_data["fixed_params"]["v0"]
        d_in = cascades_data["fixed_params"]["d0_in"]
        h0_in = cascades_data["fixed_params"]["h0_in"]
        d_out_s = cascades_data["fixed_params"]["d_out_s"]
        h_out_s = cascades_data["fixed_params"]["h_out_s"]
        
        # check validity of obj_func
        available_obj_func = ["eta_ts", 0]
        if obj_func not in available_obj_func:
            raise Exception(f'obj_func not valid. Valid options are: {available_obj_func}')
        
        # Force objective function to 0 and constraints to be empty if no design variables are given
        if design_variables == {}:
            if obj_func != 0:
                print("No design variables given. Objective function changed to 0")
            obj_func = 0
            if constraints_eq != {}:
                print("No design variables given. constraints_eq changed to {}")
            constraints_eq = {}
            if constraints_ineq != {}:
                print("No design variables given. constraints_ineq changed to {}")
            constraints_ineq = {}
            
        # Define reference mass flow rate
        if 'mass_flow_rate' in constraints_eq:
            m = constraints_eq["mass_flow_rate"]
        else:
            A_in = cascades_data["geometry"]["A_in"][0]
            m = A_in*v0*d_in 
        cascades_data["fixed_params"]["m_ref"] = m
        
        # Calculate geometry
        # cs.get_geometry(cascades_data["geometry"], m, h0_in, d_out_s, h_out_s)

        self.cascades_data = cascades_data
        self.design_variables = design_variables
        self.constraints_eq = constraints_eq
        self.constraints_ineq = constraints_ineq
        self.bounds = bounds
        self.obj_func = obj_func
        self.h0_in = h0_in
        self.d_out_s = d_out_s
        self.h_out_s = h_out_s
        self.m = m
        self.n_cascades = cascades_data["geometry"]["n_cascades"]
        
    def get_values(self, x):
        
        m = self.m
        h0_in = self.h0_in
        h_out_s = self.h_out_s
        d_out_s = self.d_out_s
        n_cascades = self.n_cascades
        
        i = 0 # Index to assign design variables from vars
        if "w_s" in self.design_variables.keys():
            w_s = x[0] # Specific speed
            w = cs.convert_specific_speed(w_s, m, d_out_s, h0_in, h_out_s)
            self.cascades_data["BC"]["omega"] = w
            i += 1
            
        keys_geometry = [key for key in self.cascades_data["geometry"].keys() if key in self.design_variables.keys()]
        for key in keys_geometry:
            self.cascades_data["geometry"][key] = [val for val in x[i:i+n_cascades]] 
            i += n_cascades

        cs.get_geometry(self.cascades_data["geometry"], m, h0_in, d_out_s, h_out_s)

        x0 = x[i:]
        residuals = cs.evaluate_cascade_series(x0, self.cascades_data)
        
        if self.obj_func == 0:
            self.f = 0 
        elif self.obj_func == 'eta_ts':
            self.f = -self.cascades_data["overall"]["eta_ts"]/100
        
        cons = []
        keys_variables = ["plane", "cascade","stage", "overall"]
        subset_cascades_data = {key: self.cascades_data[key] for key in keys_variables}
        for key in self.constraints_eq.keys():
            variable = find_variable(subset_cascades_data, key)
            cons.append((variable-self.constraints_eq[key])/self.constraints_eq[key])
            
        
        self.c_eq = np.concatenate((residuals, cons))
        self.c_ineq = None
        
        objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)
        
        return objective_and_constraints
    
    def get_bounds(self):
        
        if self.bounds == []:
            self.bounds = get_default_bounds(self.cascades_data, self.design_variables.keys())
            
        return self.bounds
            
    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)
    
    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)
    
    
def get_default_bounds(cascades_data, keys_design_variables):
    
    """
    Gives default bounds to each design variable.
    """
    
    default_bounds_w_s  = (0.1, 1.5)  
    default_bounds_stator = {"specific_diameter" : (1, 3),
                      "r_ht_in" : (0.6, 0.95),
                      "r_ht_out" : (0.6, 0.95),
                      "ar" : (1, 2),
                      "bs" : (1, 2),
                      "theta_in" : (-15*np.pi/180, 15*np.pi/180),
                      "theta_out" : (40*np.pi/180, 80*np.pi/180),
                      "te_o" : (0.05, 0.4),
                      "le_c" : (),
                      "We" : (10*np.pi/180, 60*np.pi/180),
                      "v_in" : (0.01, 0.5),
                      "v_out" : (0.1, 0.99),
                      "beta_out" : (30*np.pi/180, 80*np.pi/180),
                      "s_out" : (1, 1.2),
                      "v_in_crit" : (0.01, 0.7),
                      "v_out_crit" : (0.1, 0.99)}
    
    default_bounds_rotor = {"specific_diameter" : (1, 3),
                      "r_ht_in" : (0.6, 0.95),
                      "r_ht_out" : (0.6, 0.95),
                      "ar" : (1, 2),
                      "bs" : (1, 2),
                      "theta_in" : (-15*np.pi/180, 80*np.pi/180),
                      "theta_out" : (-80*np.pi/180, -40*np.pi/180),
                      "te_o" : (0.05, 0.4),
                      "le_c" : (),
                      "We" : (10*np.pi/180, 60*np.pi/180),
                      "v_in" : (0.1, 0.99),
                      "v_out" : (0.1, 0.99),
                      "beta_out" : (-80*np.pi/180, -30*np.pi/180),
                      "s_out" : (1, 1.2),
                      "v_in_crit" : (0.1, 0.99),
                      "v_out_crit" : (0.1, 0.99)}
            
    n_cascades = cascades_data["geometry"]["n_cascades"]
    
    bounds = []
    # Add bounds to specific speed
    if "w_s" in keys_design_variables:
        bounds += [default_bounds_w_s]
    
    # Add bounds to geometry
    keys_geometry = [key for key in cascades_data["geometry"] if key in keys_design_variables]
    for key in keys_geometry:
        bounds += [default_bounds_stator[key]]
        bounds += [default_bounds_rotor[key]]

    # Add bounds to independant variables
    bounds += [default_bounds_stator["v_in"]]
    keys = ["v_out", "v_out", "s_out", "s_out", "beta_out"]
    keys_crit = ["v_in_crit", "v_out_crit", "s_out"]
    bounds_crit = []
    for i in range(n_cascades):
        if i%2 == 0:
            bounds += [default_bounds_stator[key] for key in keys]
            bounds_crit += [default_bounds_stator[key] for key in keys_crit]
        else:
            bounds += [default_bounds_rotor[key] for key in keys]
            bounds_crit += [default_bounds_rotor[key] for key in keys_crit]
       
    bounds += bounds_crit
    return bounds


def get_initial_guess_array(cascades_data, keys_design_variables, x = None):
    
    """
    Gives design variables given by keys_design_variables from cascades data to 
    form initial guess for design optimization. 
    At maximum it includes specific speed and complete geometry in addition to 
    variables needed to evaluate each cascade (v_in, v_trhoat, v_out, s_throat,
                                               s_out, beta_out)
    """
    
    n_cascades = cascades_data["geometry"]["n_cascades"]
    
    initial_guess = []
    if "w_s" in keys_design_variables:
        initial_guess += [cascades_data["BC"]["w_s"]]
    
    for i in range(n_cascades):
        initial_guess += [val[i] for key, val in cascades_data["geometry"].items() if key in keys_design_variables]
        
    if not isinstance(x, np.ndarray):
        x = cs.generate_initial_guess(cascades_data, R = 0.5, eta_tt = 0.95, eta_ts = 0.9, Ma_crit = 0.95)
        x = cs.scale_x0(x, cascades_data)
        
    initial_guess += list(x)
    
    return np.array(initial_guess)

def find_variable(cascades_data, variable):
    
    # Function to find variable in cascades_data
    
    for key in cascades_data.keys():
        
        if any(element == variable for element in cascades_data[key]):
            
            return cascades_data[key][variable]

    raise Exception(f"Could not find column {variable} in cascades_data")