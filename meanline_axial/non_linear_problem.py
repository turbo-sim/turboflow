# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:15:28 2023

@author: laboan
"""

from .meanline import cascade_series as cs
from .solver import NonlinearSystemProblem, OptimizationProblem

class CascadesNonlinearSystemProblem(NonlinearSystemProblem):
     
    def __init__(self, cascades_data):
        
        cs.calculate_number_of_stages(cascades_data)
        cs.update_fixed_params(cascades_data)
        cs.check_geometry(cascades_data)
        x0 = cs.generate_initial_guess(cascades_data, R = 0.4)
        x_scaled = cs.scale_x0(x0, cascades_data)
        
        self.x0 = x_scaled
        self.cascades_data = cascades_data
        
    def get_values(self, vars):
        
        residuals = cs.evaluate_cascade_series(vars, self.cascades_data)
        
        return residuals
    
class CascadesOptimizationProblem(OptimizationProblem):
    
    def __init__(self, cascades_data):
        
        cs.calculate_number_of_stages(cascades_data)
        cs.update_fixed_params(cascades_data)
        cs.check_geometry(cascades_data)
        x0 = cs.generate_initial_guess(cascades_data, R = 0.4)
        x_scaled = cs.scale_x0(x0, cascades_data)
        
        self.x0 = x_scaled
        self.cascades_data = cascades_data
        
    def get_values(self, vars):
        
        residuals = cs.evaluate_cascade_series(vars, self.cascades_data)
        
        self.f = 0
        
        self.c_eq = residuals
        
        self.c_ineq = None
        
        objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)
    
        return objective_and_constraints
    
    def get_bounds(self):
        n_cascades = self.cascades_data["geometry"]["n_cascades"]
        lb, ub = cs.get_dof_bounds(n_cascades)
        
        bounds = [(lb[i], ub[i]) for i in range(len(lb))]
        
        return bounds
    
    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)
    
    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)