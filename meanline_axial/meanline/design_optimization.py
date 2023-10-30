# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:32:24 2023

@author: laboan
"""


from ..solver import OptimizationProblem
from . import cascade_series as cs


class CascadesOptimizationProblem(OptimizationProblem):
    
    def __init__(self, cascades_data):
        
        cs.update_fixed_params(cascades_data)
        
        self.cascades_data = cascades_data
    
    def get_values(self, vars):
        
        # Rename variables
        cascades_data = self.cascades_data
        m = cascades_data["BC"]["m"]
        h0_in = cascades_data["fixed_params"]["h0_in"]
        d_out_s = cascades_data["fixed_params"]["d_out_s"]
        h_out_s = cascades_data["fixed_params"]["h_out_s"]
        
        geometry = self.cascades_data["geometry"]
        n_cascades = geometry["n_cascades"]
        keys = geometry.keys()
        
        # Assign array of design variables to respective place in cascade structure
        w_s = vars[0] # Specific speed
        w = cs.convert_specific_speed(w_s, m, d_out_s, h0_in, h_out_s)
        for i in range(n_cascades):
            for j in range(len(keys)):
                geometry[keys[j]][i] = vars[i*len(keys)+j+1]
                
        x = vars[i*len(keys)+j+1:]
        
        # Compute rest of geometry
        cs.get_geometry(geometry, m, h0_in, d_out_s, h_out_s)
        
        self.cascades_data["BC"]["omega"] = w
        
        residuals = cs.evaluate_cascade_series(x, self.cascades_data)
        
        self.f = self.cascades_data["overall"]["eta_ts"]
        m_calc = self.cscades_data["overall"]["m"]
        
        self.c_eq = np.append(residuals, (m-m_calc/m))
        
        self.c_ineq = None
        
        objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)
        
        return objective_and_constraints
    
    def get_bounds(self, lb = None, ub = None):
        pass
        
    
    def get_n_eq(self):
        return self.get_number_of_constraints(self.c_eq)
    
    def get_n_ineq(self):
        return self.get_number_of_constraints(self.c_ineq)