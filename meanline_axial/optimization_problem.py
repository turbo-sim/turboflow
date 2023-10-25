# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:32:24 2023

@author: laboan
"""

from .solver import OptimizationProblem
from .meanline import cascade_series as cs


class CascadesOptimizationProblem(OptimizationProblem):
    
    def __init__(self, cascades_data):
        pass
    
    def get_values(self, vars, cascades_data):
        
        # Rename variables
        m = cascades_data["BC"]["m"]
        
        geometry = cascades_data["geometry"]
        n_cascades = geometry["n_cascades"]
        keys = geometry.keys()
        
        # Assign array of design variables to respective place in cascade structure
        w_s = vars[0] # Specific speed
        for i in range(n_cascades):
            for j in range(len(keys)):
                geometry[keys[j]][i] = vars[i*len(keys)+j+1]
        
        # Compute rest of geometry
        cs.get_geometry(geometry, m, h0_in, d_out_s, h_out_s)
        

    
    def get_bounds(self):
        pass
    
    def get_n_eq(self):
        pass
    
    def get_n_ineq(self):
        pass