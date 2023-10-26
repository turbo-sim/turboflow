# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:40:52 2023

@author: laboan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import cascade_series as cs
from .non_linear_problem import CascadesNonlinearSystemProblem
from .optimization_problem import CascadesOptimizationProblem
from .solver import NonlinearSystemSolver, OptimizationSolver
from .utilities import set_plot_options
from datetime import datetime

set_plot_options()
    
def performance(boundary_conditions, cascades_data, method = 'hybr', x0 = None, R = 0.5, eta_tt = 0.9, eta_ts = 0.8, Ma_crit = 0.95):
    
    # Calculate performance at given boundary conditions with given geometry
    cascades_data["BC"] = boundary_conditions
    print('\n')
    print(f'Pressure ration: {cascades_data["BC"]["p0_in"]/cascades_data["BC"]["p_out"]}')
    print(f'Angular speed: {cascades_data["BC"]["omega"]}')
    

    try:
        cascades_problem = CascadesNonlinearSystemProblem(cascades_data, R, eta_tt, eta_ts, Ma_crit, x0)
        solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
        solution = solver.solve(method=method, options = {'maxfev' : 50})
        if max(solution.fun>1e-6):
            raise Exception("Convergence failed")
    except:
        print("Try different solver")
        
        try:
            cascades_problem = CascadesNonlinearSystemProblem(cascades_data, R, eta_tt, eta_ts, Ma_crit, x0)
            solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
            solution = solver.solve(method='lm', options = {'maxiter' : 50})  
            if max(solution.fun>1e-6):
                raise Exception("Convergence failed")
            
        except:        
            print("Try different initial guess")
            
            try: 
                cascades_problem = CascadesNonlinearSystemProblem(cascades_data, R, eta_tt, eta_ts, Ma_crit, x0 = None)
                solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
                solution = solver.solve(method='lm', options = {'maxiter' : 50})
                if max(solution.fun>1e-6):
                    raise Exception("Convergence failed")
                
            except:
                print("Try different initial guess and solver")
                counter = 0
                R = 0
                eta_ts = 0.6
                while (max(solution.fun)>1e-6 and counter < 10):
                    
                    try:
                        R += 0.1
                        eta_ts += 0.05
                        eta_tt = eta_ts + 0.05
                        Ma_crit = 0.9
                        counter += 1
                        cascades_problem = CascadesNonlinearSystemProblem(cascades_data, R = R, eta_tt = eta_tt, eta_ts = eta_ts, Ma_crit = Ma_crit, x0 = None)
                        solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
                        solution = solver.solve(method='lm', options = {'maxiter' : 50})
                        if max(solution.fun>1e-6):
                            raise Exception("Convergence failed")
                    except:
                        print("Try again")

    return solution


def performance_map(boundary_conditions, cascades_data, filename = None):
    
    # Evaluate the cascades series at all conditions given in boundary_conditions
    # Exports the performance to ...
    
    x0 = None
    use_previous = True

    for i in range(len(boundary_conditions["fluid_name"])):
                
        conditions = {key : val[i] for key, val in boundary_conditions.items()}
        solution = performance(conditions, cascades_data, x0 = x0)
    
        if max(solution.fun)>1e-6:
            raise Exception(f"Convergence failed. Boundary condition: {i}")
        
        if use_previous == True:
            x0 = cs.convert_scaled_x0(solution.x, cascades_data)
    
        # Save performance
        BC = {key : val for key, val in cascades_data["BC"].items() if key != 'fluid'}
        plane = cascades_data["plane"]
        cascade = cascades_data["cascade"]
        stage = cascades_data["stage"]
        overall = cascades_data["overall"]
        
        plane_stack = plane.stack()
        plane_stack.index = [f'{idx}_{col+1}' for col, idx in plane_stack.index]
        cascade_stack = cascade.stack()
        cascade_stack.index = [f'{idx}_{col+1}' for col, idx in cascade_stack.index]
        stage_stack = stage.stack()
        stage_stack.index = [f'{idx}_{col+1}' for col, idx in stage_stack.index]
        
        if i == 0:
            BC_data = pd.DataFrame({key : [val] for key, val in BC.items()})
            plane_data = pd.DataFrame({key : [val] for key, val in zip(plane_stack.index, plane_stack.values)})
            cascade_data = pd.DataFrame({key : [val] for key, val in zip(cascade_stack.index, cascade_stack.values)})
            stage_data = pd.DataFrame({key : [val] for key, val in zip(stage_stack.index, stage_stack.values)})
            overall_data = pd.DataFrame({key : [val] for key, val in overall.items()})
        
        else:
            
            BC_data.loc[len(BC_data)] = BC
            plane_data.loc[len(plane_data)] = plane_stack
            cascade_data.loc[len(cascade_data)] = cascade_stack
            stage_data.loc[len(stage_data)] = stage_stack
            overall_data.loc[len(overall_data)] = overall
            
    
    # Write performance dataframe to excel
    if filename == None:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"Performance_data_{current_time}.xlsx"
        
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        BC_data.to_excel(writer, sheet_name = 'BC', index=False)
        plane_data.to_excel(writer, sheet_name = 'plane', index=False)
        cascade_data.to_excel(writer, sheet_name = 'cascade', index=False)
        stage_data.to_excel(writer, sheet_name = 'stage', index=False)
        overall_data.to_excel(writer, sheet_name = 'overall', index=False)
        
def optimal_turbine(design_point, cascades_data):
    
    # Convert design variables (geometry, specific speed) to a flat array
    geometry = cascades_data["geometry"]
    specific_speed = cascades_data["BC"]["specific_speed"]
    x0 = np.array([specific_speed])
    
    for i in range(geometry["n_cascades"]):
        cascade_geometry = np.array([val[i] for val in geometry.values()])
        x0 = np.concatenate((x0, cascade_geometry))
    
    cascades_problem = CascadesOptimizationProblem(cascades_data)
    solver = OptimizationSolver(cascades_problem, x0)
    

    
    


        
        
    