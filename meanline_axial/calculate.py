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
from .solver import NonlinearSystemSolver
from .utilities import set_plot_options
from datetime import datetime

set_plot_options()
    
def performance(boundary_conditions, cascades_data, method = 'hybr', x0 = None, R = 0.5, eta_tt = 0.9, eta_ts = 0.8, Ma_crit = 0.9):
    
    # Calculate performance at given boundary conditions with given geometry
    cascades_data["BC"] = boundary_conditions
    cascades_problem = CascadesNonlinearSystemProblem(cascades_data, R, eta_tt, eta_ts, Ma_crit, x0)
    solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
    solution = solver.solve(method=method, options = {'maxfev' : 50})
    
    # Try different solver if solution fail to converge properly
    if max(solution.fun)>1e-6:
        cascades_problem = CascadesNonlinearSystemProblem(cascades_data, R, eta_tt, eta_ts, Ma_crit, x0)
        solver = NonlinearSystemSolver(cascades_problem, cascades_problem.x0)
        solution = solver.solve(method='lm', options = {'maxiter' : 50})
    
    return solution


def performance_map(boundary_conditions, cascades_data):
    
    # Evaluate the cascades series at all conditions given in boundary_conditions
    # Exports the performance to ...
    
    x0 = None
    use_previous = True

    for i in range(len(boundary_conditions["fluid_name"])):
                
        conditions = {key : val[i] for key, val in boundary_conditions.items()}
        solution = performance(conditions, cascades_data, x0 = x0)
        
        if use_previous == True:
            x0 = cs.convert_scaled_x0(solution.x, cascades_data)
    
        # Save performance
        BC = {key : val for key, val in cascades_data["BC"].items() if key != 'fluid'}
        plane = cascades_data["plane"]
        cascade = cascades_data["cascade"]
        stage = cascades_data["stage"]
        overall = cascades_data["overall"]
        
        plane_stack = plane.stack()
        plane_stack.index = [f'{idx}_{col}' for col, idx in plane_stack.index]
        cascade_stack = cascade.stack()
        cascade_stack.index = [f'{idx}_{col}' for col, idx in cascade_stack.index]
        stage_stack = stage.stack()
        stage_stack.index = [f'{idx}_{col}' for col, idx in stage_stack.index]
        
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
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"Performance_data_{current_time}.xlsx"
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        BC_data.to_excel(writer, sheet_name = 'BC', index=False)
        plane_data.to_excel(writer, sheet_name = 'plane', index=False)
        cascade_data.to_excel(writer, sheet_name = 'cascade', index=False)
        stage_data.to_excel(writer, sheet_name = 'stage', index=False)
        overall_data.to_excel(writer, sheet_name = 'overall', index=False)
    
    
def generic_plot_distribution(lines, fig=None, ax=None, title="", xlabel="", ylabel="", scale_x=1, scale_y=1):
    
    if not fig and not ax:

        fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    for line in lines:
        df = line.get('df')
        zone = line.get('zone', "")
        data_key = line.get('data_key', "")
        legend = line.get('legend', None)
        linestyle = line.get('linestyle', "-")
        
    ax.plot(df[zone]["Position"] * scale_x, df[zone][data_key] * scale_y, linewidth=1.25, linestyle=linestyle, label=legend)

    if len(lines) > 1:
        ax.legend()

    fig.tight_layout(pad=1)

    return fig, 

def plot_function(filename, x, y, fig=None, ax=None, title="", xlabel="", ylabel="", scale_x=1, scale_y=1):
    
    if not fig and not ax:

        fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    x, lines = get_lines(filename, x, y)
    
    if isinstance(lines, (list, tuple)):
        for line in lines:
            ax.plot(x, line)
            
        return fig, ax

    ax.plot(x, line)
    
    return fig, ax
            
        
def get_lines(filename, x, y):
     
    # Import file that contains all performance parameteres
    
    # The resulting variable 'performance_data' will be a dictionary where keys are sheet names and values are DataFrames
    performance_data = pd.read_excel(filename, sheet_name = ['BC', 'plane', 'cascade', 'stage', 'overall'])
    
    x = get_line(performance_data, x)
    
    lines = []
    if isinstance(y, (list, tuple)):
        for i in range(len(y)):
            lines.append(get_line(performance_data,y[i]))
            
        return x, lines
    
    y = get_line(performance_data,y)
    
    return x, y

def get_line(performance_data, column):
    
    for key in performance_data.keys():
        
        if any(element == column for element in performance_data[key].columns):
            
            return performance_data[key][column]
        
    raise Exception(f"Could not find column {column} in performance_data")
    

        
        
    