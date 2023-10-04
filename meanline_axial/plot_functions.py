# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 08:41:53 2023

@author: laboan
"""

import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename):
    
    # Import file that contains all performance parameteres
    # The resulting variable 'performance_data' will be a dictionary where keys are sheet names and values are DataFrames
    performance_data = pd.read_excel(filename, sheet_name = ['BC', 'plane', 'cascade', 'stage', 'overall'])
    # Round off to ignore precision loss by loading data from excel 
    for key, df in performance_data.items():
        performance_data[key] = df.round(10)
        
    return performance_data    

    
def save_figure(fig, filename):
    fig.savefig(filename, bbox_inches = 'tight')
    
def stack_lines_on_subset(performance_data, x_name, column_names, subset, fig = None, ax = None, xlabel = "", ylabel = "", title = None):
        # Plot each parameter in column_names as a function of x_name in the given subset
        # subset can only be one subset

        subset[1:] = [round(value, 10) for value in subset[1:]]
        
        if len(subset) > 2:
            raise Exception("Only one subset (i.e. (key, value)) is accepted for this function")
        
        if not fig and not ax:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title == None:
            ax.set_title(f"{subset[0]} = {subset[1]}")
        else:
            ax.set_title(title)
                
        y = get_lines(performance_data, column_names, subsets = subset)
        x = get_lines(performance_data, x_name, subsets = subset)
        
        ax.stackplot(x[0], y, labels = column_names)
            
        ax.legend()
        fig.tight_layout(pad=1)
            
        return fig, ax

def plot_lines_on_subset(performance_data, x_name, column_names, subset, fig = None, ax = None, xlabel = "", ylabel = "", title = None):
    # Plot each parameter in column_names as a function of x_name in the given subset
    # subset can only be one subset
    

    subset[1:] = [round(value, 10) for value in subset[1:]]
    
    if len(subset) > 2:
        raise Exception("Only one subset (i.e. (key, value)) is accepted for this function")
    
    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title == None:
        ax.set_title(f"{subset[0]} = {subset[1]}")
    else:
        ax.set_title(title)
            
    y = get_lines(performance_data, column_names, subsets = subset)
    x = get_lines(performance_data, x_name, subsets = subset)
    
    for i in range(len(y)):
        ax.plot(x[0], y[i], label = f"{column_names[i]}")
        
    fig.tight_layout(pad=1)
        
    return fig, ax
    

def plot_lines(performance_data, x_name, column_names, fig = None, ax = None, xlabel = "", ylabel = "", title = ""):
    
    # Plot each parameter in column_names as a function of x_name
    # filename refers to a file to be loaded containing all data needed
    # x_name should be a string referring to a column in data
    # column_names should be a list of string referring to columns in data
    
    if not fig and not ax:

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
            
    y = get_lines(performance_data, column_names)
    x = get_lines(performance_data, x_name)
    
    
    for i in range(len(y)):
        ax.plot(x, y[i], label = f"{column_names[i]}")
    
    fig.tight_layout(pad=1)
        
    return fig, ax
    

def plot_line(performance_data, x_name, column_name, fig = None, ax = None, xlabel = "", ylabel = "", title = ""):
    
    if not fig and not ax:

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
            
    y = get_lines(performance_data, column_name)
    x = get_lines(performance_data, x_name)
    
    ax.plot(x,y)
    
    fig.tight_layout(pad=1)
        
    return fig, ax
        

def plot_subsets(performance_data, x, y, subsets, fig = None, ax = None, xlabel = "", ylabel = "", title = ""):
    
    # Plot variable y as a function of x on the subset defined by subsets
    # *Subset is a tuple where first element is the key of a column in performance data
    # The subsequent values defines each subset of performance data 
    
    subsets[1:] = [round(value, 10) for value in subsets[1:]]
    
    if not fig and not ax:

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
            
    y = get_lines(performance_data, y, subsets = subsets)
    x = get_lines(performance_data, x, subsets = subsets)
        
    for i in range(len(y)):
        ax.plot(x[i], y[i], label = f"{subsets[0]} = {subsets[i+1]}")
    
    ax.legend()

    fig.tight_layout(pad=1)
        
    return fig, ax

def get_lines(performance_data, column_name, subsets = None):
    
    # Return an array of parameter column_name in performance_data
    # subset is a tuple where the first value determines the parameter, while the subsequent values 
    # determines the values of the parameter that defines the subset
     
    # Get lines covering all rows in performance_data
    if subsets == None:
        # Get single line
        if isinstance(column_name, str):
            return get_column(performance_data, column_name)
        # Get several columns
        else:
            lines = []
            for column in column_name:
                lines.append(get_column(performance_data, column))
                
            return lines
    
    subsets[1:] = [round(value, 10) for value in subsets[1:]]
    
    # Get lines covering given subset 
    lines = []
    for val in subsets[1:]:
        if isinstance(column_name, str):
            indices = get_subset(performance_data, subsets[0], val)
            lines.append(get_column(performance_data, column_name)[indices])
        else:
            for column in column_name:
                indices = get_subset(performance_data, subsets[0], val)
                lines.append(get_column(performance_data, column)[indices])
                
    return lines

def find_column(performance_data, column_name):
    
    # Function for column named column_name in all sheets in performance_data
    
    for key in performance_data.keys():
        
        if any(element == column_name for element in performance_data[key].columns):
            
            return key

    raise Exception(f"Could not find column {column_name} in performance_data")

def get_column(performance_data, column_name):
    
    sheet = find_column(performance_data, column_name)
    
    return performance_data[sheet][column_name]
        
    
def get_subset(performance_data, column_name, row_value):
    
    sheet = find_column(performance_data, column_name)
    
    return performance_data[sheet][performance_data[sheet][column_name] == row_value].index
    
        
    