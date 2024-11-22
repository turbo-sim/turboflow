# %%
import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import turboflow as tf

import jax
import jax.numpy as jnp

import pickle

# import dill

from collections.abc import Mapping, Sequence

dir_figs = "figures"
os.makedirs(dir_figs, exist_ok=True)
 
tf.set_plot_options()

def plot_grad_comparison(config, step_sizes):
    # Assuming grad_jax and grad_FD are JAX arrays of shape (11, 29)
    

    # Create a scatter plot
    # plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlabel('JAX gradient')
    ax.set_ylabel('Finite difference gradient')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect('equal')

    min_val = np.finfo(float).eps
    # min_val = 1e-3
    max_val = 1000
    # max_val = max(jnp.max(gradient_jax_flat), jnp.max(gradient_fd_flat))*10

    # Plot the y=x line from min_val to max_val
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label="y = x")

    # Add labels and title
    # ax.set_title('Comparison of Jacobian: JAX vs Finite Difference')
    
    markers = ["s", "o", "+"]
    markersizes = [8.0, 5.0, 3.5]

    for i,step_size in enumerate(step_sizes):
        config["design_optimization"]["solver_options"]["derivative_abs_step"] = step_size

        x_keys, output_dict, output, jax_jac, fd_jac = tf.fitness_gradient(config,step_size)
        
        # Flatten both grad_jax and grad_FD into 1D arrays
        gradient_jax_flat = np.abs(jax_jac.ravel())  # Flatten to 1D
        gradient_fd_flat = np.abs(fd_jac.ravel())  # Flatten to 1D

        ax.plot(np.abs(gradient_jax_flat), np.abs(gradient_fd_flat), linestyle="none" ,marker=markers[i], markerfacecolor="w", markersize=markersizes[i], label=f"step_size {step_size:0.0e}")


    ax.grid(False)

    # Plot the y = x line (perfect agreement)
    # We will set the range of the line to cover the range of the data in both axes

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])


    # Create an inset axes for zooming into a region of interest
    ax_inset = inset_axes(ax, width=1.0, height=1.0, loc="center right", bbox_to_anchor=(0.77, 0.5), bbox_transform=fig.transFigure)

    # Choose a zoomed-in region for the inset (adjust as necessary)
    inset_x_range = np.linspace(0.01, 1, 100)  # Example zoom range
    inset_y_range = np.linspace(0.01, 1, 100)  # Example zoom range

    # Plot the zoomed-in data in the inset plot
    # Select a small subset of the data for the inset (adjust these ranges as needed)
    for i, step_size in enumerate(step_sizes):
        config["design_optimization"]["solver_options"]["derivative_abs_step"] = step_size
        x_keys, output_dict, output, jax_jac, fd_jac = tf.fitness_gradient(config, step_size)
        
        gradient_jax_flat = np.abs(jax_jac.ravel())
        gradient_fd_flat = np.abs(fd_jac.ravel())
        
        # Only plot the data within the zoomed-in range
        mask = (gradient_jax_flat > 0.01) & (gradient_jax_flat < 1) & (gradient_fd_flat > 0.01) & (gradient_fd_flat < 1)
        ax_inset.plot(gradient_jax_flat[mask], gradient_fd_flat[mask], linestyle="none", 
                      marker=markers[i], markerfacecolor="w", markersize=markersizes[i])

    # Set limits and grid for the inset plot
    ax_inset.set_xlim(0.01, 1)
    ax_inset.set_ylim(0.01, 1)
    ax_inset.grid(True)
    
    # Optionally, draw a rectangle to highlight the zoomed region on the main plot
    mark_inset(ax, ax_inset, loc1=2, loc2=1, fc="none", ec="red")
    

    # Display the plot
    ax.legend()

    fig.tight_layout(pad=1)

    return fig, ax

def plot_grad_deviation_step_size(gradient_deviation_dataframe, column_name):
    # Assuming grad_jax and grad_FD are JAX arrays of shape (11, 29)
    
    
    # Create a scatter plot
    # plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlabel('Step size')
    ax.set_ylabel('Gradient deviation')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect('equal')

    # Extract the x-axis (first column)
    x = gradient_deviation_dataframe.iloc[:, 0]  # First column as x-axis

    # Extract the y-axis (specified column)
    y = gradient_deviation_dataframe[column_name]  # Column passed as argument for y-axis

    ax.plot(x, y, label=column_name, color='b', linewidth=2)  # Plot y vs. x with labe
    
    ax.grid(False)

    fig.tight_layout(pad=1)

    return fig, ax


def FD_grad_excel(config, step_sizes):

    x_keys, output_dict, output, grad_jax, grad_FD = tf.fitness_gradient(config, 1e-6)
    output_keys = list(output_dict.keys()) # 18 string elements
    design_variable_keys = x_keys # 29 string elements

    # Initializing the gradient_dataframe keys row
    keys_dict = {}
    for i in range(grad_FD.shape[0]):
            for j in range(grad_FD.shape[1]):
                keys_dict[f"{output_keys[i]}_{design_variable_keys[j]}"] = 1.0
        
    gradient_dataframe = pd.DataFrame(columns=keys_dict.keys()) # We have all the columns here
    gradient_deviation_dataframe = pd.DataFrame(columns=keys_dict.keys()) # We have all the columns here
    
    gradient_list = []
    gradient_deviation_list = []
    for step_size in step_sizes:
        x_keys, output_dict, output, grad_jax, grad_FD = tf.fitness_gradient(config, step_size)

        gradient_list.append(list(grad_FD.ravel()))
        gradient_deviation_list.append(list(abs(grad_FD.ravel() - grad_jax.ravel())))
        
    gradient_rows = pd.DataFrame(gradient_list, columns=gradient_dataframe.columns)
    gradient_deviation_rows = pd.DataFrame(gradient_deviation_list, columns=gradient_deviation_dataframe.columns)

    
    gradient_dataframe = pd.concat([gradient_dataframe, gradient_rows], ignore_index=True)
    gradient_deviation_dataframe = pd.concat([gradient_deviation_dataframe, gradient_deviation_rows], ignore_index=True)

    # Add the step_sizes as the first column in the gradient rows
    gradient_dataframe.insert(0, 'Step Size', step_sizes)
    gradient_deviation_dataframe.insert(0, 'Step Size', step_sizes)

    grad_jax_row = grad_jax.ravel()
    grad_jax_row = np.insert(grad_jax_row, 0, 1.0)
    grad_jax_df = pd.DataFrame([grad_jax_row], columns=gradient_dataframe.columns)
 
    gradient_dataframe = pd.concat([gradient_dataframe, grad_jax_df], ignore_index=True)

    return gradient_deviation_dataframe, gradient_dataframe








#By default jax uses 32 bit, for scientific computing we need 64 bit precision
# jax.config.update("jax_enable_x64", True)

# Define mode 
# MODE = "performance_analysis"
MODE = "design_optimization"

# Load configuration file
CONFIG_FILE = os.path.abspath("one_stage_config.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

if MODE == "performance_analysis":
    
    # Compute performance at operation point(s) according to config file
    operation_points = config["operation_points"]
    solvers = tf.compute_performance(
        operation_points,
        config,
        export_results=False,
        stop_on_failure=True,
    )

elif MODE == "performance_map":

    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, config, export_results=True)


elif MODE == "design_optimization":

    # Compute optimal turbine

    operation_points = config["operation_points"]

    # step_sizes = np.logspace(-16, -1, num=200)
    # x_keys, output_dict, output, grad_jax, grad_FD = tf.fitness_gradient(config, step_sizes[0])
    # gradient_deviation_dataframe, gradient_dataframe = FD_grad_excel(config, step_sizes)

    # gradient_deviation_dataframe = pd.read_excel('Gradient_Deviation_Data.xlsx')

    # fig, ax = plot_grad_deviation_step_size(gradient_deviation_dataframe, "efficiency_blade_jet_ratio")   
    # filename = f"Gradient Deviation efficiency_blade_jet_ratio"
    # filepath = os.path.join(dir_figs, filename)
    # tf.savefig_in_formats(fig, filepath)
    
    # gradient_dataframe.to_excel('Gradient_Data.xlsx', index=False)
    # gradient_deviation_dataframe.to_excel('Gradient_Deviation_Data.xlsx', index=False)

    # for i in range(grad_FD.shape[0]):
    #     for j in range(grad_FD.shape[1]):
    #         print(i, j, f"{grad_FD[i, j]:+0.8e}", f"{grad_jax[i, j]:+0.8e}", f"error {grad_FD[i, j]-grad_jax[i, j]:+0.8e}")
    
    
    # gradient_deviation_plot = tf.plot_functions.plot_grad_deviation(grad_jax, grad_FD)   
    # fig, ax = plot_grad_comparison(config, [1e-3, 1e-6, 1e-9])   
    # filename = "Gradient Validation zoomed in version"
    # filepath = os.path.join(dir_figs, filename)
    # tf.savefig_in_formats(fig, filepath)



    # fig, ax = plot_grad_deviation_step_size(config, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16])   
    # filename = "Gradient Deviation with Step Sizes"
    # filepath = os.path.join(dir_figs, filename)
    # tf.savefig_in_formats(fig, filepath)

    # plt.show()

    
    solver = tf.compute_optimal_turbine(config, export_results=True)




    file_path = os.path.join('output', f"pickle_1stage_jax_{config['design_optimization']['solver_options']['method']}.pkl")
    # Open a file in write-binary mode
    # with open(file_path, 'wb') as file:
    #     # Serialize the object and write it to the file
    #     pickle.dump(numpy_solver, file)

    # # tf.save_to_pickle(solver, filename = f"pickle_1stage_{config['design_optimization']['solver_options']['method']}", out_dir = "output")
    # # dump(solver, f"output/pickle_1stage_{config['design_optimization']['solver_options']['method']}.joblib")

    # fig, ax = tf.plot_functions.plot_axial_radial_plane(solver.problem.geometry)
    # # fig, ax = tf.plot_functions.plot_velocity_triangle(solver.problem.results["planes"])

# %%
