import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import turboflow as tf

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

    for i, step_size in enumerate(step_sizes):
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


# Load configuration file
CONFIG_FILE = os.path.abspath("../config_files/one_stage_config.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

# Plot gradient validation
fig, ax = plot_grad_comparison(config, [1e-3, 1e-6, 1e-9])   
filename = "gradient_validation"
filepath = os.path.join(dir_figs, filename)
tf.savefig_in_formats(fig, filepath)

# Show figure
plt.show()


# Print gradient to check if they are correct
# step_size = 1e-3
# x_keys, output_dict, output, grad_jax, grad_FD = tf.fitness_gradient(config, step_size)
# for i in range(grad_FD.shape[0]):
#     for j in range(grad_FD.shape[1]):
#         print(i, j, f"{grad_FD[i, j]:+0.8e}", f"{grad_jax[i, j]:+0.8e}", f"error {grad_FD[i, j]-grad_jax[i, j]:+0.8e}")

