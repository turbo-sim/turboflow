import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import turboflow as tf
from sklearn.linear_model import LinearRegression

from collections.abc import Mapping, Sequence

dir_figs = "figures"
os.makedirs(dir_figs, exist_ok=True)

tf.set_plot_options()

def plot_grad_comparison(config, step_sizes):
    # Assuming grad_jax and grad_FD are JAX arrays of shape (11, 29)
    
    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlabel('AD gradient')
    ax.set_ylabel('FD gradient')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect('equal')

    min_val = np.finfo(float).eps
    max_val = 1000

    # Plot the y=x line from min_val to max_val
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label="y = x")

    markers = ["s", "o", "+"]
    markersizes = [8.0, 5.0, 3.5]

    all_x = []
    all_y = []
    deviations = []

    for i, step_size in enumerate(step_sizes):
        config["design_optimization"]["solver_options"]["derivative_abs_step"] = step_size
        x_keys, output_dict, output, jax_jac, fd_jac = tf.fitness_gradient(config, step_size)
        
        # Flatten both grad_jax and grad_FD into 1D arrays
        gradient_jax_flat = np.abs(jax_jac.ravel())  # Flatten to 1D
        gradient_fd_flat = np.abs(fd_jac.ravel())  # Flatten to 1D

        ax.plot(gradient_jax_flat, gradient_fd_flat, linestyle="none", 
                marker=markers[i], markerfacecolor="w", markersize=markersizes[i], label=f"step_size {step_size:0.0e}")

        # Collect all data points for R-squared and absolute deviation calculation
        all_x.extend(gradient_jax_flat)
        all_y.extend(gradient_fd_flat)
        
        # Calculate absolute deviations from y=x line (y=x is the perfect agreement)
        deviations.extend(np.abs(gradient_jax_flat - gradient_fd_flat))

    ax.grid(True)
    
    # Set the limits for the main plot
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    # Create an inset axes for zooming into a region of interest
    ax_inset = inset_axes(ax, width=1.0, height=1.0, loc="center right", bbox_to_anchor=(0.77, 0.5), bbox_transform=fig.transFigure)

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
    
    # Convert all_x and all_y to numpy arrays for easier manipulation
    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Fit a linear regression model to calculate R-squared
    regressor = LinearRegression()
    regressor.fit(all_x.reshape(-1, 1), all_y)  # Fit to the data
    y_pred = regressor.predict(all_x.reshape(-1, 1))
    
    # Calculate R-squared
    ss_total = np.sum((all_y - np.mean(all_y))**2)
    ss_residual = np.sum((all_y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)

    print(f"R² = {r_squared:.2f}")
    
    # Calculate the maximum absolute deviation from the y=x line
    max_abs_deviation = np.max(deviations)
    max_deviation_index = np.argmax(deviations)

    # Find the corresponding AD and FD gradient values for the maximum deviation
    max_ad_grad = all_x[max_deviation_index]
    max_fd_grad = all_y[max_deviation_index]

    print(f"Maximum Absolute Deviation from y=x line: {max_abs_deviation:.4e}")
    print(f"AD Gradient (at max deviation): {max_ad_grad:.4e}")
    print(f"FD Gradient (at max deviation): {max_fd_grad:.4e}")

    # Display R-squared and maximum deviation in the legend
    ax.text(0.7, 0.9, f"R² = {r_squared:.2f}\nMax Abs Deviation = {max_abs_deviation:.4e}", 
            transform=ax.transAxes, fontsize=12, color='red')

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
filename = "gradient_verification"
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

