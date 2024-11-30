# %%
import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import turboflow as tf


dir_figs = "figures"
os.makedirs(dir_figs, exist_ok=True)

tf.set_plot_options()


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

    ax.plot(x, y, label=column_name, color='b', linewidth=2)  # Plot y vs. x with label
    
    ax.grid(False)

    fig.tight_layout(pad=1)

    return fig, ax


# Load all gradient deviations
gradient_deviation_dataframe = pd.read_excel('gradient_deviation_data.xlsx')

# Create plot for step size sensitivity
fig, ax = plot_grad_deviation_step_size(gradient_deviation_dataframe, "efficiency_blade_jet_ratio")   
filename = f"gradient_deviation_efficiency_blade_jet_ratio"
filepath = os.path.join(dir_figs, filename)
tf.savefig_in_formats(fig, filepath)

plt.show()


