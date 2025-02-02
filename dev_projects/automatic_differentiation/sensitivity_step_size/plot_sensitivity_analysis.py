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

def plot_grad_deviation_step_size(gradient_deviation_dataframe, column_names):
    # Define the color scheme
    colors = plt.get_cmap('plasma_r')(np.linspace(0.2, 1.0, len(column_names)))
    
    # Define a set of line styles
    line_styles = ['-', '-', '-', ':', '-', '--', '-.']  # List of line styles
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlabel('Step size', fontsize=16)
    ax.set_ylabel('Gradient deviation', fontsize=16)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect('equal')

    ax.set_xlim(1e-16, 1e0)
    

    # Create a mapping for column names to mathematical representations of partial derivatives
    label_mapping = {
        "efficiency_blade_jet_ratio": r"$\partial \eta_{\text{ts}} / \partial \lambda$",  # Partial derivative with respect to blade-jet ratio
        "efficiency_beta_out_1": r"$\partial \eta_{\text{ts}} / \partial \beta_{\text{out1}}$",  # Partial derivative with respect to beta_out1
        "efficiency_beta_out_2": r"$\partial \eta_{\text{ts}} / \partial \beta_{\text{out2}}$"  # Partial derivative with respect to beta_out2
    }
    
    # Extract the x-axis (first column, step size)
    x = gradient_deviation_dataframe.iloc[:, 0]  # First column as x-axis

    # Loop over the column names and plot each
    for i, column_name in enumerate(column_names):
        y = gradient_deviation_dataframe[column_name]  # Extract the y-values for the current column
        ax.plot(x, y, label=label_mapping[column_name], color=colors[i], linestyle=line_styles[i % len(line_styles)], linewidth=2)  # Plot with different colors and line styles
    
    # Adding a grid and legends below the plot
    ax.grid(True)
    
    # ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # Adjusting the legend to be at the bottom-right, in 3 rows, and increasing font size
    ax.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1, fontsize=16, borderaxespad=0.3)



    fig.tight_layout(pad=1)

    return fig, ax

# Load all gradient deviations
gradient_deviation_dataframe = pd.read_excel('gradient_deviation_data.xlsx')

# List of column names to plot
columns_to_plot = ["efficiency_blade_jet_ratio", "efficiency_beta_out_1", "efficiency_beta_out_2"]

# Create the plot for step size sensitivity
fig, ax = plot_grad_deviation_step_size(gradient_deviation_dataframe, columns_to_plot)

# Save the plot
filename = "gradient_deviation_multiple_columns_with_grid"
filepath = os.path.join(dir_figs, filename)
fig.savefig(filepath)

plt.show()




