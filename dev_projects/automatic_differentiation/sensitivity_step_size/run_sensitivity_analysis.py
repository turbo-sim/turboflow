# %%
import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import turboflow as tf


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


# Load configuration file
CONFIG_FILE = os.path.abspath("../config_files/one_stage_config.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

# Compute optimal turbine
operation_points = config["operation_points"]

# Create dataframe of gradient deviation
step_sizes = np.logspace(-16, -1, num=200)
gradient_deviation_dataframe, gradient_dataframe = FD_grad_excel(config, step_sizes)

# Save dataframes to Excel
gradient_dataframe.to_excel('gradient_data.xlsx', index=False)
gradient_deviation_dataframe.to_excel('gradient_deviation_data.xlsx', index=False)

