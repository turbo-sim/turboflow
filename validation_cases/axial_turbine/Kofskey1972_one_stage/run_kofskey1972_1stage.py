import os
import sys
import copy
import numpy as np
import pandas as pd
import datetime
import yaml
import matplotlib.pyplot as plt
import turboflow as tf

# Define running case
CASE = "performance_map"

# Load configuration file
CONFIG_FILE = os.path.abspath("kofskey1972_1stage.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

if CASE == "performance_map":
    # Compute performance map according to config file
    operation_points = config["performance_analysis"]["performance_map"]
    solvers = tf.compute_performance(operation_points, 
                                     config, 
                                     out_filename="performance_map_critical_mach",
                                     export_results=True)

elif CASE == "experimental_points":

    # Load experimental dataset
    sheets = ["Mass flow rate", "Torque", "Total-to-static efficiency", "alpha_out"]
    data = pd.read_excel(
        "experimental_data/experimental_data_kofskey1972_1stage_raw.xlsx", sheet_name=sheets
    )

    # Merge all experimental points to one array of points
    pressure_ratio_exp = []
    speed_frac_exp = []
    for sheet in sheets:
        pressure_ratio_exp += list(data[sheet]["PR"].values)
        speed_frac_exp += list(data[sheet]["omega"].values / 100)
    pressure_ratio_exp = np.array(pressure_ratio_exp)
    speed_frac_exp = np.array(speed_frac_exp)

    # Generate operating points with same conditions as dataset
    operation_points = []
    design_point = config["operation_points"]
    for PR, speed_frac in zip(pressure_ratio_exp, speed_frac_exp):
        if not speed_frac in [
            0.3,
            0.5,
        ]:  # 30 and 50% desing speed not included in validation plot
            current_point = copy.deepcopy(design_point)
            current_point["p_out"] = design_point["p0_in"] / PR
            current_point["omega"] = design_point["omega"] * speed_frac
            operation_points.append(current_point)

    # Compute performance at experimental operating points
    tf.compute_performance(operation_points, 
                            config, 
                            out_filename="experimental_points",
                            export_results=True)

