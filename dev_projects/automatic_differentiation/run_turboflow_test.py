import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill
import turboflow as tf

# === SETTINGS ===
MODE = "optimization"  # Choose between "performance" or "optimization"
CASE_NUMBERS = [1]    # List of case numbers to run

# === INITIALIZATION ===
tf.print_package_info()
tf.set_plot_options()

logger = tf.create_logger(name="console_logger", path=None, use_datetime=False, to_console=True)


# Read appropriate case summary file
DATAFILE = "./validation/validation_cases_summary.xlsx" if MODE == "performance" else "./cases_summary.xlsx"
case_data = pd.read_excel(DATAFILE)
case_data = case_data[case_data["case"].isin(CASE_NUMBERS)]

print("Cases scheduled for simulation:")
print(", ".join(case_data["case"].astype(str)))

# === EXECUTION ===
if MODE == "performance":
    config_file = f"./config_files/one_stage_config.yaml"
    config = tf.load_config(config_file, print_summary=False)

    operation_points_list = []

    for _, row in case_data.iterrows():
        vars = ["fluid_name", "T0_in", "p0_in", "p_out", "omega", "alpha_in"]
        operation_points = {var: row[var] for var in vars}
        operation_points_list.append(operation_points)

    print("-" * 60)
    print(f"Running performance analysis using: {config_file}")
    # tf.log_dict(print, config["performance_analysis"]["solver_options"])
    print("-" * 60)

    solvers = tf.compute_performance(
        operation_points_list,
        config,
        export_results=False,
        stop_on_failure=True,
        logger=logger  # No logging
    )

elif MODE == "optimization":
    for _, row in case_data.iterrows():
        config_file = f"./config_files/{row['config_file']}"
        config = tf.load_config(config_file, print_summary=False)

        vars = ["library", "method", "derivative_method", "derivative_abs_step", "tolerance", "max_iterations"]
        for var in vars:
            config["design_optimization"]["solver_options"][var] = row[var]

        print("-" * 60)
        print(f"Running optimization using: {config_file}")
        # tf.log_dict(print, config["design_optimization"]["solver_options"])
        print("-" * 60)

        solver = tf.compute_optimal_turbine(
            config,
            export_results=False,
            logger=logger  # No logging
        )

else:
    raise ValueError("MODE must be either 'performance' or 'optimization'")
