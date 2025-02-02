import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill

import turboflow as tf

tf.print_package_info()
tf.set_plot_options()

# Main directory
OUTPUT_DIR = "output_map"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create logger to save results
logger = tf.create_logger(name="run_validation_cases", path="logs", use_datetime=True, to_console=True)
tf.print_package_info(logger)

# Read case summary
# DATAFILE = "./validation_cases_summary.xlsx"
# case_data = pd.read_excel(DATAFILE)

# # Run cases based on list of tags
# filter = ["ipopt"]
# case_data = case_data[case_data["method"].isin(filter)]

# Run cases based on case number
# case_data = case_data[case_data["case"].isin([1,2])]
# case_data = case_data[case_data["case"].isin([200, 201, 300, 301])]

# Loop over cases
logger.info("Cases scheduled for simulation:")
# logger.info(", ".join(case_data["case"].astype(str)))

# Load configuration file
config_file = f"../config_files/one_stage_config.yaml"
config = tf.load_config(config_file, print_summary=False)

operation_points = config["performance_analysis"]["performance_map"]



# Log config details
logger.info("-" * 80)
logger.info(f"Running {config_file} with solver settings:")
tf.log_dict(logger, config["performance_analysis"]["solver_options"])
logger.info("-" * 80)

# operation_points = config["operation_points"]
# out_dir = os.path.join(OUTPUT_DIR, f"case_{row['case']}")
# out_filename = os.path.splitext(row['config_file'])[0]  # No .yaml extension
solvers = tf.compute_performance(
    operation_points,
    config,
    out_dir=OUTPUT_DIR,
    export_results=True,
    stop_on_failure=True,
    logger = logger
)

# print(operation_points_list)
#     # # Load solver object example
#     # with open('solver.dill', 'rb') as f: 
#     #     solver = dill.load(f)
#     # config_filename = f"./output/folder_name/config_file.yaml"
#     # config = tf.load_config(config_file, print_summary=False)
#     # problem = tf.CascadesOptimizationProblem(config)
#     # problem.fitness(solver.x_final)



