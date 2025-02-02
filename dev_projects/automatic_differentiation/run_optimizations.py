import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill

import turboflow as tf

tf.print_package_info()
tf.set_plot_options()

# Main directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create logger to save results
logger = tf.create_logger(name="run_optimizations", path=f"{OUTPUT_DIR}/logs", use_datetime=True, to_console=True)
tf.print_package_info(logger)

# Read case summary
DATAFILE = "./cases_summary.xlsx"
case_data = pd.read_excel(DATAFILE)

# # Run cases based on list of tags
# filter = ["ipopt"]
# case_data = case_data[case_data["method"].isin(filter)]

# Run cases based on case number
case_data = case_data[case_data["case"].isin([86,87,88,89,90])]
# case_data = case_data[case_data["case"].isin([200, 201, 300, 301])]

# Loop over cases
logger.info("Cases scheduled for simulation:")
logger.info(", ".join(case_data["case"].astype(str)))
for i, row in case_data.iterrows():

    # Load configuration file
    config_file = f"./config_files/{row['config_file']}"
    config = tf.load_config(config_file, print_summary=False)

    # Load config parameters from Excel file
    vars = ["library", "method", "derivative_method", "derivative_abs_step", "tolerance", "max_iterations"]
    for var in vars:
        config["design_optimization"]["solver_options"][var] = row[var]

     # Log config details
    logger.info("-" * 80)
    logger.info(f"Running {row['config_file']} with solver settings:")
    tf.log_dict(logger, config["design_optimization"]["solver_options"])
    logger.info("-" * 80)

    # Compute optimal turbine
    operation_points = config["operation_points"]
    out_dir = os.path.join(OUTPUT_DIR, f"case_{row['case']}")
    out_filename = os.path.splitext(row['config_file'])[0]  # No .yaml extension
    solver = tf.compute_optimal_turbine(config,
                                        out_dir=out_dir,
                                        out_filename=out_filename,
                                        export_results=True,
                                        logger=logger)

#     # # Load solver object example
#     # with open('solver.dill', 'rb') as f: 
#     #     solver = dill.load(f)
#     # config_filename = f"./output/folder_name/config_file.yaml"
#     # config = tf.load_config(config_file, print_summary=False)
#     # problem = tf.CascadesOptimizationProblem(config)
#     # problem.fitness(solver.x_final)



