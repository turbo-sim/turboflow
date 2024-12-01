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

# Read case summary
DATAFILE = "./two_stage_tol_1e-6.xlsx"
case_data = pd.read_excel(DATAFILE)

# Run cases based on list of tags
# filter = []
# case_data = case_data[case_data["stages"].isin(filter)]

# # Run cases based on case number
# case_data = case_data[case_data["case"].isin([2, 3, 4])]

# # Run cases based on numerical value
# case_data = case_data[np.isclose(case_data["max_iterations"], 500)]


# Loop over cases
print("Cases scheduled for simulation:")
print(", ".join(case_data["case"].astype(str)))
solver = None
for i, row in case_data.iterrows():

    # Load configuration file
    config_file = f"./config_files/{row['config_file']}"
    config = tf.load_config(config_file, print_summary=False)

    # Load config parameters from Excel file
    vars = ["library", "method", "derivative_method", "derivative_abs_step", "tolerance", "max_iterations"]
    for var in vars:
        config["design_optimization"]["solver_options"][var] = row[var]

    # Print config to check
    # tf.print_dict(config)

    # Compute optimal turbine
    operation_points = config["operation_points"]
    out_dir = os.path.join(OUTPUT_DIR, f"case_{row['case']}_{row['stages']}stage_{row['method']}_{row['derivative_method']}_{row['derivative_abs_step']}_{row['tolerance']}")
    solver = tf.compute_optimal_turbine(config,
                                        out_dir=out_dir,
                                        # out_filename=,
                                        export_results=True)
    
    

    # # Load solver object example
    # with open('solver.dill', 'rb') as f: 
    #     solver = dill.load(f)
    # config_filename = f"./output/folder_name/config_file.yaml"
    # config = tf.load_config(config_file, print_summary=False)
    # problem = tf.CascadesOptimizationProblem(config)
    # problem.fitness(solver.x_final)

