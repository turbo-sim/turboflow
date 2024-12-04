import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import turboflow as tf

tf.set_plot_options()

# Main directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create logger to save results
logger = tf.create_logger(
    name="run_optimizations",
    path=f"{OUTPUT_DIR}/logs",
    use_datetime=True,
    to_console=True,
)
tf.print_package_info(logger)

# Read case summary
DATAFILE = "./cases_summary.xlsx"
case_data = pd.read_excel(DATAFILE)

# # Run cases based on list of tags
# filter = ["ipopt"]
# case_data = case_data[case_data["method"].isin(filter)]

# # Run cases based on case number
# case_data = case_data[case_data["case"].isin([100, 101, 102])]
# # case_data = case_data[case_data["case"].isin([200, 201, 300, 301])]

# Loop over cases
logger.info("Cases scheduled for simulation:")
logger.info(", ".join(case_data["case"].astype(str)))
for i, row in case_data.iterrows():

    # Load configuration file
    config_file = f"./config_files/{row['config_file']}"
    config = tf.load_config(config_file, print_summary=False)

    # Load config parameters from Excel file
    vars = [
        "library",
        "method",
        "derivative_method",
        "derivative_abs_step",
        "tolerance",
        "max_iterations",
    ]
    for var in vars:
        config["design_optimization"]["solver_options"][var] = row[var]

    config["simulation_options"]["loss_model"]["model"] = "isentropic"
    config["design_optimization"]["solver_options"]["plot_convergence"] = True
    config["design_optimization"]["solver_options"]["plot_scale_constraints"] = "log"
    # config["design_optimization"]["solver_options"]["extra_options"] = {"hessian_approximation": "exact"}
    
    # Log config details
    logger.info("-" * 80)
    logger.info(f"Running {row['config_file']} with solver settings:")
    tf.log_dict(logger, config["design_optimization"]["solver_options"])
    logger.info("-" * 80)

    # # Initialize problem object
    # problem = tf.CascadesOptimizationProblem(config)

    # # Check that the initial guess is within bounds and clip if necessary
    # problem.initial_guess = tf.check_and_clip_initial_guess(
    #     problem.initial_guess,
    #     problem.bounds,
    #     problem.design_variables_keys
    # )

    # fitness = problem.fitness(problem.initial_guess)
    # HessianFull = problem.hessians_jax(problem.initial_guess, lower_triangular=False)
    # # Iterate over the components of the fitness function
    # ii = 0
    # for k, hessian_matrix in enumerate(HessianFull):  # k is the index of the fitness component
    #     print(f"Hessian for fitness component {k}:")
    #     rows, cols = hessian_matrix.shape

    #     # Iterate through all i, j pairs in the matrix
    #     for i in range(rows):
    #         print(hessian_matrix[i, :])
    #         # for j in range(cols):
    #         #     if np.isnan(hessian_matrix[i, j]):
    #         #         print(ii, f"H[{k}][{i}][{j}] = {hessian_matrix[i, j]}")
    #         #         ii += 1

    #     print("\n")  # Add a newline for readability between components
        
    # print("Full size", np.size(HessianFull))


    # Compute optimal turbine
    operation_points = config["operation_points"]
    out_dir = os.path.join(OUTPUT_DIR, f"case_{row['case']}")
    out_filename = os.path.splitext(row["config_file"])[0]  # No .yaml extension
    solver = tf.compute_optimal_turbine(
        config,
        out_dir=out_dir,
        out_filename=out_filename,
        export_results=True,
        logger=logger,
    )

    solver.plot_convergence_history()
    plt.show()

#     # # Load solver object example
#     # with open('solver.dill', 'rb') as f:
#     #     solver = dill.load(f)
#     # config_filename = f"./output/folder_name/config_file.yaml"
#     # config = tf.load_config(config_file, print_summary=False)
#     # problem = tf.CascadesOptimizationProblem(config)
#     # problem.fitness(solver.x_final)
