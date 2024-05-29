import os
import sys
import numpy as np
import matplotlib.pyplot as plt    
import meanline_axial as ml


# Define configuration filename
CONFIG_FILE = "case_sCO2_recuperated.yaml"



# Usage example
optimizer = ml.cycles.ThermodynamicCycleOptimization(CONFIG_FILE)
# TODO options to add as configuration
# Use plot callback or not, if yes, then make sure a function call to the plot_cycle function is done
# Move the configuration function handling to the configuration management class
# Move the plotting outside of the cycleProblem class
# Update code to handle parameter sweeps gracefully (create ranges of variabtion by permutations of the parameters to be checked, and pass a single list of dctionaries with the updated variable names. Update the keys of the configu dictionary with the keys provided)

# optimizer.problem.plot_cycle()  # TODO make sure that the optimization can run even if plot not initialized
optimizer.run_optimization()
optimizer.save_results()
# optimizer.generate_output_files()
# optimizer.create_animation()


print("hello")



# # Initialize Brayton cycle problem
# config = ml.utilities.read_configuration_file(CONFIG_FILE)
# thermoCycle = ml.cycles.ThermodynamicCycleProblem(config["problem_formulation"])
# thermoCycle.get_optimization_values(thermoCycle.x0)
# # thermoCycle.plot_cycle()
# # thermoCycle.to_excel(filename="initial_configuration.xlsx")

# # Create interactive plot
# thermoCycle.plot_cycle_realtime(CONFIG_FILE)

# # Optimize the thermodynamic cycle
# solver = ml.OptimizationSolver(
#     thermoCycle,
#     thermoCycle.x0,
#     **config["solver_options"],
#     callback_func=[
#         thermoCycle.plot_cycle_callback,
#         thermoCycle.save_config_callback,
#     ],
# )

# # save_current_configuration
# solver.solve()
# solver.problem.to_excel(filename="optimal_solution.xlsx")

# # Print final solution values
# print()
# print("Optimal set of design variables (normalized)")
# for key, value in solver.problem.variables.items():
#     print(f"{key:40}: {value:0.3f}")

# # Make an animation of the optimization history
# opt_dir = solver.problem.optimization_dir
# ml.utils.create_mp4(
#     opt_dir,
#     os.path.join(os.path.dirname(opt_dir), "optimization_history.mp4"),
#     fps=1,
# )

# # # Keep figures open
# # plt.show()



