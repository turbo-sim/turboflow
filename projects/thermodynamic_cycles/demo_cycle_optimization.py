import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import turboflow as tf

# TODO
# The cycle problem class only takes information about the problem formulation, not the overall config file
# specifying also the settings for the optimization problem
# As a result, the output YAML is not as meaninful as it could be

# Should I have the plotting within the CycleProblem class? or move it somewhere else
# Keep in mind that I wanbt to updat eplot iteratively when calling function plot_cycle_realtime,
# but also update the plots with the latest data during optimization when a callback function is called

# Now I export properly the YAML file. Check the naming and the folder structure to comly with the exporting of figures
# Export initial configuration and final configuration separately

# I have to create a wraper class for doing hte optimization?

# # Would it be better to have a class for optimization where I also handle the solution of the optimization problem
# # In this class I could have the extra functionaliyt about plotting and exporting yaml file
# # Would this apporach be more suited to save confguration file at start and end of the optimiazation
# # Would this approach help me make parameter sweeps more, easily, or it does not really matter?


# Define configuration filename
# CONFIG_FILE = "case_water.yaml"
# CONFIG_FILE = "case_butane.yaml"
# CONFIG_FILE = "case_toluene.yaml"
# CONFIG_FILE = "case_sCO2_recuperated.yaml"
CONFIG_FILE = "case_sCO2_recompression.yaml"


# Usage example
optimizer = tf.thermodynamic_cycles.ThermodynamicCycleOptimization(CONFIG_FILE)
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
