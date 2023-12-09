import os
import sys
import numpy as np
import matplotlib.pyplot as plt

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    

import meanline_axial as ml

# Initialize Brayton cycle problem
CONFIG_FILE = "case_sCO2.yaml"
# CONFIG_FILE = "case_toluene.yaml"
config = ml.utils.read_configuration_file(CONFIG_FILE)
braytonCycle = ml.cycles.BraytonCycleProblem(config["problem_formulation"])
braytonCycle.get_optimization_values(braytonCycle.x0)
# braytonCycle.to_excel(filename="initial_configuration.xlsx")

# Create interactive plot
braytonCycle.plot_cycle_realtime(CONFIG_FILE)

# Optimize the thermodynamic cycle
solver = ml.OptimizationSolver(
    braytonCycle,
    braytonCycle.x0,
    **config["solver_options"],
    callback_func=braytonCycle.plot_cycle_callback,
)
solver.solve()
solver.problem.to_excel(filename="optimal_solution.xlsx")
for key, value in solver.problem.variables.items():
    print(f"{key:30}: {value:0.3f}")

# # Make an animation of the optimization history
# opt_dir = solver.problem.optimization_dir
# ml.utils.create_mp4(
#     opt_dir,
#     os.path.join(os.path.dirname(opt_dir), "optimization_history.mp4"),
#     fps=1,
# )

# Keep figures open
plt.show()


