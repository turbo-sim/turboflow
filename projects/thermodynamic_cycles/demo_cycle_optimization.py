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
config = ml.utils.read_configuration_file(CONFIG_FILE, validate=False)
braytonCycle = ml.cycles.BraytonCycleProblem(config["problem_formulation"])
braytonCycle.get_optimization_values(braytonCycle.x0)
braytonCycle.to_excel(filename="initial_configuration.xlsx")

# Create interactive plot
braytonCycle.plot_cycle_diagram_realtime(CONFIG_FILE)

# Optimize the thermodynamic cycle
solver = ml.OptimizationSolver(
    braytonCycle,
    braytonCycle.x0,
    **config["solver_options"],
    callback_func=braytonCycle.plot_optimization_history,
)
solver.solve()
solver.problem.to_excel(filename="optimal_solution.xlsx")
for key, value in solver.problem.variables.items():
    print(f"{key:30}: {value:0.3f}")


# Make an animation of the optimization history
opt_dir = solver.problem.optimization_dir
ml.utils.create_mp4(
    opt_dir,
    os.path.join(os.path.dirname(opt_dir), "optimization_history.mp4"),
    fps=1,
)

# Keep figures open
plt.show()



# WORK DONE
# Main translation for mthe MATLAB code to Python
# Implemented functions and classes to evaluate/optimize recuperated brayton cycle
# Plotting of the brayton cycle in a nice way
# Polytropic process for expansion compression
# Also isentropic calculations for faster computation speed
# Descretization and temperature difference calculation for heat exchangers
# I formulated the problem using components instead of states as I did before in matlab (more flexible and intuitive)
# Added functionality to plot cycle at each optimization iteration (callback function).
# Make sure not to recompute the saturation line each time (lazy initialization)
# Implemented all problem parameters in YAML configuration file
# Added fields to formulate the objective function and constraints automatically
# Very flexible constraint definition by accessing variables in nested dictionary
# Implemented interactive plotting function that reads from configuration file
# Initial guess is easily generated in an interactive way with some practice
# Optimization starts when the interactive plot is closed or the user pressed enter
# Close the plot when the initial guess is decent and optimization can start
# Add constraint and objective normalization option
# Added smart way to normalize constraints according to configuration file
# Added a smart way to calculate the degree of subcooling or superheating that works insider and outside the twophase region
# Generalized the vapor quality calculation so that it takes sensible values outside of the two-phase region
# Added smart way to plot the phase diagram or update if it is already plot including, lines, quality grid, and response to changes in property pairs
# Added options ot control the visibility of phase diagram items according to YAML file
# Added code to use default plotting options when not provided
# Added improved callback function to save convergence history
# Added functionality to create an animation of the optimization
# Added functionality to export results to an organized folder with unique datatime identifiers
# Added functionality to export configuration file
# Added additional checks to ensure objective function and constraints are numeric
# Added additional checks to ensure objective function is scalar
# Add functionality to export cycle performance as pandas dataframe
# Implemented modest expression rendering engine using the $ character to be able to define some simple expressions in the YAML file
# Bounds for the heat sink and heat source directly taken from the fixed parameters
# Now it is possible to define the pressure/enthalpy bounds as a multiple of some key states like critical point, triple point, saturation at heat sink temperature, or dilute gas at heat source temperature





# TODO

# I should create a cycle optimization problem class.
# It should be very generic instead of having a brayton cycle problem class that is very sophiusticated
# The configuration should be given to the solver class
# The solver class should have a plot interactive method, and then a start_optimization method
# Finally it should have a export configuration method 
# Also one method to create gif from optimization run

# This class could be called in a loop to plot cycle for different inlet temperatures or whatever

# Add directory of examples showing optimization cases
# Further improve the generalization of the vapor quality calculation so that supercritical vapr qualities also collapse at the critical point
# Check for any differences with respect to matlab code
# Calculation of first principle closure
# Calculation of second principle (exergy) closure
# Add functionality to verify that all the required variables/fixed_parameters are defined
# Implement recompression cycle?
# Implement re-start from previous converged solution strategy

# Studies to do
# Parametric plot of cycle / system efficiency vs turbine inlet temperature
# Influence of inlet temperature and pressure on efficiency
# Comparison of turbomachinery influence:
# 1) isentropic machines
# 2) given value of isentropic efficiency
# 3) integration with meanline model
# 4) meanline model with shaft speed constraints
