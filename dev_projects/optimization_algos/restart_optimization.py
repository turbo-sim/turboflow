
import turboflow as tf
import numpy as np
import os

# Load configuration file
# CONFIG_FILE = os.path.abspath("kofskey_constrained.yaml")
CONFIG_FILE = os.path.abspath("kofskey1972_1stage.yaml")
config = tf.load_config(CONFIG_FILE, print_summary=False)

# Load solver containes
filename = "output/pickle_multistart_slsqp_2024-09-01_02-03-20.pkl"
solver_container = tf.load_from_pickle(filename)
solvers = solver_container.solver_container

# Filter the succesful solvers
objective_functions = np.array([])
successes = 0
sols = []
initial_guesses = []
solver_container = []
func_count_total = 0
for solver in solvers:
    func_count_total += solver.convergence_history["func_count_total"][-1]
    if solver.success:
        sols.append(solver.convergence_history["x"][-1])
        initial_guesses.append(solver.convergence_history["x"][0])
        solver_container.append(solver)
        objective_functions = np.append(objective_functions, solver.convergence_history["objective_value"][-1])

# Kill horrible solutions
kill_solutions = [2]
for i in kill_solutions:
    del solver_container[i]
    del sols[i]
    del initial_guesses[i]
    objective_functions = np.delete(objective_functions, i)

# Sort solutions from least to most efficiency upgrade
combined = list(zip(objective_functions, sols, solver_container, initial_guesses))
combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
sorted_array, sorted_sols, solvers_sorted, initial_guesses_sorted = zip(*combined_sorted)

# Define which solution to start from
index = 6
# initial_guess = sorted_sols[index]
initial_guess = initial_guesses_sorted[index]

# Restart optimization
operation_points = config["operation_points"]

# for key in config["design_optimization"]["variables"].keys():
#     config["design_optimization"]["variables"]["key"]
# solver = tf.compute_optimal_turbine(config, export_results=False)
solver = tf.compute_optimal_turbine(config, export_results=False, initial_guess = initial_guess)

# Compare solutions
initial_efficiency = solvers[0].convergence_history["objective_value"][0]
print(f"New optimization: {solver.convergence_history['objective_value'][-1]-initial_efficiency}")
print(f"Old optimization: {sorted_array[index]-initial_efficiency}")
