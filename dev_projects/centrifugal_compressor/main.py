
import turboflow as tf
import numpy as np
import CoolProp as cp
import os

# Load configuration file
# CONFIG_FILE = os.path.abspath("example_zhang.yml")
# CONFIG_FILE = os.path.abspath("example_volute.yml")
# CONFIG_FILE = os.path.abspath("validation_cases/Eckardt/eckardt_impeller_A/eckardt_impeller_A_case.yaml")
CONFIG_FILE = os.path.abspath("validation_cases/Eckardt/eckardt_impeller_O/eckardt_impeller_O_case.yaml")

config = tf.load_config(CONFIG_FILE, print_summary=False)

# operating_point = config["performance_analysis"]["performance_map"]
operating_point = config["operation_points"]
solvers = tf.centrifugal_compressor.compute_performance(
    config,
    operating_point,
    export_results=False,
    stop_on_failure=False,
)
solver = solvers[0]
results = solver.problem.results

# print(f'Velocity inlet: {solver.problem.results["vaned_diffuser"]["inlet_plane"]["v"]}')
# print(f'Velocity exit: {solver.problem.results["vaned_diffuser"]["exit_plane"]["v"]}')
# print(f'Velocity throat: {solver.problem.results["vaned_diffuser"]["throat_plane"]["w"]}')
# print(f'Entropy exit: {solver.problem.results["vaned_diffuser"]["exit_plane"]["s"]}')

print(f'Temperature: {solvers[0].problem.results["planes"]["T0"].values[-1]}')


# print(f'Temperature: {solvers[0].problem.results["boundary_conditions"]["T0_in"]}')
print(f'Temperature: {solvers[0].problem.results["boundary_conditions"]["p0_in"]}')
print(f'Temperature: {solvers[0].problem.results["boundary_conditions"]["mass_flow_rate"]}')
print(f'Temperature: {solvers[0].problem.results["boundary_conditions"]["omega"]}')
print(f'Temperature: {solvers[0].problem.results["boundary_conditions"]["alpha_in"]}')

print(f'Total-to-total efficiency: {solvers[0].problem.results["overall"]["efficiency_tt"]}')
print(f'Total-to-total pressure ratio: {solvers[0].problem.results["overall"]["PR_tt"]}')

print(f'Mass flow residual: {solver.problem.results["impeller"]["throat_plane"]["mass_flow_residual"]}')
print(f'Choked: {solver.problem.results["impeller"]["throat_plane"]["choked"]}')
# print(f'Correct solution: {solver.problem.results["impeller"]["throat_plane"]["correct_solution"]}')
print(solver.problem.vars_real)
