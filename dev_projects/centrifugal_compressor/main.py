
import turboflow as tf
import numpy as np
import CoolProp as cp
import os

# Load configuration file
# CONFIG_FILE = os.path.abspath("example_zhang.yml")
# CONFIG_FILE = os.path.abspath("validation_cases/Eckardt/eckardt_impeller_A/eckardt_impeller_A_case.yaml")
CONFIG_FILE = os.path.abspath("validation_cases/Eckardt/eckardt_impeller_O/eckardt_impeller_O_case.yaml")

config = tf.load_config(CONFIG_FILE, print_summary=True)

operating_point = config["performance_analysis"]["performance_map"]
# operating_point = config["operation_points"]
solvers = tf.centrifugal_compressor.compute_performance(
    config,
    operating_point,
    export_results=True,
)

solver = solvers[0]
results = solver.problem.results

print(solver.problem.results["impeller"]["throat_plane"])

# print(results["volute"]["exit_plane"])

print(results["overall"]["PR_tt"])
# print(results["overall"]["efficiency_ts"])
print(results["overall"]["efficiency_tt"])
# print(solver.problem.vars_real)
# print(results["planes"]["alpha"])
# print(results["planes"]["v_m"])
# print(results["planes"]["p0"])
# print(results["planes"]["h0"])
# print(results["planes"]["p"])