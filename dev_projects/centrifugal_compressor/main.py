
import turboflow as tf
import numpy as np
import CoolProp as cp
import os

# Load configuration file
CONFIG_FILE = os.path.abspath("example_zhang.yml")
config = tf.load_config(CONFIG_FILE, print_summary=True)

# operating_point = config["performance_analysis"]["performance_map"]
operating_point = config["operation_points"]
solvers = tf.centrifugal_compressor.compute_performance(
    config,
    operating_point,
    export_results=False,
)

solver = solvers[0]
results = solver.problem.results

print(results["impeller"]["exit_plane"]["throat_mass_flow_res"])

# print(solver.problem.results.keys())

# print(results["volute"]["exit_plane"])

# print(results["overall"]["PR_tt"])
# print(results["overall"]["efficiency_ts"])
# print(results["overall"]["efficiency_tt"])
# print(solver.problem.vars_real)
# print(results["planes"]["alpha"])
# print(results["planes"]["v_m"])
# print(results["planes"]["p0"])
# print(results["planes"]["h0"])
# print(results["planes"]["p"])