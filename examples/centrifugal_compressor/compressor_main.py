

import turboflow as tf
import numpy as np
import CoolProp as cp
import os

# Load configuration file
CONFIG_FILE = os.path.abspath("compressor_config.yaml")
config = tf.load_config(CONFIG_FILE)

operating_point = config["operation_points"]
solvers = tf.centrifugal_compressor.compute_performance(
    config,
    operating_point,
    export_results=False,
)

print(f'Total-to-total efficiency: {solvers[0].problem.results["overall"]["efficiency_tt"]}')
print(f'Total-to-total pressure ratio: {solvers[0].problem.results["overall"]["PR_tt"]}')
print(f'Choked: {solvers[0].problem.results["impeller"]["throat_plane"]["choked"]}')


