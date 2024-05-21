import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import meanline_axial as ml 

# Define running option
CASE = 1

# Load configuration file
CONFIG_FILE = os.path.abspath("MM_base.yaml")
config = ml.read_configuration_file(CONFIG_FILE)

print(config["geometry"]["radius_hub"])
# stop

x0 = {'w_out_1': 117.45901482502731, 
      's_out_1': 1749.2457442847763, 
      'beta_out_1': 79.91058397648064, 
      'v*_in_1': 19.2061513238162, 
      'w*_throat_1': 183.25753196738836, 
      's*_throat_1': 1749.8237819845808, 
      'w*_out_1': 183.25343183989673, 
      'beta*_out_1': 79.99999745221794, 
      's*_out_1': 1749.8237436923164, 
      'w_out_2': 262.72848808964204, 
      's_out_2': 1760.4542001888556, 
      'beta_out_2': -64.97129612831986, 
      'v*_in_2': 117.46425093270062, 
      'w*_throat_2': 169.0002984397546, 
      'beta*_out_2': -70.22379260904015, 
      's*_throat_2': 1755.7223450705458, 
      'w*_out_2': 169.00661529641775, 
      's*_out_2': 1755.8389723307678, 
      'v_in': 16.701956605533116}

# Run calculations
if CASE == 1:
    # Compute performance map according to config file
    operation_points = config["operation_points"]
    solvers = ml.compute_performance(operation_points, config, initial_guess = x0, export_results=False, stop_on_failure=True)
