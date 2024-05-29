import os
import sys
import numpy as np
import matplotlib.pyplot as plt

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import turbo_flow as tf
from turbo_flow import math
from turbo_flow import meanline


# Load configuration file
CONFIG_FILE = "config_one_stage.yaml"
case_data = tf.read_configuration_file(CONFIG_FILE)

# ml.print_dict(case_data)

# # Create partial geometry from optimization variables
# geom = meanline.create_partial_geometry_from_optimization_variables(case_data["_geometry_optimization"], None)
# ml.print_dict(geom)
# print()

# # Create full geometry
# ml.meanline.validate_turbine_geometry(geom)
# geom = ml.meanline.calculate_full_geometry(geom)
# ml.print_dict(geom)
# geom_info = ml.meanline.check_turbine_geometry(geom)



