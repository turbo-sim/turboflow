import os
import sys
import numpy as np
import matplotlib.pyplot as plt

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml
from meanline_axial import math
from meanline_axial import meanline


# Load configuration file
CONFIG_FILE = "config_one_stage.yaml"
case_data = ml.read_configuration_file(CONFIG_FILE)

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



