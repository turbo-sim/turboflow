import os
import sys
import numpy as np
import matplotlib.pyplot as plt

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml
from meanline_axial import math


# Load configuration file
CONFIG_FILE = "config_one_stage.yaml"
case_data = ml.read_configuration_file(CONFIG_FILE)


ml.meanline.validate_axial_turbine_geometry(case_data["geometry_new"])
geom = ml.meanline.calculate_full_geometry(case_data["geometry_new"])
ml.print_dict(geom)
geom_info = ml.meanline.check_axial_turbine_geometry(geom)

print()
for msg in geom_info:
    print(msg)


# print(np.rad2deg(geom["flaring_angle"]))

# DONE: improve logic for angle conventions of rotor/stator
# DONE: improve logic so tip clearance can be zero for stator



