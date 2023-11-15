import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp
from scipy.optimize._numdiff import approx_derivative

import os
import sys

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml


# Load configuration file
CONFIG_FILE = "config_one_stage.yaml"
case_data = ml.read_configuration_file(CONFIG_FILE)


# print(case_data)

# ml.print_dict(case_data["model_options"])
config = ml.validate_configuration_file(config=case_data, schema=ml.configuration_options)


# ml.print_dict(config)


