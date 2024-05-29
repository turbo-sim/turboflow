import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp
from scipy.optimize._numdiff import approx_derivative
import turboflow as tf


# Load configuration file
CONFIG_FILE = "config_one_stage.yaml"
case_data = tf.read_configuration_file(CONFIG_FILE)


# print(case_data)

# ml.print_dict(case_data["model_options"])
config, _, _ = tf.validate_configuration_options(config=case_data, schema=ml.CONFIGURATION_OPTIONS)

# ml.print_dict(config)

# ml.print_dict(config)



