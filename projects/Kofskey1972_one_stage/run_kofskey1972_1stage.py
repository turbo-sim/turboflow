import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml


CONFIG_FILE = "kofskey1972_1stage.yaml"
cascades_data = ml.read_configuration_file(CONFIG_FILE)

Case = 3





def validate_geometry_config(geometry_config):
    required_keys = {
        'n_cascades', 's', 'c', 'b', 'H', 't_max', 'o', 'We', 'le', 'te',
        'xi', 'theta_in', 'theta_out', 't_cl', 'radius', 'r_in', 'r_out',
        'r_ht_in', 'A_in', 'A_out'
    }
        # Check for required fields
    missing_keys = required_keys - geometry_config.keys()
    if missing_keys:
        return f"Missing geometry configuration keys: {missing_keys}", False
    
    # Check for extra fields
    extra_keys = geometry_config.keys() - required_keys
    if extra_keys:
        return f"Extra geometry configuration keys: {extra_keys}", False

    # If all checks pass
    return "Geometry configuration is valid.", True

# Example usage:
# Assuming 'config' is your dictionary containing the YAML data
geometry_validation_result, is_valid = validate_geometry_config(cascades_data["geometry"])
if is_valid:
    print(geometry_validation_result)
else:
    print(geometry_validation_result)


if Case == 1:
    # Compute performance map according to config file
    operation_points = cascades_data["operation_points"]
    ml.compute_performance(operation_points, cascades_data)

elif Case == 2:
    
    # Gnerate dataset with same conditions as dataset
    data = pd.read_excel("interpolated_dataset_kofskey1972_1stage.xlsx")
    pr_ts = data["pr_ts"]
    omega = data["omega"]
    
    performance_map  = {'fluid_name' : 'air',
                        'p0_in' : 13.8e4,
                        'T0_in' : 295.6,
                        'p_out' : 13.8e4/pr_ts.values,
                        'omega' : omega.values/100*1627}
    
    ml.compute_performance(performance_map, cascades_data)

elif Case == 3:
    # Compute performance map according to config file
    operation_points = cascades_data["performance_map"]
    omega_frac = np.asarray([0.5, 0.7, 0.9, 1.0])
    operation_points["omega"] = operation_points["omega"]*omega_frac
    ml.compute_performance(operation_points, cascades_data)



# Show plots
# plt.show()

    # DONE add option to give operation points as list of lists to define several speed lines
    # DONE add option to define range of values for all the parameters of the operating point, including T0_in, p0_in and alpha_in
    # DONE all variables should be ranged to create a nested list of lists
    # DONE the variables should work if they are scalars as well
    # DONE implemented closest-point strategy for initial guess of performance map
    # DONE implement two norm of relative deviation as metric

    # TODO update plotting so the different lines are plotted separately
    # TODO seggregate solver from initial guess in the single point evaluation
    
