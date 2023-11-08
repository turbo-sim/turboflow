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






def caluclate_geometry(geometry):
    
    A_in = []
    A_out = []
    H_m = []
    r_ht_in_vec = []
    r_m_in_vec = []
    r_m_out_vec = []
    
    for r_in, r_out, H_in, H_out in geometry["n_cascades"]:
        
        
        if geometry["radius_reference"] == 'mean':
            
            r_ht_in = (r_in-H_in/2)(r_in + H_in/2)
            r_ht_out = (r_out-H_out/2)(r_out + H_out/2)
            r_m_in = r_in
            r_m_out = r_out
            
        elif geometry["radius_reference"] == 'hub':
            
            r_t_in = r_in+H_in
            r_t_out = r_out+H_out
            r_ht_in = r_in/r_t_in
            r_ht_out = r_out/r_t_out
            r_m_in = (r_t_in+r_in)/2
            r_m_out = (r_t_out+r_out)/2
            
        elif geometry["radius_reference"] == 'tip':
            
            r_h_in = r_in-H_in
            r_h_out = r_out-H_out
            r_ht_in = r_h_in/r_in
            r_ht_out = r_h_out/r_out
            r_m_in = (r_h_in+r_in)/2
            r_m_out = (r_h_out+r_out)/2        
        
        H_m.append((H_in+H_out)/2)             
        A_in.append(2*np.pi*H_in*r_in)
        A_out.append(2*np.pi*H_out*r_out)
        r_m_in_vec.append(r_m_in)
        r_m_out_vec.append(r_m_out)
        r_ht_in_vec.append(r_ht_in)
        
    geometry["r_ht_in"] = r_ht_in
    geometry["r_m_in"] = r_m_in
    geometry["r_m_out"] = r_m_out
        





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






CONFIG_FILE = "config_one_stage.yaml"
case_data = ml.read_configuration_file(CONFIG_FILE)


radius_reference = 'mean'


# Example usage:
# Assuming 'config' is your dictionary containing the YAML data
geometry_validation_result, is_valid = validate_geometry_config(case_data["geometry"])
if is_valid:
    print(geometry_validation_result)
else:
    print(geometry_validation_result)