# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:05:13 2023

@author: laboan
"""

import numpy as np
import pandas as pd

import os
import sys

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml



filename = "R125.yaml"
data_structure = ml.read_configuration_file(filename)

fluid_name = data_structure["BC"]["fluid_name"]

data_structure["geometry"] = {key: np.asarray(value) for key, value in data_structure["geometry"].items()}

data_structure["BC"]["fluid"] = ml.FluidCoolProp_2Phase(fluid_name)
data_structure["fixed_params"] = {}
data_structure["overall"] = {}
data_structure["plane"] = pd.DataFrame(columns = ml.meanline.keys_plane)
data_structure["cascade"] = pd.DataFrame(columns = ml.meanline.keys_cascade)

# ml.print_dict(data_structure)

if __name__ == '__main__':
    
    import time
    
    starttime = time.time()
    
    ml.meanline.number_stages(data_structure)
    ml.meanline.update_fixed_params(data_structure)
    x0 = ml.meanline.generate_initial_guess(data_structure, R = 0.4)
    x_scaled = ml.meanline.scale_x0(x0, data_structure)
    solution, convergence_history = ml.meanline.cascade_series_analysis(data_structure, x_scaled)
    endtime = time.time()-starttime
    

