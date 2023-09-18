# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:46:23 2023

@author: laboan
"""

import Cascade_series as CS
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
from turbo_functions import FluidCoolProp_2Phase
import CoolProp as CP
from df_keys import *


# Define boundary conditions
fluid_name = "R125" # Fluid
p0_in      = 36.5e5 # Inlet total pressure
T0_in      = 190+273.15 # Inlet total temperature
p_out      = 15.85e5 # Exit static pressure
angle_in   = 0 # Flow angle at first stator
omega      = 3458 # Rotational speed
lossmodel  = 'Benner' # loss model (Benner only for off-design, KO only other option)
n_cascades = 2

# Define geometry (each element in tuple belong to a different cascade)
geometry = {"r_ht_in" : (0.6, 0.6), # Hub to tip ratio
            "s" : (0.0081, 0.0073), # Pitch
            "c" : (0.0132, 0.0109), # Chord
            "b" : (0.0101, 0.0076), # Axial chord
            "H" : (0.0204, 0.0218), # Blade halloght
            "t_max" : (0.0026, 0.0019), # Maximum blade thickness
            "o" : (0.0014, 0.0017), # Throat opening
            "We" : (45*np.pi/180, 45*np.pi/180),  # Wedge angle
            "le" : (0.006, 0.006), # Leading edge radius
            "te" : (0.00076, 0.0076), # Trialing egde thickness
            "xi" : (0.698132, -0.80116), # Stagger anlge
            "theta_in" : (0, -0.2618), # Inlet metal anlge
            "theta_out" : (1.3963, -1.3405), # Exit metal angle
            "A_in" : (0.0052, 0.0052), # Inlet area
            "A_throat" : (0.0009, 0.001352), # Throat area
            "A_out" : (0.0052, 0.005952*1.4),  # Exit area
            "t_cl" : (0, 5e-4), # Tip clearance
            "radius" : (0.0408, 0.0408) # Radius (should be constant) 
                } 

##################################################################################

# Create structure for turbine data
BC = {}
BC["fluid"] = FluidCoolProp_2Phase(fluid_name)
BC["p0_in"] = p0_in
BC["T0_in"] = T0_in
BC["p_out"] = p_out
BC["alpha_in"] = angle_in

fixed_params = {}
fixed_params["n_cascades"] = n_cascades
fixed_params["loss_model"] = lossmodel

overall = {}
overall["omega"] = omega

# Organize_data
data_structure = {}
data_structure["BC"] = BC
data_structure["fixed_params"] = fixed_params
data_structure["geometry"] = pd.DataFrame(geometry)
data_structure["plane"] = pd.DataFrame(columns = keys_plane)
data_structure["cascade"] = pd.DataFrame(columns = keys_cascade)
data_structure["overall"] = overall


if __name__ == '__main__':
    
    R = np.linspace(0.2,0.6, 3)
    
    # starttime = time.time()
    # solution, convergence_history = CS.cascade_series_analysis(data_structure, x = x0)
    # endtime = time.time()-starttime
    
    fastest_result = CS.multiprocess(data_structure, R)


