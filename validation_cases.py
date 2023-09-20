# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:05:13 2023

@author: laboan
"""

import numpy as np
import pandas as pd
from turbo_functions import FluidCoolProp_2Phase
from df_keys import *

Case = 0

# Validation data
# Kofskey et al. 1972

if Case == 0:
    fluid_name = "air" # Fluid
    p0_in      = 13.8e4 # Inlet total pressure
    T0_in      = 295.6 # Inlet total temperature
    p_out      = p0_in/2.298 # Exit static pressure
    angle_in   = 0 # Flow angle at first stator
    omega      = 1627 # Rotational speed
    lossmodel  = 'KO' # loss model (Benner only for off-design, KO only other option)
    n_cascades = 2
    
    geometry = {"r_ht_in" : (0.7159, 0.7156), # Hub to tip ratio
                "s" : (1.8294e-2, 1.524e-2), # Pitch
                "c" : (2.616e-2, 2.606e-2), # Chord
                "b" : (0.022063, 0.025025), # Axial chord
                "H" : (3.363e-2, 3.658e-2), # Blade halloght
                "t_max" : (0.229e-2, 0.389e-2), # Maximum blade thickness
                "o" : (0.007731, 0.007155), # Throat opening
                "We" : (45*np.pi/180, 45*np.pi/180),  # Wedge angle
                "le" : (0.006, 0.006), # Leading edge radius
                "te" : (0.5e-4, 0.5e-4), # Trialing egde thickness (GUESS)
                "xi" : (0.567232, -0.282743), # Stagger anlge
                "theta_in" : (0, 29.6*np.pi/180), # Inlet metal anlge
                "theta_out" : (65*np.pi/180,-61.6*np.pi/180), # Exit metal angle
                "A_in" : (0.021468, 0.021468), # Inlet area
                "A_throat" : (0.021468, 0.02503), # Throat area
                "A_out" : (0.021468, 0.025197),  # Exit area
                "t_cl" : (0, 0.03e-2), # Tip clearance
                "radius" : (10.16e-2, 10.16e-2) # Radius (should be constant) 
                    } 
elif Case == 1:
    # Kofskey et al. 1974
    
    fluid_name = "air" # Fluid
    p0_in      = 10.82e4 # Inlet total pressure
    T0_in      = 310 # Inlet total temperature
    p_out      = p0_in/2.144 # Exit static pressure
    angle_in   = 0 # Flow angle at first stator
    omega      = 1963 # Rotational speed
    lossmodel  = 'Benner' # loss model (Benner only for off-design, KO only other option)
    n_cascades = 2
    
    geometry = {"r_ht_in" : (0.6769, 0.6769), # Hub to tip ratio
                "s" : (1.1858e-2, 1.1409e-2), # Pitch
                "c" : (1.9-2, 1.757e-2), # Chord
                "b" : (0.015935, 0.016914), # Axial chord
                "H" : (3.99e-2, 3.99e-2), # Blade halloght
                "t_max" : (0.003467, 0.002636), # Maximum blade thickness
                "o" : (0.7505e-2, 0.65e-2), # Throat opening
                "We" : (45*np.pi/180, 45*np.pi/180),  # Wedge angle
                "le" : (0.006, 0.006), # Leading edge radius
                "te" : (0.05e-2, 0.05e-2), # Trialing egde thickness (GUESS)
                "xi" : (0.575959, -0.274017), # Stagger anlge
                "theta_in" : (0, 23.2*np.pi/180), # Inlet metal anlge
                "theta_out" : (66*np.pi/180,-54.6*np.pi/180), # Exit metal angle
                "A_in" : (0.025947, 0.025947), # Inlet area
                "A_throat" : (0.025947, 0.027659), # Throat area
                "A_out" : (0.025947, 0.027659),  # Exit area
                "t_cl" : (0, 0.28e-3), # Tip clearance
                "radius" : (10.35e-2, 10.35e-2) # Radius (should be constant) 
                    } 
else:
    raise Exception('Case not valid')

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

