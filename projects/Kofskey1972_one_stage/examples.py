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


Case = 4

# Validation data


if Case == 0:
    # One stage Kofskey et al. 1972
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
    
elif Case == 3:
    # Two stage Kofskey et al. (1972)
    fluid_name = "air" # Fluid
    p0_in      = 12.4e4 # Inlet total pressure
    T0_in      = 298.9 # Inlet total temperature
    p_out      = p0_in/4.640 # Exit static pressure
    angle_in   = 0 # Flow angle at first stator
    omega      = 1627 # Rotational speed
    lossmodel  = 'Benner' # loss model (Benner only for off-design, KO only other option)
    n_cascades = 4
    
    
    geometry = {"r_ht_in" : (0.7160, 0.7160, 0.6748, 0.6385), # Hub to tip ratio
                "s" : (1.8294e-2, 1.524e-2, 1.484e-2, 1.451e-2), # Pitch
                "c" : (2.616e-2, 2.606e-2, 2.182e-2, 2.408e-2), # Chord
                "b" : (1.9123e-2, 2.2326e-2, 1.9136e-2, 2.2366e-2), # Axial chord
                "H" : (3.363e-2, 3.658e-2, 4.214e-2, 4.801e-2), # Blade halloght
                "t_max" : (0.505e-2, 0.447e-2, 0.328e-2, 0.280e-2), # Maximum blade thickness
                "o" : (0.7731e-2, 0.7249e-2, 0.8469e-2, 0.9538e-2), # Throat opening
                "We" : (43.03*np.pi/180, 31.05*np.pi/180, 28.72*np.pi/180, 21.75*np.pi/180),  # Wedge angle
                "le" : (0.127e-2, 0.081e-2, 0.097e-2, 0.081e-2), # Leading edge radius
                "te" : (2*0.025e-2, 2*0.025e-2, 2*0.025e-2, 2*0.025e-2), # Trialing egde thickness (GUESS)
                "xi" : (43.03*np.pi/180, 31.05*np.pi/180, 28.72*np.pi/180, 21.75*np.pi/180), # Stagger angle
                "theta_in" : (0, 29.6*np.pi/180, -26.1*np.pi/180, 13.9*np.pi/180), # Inlet metal anlge
                "theta_out" : (65*np.pi/180,-61.6*np.pi/180, 55.2*np.pi/180, -48.9*np.pi/180), # Exit metal angle
                "A_in" : (0.021468, 0.021468, 0.025197, 0.028618), # Inlet area
                "A_throat" : (9.10e-03, 1.20e-02, 1.63e-02, 2.15e-02), # Throat area
                "A_out" : (0.021468, 0.025197, 0.028618, 0.032678),  # Exit area
                "t_cl" : (0, 0.030e-2,0,0.038e-2), # Tip clearance
                "radius" : (10.16e-2, 10.16e-2, 10.16e-2, 10.16e-2) # Radius (should be constant) 
                    } 
    
elif Case == 4:
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
    
    
else:
    raise Exception('Case not valid')

# Create structure for turbine data
BC = {}
BC["fluid"] = ml.FluidCoolProp_2Phase(fluid_name)
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
data_structure["plane"] = pd.DataFrame(columns = ml.meanline.keys_plane)
data_structure["cascade"] = pd.DataFrame(columns = ml.meanline.keys_cascade)
data_structure["overall"] = overall

if __name__ == '__main__':
    
    import time
    
    starttime = time.time()
    
    ml.meanline.number_stages(data_structure)
    ml.meanline.update_fixed_params(data_structure)
    x0 = ml.meanline.generate_initial_guess(data_structure, R = 0.4)
    x_scaled = ml.meanline.scale_x0(x0, data_structure)
    solution, convergence_history = ml.meanline.cascade_series_analysis(data_structure, x_scaled)
    endtime = time.time()-starttime
    

