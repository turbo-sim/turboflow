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


# filename = "R125_opt.yaml"

filename = "R125.yaml"
cascades_data = ml.get_cascades_data(filename)

Case = 0

if Case == 0:
    
    # Solve using nonlinear equation solver
    BC = cascades_data["BC"]
    sol = ml.compute_operating_point(BC, cascades_data)
    
elif Case == 2:
    design_point = cascades_data["BC"]
    
    # x0 = np.array([0.6367030438527637, 0.07576436,  0.66517235,  0.66227372,  1.00207725,  1.00206263,
    #         1.39694108,  0.65288985,  0.6475509 ,  1.005179  ,  1.00513671,
    #         -1.33543023,  0.07655413,  0.74213551,  1.00246706,  0.66882363,
    #         0.72243958,  1.00572839])
    
    x0 = 1.01*np.array([ 0.0756511 ,  0.66000876,  0.65678993,  1.00205134,  1.00203512,
            1.39691342,  0.65185969,  0.6460505 ,  1.00491815,  1.00487539,
           -1.33537204,  0.07655411,  0.74213538,  1.00246707,  0.66321498,
            0.72677942,  1.00546961])
    
    sol = ml.compute_optimal_turbine(design_point, cascades_data, x0)
