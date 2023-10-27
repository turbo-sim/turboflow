# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:34:28 2023

@author: laboan
"""

import numpy as np

from . import deviation_aungier as ar
from . import deviation_ainley_mathesion as am

available_deviation_models = ["aungier", "ainley_methesion", "metal_angle"]

def deviation(deviation_model, theta_out, opening, pitch, Ma_exit, Ma_crit):
    
    if deviation_model in available_deviation_models:
        if deviation_model == 'aungier':
            beta = ar.calculate_deviation(opening, pitch, Ma_exit, Ma_crit)
        elif deviation_model == 'ainley_mathesion':
            beta = am.calculate_deviation()
        elif deviation_model == 'metal_angle':
            beta = theta_out*180/np.pi
            
        return beta
    else:
        raise ValueError(f"Invalid deviation model. Available options: {', '.join(available_deviation_models)}")