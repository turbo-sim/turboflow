# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:45:02 2023

@author: laboan
"""

import numpy as np

def calculate_deviation(opening, pitch, Ma_exit, Ma_crit):
    
    beta_g = np.arcsin(opening/pitch)*180/np.pi
    delta_0 = np.arcsin(opening/pitch*(1+(1-opening/pitch)*(beta_g/90)**2))*180/np.pi-beta_g

    if Ma_exit < 0.50:
        delta = delta_0

    elif Ma_exit > 0.50:
        X = (2*Ma_exit-1) / (2*Ma_crit-1)
        delta = delta_0*(1-10*X**3+15*X**4-6*X**5)
                        
    beta = np.arccos(opening/pitch)*180/np.pi-delta

    return beta