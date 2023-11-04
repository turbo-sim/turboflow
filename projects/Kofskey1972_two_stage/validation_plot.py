# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:54:41 2023

@author: laboan
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

desired_path = os.path.abspath('../..')

if desired_path not in sys.path:
    sys.path.append(desired_path)
    
import meanline_axial as ml

filename = "kofskey1972_2stage.yaml"
cascades_data = ml.read_configuration_file(filename)

if __name__ == '__main__':
    
    performance_map = ml.PerformanceMap()
    maps = ["m"]
    fig = performance_map.plot_omega_line(cascades_data, maps, pr_limits = [4.64, 5], N = 4, method = 'hybr', ig = {'R' : 0.4, 'eta' : 0.9, 
                                                                                                          'Ma_crit' : 0.92})
    # fig.savefig("Mass_flow_rate.png", bbox_inches = 'tight')