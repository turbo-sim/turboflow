# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:58:04 2023

@author: laboan
"""

import numpy as np
import CoolProp as cp
import matplotlib.pyplot as plt
import os
import sys
desired_path = os.path.abspath("../..")

if desired_path not in sys.path:
    sys.path.append(desired_path)

import meanline_axial as ml
ml.set_plot_options()


def throat(h, blocking):
    
    s = 3900.832659
    h0 = 436348.93224
    # m = 2.441937
    m = 2.551717
    fluid = cp.AbstractState("HEOS", "air")
    beta = 66.6846*np.pi/180
    A = 0.02595989*(1-blocking)
    
    v = np.sqrt(2*(h0-h))
    fluid.update(cp.HmassSmass_INPUTS, h, s)
    rho_1 = fluid.rhomass()
    
    rho_2 = m/(v*np.cos(beta)*A)
    
    return (rho_1-rho_2)*100
    
N = 100
eps = 0.15
h = np.linspace(403619.336532-eps*403619.336532, 403619.336532+0.02*403619.336532, N)
blocking = [0, 0.05]

fig, ax = plt.subplots()
for b in blocking:
    res = np.zeros(N)
    for i in range(len(h)):
        
        res[i] = throat(h[i], b)
    
    ax.plot(h/1e5, res, label = str(b))
        
ax.legend()
ax.set_xlabel("Enthalpy")
ax.set_ylabel(r"$\rho_{\mathrm{error}}$")
    