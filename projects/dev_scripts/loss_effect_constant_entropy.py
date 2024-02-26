# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:51:44 2024

@author: lasseba
"""

import CoolProp as cp
import numpy as np
import matplotlib.pyplot as plt

# Define inputs
p01 = 10e5
T01 = 300
fluid_name = 'air'
N = 1000
M = 5
pressure_ratio = np.linspace(0.1, 0.9, N)
Y = np.asarray([0.1, 0.2, 0.3])

# Define input state
fluid = cp.AbstractState('HEOS', fluid_name)
fluid.update(cp.PT_INPUTS, p01, T01)
h01 = fluid.hmass()
s1 = fluid.smass()

# Define result vectors
phi = np.zeros(N)
Ma = np.zeros(N)
s = np.zeros(N)

# Define figure
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()

for j in range(len(Y)):
    p02 = p01*(1-Y[j])
    for i in range(N):
        p2 = p01*pressure_ratio[i]
        fluid.update(cp.HmassP_INPUTS, h01, p02)
        s2 = fluid.smass()
        fluid.update(cp.PSmass_INPUTS, p2, s2)
        a2 = fluid.speed_sound()
        d2 = fluid.rhomass()
        h2 = fluid.hmass()
        v2 = np.sqrt(2*(h01-h2))
        phi2 = d2*v2 
        Ma2 = v2/a2
        phi[i] = phi2
        Ma[i] = Ma2
        s[i] = v2
        
    index = np.where(phi == max(phi))
    print(Ma[index])    
    ax.plot(Ma, phi, label = f'{Y[j]}')
    ax1.plot(Ma, s, label = f'{Y[j]}')

ax.legend()
# ax.set_xlim([0.5, 1.2])        

