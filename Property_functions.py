# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:06:09 2023

@author: lasseba
"""

import CoolProp.CoolProp as CP

def compute_viscosity(p, prop2, prop2_value, fluid):
    # Avoid viscosity computation in the two-phase region
    q = compute_quality(p, prop2, prop2_value, fluid)
    if 0 <= q <= 1:
        v = prop_calculation('V', 'P', p, 'Q', 1, fluid)
    else:
        v = prop_calculation('V', 'P', p, prop2, prop2_value, fluid)
    
    return v

def compute_quality(p, prop2, prop2_value, fluid):
    # Give an output for quality even if the pressure is supercritical
    p_crit = prop_calculation('P_CRITICAL', fluid)
    if p < p_crit:
        q = prop_calculation('Q', 'P', p, prop2, prop2_value, fluid)
    elif p >= p_crit:
        q = 1.1
        # print("Supercritical region occured")
    else:
        raise ValueError('Oops, something went wrong in quality computation')
    
    return q

def prop_calculation(*args):
    prop = CP.PropsSI(*args)
    return prop

def p_hs_flash(h, s, fluid, p_min, p_max):
    try:
        # CoolProp default h-s flash function
        # It is faster but may fail for some substances and states
        # It might require additional input arguments depending on the fluid
        p = prop_calculation('P', 'H', h, 'S', s, fluid)
    except:
        # This alternative uses optimization to find the pressure that satisfies the density equations
        # It is more robust but computationally more expensive
        # It requires the SciPy library to be installed
        from scipy.optimize import root_scalar

        rho_error = lambda p: prop_calculation('D', 'P', p, 'H', h, fluid) - prop_calculation('D', 'P', p, 'S', s, fluid)
        p = root_scalar(rho_error, x0 = p_min, x1 = p_max)
        
    
    return p

def compute_speed_of_sound(p, prop2, prop2_value, fluid):
    # Avoid speed of sound computation in the two-phase region
    q = compute_quality(p, prop2, prop2_value, fluid)
    if 0 <= q <= 1:
        a = prop_calculation('A', 'P', p, 'Q', 1, fluid)
    else:
        a = prop_calculation('A', 'P', p, prop2, prop2_value, fluid)
    
    return a

